import Foundation
import MLX
import MLXFast
import MLXNN

/// MLX port of OmniVoice's `ECAPA_TDNN_WAVLM` (s3prl WavLM-large upstream with gated
/// relative position bias, softmax-weighted sum of the 25 hidden states, instance norm,
/// ECAPA-TDNN head). Activations use the `[batch, time, channels]` layout; PyTorch conv
/// weights `[out, in, k]` are transposed once at load time to MLX's `[out, k, in]`.
final class SimModel {
    private let w: SimWeights

    // Pre-transposed conv weights keyed by state-dict prefix.
    private var convWeights: [String: MLXArray] = [:]
    private let posConvWeight: MLXArray
    private let hiddenWeights: [Float]

    /// Max query rows per attention block; bounds the `[heads, rows, T]` score tensor
    /// (~200 MB at the 120 s / 6000-frame limit). Inputs up to ~10 s use one block.
    private let attentionChunk = 512

    init(weights: SimWeights) {
        self.w = weights
        let p = SimWeights.wavlmPrefix
        for i in 0..<SimWeights.convLayers.count {
            let key = "\(p)feature_extractor.conv_layers.\(i).0"
            convWeights[key] = weights["\(key).weight"].transposed(0, 2, 1)
        }
        var ecapaConvs = ["layer1.conv", "conv", "pooling.linear1", "pooling.linear2"]
        for b in 2...4 {
            ecapaConvs.append("layer\(b).Conv1dReluBn1.conv")
            ecapaConvs.append("layer\(b).Conv1dReluBn2.conv")
            for i in 0..<(SimWeights.res2Scale - 1) {
                ecapaConvs.append("layer\(b).Res2Conv1dReluBn.convs.\(i)")
            }
        }
        for key in ecapaConvs {
            convWeights[key] = weights["\(key).weight"].transposed(0, 2, 1)
        }

        // torch.nn.utils.weight_norm(dim=2): weight = g * v / ||v||, norm over dims (0, 1).
        let g = weights["\(p)encoder.pos_conv.0.weight_g"]
        let v = weights["\(p)encoder.pos_conv.0.weight_v"]
        let norm = sqrt(sum(v * v, axes: [0, 1], keepDims: true))
        posConvWeight = (g * v / norm).transposed(0, 2, 1)

        let fw = softmax(weights["feature_weight"], axis: -1, precise: true)
        hiddenWeights = fw.asArray(Float.self)

        var all = Array(convWeights.values)
        all.append(posConvWeight)
        eval(all)
    }

    /// Minimum number of 16 kHz samples that yields at least one WavLM frame.
    static let minimumSamples: Int = {
        var n = 1
        for layer in SimWeights.convLayers.reversed() { n = (n - 1) * layer.stride + layer.kernel }
        return n
    }()

    static func frameCount(samples: Int) -> Int {
        var n = samples
        for layer in SimWeights.convLayers {
            n = n < layer.kernel ? 0 : (n - layer.kernel) / layer.stride + 1
        }
        return n
    }

    // MARK: - Primitives

    private func linear(_ x: MLXArray, _ prefix: String) -> MLXArray {
        addMM(w["\(prefix).bias"], x, w["\(prefix).weight"].T)
    }

    private func conv(
        _ x: MLXArray, _ prefix: String, padding: Int = 0, dilation: Int = 1, bias: Bool = true
    ) -> MLXArray {
        let y = conv1d(x, convWeights[prefix]!, padding: padding, dilation: dilation)
        return bias ? y + w["\(prefix).bias"] : y
    }

    private func layerNorm(_ x: MLXArray, _ prefix: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w["\(prefix).weight"], bias: w["\(prefix).bias"], eps: 1e-5)
    }

    /// BatchNorm1d in eval mode over the channel (last) axis.
    private func batchNorm(_ x: MLXArray, _ prefix: String) -> MLXArray {
        let scale = w["\(prefix).weight"] * rsqrt(w["\(prefix).running_var"] + 1e-5)
        return (x - w["\(prefix).running_mean"]) * scale + w["\(prefix).bias"]
    }

    // MARK: - WavLM

    /// s3prl relative position bucket for every offset `key - query` in `-(T-1)...(T-1)`,
    /// computed in Float32 like the PyTorch reference.
    private func bucketTable(_ t: Int) -> MLXArray {
        let numBuckets = SimWeights.numBuckets / 2
        let maxExact = numBuckets / 2
        let maxDistance: Float = 800
        let logRatio = Float(log(Double(maxDistance) / Double(maxExact)))
        var table = [Int32](repeating: 0, count: 2 * t - 1)
        for i in 0..<(2 * t - 1) {
            let rel = i - (t - 1)
            var bucket = rel > 0 ? numBuckets : 0
            let a = abs(rel)
            if a < maxExact {
                bucket += a
            } else {
                let scaled = Foundation.log(Float(a) / Float(maxExact)) / logRatio
                    * Float(numBuckets - maxExact)
                bucket += min(maxExact + Int(scaled), numBuckets - 1)
            }
            table[i] = Int32(bucket)
        }
        return MLXArray(table)
    }

    /// One pre-LN transformer layer with gated relative position bias.
    /// `biasTable` is the ungated relative position bias per head and offset, `[H, 2T-1]`
    /// (layer 0's embedding; s3prl shares it across layers, each layer applies its gate).
    private func encoderLayer(_ x: MLXArray, layer l: Int, biasTable: MLXArray, t: Int) -> MLXArray {
        let p = "\(SimWeights.wavlmPrefix)encoder.layers.\(l)."
        let heads = SimWeights.numHeads
        let headDim = SimWeights.embedDim / heads

        let h = layerNorm(x, "\(p)self_attn_layer_norm")  // [1, T, C]
        func heads4(_ a: MLXArray) -> MLXArray {  // [1, T, C] -> [1, H, T, D]
            a.reshaped(1, t, heads, headDim).transposed(0, 2, 1, 3)
        }
        let q = heads4(linear(h, "\(p)self_attn.q_proj"))
        let k = heads4(linear(h, "\(p)self_attn.k_proj"))
        let v = heads4(linear(h, "\(p)self_attn.v_proj"))
        let scale = Float(1.0 / Double(headDim).squareRoot())

        // Gate from the un-projected (layer-normed) query split into heads.
        let qh = h.reshaped(t, heads, headDim).transposed(1, 0, 2)  // [H, T, D]
        let g = linear(qh, "\(p)self_attn.grep_linear").reshaped(heads, t, 2, 4).sum(axis: -1)
        let gates = sigmoid(g)
        let gateA = gates[0..., 0..., 0 ..< 1]
        let gateB = gates[0..., 0..., 1 ..< 2]
        let grepA = w["\(p)self_attn.grep_a"].reshaped(heads, 1, 1)
        let gate = gateA * (gateB * grepA - 1.0) + 2.0  // [H, T, 1]

        let keyPos = MLXArray(Int32(0) ..< Int32(t)).reshaped(1, -1)
        var outs: [MLXArray] = []
        var start = 0
        while start < t {
            let end = min(start + attentionChunk, t)
            let queryPos = MLXArray(Int32(start) ..< Int32(end)).reshaped(-1, 1)
            let idx = keyPos - queryPos + Int32(t - 1)  // [c, T] offset index
            let bias = take(biasTable, idx, axis: 1)  // [H, c, T]
            let mask = (gate[0..., start ..< end] * bias).expandedDimensions(axis: 0)  // [1, H, c, T]
            let out = MLX.scaledDotProductAttention(
                queries: q[0..., 0..., start ..< end], keys: k, values: v, scale: scale, mask: mask)
            // Evaluate per block so only one block's scores are alive at a time.
            if t > attentionChunk { eval(out) }
            outs.append(out)
            start = end
        }
        let attn = (outs.count == 1 ? outs[0] : concatenated(outs, axis: 2))  // [1, H, T, D]
            .transposed(0, 2, 1, 3).reshaped(1, t, SimWeights.embedDim)
        var y = x + linear(attn, "\(p)self_attn.out_proj")

        let f = layerNorm(y, "\(p)final_layer_norm")
        y = y + linear(gelu(linear(f, "\(p)fc1")), "\(p)fc2")
        return y
    }

    /// Softmax-weighted sum of the 25 WavLM hidden states, `[1, T, 1024]`.
    private func wavlmFeatures(_ wav: MLXArray) -> MLXArray {
        let p = SimWeights.wavlmPrefix
        // cfg.normalize: F.layer_norm over the whole waveform.
        var x = MLXFast.layerNorm(wav, weight: nil, bias: nil, eps: 1e-5).reshaped(1, -1, 1)
        for (i, layer) in SimWeights.convLayers.enumerated() {
            let key = "\(p)feature_extractor.conv_layers.\(i)"
            x = conv1d(x, convWeights["\(key).0"]!, stride: layer.stride)
            x = gelu(layerNorm(x, "\(key).2.1"))
            eval(x)  // bounds peak memory on long inputs (first block is [N/5, 512])
        }
        x = layerNorm(x, "\(p)layer_norm")
        x = linear(x, "\(p)post_extract_proj")
        let t = x.dim(1)

        let kernel = SimWeights.posConvKernel
        var pc = conv1d(x, posConvWeight, padding: kernel / 2, groups: SimWeights.posConvGroups)
        pc = pc + w["\(p)encoder.pos_conv.0.bias"]
        pc = gelu(pc[0..., 0 ..< t])  // SamePad drops the trailing frame (even kernel)
        x = x + pc
        eval(x)

        let relEmb = w["\(p)encoder.layers.0.self_attn.relative_attention_bias.weight"]
        let biasTable = take(relEmb, bucketTable(t), axis: 0).T  // [H, 2T-1]
        eval(biasTable)
        var acc = hiddenWeights[0] * x
        for l in 0..<SimWeights.numLayers {
            x = encoderLayer(x, layer: l, biasTable: biasTable, t: t)
            if l == SimWeights.numLayers - 1 {
                x = layerNorm(x, "\(p)encoder.layer_norm")
            }
            acc = acc + hiddenWeights[l + 1] * x
            eval(x, acc)
        }
        return acc
    }

    // MARK: - ECAPA-TDNN

    private func convReluBn(
        _ x: MLXArray, _ prefix: String, padding: Int = 0, dilation: Int = 1
    ) -> MLXArray {
        batchNorm(relu(conv(x, "\(prefix).conv", padding: padding, dilation: dilation)), "\(prefix).bn")
    }

    private func seRes2Block(_ x: MLXArray, _ prefix: String, dilation: Int) -> MLXArray {
        var y = convReluBn(x, "\(prefix).Conv1dReluBn1")
        let width = SimWeights.channels / SimWeights.res2Scale
        let parts = split(y, parts: SimWeights.res2Scale, axis: -1)
        var outs: [MLXArray] = []
        var sp = parts[0]
        for i in 0..<(SimWeights.res2Scale - 1) {
            if i > 0 { sp = sp + parts[i] }
            let c = "\(prefix).Res2Conv1dReluBn"
            sp = conv(sp, "\(c).convs.\(i)", padding: dilation, dilation: dilation)
            sp = batchNorm(relu(sp), "\(c).bns.\(i)")
            outs.append(sp)
        }
        outs.append(parts[SimWeights.res2Scale - 1])
        precondition(width * SimWeights.res2Scale == SimWeights.channels)
        y = concatenated(outs, axis: -1)
        y = convReluBn(y, "\(prefix).Conv1dReluBn2")

        let s = y.mean(axis: 1)  // [1, C]
        let e = sigmoid(linear(relu(linear(s, "\(prefix).SE_Connect.linear1")), "\(prefix).SE_Connect.linear2"))
        return y * e.expandedDimensions(axis: 1) + x
    }

    private func ecapa(_ features: MLXArray) -> MLXArray {
        // get_feat: + 1e-6, then InstanceNorm1d (no affine, biased variance) over time.
        var x = features + 1e-6
        let mean = x.mean(axis: 1, keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axis: 1, keepDims: true)
        x = (x - mean) * rsqrt(variance + 1e-5)

        let out1 = convReluBn(x, "layer1", padding: 2)
        let out2 = seRes2Block(out1, "layer2", dilation: 2)
        let out3 = seRes2Block(out2, "layer3", dilation: 3)
        let out4 = seRes2Block(out3, "layer4", dilation: 4)
        var out = relu(conv(concatenated([out2, out3, out4], axis: -1), "conv"))

        // Attentive statistics pooling (softmax over time).
        let alpha = softmax(
            conv(tanh(conv(out, "pooling.linear1")), "pooling.linear2"), axis: 1, precise: true)
        let mu = sum(alpha * out, axis: 1)
        let residuals = sum(alpha * out * out, axis: 1) - mu * mu
        let sd = sqrt(maximum(residuals, 1e-9))
        out = concatenated([mu, sd], axis: -1)  // [1, 3072]
        out = batchNorm(out, "bn")
        return linear(out, "linear")  // [1, 256]
    }

    /// Embedding for one 16 kHz waveform, `[256]`.
    func embed(samples16k: [Float]) -> [Float] {
        let features = wavlmFeatures(MLXArray(samples16k))
        let e = ecapa(features)
        eval(e)
        return e.reshaped(-1).asArray(Float.self)
    }
}
