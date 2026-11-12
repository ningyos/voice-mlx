import Foundation
import MLX

/// Errors raised by the SIM-o speaker-similarity scorer.
public enum SpeakerSimilarityError: Error, CustomStringConvertible, Equatable {
    case weightsNotFound(path: String)
    case weightsUnreadable(path: String, reason: String)
    case missingWeights(keys: [String])
    case shapeMismatch(key: String, expected: [Int], actual: [Int])
    case emptyAudio
    case audioTooShort(samples: Int, minimumSamples: Int)
    case invalidSampleRate(Int)
    case embeddingDimensionMismatch(expected: Int, actual: Int)
    case invalidParityList(path: String, line: Int, reason: String)

    public var description: String {
        switch self {
        case .weightsNotFound(let path):
            return "SIM weights not found at \(path) (export with scripts/export_sim_weights.py)"
        case .weightsUnreadable(let path, let reason):
            return "SIM weights at \(path) could not be read: \(reason)"
        case .missingWeights(let keys):
            let shown = keys.prefix(8).joined(separator: ", ")
            return "SIM weights are missing \(keys.count) tensor(s): \(shown)\(keys.count > 8 ? ", ..." : "")"
        case .shapeMismatch(let key, let expected, let actual):
            return "SIM weight \(key) has shape \(actual), expected \(expected)"
        case .emptyAudio:
            return "audio is empty"
        case .audioTooShort(let samples, let minimumSamples):
            return "audio has \(samples) samples at 16 kHz; the WavLM CNN needs at least \(minimumSamples)"
        case .invalidSampleRate(let rate):
            return "invalid sample rate \(rate)"
        case .embeddingDimensionMismatch(let expected, let actual):
            return "embedding has \(actual) values, expected \(expected)"
        case .invalidParityList(let path, let line, let reason):
            return "parity list \(path):\(line): \(reason)"
        }
    }
}

/// Validated view over the exported `ECAPA_TDNN_WAVLM` state dict.
struct SimWeights {
    private let arrays: [String: MLXArray]

    static let wavlmPrefix = "feature_extract.model."
    static let numLayers = 24
    static let embedDim = 1024
    static let ffnDim = 4096
    static let numHeads = 16
    static let numBuckets = 320
    static let convLayers: [(dim: Int, kernel: Int, stride: Int)] =
        [(512, 10, 5)] + Array(repeating: (512, 3, 2), count: 4) + Array(repeating: (512, 2, 2), count: 2)
    static let posConvKernel = 128
    static let posConvGroups = 16
    static let channels = 512
    static let catChannels = 1536
    static let attentionChannels = 128
    static let seBottleneck = 128
    static let res2Scale = 8
    static let embeddingDim = 256
    static let numHiddenStates = numLayers + 1

    /// Every tensor the forward pass reads, with its PyTorch shape.
    static func expectedShapes() -> [String: [Int]] {
        var s: [String: [Int]] = [:]
        let w = wavlmPrefix
        var inDim = 1
        for (i, layer) in convLayers.enumerated() {
            s["\(w)feature_extractor.conv_layers.\(i).0.weight"] = [layer.dim, inDim, layer.kernel]
            s["\(w)feature_extractor.conv_layers.\(i).2.1.weight"] = [layer.dim]
            s["\(w)feature_extractor.conv_layers.\(i).2.1.bias"] = [layer.dim]
            inDim = layer.dim
        }
        s["\(w)layer_norm.weight"] = [512]
        s["\(w)layer_norm.bias"] = [512]
        s["\(w)post_extract_proj.weight"] = [embedDim, 512]
        s["\(w)post_extract_proj.bias"] = [embedDim]
        s["\(w)encoder.pos_conv.0.weight_g"] = [1, 1, posConvKernel]
        s["\(w)encoder.pos_conv.0.weight_v"] = [embedDim, embedDim / posConvGroups, posConvKernel]
        s["\(w)encoder.pos_conv.0.bias"] = [embedDim]
        s["\(w)encoder.layer_norm.weight"] = [embedDim]
        s["\(w)encoder.layer_norm.bias"] = [embedDim]
        let headDim = embedDim / numHeads
        for l in 0..<numLayers {
            let p = "\(w)encoder.layers.\(l)."
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                s["\(p)self_attn.\(proj).weight"] = [embedDim, embedDim]
                s["\(p)self_attn.\(proj).bias"] = [embedDim]
            }
            s["\(p)self_attn.grep_linear.weight"] = [8, headDim]
            s["\(p)self_attn.grep_linear.bias"] = [8]
            s["\(p)self_attn.grep_a"] = [1, numHeads, 1, 1]
            if l == 0 {
                s["\(p)self_attn.relative_attention_bias.weight"] = [numBuckets, numHeads]
            }
            for ln in ["self_attn_layer_norm", "final_layer_norm"] {
                s["\(p)\(ln).weight"] = [embedDim]
                s["\(p)\(ln).bias"] = [embedDim]
            }
            s["\(p)fc1.weight"] = [ffnDim, embedDim]
            s["\(p)fc1.bias"] = [ffnDim]
            s["\(p)fc2.weight"] = [embedDim, ffnDim]
            s["\(p)fc2.bias"] = [embedDim]
        }
        s["feature_weight"] = [numHiddenStates]

        func bn(_ p: String, _ c: Int) {
            for n in ["weight", "bias", "running_mean", "running_var"] { s["\(p).\(n)"] = [c] }
        }
        func conv(_ p: String, out: Int, in inC: Int, k: Int) {
            s["\(p).weight"] = [out, inC, k]
            s["\(p).bias"] = [out]
        }
        conv("layer1.conv", out: channels, in: embedDim, k: 5)
        bn("layer1.bn", channels)
        let width = channels / res2Scale
        for b in 2...4 {
            let p = "layer\(b)"
            conv("\(p).Conv1dReluBn1.conv", out: channels, in: channels, k: 1)
            bn("\(p).Conv1dReluBn1.bn", channels)
            for i in 0..<(res2Scale - 1) {
                conv("\(p).Res2Conv1dReluBn.convs.\(i)", out: width, in: width, k: 3)
                bn("\(p).Res2Conv1dReluBn.bns.\(i)", width)
            }
            conv("\(p).Conv1dReluBn2.conv", out: channels, in: channels, k: 1)
            bn("\(p).Conv1dReluBn2.bn", channels)
            s["\(p).SE_Connect.linear1.weight"] = [seBottleneck, channels]
            s["\(p).SE_Connect.linear1.bias"] = [seBottleneck]
            s["\(p).SE_Connect.linear2.weight"] = [channels, seBottleneck]
            s["\(p).SE_Connect.linear2.bias"] = [channels]
        }
        conv("conv", out: catChannels, in: catChannels, k: 1)
        conv("pooling.linear1", out: attentionChannels, in: catChannels, k: 1)
        conv("pooling.linear2", out: catChannels, in: attentionChannels, k: 1)
        bn("bn", catChannels * 2)
        s["linear.weight"] = [embeddingDim, catChannels * 2]
        s["linear.bias"] = [embeddingDim]
        return s
    }

    init(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SpeakerSimilarityError.weightsNotFound(path: url.path)
        }
        let loaded: [String: MLXArray]
        do {
            loaded = try loadArrays(url: url)
        } catch {
            throw SpeakerSimilarityError.weightsUnreadable(path: url.path, reason: "\(error)")
        }
        // Shape errors on present tensors are reported first (they point at the wrong
        // model/export), then any absent tensors.
        let expected = Self.expectedShapes()
        var arrays: [String: MLXArray] = [:]
        var missing: [String] = []
        for (key, shape) in expected.sorted(by: { $0.key < $1.key }) {
            guard let a = loaded[key] else {
                missing.append(key)
                continue
            }
            if a.shape != shape {
                throw SpeakerSimilarityError.shapeMismatch(key: key, expected: shape, actual: a.shape)
            }
            arrays[key] = a.dtype == .float32 ? a : a.asType(.float32)
        }
        if !missing.isEmpty { throw SpeakerSimilarityError.missingWeights(keys: missing) }
        self.arrays = arrays
    }

    subscript(_ key: String) -> MLXArray { arrays[key]! }
}
