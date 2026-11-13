import Foundation
import MLX
import MLXNN
import MLXRandom

/// Where the code predictor's per-frame time goes (`voice-bench cp-probe`).
public struct CodePredictorProbe: Sendable {
    /// Real model, one compiled 15-group frame per `eval` (as in `--profile`).
    public var frameMs: Double
    /// Real model, all frames queued before one `eval` (GPU time, host sync gaps hidden).
    public var frameQueuedMs: Double
    /// Same graph (5 layers, 15 groups, same vocab and sampling, 8-bit) with hidden 64,
    /// so arithmetic and weight traffic are negligible: the per-kernel overhead floor.
    public var tinyFrameMs: Double
    /// Weight bytes the real model reads per frame (layers + norm + one lm head, × 15).
    public var weightBytesPerFrame: Double
    /// Bandwidth of a large 8-bit matrix-vector product on this GPU (GB/s).
    public var qmvBandwidthGBs: Double
    /// Weight bytes one talker step reads (28 layers + codec head).
    public var talkerWeightBytesPerFrame: Double

    public var weightFloorMs: Double { weightBytesPerFrame / (qmvBandwidthGBs * 1e9) * 1000 }
    public var talkerWeightFloorMs: Double { talkerWeightBytesPerFrame / (qmvBandwidthGBs * 1e9) * 1000 }
}

extension Qwen3Voice {
    /// Diagnostics: split the code predictor's per-frame time into weight traffic and
    /// fixed per-kernel cost. Uses synthetic inputs; does not touch generation state.
    public func probeCodePredictor(frames: Int = 200, sampling: SamplingOptions = SamplingOptions()) -> CodePredictorProbe {
        let D = cpConfig.hiddenSize
        let hidden = MLXRandom.normal([1, 1, D], key: MLXRandom.key(1)).asType(computeDType)
        let code0 = MLXArray(Int32(100))
        func timeFrames(_ run: (MLXArray) -> MLXArray, queued: Bool) -> Double {
            for i in 0 ..< 5 { eval(run(MLXRandom.key(UInt64(i)))) }
            let t0 = now()
            var outs: [MLXArray] = []
            for i in 0 ..< frames {
                let r = run(MLXRandom.key(UInt64(1000 + i)))
                if queued { outs.append(r) } else { eval(r) }
            }
            if queued { eval(outs) }
            return (now() - t0) / Double(frames) * 1000
        }
        let real = { (k: MLXArray) in self.predictFrame(hidden: hidden, code0: code0, sampling: sampling, key: k) }
        let frameMs = timeFrames(real, queued: false)
        let frameQueuedMs = timeFrames(real, queued: true)

        // Tiny model with the same graph.
        var tc = cpConfig
        tc.hiddenSize = 64
        tc.embeddingDim = 64
        tc.numHeads = 2
        tc.numKVHeads = 1
        tc.headDim = 32
        tc.intermediateSize = 128
        let tiny = CodePredictorModel(config: tc)
        let cast = tiny.parameters().flattened().map { (k, v) in
            (k, v.dtype.isFloatingPoint ? v.asType(computeDType) : v)
        }
        tiny.update(parameters: ModuleParameters.unflattened(cast))
        eval(tiny)
        let greedy = !sampling.subtalkerDoSample
        let topK = greedy ? 0 : sampling.subtalkerTopK
        let tinyFn = compile(inputs: [tiny], outputs: [tiny], shapeless: false) { inputs in
            [tiny.predictFrame(inputsEmbeds: inputs[0], temperature: inputs[1], gumbels: inputs[2], topK: topK, greedy: greedy)]
        }
        let tinyIn = MLXRandom.normal([1, 2, 64], key: MLXRandom.key(2)).asType(computeDType)
        let temp = MLXArray(sampling.subtalkerTemperature)
        let groups = tc.numCodeGroups - 1
        let tinyRun = { (k: MLXArray) in tinyFn([tinyIn, temp, MLXRandom.gumbel([groups, tc.vocabSize], key: k)])[0] }
        let tinyFrameMs = timeFrames(tinyRun, queued: false)

        // Weight bytes per frame.
        func bytes(_ m: Module) -> Int { m.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } }
        let perGroup = codePredictor.layers.reduce(0) { $0 + bytes($1) } + bytes(codePredictor.norm)
            + bytes(codePredictor.lmHeads[0])
        let weightBytesPerFrame = Double(perGroup * groups)

        // Achievable 8-bit qmv bandwidth: [1, 1024] × [1024, 32768] products over two
        // alternating matrices (2 × 35 MB, beyond the system cache), best of 5 runs.
        let bigs = (0 ..< 2).map { _ in
            QuantizedLinear(1024, 32768, bias: false, groupSize: cpConfig.groupSize, bits: max(cpConfig.bits, 8))
        }
        for big in bigs {
            let cast = big.parameters().flattened().map { (k, v) in
                (k, v.dtype.isFloatingPoint ? v.asType(computeDType) : v)
            }
            big.update(parameters: ModuleParameters.unflattened(cast))
            eval(big)
        }
        let reps = 60
        let xs = (0 ..< reps).map { MLXRandom.normal([1, 1, 1024], key: MLXRandom.key(UInt64(10 + $0))).asType(computeDType) }
        eval(xs)
        eval(bigs[0](xs[0]), bigs[1](xs[1]))
        var best = Double.infinity
        for _ in 0 ..< 5 {
            let t0 = now()
            eval(xs.enumerated().map { bigs[$0.offset % 2]($0.element) })
            best = min(best, now() - t0)
        }
        let qmvBandwidthGBs = Double(bytes(bigs[0]) * reps) / best / 1e9

        let talkerBytes = talker.layers.reduce(0) { $0 + bytes($1) } + bytes(talker.codecHead)
        return CodePredictorProbe(
            frameMs: frameMs, frameQueuedMs: frameQueuedMs, tinyFrameMs: tinyFrameMs,
            weightBytesPerFrame: weightBytesPerFrame, qmvBandwidthGBs: qmvBandwidthGBs,
            talkerWeightBytesPerFrame: Double(talkerBytes))
    }
}
