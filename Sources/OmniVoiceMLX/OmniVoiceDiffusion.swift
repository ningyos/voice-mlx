import Foundation
import MLX
import MLXNN
import MLXRandom

/// Knobs of the iterative-unmasking decode. Defaults mirror the reference
/// `OmniVoiceGenerationConfig` except `steps` (16 instead of 32, see README).
public struct DiffusionOptions: Sendable, Equatable {
    /// Number of unmasking steps (`num_step`).
    public var steps: Int = 16
    /// Classifier-free guidance scale in log-prob space (`guidance_scale`); 0 disables CFG.
    public var guidance: Float = 2.0
    /// Time-step shift of the unmasking schedule (`t_shift`).
    public var tShift: Float = 0.1
    /// Penalty per codebook index so earlier codebooks unmask first (`layer_penalty_factor`).
    public var layerPenalty: Float = 5.0
    /// Gumbel temperature on the position scores (`position_temperature`); 0 = greedy.
    public var positionTemperature: Float = 5.0
    /// Gumbel temperature on the token choice (`class_temperature`); 0 = argmax.
    public var classTemperature: Float = 0.0
    /// Fraction of the vocabulary kept before class sampling (`_filter_top_k` ratio).
    public var classTopKRatio: Float = 0.1

    public init() {}

    /// True when decoding draws random numbers and therefore needs a seed.
    public var isStochastic: Bool { positionTemperature > 0 || classTemperature > 0 }

    /// Fully deterministic decoding (both temperatures 0) — used for parity.
    public static var greedy: DiffusionOptions {
        var o = DiffusionOptions()
        o.positionTemperature = 0
        o.classTemperature = 0
        return o
    }
}

/// Wall-clock accumulator for named stages. When attached to a decode it also
/// forces evaluation at stage boundaries, which serialises GPU work that would
/// otherwise overlap — use it to attribute time, not to measure end-to-end latency.
public final class StageProfiler: @unchecked Sendable {
    public private(set) var totals: [String: Double] = [:]
    public private(set) var counts: [String: Int] = [:]
    public private(set) var order: [String] = []

    public init() {}

    @discardableResult
    public func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let r = try body()
        add(name, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return r
    }

    public func add(_ name: String, _ seconds: Double) {
        if totals[name] == nil { order.append(name) }
        totals[name, default: 0] += seconds
        counts[name, default: 0] += 1
    }

    public func reset() {
        totals = [:]
        counts = [:]
        order = []
    }
}

extension OmniVoiceModel {
    /// Number of tokens unmasked at each step; mirrors `_generate_iterative`.
    static func unmaskSchedule(totalMask: Int, steps: Int, tShift: Float) -> [Int] {
        // `torch.linspace(0, 1, steps + 1)` in float32, then the shift, in float32.
        let ts: [Float] = (0 ... steps).map {
            let t = Float($0) / Float(steps)
            return tShift * t / (1 + (tShift - 1) * t)
        }
        var schedule = [Int]()
        var rem = totalMask
        for step in 0 ..< steps {
            let n = step == steps - 1
                ? rem
                : min(Int(ceil(Double(totalMask) * (Double(ts[step + 1]) - Double(ts[step])))), rem)
            schedule.append(n)
            rem -= n
        }
        return schedule
    }

    /// N-step iterative-unmasking diffusion decode → audio token matrix `[1, C, T]`.
    ///
    /// Each step runs a cond forward (full style+text+ref+target context) and an
    /// uncond forward (target only), combines them with classifier-free guidance
    /// in log-prob space, scores each still-masked position by confidence minus
    /// a per-codebook penalty (plus Gumbel noise when `positionTemperature > 0`),
    /// and unmasks the top-k positions. Port of `OmniVoice._generate_iterative`
    /// for batch size 1; the reference's batched uncond row attends only to its
    /// own `T` positions, so running it as a separate length-`T` forward is exact.
    ///
    /// - Parameters:
    ///   - condInputIds: `[1, C, condLen]` with the target region all-mask.
    ///   - audioMask: `[1, condLen]` (0/1).
    ///   - targetLen: number of audio frames to generate (the trailing positions).
    ///   - key: MLX PRNG key; required when `options.isStochastic`.
    ///   - profiler: when set, evaluates and times cond / uncond / select per step.
    public func generateTokens(
        condInputIds: MLXArray, audioMask: MLXArray, targetLen: Int,
        options: DiffusionOptions, key: MLXArray?, profiler: StageProfiler? = nil
    ) -> MLXArray {
        let C = cfg.numAudioCodebook
        let V = cfg.audioVocabSize
        let maskId = Int32(cfg.audioMaskId)
        let condLen = condInputIds.dim(2)
        let tStart = condLen - targetLen
        precondition(options.steps > 0 && targetLen > 0 && tStart >= 0)
        precondition(!options.isStochastic || key != nil, "stochastic decoding needs a PRNG key")

        var tokens = MLXArray.full([1, C, targetLen], values: MLXArray(maskId))
        var cond = condInputIds
        cond[0..., 0..., tStart ..< condLen] = tokens
        let uncondMask = MLXArray.ones([1, targetLen], type: Int32.self)
        let layerPenalty = (MLXArray(0 ..< C).asType(.float32) * options.layerPenalty)
            .reshaped([1, C, 1])
        let schedule = Self.unmaskSchedule(
            totalMask: targetLen * C, steps: options.steps, tShift: options.tShift)

        let negInf = MLXArray(-Float.infinity)
        // Additive column that forbids the mask token (`log_probs[..., mask] = -inf`).
        let maskColumn = MLX.where(
            MLXArray(0 ..< V) .== MLXArray(maskId), negInf, MLXArray(Float(0)))
        var rngKey = key

        for step in 0 ..< options.steps {
            let k = schedule[step]
            if k <= 0 { continue }

            var condLogits = self(inputIds: cond, audioMask: audioMask, lastPositions: targetLen)
                                                                              // [1, C, T, V]
            if let p = profiler { p.measure("diffusion.cond_forward") { eval(condLogits) } }
            var logProbs: MLXArray
            condLogits = condLogits.asType(.float32)
            if options.guidance != 0 {
                let uncondLogits = self(inputIds: tokens, audioMask: uncondMask).asType(.float32)
                if let p = profiler { p.measure("diffusion.uncond_forward") { eval(uncondLogits) } }
                let cLog = logSoftmax(condLogits, axis: -1)
                let uLog = logSoftmax(uncondLogits, axis: -1)
                logProbs = logSoftmax(cLog + options.guidance * (cLog - uLog), axis: -1)
            } else {
                logProbs = logSoftmax(condLogits, axis: -1)
            }
            logProbs = logProbs + maskColumn

            var stepKey: MLXArray? = nil
            if options.isStochastic, let kk = rngKey {
                let (next, sub) = MLXRandom.split(key: kk)
                rngKey = next
                stepKey = sub
            }
            let (posKey, classKey): (MLXArray?, MLXArray?) = stepKey.map {
                let (a, b) = MLXRandom.split(key: $0)
                return (a, b)
            } ?? (nil, nil)

            let predTokens: MLXArray
            if options.classTemperature > 0, let ck = classKey {
                let keep = Int(ceil(Double(options.classTopKRatio) * Double(V)))
                let kth = MLX.sorted(logProbs, axis: -1)[.ellipsis, (V - keep) ..< (V - keep + 1)]
                let filtered = MLX.where(logProbs .>= kth, logProbs, negInf)
                predTokens = Self.gumbel(filtered, temperature: options.classTemperature, key: ck)
                    .argMax(axis: -1).asType(.int32)
            } else {
                predTokens = logProbs.argMax(axis: -1).asType(.int32)                // [1, C, T]
            }
            var scores = logProbs.max(axis: -1) - layerPenalty                       // [1, C, T]
            if options.positionTemperature > 0, let pk = posKey {
                scores = Self.gumbel(scores, temperature: options.positionTemperature, key: pk)
            }
            scores = MLX.where(tokens .!= MLXArray(maskId), negInf, scores)

            // Top-k positions to unmask this step.
            let flat = scores.reshaped([C * targetLen])
            let topk = MLX.argSort(-flat)[0 ..< k]
            var pick = MLXArray.zeros([C * targetLen], type: Bool.self)
            pick[topk] = MLXArray.ones([k], type: Bool.self)

            tokens = MLX.where(pick.reshaped([1, C, targetLen]), predTokens, tokens)
            cond[0..., 0..., tStart ..< condLen] = tokens
            if let p = profiler {
                p.measure("diffusion.cfg_select") { eval(tokens, cond) }
            } else {
                eval(tokens, cond)
            }
        }
        return tokens
    }

    /// `_gumbel_sample`: `logits / T + (-log(-log(u + 1e-10) + 1e-10))`.
    static func gumbel(_ logits: MLXArray, temperature: Float, key: MLXArray) -> MLXArray {
        let u = MLXRandom.uniform(0 ..< 1, logits.shape, key: key)
        let g = -MLX.log(-MLX.log(u + 1e-10) + 1e-10)
        return logits / temperature + g
    }
}
