import Foundation

public enum AudioPostprocessError: Error, CustomStringConvertible, Equatable {
    /// Mirrors the `ValueError` raised by `OmniVoice.create_voice_clone_prompt`.
    case emptyAfterSilenceRemoval

    public var description: String {
        switch self {
        case .emptyAfterSilenceRemoval:
            return "Reference audio is empty after silence removal. Try setting preprocess=false."
        }
    }
}

/// Swift port of OmniVoice's audio pre/post-processing
/// (`omnivoice/utils/audio.py` and the helpers inside `OmniVoice.create_voice_clone_prompt`
/// / `OmniVoice._post_process_audio`).
///
/// Silence handling reproduces pydub exactly: the float signal is quantized to int16
/// (`(x * 32768).clip(-32768, 32767).astype(int16)`), all positions are milliseconds that
/// map to frames the way `AudioSegment.__getitem__` does (banker's-rounded segment length,
/// truncated `ms * rate / 1000` frame index, up to 2 ms of zero padding at the end), and
/// loudness uses `audioop.rms` (truncated integer RMS). Outputs of `removeSilence` are
/// therefore `int16 / 32768` values.
///
/// Float arithmetic mirrors numpy 2 (NEP 50) float32 semantics, including numpy's pairwise
/// float32 summation in `rms`.
public enum AudioPostprocess {
    /// pydub's `AudioSegment.silent(duration=0)` uses this frame rate; concatenating real
    /// segments onto it only avoids `audioop.ratecv` when the real rate is at least this.
    static let pydubSilentFrameRate = 11025

    // MARK: - numpy-compatible helpers

    /// `float(np.sqrt(np.mean(x ** 2)))` for a float32 array: squares in float32, numpy's
    /// pairwise float32 sum, float32 division by the count. Empty input yields NaN like numpy.
    public static func rms(_ s: [Float]) -> Float {
        var squares = [Float](repeating: 0, count: s.count)
        for i in s.indices { squares[i] = s[i] * s[i] }
        let total: Float = squares.withUnsafeBufferPointer { 0 + pairwiseSum($0[...]) }
        return (total / Float(s.count)).squareRoot()
    }

    /// numpy `@TYPE@_pairwise_sum` (8-way unrolled, block size 128) with a float32 accumulator.
    static func pairwiseSum(_ a: Slice<UnsafeBufferPointer<Float>>) -> Float {
        let n = a.count
        let b = a.startIndex
        if n < 8 {
            var res: Float = 0
            for i in 0 ..< n { res += a[b + i] }
            return res
        } else if n <= 128 {
            var r = SIMD8<Float>(a[b ..< b + 8])
            var i = 8
            while i < n - (n % 8) {
                r += SIMD8<Float>(a[(b + i) ..< (b + i + 8)])
                i += 8
            }
            var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n {
                res += a[b + i]
                i += 1
            }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwiseSum(a[b ..< b + n2]) + pairwiseSum(a[(b + n2)...])
        }
    }

    // MARK: - Silence removal (pydub port)

    /// `omnivoice.utils.audio.remove_silence`: `split_on_silence(min_silence_len=mid,
    /// silence_thresh=-50, keep_silence=mid, seek_step=10)` (skipped when `midSilenceMs == 0`),
    /// concatenation, then `remove_silence_edges(lead, trail, -50)`.
    ///
    /// Requires `sampleRate >= 11025` when `midSilenceMs > 0`: below that pydub would resample
    /// the concatenation with `audioop.ratecv`, which is not ported.
    public static func removeSilence(
        _ s: [Float], sampleRate: Int, midSilenceMs: Int, leadSilenceMs: Int, trailSilenceMs: Int,
        silenceThresholdDB: Double = -50, seekStepMs: Int = 10, edgeChunkMs: Int = 10
    ) -> [Float] {
        precondition(sampleRate > 0, "sample rate must be positive")
        precondition(midSilenceMs >= 0 && leadSilenceMs >= 0 && trailSilenceMs >= 0,
                     "silence durations must be non-negative")
        var wave = PCM16Segment(samples: quantizeInt16(s), frameRate: sampleRate)

        if midSilenceMs > 0 {
            let ranges = splitOnSilence(
                wave, minSilenceLen: midSilenceMs, silenceThresh: silenceThresholdDB,
                keepSilence: midSilenceMs, seekStep: seekStepMs)
            if !ranges.isEmpty {
                precondition(sampleRate >= pydubSilentFrameRate,
                             "pydub resamples segments below \(pydubSilentFrameRate) Hz; unsupported")
            }
            var joined: [Int16] = []
            joined.reserveCapacity(wave.samples.count)
            for (start, end) in ranges { wave.appendSlice(startMs: start, endMs: end, to: &joined) }
            wave = PCM16Segment(samples: joined, frameRate: sampleRate)
        }

        wave = removeSilenceEdges(wave, lead: leadSilenceMs, trail: trailSilenceMs,
                                  threshold: silenceThresholdDB, chunkSize: edgeChunkMs)
        return wave.samples.map { Float($0) / 32768 }
    }

    /// `(x * 32768.0).clip(-32768, 32767).astype(np.int16)` in float32.
    static func quantizeInt16(_ s: [Float]) -> [Int16] {
        s.map { x in
            let v = min(max(x * 32768, -32768), 32767)
            return Int16(v)  // truncation toward zero, like astype
        }
    }

    /// `remove_silence_edges`.
    static func removeSilenceEdges(
        _ audio: PCM16Segment, lead: Int, trail: Int, threshold: Double, chunkSize: Int
    ) -> PCM16Segment {
        var a = audio
        var start = detectLeadingSilence(a, threshold: threshold, chunkSize: chunkSize)
        start = max(0, start - lead)
        a = a.slice(startMs: start, endMs: nil)
        a.samples.reverse()
        start = detectLeadingSilence(a, threshold: threshold, chunkSize: chunkSize)
        start = max(0, start - trail)
        a = a.slice(startMs: start, endMs: nil)
        a.samples.reverse()
        return a
    }

    /// `pydub.silence.detect_leading_silence`.
    static func detectLeadingSilence(_ sound: PCM16Segment, threshold: Double, chunkSize: Int) -> Int {
        precondition(chunkSize > 0, "chunk size must be positive")
        let len = sound.lengthMs
        var trimMs = 0
        while sound.dBFS(startMs: trimMs, endMs: trimMs + chunkSize) < threshold && trimMs < len {
            trimMs += chunkSize
        }
        return min(trimMs, len)
    }

    /// `pydub.silence.detect_silence`; returns `[start, end]` ms ranges.
    static func detectSilence(
        _ seg: PCM16Segment, minSilenceLen: Int, silenceThresh: Double, seekStep: Int
    ) -> [(Int, Int)] {
        precondition(seekStep > 0, "seek step must be positive")
        let segLen = seg.lengthMs
        if segLen < minSilenceLen { return [] }
        let thresh = pow(10.0, silenceThresh / 20) * PCM16Segment.maxPossibleAmplitude

        let lastSliceStart = segLen - minSilenceLen
        var sliceStarts = Array(stride(from: 0, through: lastSliceStart, by: seekStep))
        if lastSliceStart % seekStep != 0 { sliceStarts.append(lastSliceStart) }

        // Sliding windows overlap heavily (seek 10 ms vs. window = min silence); squared int16
        // values are integers, so prefix sums give the exact same sum as audioop's double loop.
        let prefix = seg.squarePrefixSums()
        var silenceStarts: [Int] = []
        for i in sliceStarts
        where Double(seg.rms(startMs: i, endMs: i + minSilenceLen, prefix: prefix)) <= thresh {
            silenceStarts.append(i)
        }
        guard var prev = silenceStarts.first else { return [] }

        var ranges: [(Int, Int)] = []
        var currentStart = prev
        for s in silenceStarts.dropFirst() {
            let continuous = s == prev + seekStep
            let hasGap = s > prev + minSilenceLen
            if !continuous && hasGap {
                ranges.append((currentStart, prev + minSilenceLen))
                currentStart = s
            }
            prev = s
        }
        ranges.append((currentStart, prev + minSilenceLen))
        return ranges
    }

    /// `pydub.silence.detect_nonsilent`.
    static func detectNonsilent(
        _ seg: PCM16Segment, minSilenceLen: Int, silenceThresh: Double, seekStep: Int
    ) -> [(Int, Int)] {
        let silent = detectSilence(seg, minSilenceLen: minSilenceLen, silenceThresh: silenceThresh,
                                   seekStep: seekStep)
        let len = seg.lengthMs
        guard let first = silent.first, let last = silent.last else { return [(0, len)] }
        if first.0 == 0 && first.1 == len { return [] }
        var prevEnd = 0
        var out: [(Int, Int)] = []
        for (s, e) in silent {
            out.append((prevEnd, s))
            prevEnd = e
        }
        if last.1 != len { out.append((prevEnd, len)) }
        if out[0] == (0, 0) { out.removeFirst() }
        return out
    }

    /// `pydub.silence.split_on_silence` (integer `keep_silence`); returns the clamped
    /// `[max(start, 0), min(end, len)]` ms ranges that pydub slices out.
    static func splitOnSilence(
        _ seg: PCM16Segment, minSilenceLen: Int, silenceThresh: Double, keepSilence: Int, seekStep: Int
    ) -> [(Int, Int)] {
        var ranges = detectNonsilent(seg, minSilenceLen: minSilenceLen, silenceThresh: silenceThresh,
                                     seekStep: seekStep)
            .map { [$0.0 - keepSilence, $0.1 + keepSilence] }
        if ranges.count > 1 {
            for i in 0 ..< ranges.count - 1 {
                let lastEnd = ranges[i][1]
                let nextStart = ranges[i + 1][0]
                if nextStart < lastEnd {
                    ranges[i][1] = floorDiv(lastEnd + nextStart, 2)
                    ranges[i + 1][0] = ranges[i][1]
                }
            }
        }
        let len = seg.lengthMs
        return ranges.map { (max($0[0], 0), min($0[1], len)) }
    }

    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
    }

    // MARK: - Fade and pad

    /// `fade_and_pad_audio`: linear `np.linspace` fades (float64 → float32) over
    /// `min(int(fade * sr), T // 2)` samples on both ends, then `int(pad * sr)` zeros per side.
    /// Empty input is returned unchanged (no padding), like Python.
    public static func fadeAndPad(
        _ s: [Float], sampleRate: Int, padSeconds: Double = 0.1, fadeSeconds: Double = 0.1
    ) -> [Float] {
        if s.isEmpty { return s }
        let fadeSamples = Int(fadeSeconds * Double(sampleRate))
        let padSamples = Int(padSeconds * Double(sampleRate))
        var out = s
        if fadeSamples > 0 {
            let k = min(fadeSamples, out.count / 2)
            if k > 0 {
                let fadeIn = linspace(0, 1, k)
                let fadeOut = linspace(1, 0, k)
                let tail = out.count - k
                for i in 0 ..< k {
                    out[i] *= fadeIn[i]
                    out[tail + i] *= fadeOut[i]
                }
            }
        }
        if padSamples > 0 {
            let zeros = [Float](repeating: 0, count: padSamples)
            out = zeros + out + zeros
        }
        return out
    }

    /// `np.linspace(start, stop, num, dtype=np.float32)`: computed in float64, then cast.
    static func linspace(_ start: Double, _ stop: Double, _ num: Int) -> [Float] {
        let div = num - 1
        let delta = stop - start
        var y = (0 ..< num).map { Double($0) }
        if div > 0 {
            let step = delta / Double(div)
            if step == 0 {
                for i in y.indices { y[i] = y[i] / Double(div) * delta }
            } else {
                for i in y.indices { y[i] *= step }
            }
        } else {
            for i in y.indices { y[i] *= delta }
        }
        for i in y.indices { y[i] += start }
        if num > 1 { y[num - 1] = stop }
        return y.map { Float($0) }
    }

    // MARK: - Composite pipelines

    /// Reference-audio part of `OmniVoice.create_voice_clone_prompt` for an already
    /// resampled mono signal with a user-provided transcript (so `trim_long_audio` is not run):
    /// RMS-normalize to `targetRMS` when `0 < rms < targetRMS`, optionally remove silence
    /// (mid 200 / lead 100 / trail 200 ms), then drop the trailing `count % hopLength` samples.
    /// Returns the processed samples and the RMS measured before normalization (`ref_rms`).
    public static func prepareReference(
        _ s: [Float], sampleRate: Int, hopLength: Int, preprocess: Bool,
        targetRMS: Double = 0.1, midSilenceMs: Int = 200, leadSilenceMs: Int = 100,
        trailSilenceMs: Int = 200
    ) throws -> (samples: [Float], rms: Float) {
        precondition(hopLength > 0, "hop length must be positive")
        let refRMS = rms(s)
        var wav = s
        if 0 < Double(refRMS) && Double(refRMS) < targetRMS {
            let target = Float(targetRMS)
            wav = wav.map { $0 * target / refRMS }
        }
        if preprocess {
            wav = removeSilence(wav, sampleRate: sampleRate, midSilenceMs: midSilenceMs,
                                leadSilenceMs: leadSilenceMs, trailSilenceMs: trailSilenceMs)
            if wav.isEmpty { throw AudioPostprocessError.emptyAfterSilenceRemoval }
        }
        let clip = wav.count % hopLength
        if clip > 0 { wav.removeLast(clip) }
        return (wav, refRMS)
    }

    /// `OmniVoice._post_process_audio`: optional silence removal (mid 500 / lead 100 /
    /// trail 100 ms), volume match (`x * ref_rms / targetRMS` when `ref_rms < targetRMS`;
    /// peak-normalize to `peakTarget` when `referenceRMS` is nil and peak > 1e-6), then
    /// `fadeAndPad`. Python raises on empty audio with `ref_rms=None` (max of an empty array);
    /// here empty audio is returned unchanged since every step is a no-op on it.
    public static func postprocessGenerated(
        _ s: [Float], sampleRate: Int, referenceRMS: Float?, removeSilence: Bool,
        padSeconds: Double = 0.1, fadeSeconds: Double = 0.1,
        targetRMS: Double = 0.1, peakTarget: Float = 0.5, midSilenceMs: Int = 500,
        leadSilenceMs: Int = 100, trailSilenceMs: Int = 100
    ) -> [Float] {
        var audio = s
        if removeSilence {
            audio = Self.removeSilence(audio, sampleRate: sampleRate, midSilenceMs: midSilenceMs,
                                       leadSilenceMs: leadSilenceMs, trailSilenceMs: trailSilenceMs)
        }
        if let ref = referenceRMS {
            if Double(ref) < targetRMS {
                let target = Float(targetRMS)
                audio = audio.map { $0 * ref / target }
            }
        } else if !audio.isEmpty {
            var peak: Float = 0
            for x in audio { peak = max(peak, abs(x)) }
            if peak > Float(1e-6) { audio = audio.map { $0 / peak * peakTarget } }
        }
        return fadeAndPad(audio, sampleRate: sampleRate, padSeconds: padSeconds, fadeSeconds: fadeSeconds)
    }
}

/// Mono 16-bit pydub `AudioSegment` with millisecond-based accessors.
struct PCM16Segment {
    var samples: [Int16]
    let frameRate: Int

    static let maxPossibleAmplitude = 32768.0

    /// `len(segment)`: `round(1000 * (frames / rate))` with Python's half-to-even rounding.
    var lengthMs: Int {
        Int((1000 * (Double(samples.count) / Double(frameRate))).rounded(.toNearestOrEven))
    }

    /// `_parse_position` for non-negative ms: `int(ms * (rate / 1000.0))`.
    func frameIndex(ms: Int) -> Int {
        precondition(ms >= 0, "negative positions are not supported")
        return Int(Double(ms) * (Double(frameRate) / 1000))
    }

    /// Frames copied by `segment[startMs:endMs]` and the zero frames pydub appends when the
    /// ms→frame mapping overruns the data (at most 2 ms; none if the copied data is empty).
    func sliceLayout(startMs: Int, endMs: Int?) -> (range: Range<Int>, padding: Int) {
        let len = lengthMs
        let s = frameIndex(ms: min(startMs, len))
        let e = frameIndex(ms: min(endMs ?? len, len))
        let n = samples.count
        let lo = min(s, n)
        let hi = max(lo, min(e, n))
        let missing = (e - s) - (hi - lo)
        if missing > 0 {
            precondition(Double(missing) <= 2 * (Double(frameRate) / 1000),
                         "pydub TooManyMissingFrames: \(missing)")
        }
        let padding = (missing > 0 && hi > lo) ? missing : 0
        return (lo ..< hi, padding)
    }

    func slice(startMs: Int, endMs: Int?) -> PCM16Segment {
        var out: [Int16] = []
        appendSlice(startMs: startMs, endMs: endMs, to: &out)
        return PCM16Segment(samples: out, frameRate: frameRate)
    }

    func appendSlice(startMs: Int, endMs: Int?, to out: inout [Int16]) {
        let (range, padding) = sliceLayout(startMs: startMs, endMs: endMs)
        out.append(contentsOf: samples[range])
        if padding > 0 { out.append(contentsOf: repeatElement(0, count: padding)) }
    }

    /// `segment[startMs:endMs].rms` (`audioop.rms`: truncated integer RMS).
    func rms(startMs: Int, endMs: Int) -> Int {
        let (range, padding) = sliceLayout(startMs: startMs, endMs: endMs)
        var sumSquares: Int64 = 0
        for i in range {
            let v = Int64(samples[i])
            sumSquares += v * v
        }
        return Self.audioopRMS(sumSquares: sumSquares, count: range.count + padding)
    }

    /// Same as `rms(startMs:endMs:)` using `squarePrefixSums()` of this segment.
    func rms(startMs: Int, endMs: Int, prefix: [Int64]) -> Int {
        let (range, padding) = sliceLayout(startMs: startMs, endMs: endMs)
        let sumSquares = prefix[range.upperBound] - prefix[range.lowerBound]
        return Self.audioopRMS(sumSquares: sumSquares, count: range.count + padding)
    }

    /// `prefix[i]` = sum of squared samples before index `i` (count + 1 entries).
    func squarePrefixSums() -> [Int64] {
        var prefix = [Int64](repeating: 0, count: samples.count + 1)
        for i in samples.indices {
            let v = Int64(samples[i])
            prefix[i + 1] = prefix[i] + v * v
        }
        return prefix
    }

    /// `(unsigned int)sqrt(sum_squares / n)`; window sums stay far below 2^53, so the double
    /// conversion is exact.
    static func audioopRMS(sumSquares: Int64, count: Int) -> Int {
        if count == 0 { return 0 }
        return Int((Double(sumSquares) / Double(count)).squareRoot())
    }

    /// `segment[startMs:endMs].dBFS` (`20 * math.log(rms / 32768, 10)`, `-inf` for silence).
    func dBFS(startMs: Int, endMs: Int) -> Double {
        let r = rms(startMs: startMs, endMs: endMs)
        if r == 0 { return -.infinity }
        let ratio = Double(r) / Self.maxPossibleAmplitude
        return 20 * (log(ratio) / log(10.0))
    }
}
