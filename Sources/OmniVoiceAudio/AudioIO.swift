import AVFoundation
import Foundation

public enum AudioIOError: Error, CustomStringConvertible {
    case unreadable(URL, underlying: Error)
    case empty(URL)
    case bufferAllocation(URL)
    case writeFailed(URL, underlying: Error)
    case shortRead(URL, got: Int, expected: Int)

    public var description: String {
        switch self {
        case .unreadable(let url, let e): return "cannot read audio \(url.path): \(e)"
        case .empty(let url): return "audio file has no samples: \(url.path)"
        case .bufferAllocation(let url): return "cannot allocate PCM buffer for \(url.path)"
        case .writeFailed(let url, let e): return "cannot write audio \(url.path): \(e)"
        case .shortRead(let url, let got, let expected):
            return "short read from \(url.path): \(got) of \(expected) frames"
        }
    }
}

/// Mono float audio at a known sample rate.
public struct AudioClip: Sendable {
    public var samples: [Float]
    public var sampleRate: Int

    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    public var duration: Double { Double(samples.count) / Double(sampleRate) }

    /// Resample with the torchaudio-compatible kernel (no-op at equal rates).
    public func resampled(to rate: Int) -> AudioClip {
        AudioClip(samples: Resampler.resample(samples, from: sampleRate, to: rate), sampleRate: rate)
    }
}

public enum AudioIO {
    /// Read an audio file as Float32 and average channels to mono — the same
    /// as `soundfile.read(path, dtype="float32")` followed by `mean(axis=channel)`.
    public static func read(_ url: URL) throws -> AudioClip {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch {
            throw AudioIOError.unreadable(url, underlying: error)
        }
        let format = file.processingFormat  // deinterleaved Float32
        let total = Int(file.length)
        guard total > 0 else { throw AudioIOError.empty(url) }
        let channels = Int(format.channelCount)
        let chunk = AVAudioFrameCount(min(total, 1 << 16))
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunk) else {
            throw AudioIOError.bufferAllocation(url)
        }
        // AVAudioFile may return short reads (e.g. float WAVs with an FLLR chunk),
        // so read until the declared length is reached.
        var mono = [Float]()
        mono.reserveCapacity(total)
        let inv = 1 / Float(channels)
        while mono.count < total {
            do {
                try file.read(into: buf, frameCount: AVAudioFrameCount(min(Int(chunk), total - mono.count)))
            } catch { throw AudioIOError.unreadable(url, underlying: error) }
            let n = Int(buf.frameLength)
            guard n > 0, let data = buf.floatChannelData else { break }
            if channels == 1 {
                mono.append(contentsOf: UnsafeBufferPointer(start: data[0], count: n))
            } else {
                for i in 0 ..< n {
                    var s: Float = 0
                    for c in 0 ..< channels { s += data[c][i] }
                    mono.append(s * inv)
                }
            }
        }
        guard mono.count == total else { throw AudioIOError.shortRead(url, got: mono.count, expected: total) }
        return AudioClip(samples: mono, sampleRate: Int(format.sampleRate))
    }

    /// Read and resample to `sampleRate` with the torchaudio-compatible kernel.
    public static func read(_ url: URL, sampleRate: Int) throws -> AudioClip {
        try read(url).resampled(to: sampleRate)
    }

    /// Write mono Float32 WAV.
    public static func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        guard let fmt = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1),
            let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(max(1, samples.count)))
        else { throw AudioIOError.bufferAllocation(url) }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress { buf.floatChannelData![0].update(from: base, count: samples.count) }
        }
        do {
            let file = try AVAudioFile(
                forWriting: url, settings: fmt.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            try file.write(from: buf)
        } catch { throw AudioIOError.writeFailed(url, underlying: error) }
    }
}
