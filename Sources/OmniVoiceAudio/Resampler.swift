import Accelerate
import Foundation

/// Windowed-sinc resampler matching `torchaudio.functional.resample` with its
/// defaults (`sinc_interp_hann`, `lowpass_filter_width=6`, `rolloff=0.99`).
///
/// The kernel is built in Float64 exactly like torchaudio's
/// `_get_sinc_resample_kernel` and then cast to Float32; the strided
/// convolution (`_apply_sinc_resample_kernel`) runs on the CPU in Float32.
/// Both the OmniVoice reference loader and the SIM-o scorer's
/// `load_eval_waveform` resample with this function, so matching it keeps the
/// Swift inputs bit-close to the Python ones.
public enum Resampler {
    public static func resample(
        _ samples: [Float], from origFreq: Int, to newFreq: Int,
        lowpassFilterWidth: Int = 6, rolloff: Double = 0.99
    ) -> [Float] {
        precondition(origFreq > 0 && newFreq > 0, "sample rates must be positive")
        if origFreq == newFreq || samples.isEmpty { return samples }
        let g = gcd(origFreq, newFreq)
        let orig = origFreq / g
        let neu = newFreq / g

        let baseFreq = Double(min(orig, neu)) * rolloff
        let width = Int((Double(lowpassFilterWidth) * Double(orig) / baseFreq).rounded(.up))
        let lfw = Double(lowpassFilterWidth)
        let klen = 2 * width + orig
        let scale = baseFreq / Double(orig)

        // kernels[j][i], j = output phase, i = tap.
        var kernel = [Float](repeating: 0, count: neu * klen)
        for j in 0 ..< neu {
            let tj = -Double(j) / Double(neu)
            for i in 0 ..< klen {
                var t = (tj + Double(-width + i) / Double(orig)) * baseFreq
                t = min(max(t, -lfw), lfw)
                let window = pow(cos(t * Double.pi / lfw / 2.0), 2.0)
                let tp = t * Double.pi
                let sinc = tp == 0 ? 1.0 : sin(tp) / tp
                kernel[j * klen + i] = Float(sinc * window * scale)
            }
        }

        // Pad (width, width + orig) with zeros, then a stride-`orig` conv.
        let length = samples.count
        var padded = [Float](repeating: 0, count: width + length + width + orig)
        padded.replaceSubrange(width ..< width + length, with: samples)
        let numFrames = (padded.count - klen) / orig + 1
        let targetLength = Int((Double(neu) * Double(length) / Double(orig)).rounded(.up))
        var out = [Float](repeating: 0, count: numFrames * neu)
        padded.withUnsafeBufferPointer { p in
            kernel.withUnsafeBufferPointer { k in
                out.withUnsafeMutableBufferPointer { o in
                    for f in 0 ..< numFrames {
                        let src = p.baseAddress! + f * orig
                        for j in 0 ..< neu {
                            var acc: Float = 0
                            vDSP_dotpr(src, 1, k.baseAddress! + j * klen, 1, &acc, vDSP_Length(klen))
                            o[f * neu + j] = acc
                        }
                    }
                }
            }
        }
        if out.count > targetLength { out.removeSubrange(targetLength...) }
        return out
    }

    static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = a, y = b
        while y != 0 { (x, y) = (y, x % y) }
        return x
    }
}
