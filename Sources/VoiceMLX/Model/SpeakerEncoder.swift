import Accelerate
import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Squeeze-Excitation Block

class SEBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(channels: Int, bottleneck: Int = 128) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: bottleneck,
            kernelSize: 1, bias: true)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: bottleneck, outputChannels: channels,
            kernelSize: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C]
        let s = x.mean(axis: 1, keepDims: true)  // [B, 1, C]
        let h = sigmoid(conv2(relu(conv1(s))))    // [B, 1, C]
        return x * h
    }
}

// MARK: - Res2Net Block

class Res2NetBlock: Module {
    @ModuleInfo var blocks: [Conv1d]
    let scaleCount: Int

    init(channels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        let width = channels / scale
        self.scaleCount = scale
        let pad = ((kernelSize - 1) * dilation) / 2
        self._blocks.wrappedValue = (0..<(scale - 1)).map { _ in
            Conv1d(inputChannels: width, outputChannels: width,
                   kernelSize: kernelSize, padding: pad,
                   dilation: dilation, bias: true)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C]
        let width = x.dim(2) / scaleCount
        var outputs: [MLXArray] = []

        for i in 0..<scaleCount {
            let chunk = x[0..., 0..., (i * width)..<((i + 1) * width)]
            if i == 0 {
                outputs.append(chunk)
            } else {
                let input = i == 1 ? chunk : chunk + outputs[i - 1]
                // Each sub-block is a TimeDelayNetBlock (Conv1d + ReLU)
                outputs.append(relu(blocks[i - 1](input)))
            }
        }

        return concatenated(outputs, axis: 2)
    }
}

// MARK: - ECAPA-TDNN SE-Res2Net Block

class ECAPABlock: Module {
    @ModuleInfo var tdnn1: Conv1d
    @ModuleInfo var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: Conv1d
    @ModuleInfo var seBlock: SEBlock

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1) {
        self._tdnn1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, bias: true)
        self._res2netBlock.wrappedValue = Res2NetBlock(
            channels: channels, scale: 8, kernelSize: kernelSize, dilation: dilation)
        self._tdnn2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, bias: true)
        self._seBlock.wrappedValue = SEBlock(channels: channels, bottleneck: 128)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        // tdnn1 and tdnn2 are TimeDelayNetBlocks (Conv1d + ReLU) in Python
        var h = relu(tdnn1(x))
        h = res2netBlock(h)
        h = relu(tdnn2(h))
        h = seBlock(h)
        return h + residual
    }
}

// MARK: - Attentive Statistics Pooling

/// Attentive statistics pooling matching Python's implementation.
/// Internally computes global mean/std, concatenates [x, mean, std] for attention context,
/// then uses attention weights to compute weighted mean/std statistics.
class AttentiveStatisticsPooling: Module {
    @ModuleInfo var tdnn: Conv1d   // attention: channels*3 → attention_channels
    @ModuleInfo var conv: Conv1d   // attention_channels → channels

    init(channels: Int, attention: Int = 128) {
        // tdnn takes [x, global_mean, global_std] concatenated = 3*channels
        self._tdnn.wrappedValue = Conv1d(
            inputChannels: channels * 3, outputChannels: attention,
            kernelSize: 1, bias: true)
        // conv projects back to channels (NOT channels*3)
        self._conv.wrappedValue = Conv1d(
            inputChannels: attention, outputChannels: channels,
            kernelSize: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C] where C = 1536 (concatenated block outputs after MFA)

        // Compute global mean and std over time
        let globalMean = x.mean(axis: 1, keepDims: true)  // [B, 1, C]
        let globalVar = ((x - globalMean) * (x - globalMean)).mean(axis: 1, keepDims: true)
        let globalStd = sqrt(clip(globalVar, min: 1e-12))  // [B, 1, C]

        // Expand to time dimension
        let meanExpanded = broadcast(globalMean, to: x.shape)  // [B, T, C]
        let stdExpanded = broadcast(globalStd, to: x.shape)    // [B, T, C]

        // Concatenate [x, mean, std] for attention context: [B, T, 3C]
        let attnInput = concatenated([x, meanExpanded, stdExpanded], axis: 2)

        // Compute attention weights: [B, T, 3C] → [B, T, C]
        let alpha = softmax(conv(tanh(tdnn(attnInput))), axis: 1)  // softmax over T

        // Weighted statistics
        let weightedMean = (alpha * x).sum(axis: 1)  // [B, C]
        let weightedVar = (alpha * (x - weightedMean.expandedDimensions(axis: 1)) * (x - weightedMean.expandedDimensions(axis: 1))).sum(axis: 1)
        let weightedStd = sqrt(clip(weightedVar, min: 1e-12))  // [B, C]

        // Concatenate mean and std: [B, 2C]
        return concatenated([weightedMean, weightedStd], axis: 1)
    }
}

// MARK: - Multi-Feature Aggregation (TimeDelayNetBlock: Conv1d + ReLU)

class MFA: Module {
    @ModuleInfo var conv: Conv1d

    init(channels: Int) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(conv(x))
    }
}

// MARK: - ECAPA-TDNN Speaker Encoder

/// ECAPA-TDNN speaker encoder for extracting 1024-dim speaker embeddings.
/// Used for voice cloning in Qwen3-TTS Base model.
///
/// Architecture: TDNN(128→512) → 3x SE-Res2Net blocks(512) → cat(1536) → MFA(1536) → ASP → FC(3072→1024)
/// Input: 128-bin mel spectrogram (24kHz, n_fft=1024, hop=256)
/// Output: 1024-dim speaker embedding
public class SpeakerEncoder: Module {

    @ModuleInfo(key: "blocks") var initialConv: Conv1d
    @ModuleInfo var block1: ECAPABlock
    @ModuleInfo var block2: ECAPABlock
    @ModuleInfo var block3: ECAPABlock
    @ModuleInfo var mfa: MFA
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo var fc: Conv1d

    /// - Parameter embeddingDim: speaker-embedding output dimension. This equals
    ///   the model hidden size — 1024 for the 0.6B, **2048 for the 1.7B**. The
    ///   1.7B's speaker-encoder `fc` is `3072 → 2048` (the 0.6B's is `3072 → 1024`),
    ///   so hardcoding 1024 crashes ICL cloning on the 1.7B. The downstream ICL
    ///   prefill reshapes the embedding to `[1, 1, hiddenSize]`, so this must match.
    public init(embeddingDim: Int = 1024) {
        // Initial TDNN: mel(128) → 512 channels, kernel=5
        self._initialConv.wrappedValue = Conv1d(
            inputChannels: 128, outputChannels: 512,
            kernelSize: 5, padding: 2, bias: true)

        // 3 SE-Res2Net blocks with increasing dilation
        self._block1.wrappedValue = ECAPABlock(channels: 512, kernelSize: 3, dilation: 2)
        self._block2.wrappedValue = ECAPABlock(channels: 512, kernelSize: 3, dilation: 3)
        self._block3.wrappedValue = ECAPABlock(channels: 512, kernelSize: 3, dilation: 4)

        // Multi-feature aggregation: cat of 3 block outputs (1536) → 1536
        self._mfa.wrappedValue = MFA(channels: 1536)

        // Attentive statistics pooling: 1536 channels
        self._asp.wrappedValue = AttentiveStatisticsPooling(channels: 1536, attention: 128)

        // Final projection: 3072 (1536*2 from ASP mean+std) → embeddingDim
        // (1024 for 0.6B, 2048 for 1.7B).
        self._fc.wrappedValue = Conv1d(
            inputChannels: 3072, outputChannels: embeddingDim,
            kernelSize: 1, bias: true)

        super.init()
    }

    /// Extract speaker embedding from mel spectrogram
    /// - Parameter mels: [B, T, 128] mel spectrogram
    /// - Returns: [B, 1024] speaker embedding
    public func callAsFunction(_ mels: MLXArray) -> MLXArray {
        // Initial TDNN (Conv1d + ReLU)
        let h0 = relu(initialConv(mels))  // [B, T, 512]

        // 3 SE-Res2Net blocks
        let out1 = block1(h0)
        let out2 = block2(out1)
        let out3 = block3(out2)

        // Concatenate block outputs (skip initial conv)
        var h = concatenated([out1, out2, out3], axis: 2)  // [B, T, 1536]

        // Multi-feature aggregation (Conv1d + ReLU)
        h = mfa(h)  // [B, T, 1536]

        // Attentive statistics pooling → [B, 3072]
        h = asp(h)

        // Final projection → [B, 1024]
        h = h.expandedDimensions(axis: 1)  // [B, 1, 3072] for Conv1d
        h = fc(h)  // [B, 1, 1024]
        h = h.squeezed(axis: 1)  // [B, 1024]

        return h
    }
}

// MARK: - Mel Spectrogram for Speaker Encoder

/// 128-bin log-mel spectrogram, a port of `qwen_tts`'s `mel_spectrogram`
/// (n_fft 1024, hop 256, win 1024, fmin 0, fmax 12000, 24 kHz, `center=False` after a
/// `(n_fft - hop) / 2` reflect pad, magnitude `sqrt(re² + im² + 1e-9)`, librosa slaney
/// mel filterbank with slaney norm, `log(clamp(x, 1e-5))`).
public enum SpeakerMel {
    public static let sampleRate = 24000
    static let nFFT = 1024
    static let hop = 256
    static let nMels = 128
    static let fMax = 12000.0

    /// - Parameter audio: mono samples at 24 kHz.
    /// - Returns: `[1, frames, 128]` log-mel.
    public static func compute(audio: [Float]) -> MLXArray {
        let mags = stftMagnitudes(audio)
        let frames = mags.count / (nFFT / 2 + 1)
        let spec = MLXArray(mags, [frames, nFFT / 2 + 1])
        let mel = matmul(spec, filterbank)  // [frames, nMels]
        return log(clip(mel, min: 1e-5)).expandedDimensions(axis: 0)
    }

    /// Reflect padding like `torch.nn.functional.pad(mode="reflect")`.
    static func reflectPad(_ x: [Float], _ pad: Int) -> [Float] {
        precondition(x.count > pad, "audio shorter than the STFT padding")
        var out = [Float](repeating: 0, count: x.count + 2 * pad)
        for i in 0 ..< pad { out[pad - 1 - i] = x[i + 1] }
        for i in 0 ..< x.count { out[pad + i] = x[i] }
        for i in 0 ..< pad { out[pad + x.count + i] = x[x.count - 2 - i] }
        return out
    }

    static func stftMagnitudes(_ samples: [Float]) -> [Float] {
        let padded = reflectPad(samples, (nFFT - hop) / 2)
        let numBins = nFFT / 2 + 1
        let numFrames = (padded.count - nFFT) / hop + 1
        var mags = [Float](repeating: 0, count: numFrames * numBins)
        // torch.hann_window (periodic).
        let window = (0 ..< nFFT).map { 0.5 * (1 - cos(2 * Float.pi * Float($0) / Float(nFFT))) }
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed")
        }
        defer { vDSP_destroy_fftsetup(setup) }
        var re = [Float](repeating: 0, count: nFFT / 2)
        var im = [Float](repeating: 0, count: nFFT / 2)
        var windowed = [Float](repeating: 0, count: nFFT)
        for f in 0 ..< numFrames {
            let start = f * hop
            for i in 0 ..< nFFT { windowed[i] = padded[start + i] * window[i] }
            windowed.withUnsafeMutableBufferPointer { w in
                re.withUnsafeMutableBufferPointer { r in
                    im.withUnsafeMutableBufferPointer { m in
                        var split = DSPSplitComplex(realp: r.baseAddress!, imagp: m.baseAddress!)
                        w.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) {
                            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(nFFT / 2))
                        }
                        vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                        // vDSP's real FFT is scaled by 2; DC and Nyquist are packed in [0].
                        let dc = split.realp[0] / 2, ny = split.imagp[0] / 2
                        mags[f * numBins] = (dc * dc + 1e-9).squareRoot()
                        for b in 1 ..< nFFT / 2 {
                            let x = split.realp[b] / 2, y = split.imagp[b] / 2
                            mags[f * numBins + b] = (x * x + y * y + 1e-9).squareRoot()
                        }
                        mags[f * numBins + nFFT / 2] = (ny * ny + 1e-9).squareRoot()
                    }
                }
            }
        }
        return mags
    }

    /// `librosa.filters.mel(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)`
    /// (slaney mel scale, slaney norm), transposed to `[bins, mels]`.
    static let filterbank: MLXArray = {
        let numBins = nFFT / 2 + 1
        let fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logStep = log(6.4) / 27
        func hzToMel(_ hz: Double) -> Double { hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logStep : hz / fSp }
        func melToHz(_ m: Double) -> Double { m >= minLogMel ? minLogHz * exp(logStep * (m - minLogMel)) : fSp * m }
        let lo = hzToMel(0), hi = hzToMel(fMax)
        let melF = (0 ..< nMels + 2).map { melToHz(lo + (hi - lo) * Double($0) / Double(nMels + 1)) }
        let fft = (0 ..< numBins).map { Double(sampleRate) / 2 * Double($0) / Double(numBins - 1) }
        var w = [Float](repeating: 0, count: numBins * nMels)
        for m in 0 ..< nMels {
            let d0 = melF[m + 1] - melF[m], d1 = melF[m + 2] - melF[m + 1]
            let enorm = 2 / (melF[m + 2] - melF[m])
            for k in 0 ..< numBins {
                let lower = (fft[k] - melF[m]) / d0
                let upper = (melF[m + 2] - fft[k]) / d1
                w[k * nMels + m] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return MLXArray(w, [numBins, nMels])
    }()
}
