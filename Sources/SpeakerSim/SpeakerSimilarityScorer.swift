import Foundation
import MLX
import VoiceAudio

/// SIM-o speaker-similarity scorer: MLX port of OmniVoice's evaluation model
/// (`omnivoice.eval.models.ecapa_tdnn_wavlm.ECAPA_TDNN_WAVLM`, WavLM-large + ECAPA-TDNN,
/// 256-d embeddings compared with cosine similarity).
///
/// Weights come from `scripts/export_sim_weights.py`. Not thread-safe: use one instance
/// per thread.
public final class SpeakerSimilarityScorer {
    /// Sample rate the model consumes.
    public static let sampleRate = 16_000
    /// Audio is truncated to this length after resampling, like OmniVoice's
    /// `load_eval_waveform(..., max_seconds=120)`.
    public static let maxSeconds = 120
    /// Embedding dimension.
    public static let embeddingDimension = SimWeights.embeddingDim
    /// Minimum number of 16 kHz samples the WavLM CNN front end accepts.
    public static var minimumSamples: Int { SimModel.minimumSamples }

    /// Default weights location, overridable with `OMNIVOICE_SIM_WEIGHTS`.
    public static var defaultWeightsURL: URL {
        if let env = ProcessInfo.processInfo.environment["OMNIVOICE_SIM_WEIGHTS"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/omnivoice-mlx/sim/wavlm_ecapa.safetensors")
    }

    private let model: SimModel

    /// Loads and validates every tensor the forward pass needs (presence and shape).
    public init(weightsURL: URL) throws {
        model = SimModel(weights: try SimWeights(url: weightsURL))
    }

    /// 256-d speaker embedding. Resamples to 16 kHz (torchaudio-compatible) and keeps at
    /// most `maxSeconds` of audio.
    public func embed(_ samples: [Float], sampleRate: Int) throws -> [Float] {
        guard sampleRate > 0 else { throw SpeakerSimilarityError.invalidSampleRate(sampleRate) }
        guard !samples.isEmpty else { throw SpeakerSimilarityError.emptyAudio }
        var wav = sampleRate == Self.sampleRate
            ? samples
            : Resampler.resample(samples, from: sampleRate, to: Self.sampleRate)
        let maxLength = Self.sampleRate * Self.maxSeconds
        if wav.count > maxLength { wav = Array(wav[0 ..< maxLength]) }
        guard wav.count >= Self.minimumSamples else {
            throw SpeakerSimilarityError.audioTooShort(samples: wav.count, minimumSamples: Self.minimumSamples)
        }
        return model.embed(samples16k: wav)
    }

    /// Embedding of an audio file read with `AudioIO.read` (channel mean, native rate).
    public func embed(contentsOf url: URL) throws -> [Float] {
        let clip = try AudioIO.read(url)
        return try embed(clip.samples, sampleRate: clip.sampleRate)
    }

    /// Cosine similarity between a reference embedding and the embedding of `samples`.
    public func similarity(_ refEmbedding: [Float], _ samples: [Float], sampleRate: Int) throws -> Float {
        guard refEmbedding.count == Self.embeddingDimension else {
            throw SpeakerSimilarityError.embeddingDimensionMismatch(
                expected: Self.embeddingDimension, actual: refEmbedding.count)
        }
        return Self.cosine(refEmbedding, try embed(samples, sampleRate: sampleRate))
    }

    /// Cosine similarity matching `torch.nn.functional.cosine_similarity` (eps 1e-8 on
    /// each norm). Vectors must have equal length.
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "cosine: length mismatch \(a.count) vs \(b.count)")
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< a.count {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        let eps = 1e-8
        return Float(dot / (max(na.squareRoot(), eps) * max(nb.squareRoot(), eps)))
    }
}
