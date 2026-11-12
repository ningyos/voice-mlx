import Foundation

// Architecture configs for the Qwen3-TTS-12Hz 0.6B talker, code predictor and the
// 12 Hz speech-tokenizer decoder (dimensions from soniqo/speech-swift `468db5f`,
// cross-checked against Qwen/Qwen3-TTS-12Hz-0.6B-Base `config.json`).

public struct TalkerConfig: Sendable {
    public var hiddenSize = 1024
    public var numLayers = 28
    public var numHeads = 16
    public var numKVHeads = 8
    public var headDim = 128
    public var intermediateSize = 3072
    public var ropeTheta: Float = 1_000_000
    public var rmsNormEps: Float = 1e-6
    public var textVocabSize = 151936
    public var textHiddenSize = 2048
    public var codecVocabSize = 3072
    /// Quantization group size and bits of the linear layers; `bits == 0` = unquantized.
    public var groupSize = 64
    public var bits = 0

    public init() {}
}

public struct CodePredictorConfig: Sendable {
    public var hiddenSize = 1024
    /// Embedding width (equals `hiddenSize` for 0.6B; 2048 with a projection for 1.7B).
    public var embeddingDim = 1024
    public var numLayers = 5
    public var numHeads = 16
    public var numKVHeads = 8
    public var headDim = 128
    public var intermediateSize = 3072
    public var ropeTheta: Float = 1_000_000
    public var rmsNormEps: Float = 1e-6
    public var vocabSize = 2048
    public var numCodeGroups = 16
    public var groupSize = 64
    public var bits = 0

    public var needsProjection: Bool { embeddingDim != hiddenSize }

    public init() {}
}

public struct SpeechTokenizerDecoderConfig: Sendable {
    public var latentDim = 1024
    public var decoderDim = 1536
    public var hiddenSize = 512
    public var numHeads = 16
    public var numKVHeads = 16
    public var headDim = 64
    public var numLayers = 8
    public var upsampleRates = [8, 5, 4, 3]
    public var upsamplingRatios = [2, 2]
    public var numQuantizers = 16
    public var semanticCodebookSize = 2048
    public var acousticCodebookSize = 2048
    public var codebookDim = 256
    public var slidingWindow = 72
    public var sampleRate = 24000
    public var frameRate = 12.5
    /// `decoder_config.rms_norm_eps` of Qwen/Qwen3-TTS-Tokenizer-12Hz (speech-swift used 1e-8).
    public var rmsNormEps: Float = 1e-5

    /// Output samples per codec frame (1920 at 24 kHz / 12.5 Hz).
    public var samplesPerFrame: Int { upsampleRates.reduce(1, *) * upsamplingRatios.reduce(1, *) }

    public init() {}
}

/// Special token ids (`Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`).
public enum SpecialTokens {
    public static let codecPad = 2148
    public static let codecBos = 2149
    public static let codecEos = 2150
    public static let codecThink = 2154
    public static let codecNothink = 2155
    public static let codecThinkBos = 2156
    public static let codecThinkEos = 2157
    public static let ttsPad = 151671
    public static let ttsBos = 151672
    public static let ttsEos = 151673
    public static let imStart = 151644
    public static let imEnd = 151645
    public static let assistant = 77091
    public static let newline = 198
}
