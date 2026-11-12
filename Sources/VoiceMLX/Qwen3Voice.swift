import Foundation
import MLX
import MLXNN
import MLXRandom
import VoiceAudio

/// A reference voice for ICL cloning, encoded once and reused for every utterance.
public struct VoiceReference: Codable, Sendable, Equatable {
    /// Codec tokens, `frames × 16`, frame-major.
    public var codes: [Int32]
    public var frames: Int
    public var transcript: String
    /// x-vector from the speaker encoder (1024 floats).
    public var speakerEmbedding: [Float]

    public var duration: Double { Double(frames) / 12.5 }

    public init(codes: [Int32], frames: Int, transcript: String, speakerEmbedding: [Float]) {
        self.codes = codes
        self.frames = frames
        self.transcript = transcript
        self.speakerEmbedding = speakerEmbedding
    }
}

/// Who speaks: an ICL reference (Base bundle) or a named speaker of a single-speaker /
/// CustomVoice bundle.
public enum Voice: Sendable {
    case reference(VoiceReference)
    case speaker(String)
}

public struct StageTimings: Sendable {
    /// Prompt construction + talker prefill + first token.
    public var prefill: Double = 0
    /// Talker steps after the first token (forward + sampling).
    public var talker: Double = 0
    /// Code predictor (codebooks 1–15) over all frames.
    public var codePredictor: Double = 0
    public var decode: Double = 0
    public var total: Double = 0
}

public struct Synthesis: Sendable {
    /// 24 kHz mono.
    public var samples: [Float]
    public var sampleRate: Int { 24000 }
    /// Generated codec tokens, `frames × 16`, frame-major.
    public var codes: [Int32]
    public var frames: Int
    public var seed: UInt64
    /// True when the talker emitted EOS (false = stopped at `maxNewTokens`).
    public var finished: Bool
    public var timings: StageTimings
}

public struct AudioChunk: Sendable {
    public var samples: [Float]
    /// Generated frames covered by this chunk.
    public var frames: Range<Int>
    public var isFinal: Bool
    /// Seconds since the generate call started.
    public var elapsed: Double
}

/// Qwen3-TTS-12Hz 0.6B on MLX: Base (ICL voice cloning) or single-speaker
/// (CustomVoice-style, speaker name) bundles.
public final class Qwen3Voice: @unchecked Sendable {
    public let precision: WeightPrecision
    public let computeDType: DType
    /// Speaker names the bundle knows (empty for Base).
    public var speakers: [String] { speakerIds.keys.sorted() }
    public var supportsReference: Bool { speakerEncoder != nil && codecEncoder != nil }
    /// Accepted language names (plus ISO codes and `auto`).
    public var languages: [String] { languageIds.keys.sorted() }

    let talkerConfig: TalkerConfig
    let cpConfig: CodePredictorConfig
    let codecConfig = SpeechTokenizerDecoderConfig()
    let talker: TalkerModel
    let codePredictor: CodePredictorModel
    let speakerEncoder: SpeakerEncoder?
    let codecEncoder: SpeechTokenizerEncoder?
    let decoder: SpeechTokenizerDecoder
    let tokenizer: TextTokenizer
    let speakerIds: [String: Int]
    let speakerDialects: [String: String]
    let languageIds: [String: Int]

    var compiledTalkerStep: (([MLXArray]) -> [MLXArray])?
    var compiledFrames: [String: ([MLXArray]) -> [MLXArray]] = [:]
    var suppressMask: MLXArray

    // MARK: Loading

    /// Load a talker bundle (Base or single-speaker) and the 12 Hz codec.
    public static func load(
        model: ModelSource = .base8bit, codec: ModelSource = .codec, options: LoadOptions = LoadOptions()
    ) async throws -> Qwen3Voice {
        let modelDir = try await BundleResolver.resolve(model, required: BundleResolver.talkerFiles, options: options)
        let codecDir = try await BundleResolver.resolve(codec, required: ["config.json"], options: options)
        return try Qwen3Voice(modelDirectory: modelDir, codecDirectory: codecDir, options: options)
    }

    public init(modelDirectory: URL, codecDirectory: URL, options: LoadOptions = LoadOptions()) throws {
        try BundleResolver.check(modelDirectory, required: BundleResolver.talkerFiles)
        try BundleResolver.check(codecDirectory, required: ["config.json"])
        let cfg = try BundleConfig(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        if cfg.storedBits != 0 && options.precision == .bf16 {
            throw VoiceError.precisionUnavailable(
                "\(modelDirectory.lastPathComponent) is \(cfg.storedBits)-bit; bf16 needs an unquantized bundle")
        }
        if cfg.storedBits != 0 && cfg.storedGroupSize != options.quantizationGroupSize {
            throw VoiceError.precisionUnavailable(
                "bundle group size \(cfg.storedGroupSize) != requested \(options.quantizationGroupSize)")
        }
        precision = options.precision
        computeDType = options.computeDType
        var tc = cfg.talker
        var cc = cfg.codePredictor
        tc.bits = cfg.storedBits
        cc.bits = cfg.storedBits
        tc.groupSize = cfg.storedGroupSize
        cc.groupSize = cfg.storedGroupSize
        talkerConfig = tc
        cpConfig = cc
        speakerIds = cfg.speakers
        speakerDialects = cfg.speakerDialects
        languageIds = cfg.languages
        tokenizer = try TextTokenizer(directory: modelDirectory)

        // Quantizing at load must start from bf16 so scales/biases come out bf16, which
        // is how the aufklarer / voice-train 8-bit bundles are made (`mx.quantize` on the
        // bf16 tensors); other float dtypes are cast after quantization.
        let quantizeOnLoad = cfg.storedBits == 0 && options.precision == .int8
        let weights = try BundleResolver.loadWeights(
            modelDirectory, dtype: quantizeOnLoad ? .bfloat16 : options.computeDType)
        let talker = TalkerModel(config: tc)
        let codePredictor = CodePredictorModel(config: cc)
        try applyStrict([("talker", talker), ("talker.code_predictor", codePredictor)]) {
            TTSWeightLoader.loadTalkerAndCodePredictorWeights(
                talker: talker, codePredictor: codePredictor, from: weights)
        }
        self.talker = talker
        self.codePredictor = codePredictor
        if weights.keys.contains(where: { $0.hasPrefix("speaker_encoder.") }) {
            let se = SpeakerEncoder()
            try applyStrict([("speaker_encoder", se)]) {
                TTSWeightLoader.loadSpeakerEncoderWeights(into: se, from: weights)
            }
            speakerEncoder = se
        } else {
            speakerEncoder = nil
        }
        if quantizeOnLoad {
            // Same quantized set as the aufklarer 8-bit bundles: every Linear of the
            // talker and code predictor (incl. text projection, codec head, lm heads);
            // embeddings and norms stay float.
            let gs = options.quantizationGroupSize
            let filter: (String, Module) -> Bool = { _, m in m is Linear && !(m is QuantizedLinear) }
            quantize(model: talker, groupSize: gs, bits: 8, filter: filter)
            quantize(model: codePredictor, groupSize: gs, bits: 8, filter: filter)
            if options.computeDType != .bfloat16 {
                for m in [talker, codePredictor, speakerEncoder].compactMap({ $0 as Module? }) {
                    let cast = m.parameters().flattened().map { (k, v) in
                        (k, v.dtype.isFloatingPoint ? v.asType(options.computeDType) : v)
                    }
                    m.update(parameters: ModuleParameters.unflattened(cast))
                }
            }
        }

        // The codec stays float32 regardless of computeDType.
        let codecWeights = try BundleResolver.loadWeights(codecDirectory, dtype: .float32)
        let decoder = SpeechTokenizerDecoder(config: codecConfig)
        try applyStrict([("codec decoder", decoder)]) {
            TTSWeightLoader.loadSpeechTokenizerDecoderWeights(into: decoder, from: codecWeights)
        }
        self.decoder = decoder
        if speakerEncoder != nil {
            let enc = SpeechTokenizerEncoder(config: codecConfig)
            try applyStrict([("codec encoder", enc)]) {
                TTSWeightLoader.loadSpeechTokenizerEncoderWeights(into: enc, from: codecWeights)
            }
            codecEncoder = enc
        } else {
            codecEncoder = nil
        }
        if speakerIds.isEmpty && speakerEncoder == nil {
            throw VoiceError.invalidBundle(
                modelDirectory, "neither speaker_encoder weights (Base) nor talker_config.spk_id (single-speaker)")
        }

        // Suppress [vocab - 1024, vocab) except EOS (qwen_tts `suppress_tokens`).
        let v = tc.codecVocabSize
        var sup = [Bool](repeating: false, count: v)
        for i in (v - 1024) ..< v where i != SpecialTokens.codecEos { sup[i] = true }
        suppressMask = MLXArray(sup)
        eval(talker, codePredictor, decoder)
        setupCompilation()
    }

    // MARK: Languages / speakers

    static let isoLanguages = [
        "zh": "chinese", "en": "english", "de": "german", "it": "italian", "pt": "portuguese",
        "es": "spanish", "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian",
    ]

    /// `nil` = `auto` (no language token). Unknown names throw (speech-swift fell back to auto).
    func languageId(_ language: String) throws -> Int? {
        let l = language.lowercased().trimmingCharacters(in: .whitespaces)
        if l == "auto" { return nil }
        let name = Self.isoLanguages[l] ?? l
        guard let id = languageIds[name] else {
            throw VoiceError.unknownLanguage(language, known: ["auto"] + languages + Self.isoLanguages.keys.sorted())
        }
        return id
    }

    // MARK: Reference

    /// Encode a reference once: codec tokens + speaker embedding. `audio` is mono at
    /// `sampleRate`; it is resampled to 24 kHz with the torchaudio-compatible resampler.
    public func encodeReference(audio: [Float], sampleRate: Int, transcript: String) throws -> VoiceReference {
        guard let enc = codecEncoder, let se = speakerEncoder else {
            throw VoiceError.unsupported("this bundle has no speaker encoder; ICL references need a Base bundle")
        }
        let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw VoiceError.invalidArgument("reference transcript is empty") }
        guard sampleRate > 0, !audio.isEmpty else { throw VoiceError.invalidArgument("reference audio is empty") }
        let wav = Resampler.resample(audio, from: sampleRate, to: 24000)
        guard wav.count >= 1920 else { throw VoiceError.invalidArgument("reference shorter than one codec frame (80 ms)") }
        let codes = enc.encode(samples: wav)  // [1, 16, T]
        let mel = SpeakerMel.compute(audio: wav).asType(computeDType)
        let emb = se(mel).asType(.float32)  // [1, 1024]
        eval(codes, emb)
        let frames = codes.dim(2)
        let flat = codes[0].transposed(1, 0).asType(.int32).asArray(Int32.self)  // [T, 16]
        return VoiceReference(codes: flat, frames: frames, transcript: text, speakerEmbedding: emb.asArray(Float.self))
    }

    public func encodeReference(contentsOf url: URL, transcript: String) throws -> VoiceReference {
        let clip = try AudioIO.read(url)
        return try encodeReference(audio: clip.samples, sampleRate: clip.sampleRate, transcript: transcript)
    }

    // MARK: Generate

    /// ICL voice cloning with a Base bundle.
    public func generate(
        text: String, language: String, reference: VoiceReference, sampling: SamplingOptions = SamplingOptions(),
        seed: UInt64? = nil
    ) throws -> Synthesis {
        var o = GenerationOptions()
        o.sampling = sampling
        o.seed = seed
        return try generate(text: text, language: language, voice: .reference(reference), options: o)
    }

    /// Single-speaker / CustomVoice path (`generate_custom_voice`).
    public func generate(
        text: String, language: String, speaker: String, sampling: SamplingOptions = SamplingOptions(),
        seed: UInt64? = nil
    ) throws -> Synthesis {
        var o = GenerationOptions()
        o.sampling = sampling
        o.seed = seed
        return try generate(text: text, language: language, voice: .speaker(speaker), options: o)
    }

    public func generate(text: String, language: String, voice: Voice, options: GenerationOptions) throws -> Synthesis {
        try run(text: text, language: language, voice: voice, options: options, streaming: nil, onChunk: nil)
    }

    /// Streaming variant: `onChunk` receives audio as codec frames are produced. The
    /// concatenation of all chunks is the utterance (chunk boundaries are decoded with
    /// `leftContextFrames` of left context, so it differs slightly from `generate`).
    @discardableResult
    public func generateStreaming(
        text: String, language: String, voice: Voice, options: GenerationOptions,
        streaming: StreamingOptions = StreamingOptions(), onChunk: (AudioChunk) throws -> Void
    ) throws -> Synthesis {
        try withoutActuallyEscaping(onChunk) { cb in
            try run(text: text, language: language, voice: voice, options: options, streaming: streaming, onChunk: cb)
        }
    }

    /// `AsyncThrowingStream` wrapper around `generateStreaming`. MLX work runs on a
    /// detached task; do not use this model concurrently from elsewhere meanwhile.
    public func stream(
        text: String, language: String, voice: Voice, options: GenerationOptions = GenerationOptions(),
        streaming: StreamingOptions = StreamingOptions()
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try self.generateStreaming(
                        text: text, language: language, voice: voice, options: options, streaming: streaming
                    ) { chunk in
                        try Task.checkCancellation()
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Decode codec tokens (`frames × 16`, frame-major) to 24 kHz audio with the
    /// reference's chunked decode.
    public func decode(codes: [Int32], frames: Int, chunkFrames: Int = 300, leftContextFrames: Int = 25) -> [Float] {
        let arr = MLXArray(codes, [frames, 16]).transposed(1, 0).expandedDimensions(axis: 0)
        let wav = decoder.chunkedDecode(codes: arr, chunkSize: chunkFrames, leftContext: leftContextFrames)
        return wav.reshaped(-1).asArray(Float.self)
    }

    /// Text token ids exactly as the bundle's tokenizer produces them (special tokens
    /// such as `<|im_start|>` are recognised).
    public func tokenize(_ text: String) throws -> [Int] { try tokenizer.encode(text) }

    /// Diagnostics: float32 talker logits at the last prefill position (before any
    /// logits processing), for parity checks.
    public func firstStepLogits(text: String, language: String, voice: Voice, textFeed: TextFeed? = nil) throws -> [Float] {
        let p = try buildPrompt(text: text, language: language, voice: voice, textFeed: textFeed)
        let L = p.prefill.dim(1)
        let (logits, _, _) = talker(inputsEmbeds: p.prefill, offset: MLXArray(Int32(0)), cache: nil)
        return logits[0, L - 1].asType(.float32).asArray(Float.self)
    }
}
