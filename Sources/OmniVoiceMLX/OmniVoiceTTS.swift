import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import OmniVoiceAudio

public enum OmniVoiceError: Error, CustomStringConvertible, Equatable {
    case missingBundleFile(String)
    case invalidConfig(String)
    case unknownLanguage(String)
    case emptyText
    case emptyTranscript
    case invalidOption(String)
    case invalidReference(String)
    case nonFiniteOutput

    public var description: String {
        switch self {
        case .missingBundleFile(let p): return "model bundle file missing: \(p)"
        case .invalidConfig(let m): return "invalid bundle config: \(m)"
        case .unknownLanguage(let l):
            return "unknown language '\(l)'; use an OmniVoice language id (en, zh, ja, ...) or English name"
        case .emptyText: return "text to synthesize is empty"
        case .emptyTranscript: return "reference transcript is empty"
        case .invalidOption(let m): return "invalid option: \(m)"
        case .invalidReference(let m): return "invalid reference: \(m)"
        case .nonFiniteOutput: return "codec produced non-finite samples"
        }
    }
}

/// Published MLX bundles.
public enum OmniVoiceBundle: Sendable, Equatable {
    /// `aufklarer/OmniVoice-MLX-int8`: 8-bit (group 64) backbone, fp16 codec. Default.
    case int8
    /// `aufklarer/OmniVoice-MLX-fp16`: fp16 backbone, fp16 codec.
    case fp16
    /// Any Hugging Face repo with the same layout.
    case repo(String)

    public var repoId: String {
        switch self {
        case .int8: return "aufklarer/OmniVoice-MLX-int8"
        case .fp16: return "aufklarer/OmniVoice-MLX-fp16"
        case .repo(let id): return id
        }
    }

    /// Files a bundle directory must contain.
    public static let requiredFiles = [
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "audio_tokenizer/model.safetensors",
    ]

    /// Default download root; bundles land in `<root>/models/<repo id>`. Shared
    /// with speech-swift so already-downloaded bundles are reused.
    public static var defaultCacheRoot: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("qwen3-speech")
    }
}

/// An encoded reference voice: codec tokens + transcript + loudness. Encode
/// once with `OmniVoiceTTS.encodeReference` and reuse for every sentence.
/// Codable so it can be stored and reloaded without re-running the encoder.
public struct VoiceReference: Sendable, Codable, Equatable {
    /// Row-major `[codebooks, frames]` codec tokens.
    public let codes: [Int32]
    public let codebooks: Int
    public let frames: Int
    /// Transcript as fed to the model (after `add_punctuation` when preprocessed).
    public let transcript: String
    /// RMS of the reference waveform before normalisation (`ref_rms`).
    public let rms: Float

    public init(codes: [Int32], codebooks: Int, frames: Int, transcript: String, rms: Float) throws {
        guard codebooks > 0, frames > 0, codes.count == codebooks * frames else {
            throw OmniVoiceError.invalidReference(
                "codes count \(codes.count) != \(codebooks) x \(frames)")
        }
        self.codes = codes
        self.codebooks = codebooks
        self.frames = frames
        self.transcript = transcript
        self.rms = rms
    }

    var tokens: MLXArray { MLXArray(codes, [codebooks, frames]) }
}

/// Options for reference encoding (mirrors `create_voice_clone_prompt`).
public struct ReferenceOptions: Sendable {
    /// RMS-normalise, remove silences (mid 200 ms, lead 100 ms, trail 200 ms) and
    /// add final punctuation to the transcript, as `preprocess_prompt=True` does.
    public var preprocess: Bool = true
    /// Keep only the first `maxSeconds` of the (preprocessed) reference; `nil` keeps
    /// everything. The transcript is NOT cut, so only use this with a transcript
    /// that matches the kept audio.
    public var maxSeconds: Double? = nil
    public init() {}
}

/// Options for one synthesis call.
public struct GenerationOptions: Sendable {
    public var diffusion = DiffusionOptions()
    /// PRNG seed for the Gumbel sampling. `nil` draws a fresh seed (reported in
    /// `Synthesis.seed`). Ignored when decoding is greedy.
    public var seed: UInt64? = nil
    /// Fixed output length in seconds; `nil` estimates it from the text and the
    /// reference pace (`RuleDurationEstimator`).
    public var duration: Double? = nil
    /// Speaking-rate factor for the estimate (`speed`); ignored when `duration` is set.
    public var speed: Double = 1.0
    /// Style instruction, passed through verbatim (no `_resolve_instruct` normalisation).
    public var instruct: String? = nil
    /// Prepend `<|denoise|>` when cloning (`denoise`).
    public var denoise: Bool = true
    /// Remove long silences from the output (`postprocess_output`).
    public var removeSilence: Bool = true
    /// Edge padding / linear fade in seconds (`pad_duration`, `fade_duration`).
    public var padSeconds: Double = 0.1
    public var fadeSeconds: Double = 0.1
    public init() {}
}

/// Result of `synthesize`.
public struct Synthesis: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    /// Row-major `[codebooks, frames]` generated codec tokens.
    public let codes: [Int32]
    public let frames: Int
    public let seed: UInt64?
}

/// OmniVoice zero-shot voice-cloning TTS on MLX: Qwen3 bidirectional diffusion
/// backbone + Higgs-audio v2 codec (encode + decode) + Qwen2 BPE front-end.
public final class OmniVoiceTTS {
    public let config: OmniVoiceConfig
    let model: OmniVoiceModel
    let codec: OmniVoiceCodec
    let encoder: OmniVoiceCodecEncoder
    let builder: OmniVoiceInputBuilder
    /// Directory the bundle was loaded from.
    public let bundleDirectory: URL

    /// Output sample rate (24 kHz).
    public var sampleRate: Int { config.sampleRate }
    /// Codec frames per second (24000 / hop 960 = 25).
    public var frameRate: Double { Double(config.sampleRate) / Double(config.hopLength) }

    /// Load from a local bundle directory (`model.safetensors`, `config.json`,
    /// `tokenizer.json`, `tokenizer_config.json`, `audio_tokenizer/model.safetensors`).
    /// A `quantization` block in `config.json` selects the quantized backbone.
    /// - Parameter computeDType: dtype for backbone float parameters/activations.
    public init(directory: URL, computeDType: DType = .float32) async throws {
        let fm = FileManager.default
        for f in OmniVoiceBundle.requiredFiles
        where !fm.fileExists(atPath: directory.appendingPathComponent(f).path) {
            throw OmniVoiceError.missingBundleFile(directory.appendingPathComponent(f).path)
        }
        let cfg = OmniVoiceConfig()
        let quant = try Self.readQuantization(directory.appendingPathComponent("config.json"))

        let model = OmniVoiceModel(cfg)
        try model.loadWeights(
            try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors")),
            quantization: quant, computeDType: computeDType)

        let codecRaw = try MLX.loadArrays(
            url: directory.appendingPathComponent("audio_tokenizer/model.safetensors"))
        let codec = OmniVoiceCodec()
        try codec.loadWeights(codecRaw)
        let encoder = OmniVoiceCodecEncoder()
        try encoder.loadWeights(codecRaw)

        let tokenizer = try await OmniVoiceTokenizer.load(from: directory)
        self.config = cfg
        self.model = model
        self.codec = codec
        self.encoder = encoder
        self.builder = OmniVoiceInputBuilder(tokenizer: tokenizer, config: cfg)
        self.bundleDirectory = directory
    }

    /// Resolve a bundle from the local cache, downloading missing files from
    /// Hugging Face unless `offline` is set, then load it.
    public static func load(
        _ bundle: OmniVoiceBundle = .int8,
        cacheRoot: URL = OmniVoiceBundle.defaultCacheRoot,
        offline: Bool = false,
        computeDType: DType = .float32
    ) async throws -> OmniVoiceTTS {
        let dir = try await resolveBundle(bundle, cacheRoot: cacheRoot, offline: offline)
        return try await OmniVoiceTTS(directory: dir, computeDType: computeDType)
    }

    /// Local directory of `bundle`, downloading it first when files are missing.
    public static func resolveBundle(
        _ bundle: OmniVoiceBundle, cacheRoot: URL = OmniVoiceBundle.defaultCacheRoot,
        offline: Bool = false
    ) async throws -> URL {
        let dir = cacheRoot.appendingPathComponent("models").appendingPathComponent(bundle.repoId)
        let missing = OmniVoiceBundle.requiredFiles.filter {
            !FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
        if missing.isEmpty { return dir }
        if offline {
            throw OmniVoiceError.missingBundleFile(dir.appendingPathComponent(missing[0]).path)
        }
        let hub = HubApi(downloadBase: cacheRoot)
        return try await hub.snapshot(
            from: Hub.Repo(id: bundle.repoId),
            matching: ["*.safetensors", "*.json", "audio_tokenizer/*"])
    }

    // MARK: Reference

    /// Encode a reference clip once. Loads any format AVFoundation reads,
    /// averages channels, resamples to 24 kHz (torchaudio kernel), then applies
    /// the `create_voice_clone_prompt` preprocessing.
    public func encodeReference(
        audio url: URL, transcript: String, options: ReferenceOptions = .init()
    ) throws -> VoiceReference {
        let clip = try AudioIO.read(url)
        return try encodeReference(
            samples: clip.samples, sampleRate: clip.sampleRate, transcript: transcript,
            options: options)
    }

    public func encodeReference(
        samples: [Float], sampleRate: Int, transcript: String,
        options: ReferenceOptions = .init(), profiler: StageProfiler? = nil
    ) throws -> VoiceReference {
        let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw OmniVoiceError.emptyTranscript }
        guard !samples.isEmpty else { throw OmniVoiceError.invalidReference("no audio samples") }
        guard sampleRate > 0 else { throw OmniVoiceError.invalidReference("sample rate \(sampleRate)") }
        if let m = options.maxSeconds, !(m > 0) {
            throw OmniVoiceError.invalidOption("maxSeconds must be > 0")
        }
        let wav24 = Resampler.resample(samples, from: sampleRate, to: config.sampleRate)
        var (prepared, rms) = try AudioPostprocess.prepareReference(
            wav24, sampleRate: config.sampleRate, hopLength: config.hopLength,
            preprocess: options.preprocess)
        if let m = options.maxSeconds {
            let keep = Int(m * Double(config.sampleRate)) / config.hopLength * config.hopLength
            if keep < prepared.count { prepared.removeSubrange(keep...) }
        }
        guard prepared.count >= config.hopLength else {
            throw OmniVoiceError.invalidReference(
                "reference shorter than one codec frame after preprocessing")
        }
        let tokens = try (profiler ?? StageProfiler()).measure("reference.encode") { () -> MLXArray in
            let t = encoder.encode(MLXArray(prepared).reshaped([1, 1, prepared.count]))
            eval(t)
            return t
        }
        let frames = tokens.dim(2)
        let codes = tokens.reshaped([-1]).asType(.int32).asArray(Int32.self)
        return try VoiceReference(
            codes: codes, codebooks: tokens.dim(1), frames: frames,
            transcript: options.preprocess ? TextPrep.addPunctuation(text) : text, rms: rms)
    }

    // MARK: Generation

    /// Synthesize `text` in the reference voice; returns 24 kHz mono samples.
    public func generate(
        text: String, language: String?, reference: VoiceReference,
        steps: Int = 16, guidance: Float = 2.0, duration: Double? = nil,
        instruct: String? = nil, seed: UInt64? = nil
    ) throws -> [Float] {
        var o = GenerationOptions()
        o.diffusion.steps = steps
        o.diffusion.guidance = guidance
        o.duration = duration
        o.instruct = instruct
        o.seed = seed
        return try synthesize(text: text, language: language, reference: reference, options: o).samples
    }

    /// Full synthesis with options; returns samples plus the generated tokens and
    /// the seed used. With `profiler`, stage boundaries are evaluated and timed.
    public func synthesize(
        text: String, language: String?, reference: VoiceReference?,
        options: GenerationOptions = .init(), profiler: StageProfiler? = nil
    ) throws -> Synthesis {
        let p = profiler ?? StageProfiler()
        let codes = try p.measure("tokens.generate") {
            try generateCodes(text: text, language: language, reference: reference,
                              options: options, profiler: profiler)
        }
        let wav = try decode(codes: codes.codes, frames: codes.frames, profiler: profiler)
        guard wav.allSatisfy({ $0.isFinite }) else { throw OmniVoiceError.nonFiniteOutput }
        let out = p.measure("postprocess") {
            AudioPostprocess.postprocessGenerated(
                wav, sampleRate: config.sampleRate, referenceRMS: reference?.rms,
                removeSilence: options.removeSilence, padSeconds: options.padSeconds,
                fadeSeconds: options.fadeSeconds)
        }
        return Synthesis(
            samples: out, sampleRate: config.sampleRate, codes: codes.codes,
            frames: codes.frames, seed: codes.seed)
    }

    /// Diffusion only: returns row-major `[codebooks, frames]` tokens.
    public func generateCodes(
        text: String, language: String?, reference: VoiceReference?,
        options: GenerationOptions = .init(), profiler: StageProfiler? = nil
    ) throws -> (codes: [Int32], frames: Int, seed: UInt64?) {
        let target = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { throw OmniVoiceError.emptyText }
        let d = options.diffusion
        guard d.steps > 0 else { throw OmniVoiceError.invalidOption("steps must be > 0") }
        guard d.positionTemperature >= 0, d.classTemperature >= 0 else {
            throw OmniVoiceError.invalidOption("temperatures must be >= 0")
        }
        guard d.classTopKRatio > 0, d.classTopKRatio <= 1 else {
            throw OmniVoiceError.invalidOption("classTopKRatio must be in (0, 1]")
        }
        if let dur = options.duration, !(dur > 0) {
            throw OmniVoiceError.invalidOption("duration must be > 0")
        }
        guard options.speed > 0 else { throw OmniVoiceError.invalidOption("speed must be > 0") }
        if let r = reference, r.codebooks != config.numAudioCodebook {
            throw OmniVoiceError.invalidReference(
                "reference has \(r.codebooks) codebooks, model expects \(config.numAudioCodebook)")
        }
        let p = profiler ?? StageProfiler()
        let (ids, mask, targetLen) = try p.measure("input.build") { () -> (MLXArray, MLXArray, Int) in
            let r = try prepareInputs(text: text, language: language, reference: reference, options: options)
            if profiler != nil { eval(r.0, r.1) }
            return r
        }

        var seed: UInt64? = nil
        var key: MLXArray? = nil
        if d.isStochastic {
            let s = options.seed ?? UInt64.random(in: 0 ... UInt64.max)
            seed = s
            key = MLXRandom.key(s)
        }
        let tokens = model.generateTokens(
            condInputIds: ids, audioMask: mask, targetLen: targetLen, options: d, key: key,
            profiler: profiler)
        let codes = tokens.reshaped([-1]).asArray(Int32.self)
        return (codes, targetLen, seed)
    }

    /// Conditioning ids `[1, C, L]`, audio mask `[1, L]` and target length,
    /// mirroring `_preprocess_all` + `_prepare_inference_inputs`.
    func prepareInputs(
        text: String, language: String?, reference: VoiceReference?, options: GenerationOptions
    ) throws -> (MLXArray, MLXArray, Int) {
        let lang = try Self.resolveLanguage(language)
        let targetLen: Int
        if let dur = options.duration {
            targetLen = max(1, Int(dur * frameRate))
        } else {
            // The reference estimates from the raw (unstripped) text.
            var est = RuleDurationEstimator.shared.estimateTargetTokensRaw(
                targetText: text, refText: reference?.transcript,
                numRefAudioTokens: reference?.frames)
            if options.speed != 1.0 { est /= options.speed }
            targetLen = max(1, Int(est))
        }
        let (ids, mask) = builder.buildInputs(
            text: text, refText: reference?.transcript, lang: lang,
            refAudioTokens: reference?.tokens, targetLen: targetLen,
            denoise: options.denoise, instruct: options.instruct)
        return (ids, mask, targetLen)
    }

    /// Host copy of the conditioning inputs (for parity checks): row-major
    /// `[C, L]` ids, `[L]` audio mask, and the target length.
    public func inspectInputs(
        text: String, language: String?, reference: VoiceReference?,
        options: GenerationOptions = .init()
    ) throws -> (inputIds: [Int32], length: Int, audioMask: [Int32], targetLen: Int) {
        let (ids, mask, t) = try prepareInputs(
            text: text, language: language, reference: reference, options: options)
        return (ids.asType(.int32).reshaped([-1]).asArray(Int32.self), ids.dim(2),
                mask.asType(.int32).reshaped([-1]).asArray(Int32.self), t)
    }

    /// Logits of the first diffusion step (all-mask target) for parity checks:
    /// row-major `[C, T, V]` cond (target slice) and uncond, as Float32.
    public func firstStepLogits(
        text: String, language: String?, reference: VoiceReference?,
        options: GenerationOptions = .init()
    ) throws -> (cond: [Float], uncond: [Float], targetLen: Int) {
        let (ids, mask, t) = try prepareInputs(
            text: text, language: language, reference: reference, options: options)
        let L = ids.dim(2)
        let cond = model(inputIds: ids, audioMask: mask)[0..., 0..., (L - t) ..< L, 0...].asType(.float32)
        let uncond = model(
            inputIds: ids[0..., 0..., (L - t) ..< L],
            audioMask: MLXArray.ones([1, t], type: Int32.self)).asType(.float32)
        return (cond.reshaped([-1]).asArray(Float.self), uncond.reshaped([-1]).asArray(Float.self), t)
    }

    /// Codec decode of `[codebooks, frames]` tokens to raw 24 kHz samples (no post-processing).
    public func decode(codes: [Int32], frames: Int, profiler: StageProfiler? = nil) throws -> [Float] {
        let C = config.numAudioCodebook
        guard frames > 0, codes.count == C * frames else {
            throw OmniVoiceError.invalidOption("codes count \(codes.count) != \(C) x \(frames)")
        }
        guard codes.allSatisfy({ $0 >= 0 && $0 < Int32(config.audioMaskId) }) else {
            throw OmniVoiceError.invalidOption("codes must be in 0..<\(config.audioMaskId)")
        }
        return (profiler ?? StageProfiler()).measure("codec.decode") {
            let wav = codec.decode(MLXArray(codes, [1, C, frames]))
            return wav.asType(.float32).asArray(Float.self)
        }
    }

    // MARK: Helpers

    /// `_resolve_language`: nil/"none" → nil; known id → id; English name → id;
    /// anything else throws (the reference silently falls back to no language).
    public static func resolveLanguage(_ language: String?) throws -> String? {
        guard let language, language.lowercased() != "none" else { return nil }
        if LanguageMap.ids.contains(language) { return language }
        if let id = LanguageMap.nameToId[language.lowercased()] { return id }
        throw OmniVoiceError.unknownLanguage(language)
    }

    private static func readQuantization(_ url: URL) throws -> (groupSize: Int, bits: Int)? {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OmniVoiceError.invalidConfig("config.json is not an object")
        }
        guard let q = json["quantization"] else { return nil }
        guard let qd = q as? [String: Any], let g = qd["group_size"] as? Int, let b = qd["bits"] as? Int
        else { throw OmniVoiceError.invalidConfig("quantization block lacks group_size/bits") }
        return (g, b)
    }
}
