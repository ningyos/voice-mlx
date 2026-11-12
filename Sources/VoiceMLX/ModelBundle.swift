import Foundation
import Hub
import MLX
import MLXNN

/// Where a model or codec bundle comes from.
public enum ModelSource: Sendable, Equatable {
    /// A local bundle directory.
    case directory(URL)
    /// A Hugging Face repo id, cached under `cacheRoot/models/<id>` (default
    /// `~/Library/Caches/qwen3-speech`, the layout speech-swift uses) and downloaded
    /// on first use.
    case hub(String)

    public static let base8bit = ModelSource.hub("aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit")
    public static let customVoiceBF16 = ModelSource.hub("aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-bf16")
    public static let codec = ModelSource.hub("Qwen/Qwen3-TTS-Tokenizer-12Hz")

    public static var defaultCacheRoot: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("qwen3-speech")
    }
}

/// Weight precision of the talker / code-predictor linear layers.
public enum WeightPrecision: String, Sendable, CaseIterable {
    /// MLX affine 8-bit, group size 64 (aufklarer `-MLX-8bit` layout). An unquantized
    /// bundle is quantized at load time.
    case int8
    /// Unquantized; requires an unquantized (bf16) bundle.
    case bf16
}

public struct LoadOptions: Sendable {
    public var precision: WeightPrecision = .int8
    /// Float dtype of activations, embeddings, norms and quantization scales.
    /// `.float32` (with `MLX_ENABLE_TF32=0`) is the parity mode.
    public var computeDType: DType = .bfloat16
    public var quantizationGroupSize = 64
    public var cacheRoot: URL = ModelSource.defaultCacheRoot
    /// Allow downloading `.hub` sources that are not cached yet.
    public var allowDownload = true

    public init() {}
}

/// Parsed `config.json` of a talker bundle.
struct BundleConfig {
    var talker = TalkerConfig()
    var codePredictor = CodePredictorConfig()
    var storedBits = 0
    var storedGroupSize = 64
    var speakers: [String: Int] = [:]
    var speakerDialects: [String: String] = [:]
    var languages: [String: Int] = BundleConfig.officialLanguages

    /// `talker_config.codec_language_id` of Qwen/Qwen3-TTS-12Hz-0.6B-Base (the aufklarer
    /// bundles omit it).
    static let officialLanguages: [String: Int] = [
        "chinese": 2055, "english": 2050, "german": 2053, "italian": 2070, "portuguese": 2071,
        "spanish": 2054, "japanese": 2058, "korean": 2064, "french": 2061, "russian": 2069,
    ]

    init(contentsOf url: URL) throws {
        let dir = url.deletingLastPathComponent()
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw VoiceError.invalidBundle(dir, "config.json is not an object")
        }
        if let q = json["quantization_config"] as? [String: Any] {
            guard let bits = q["bits"] as? Int, bits == 8 else {
                throw VoiceError.invalidBundle(dir, "only 8-bit quantized bundles are supported, got \(q["bits"] ?? "nil")")
            }
            storedBits = bits
            storedGroupSize = q["group_size"] as? Int ?? 64
        }
        if let size = json["model_size"] as? String ?? json["tts_model_size"] as? String,
            !["0.6B", "0b6"].contains(size)
        {
            throw VoiceError.unsupported("model size \(size); only the 0.6B model is supported")
        }
        guard let t = json["talker_config"] as? [String: Any] else { return }
        func int(_ k: String, _ v: inout Int) { if let x = t[k] as? Int { v = x } }
        int("hidden_size", &talker.hiddenSize)
        int("num_hidden_layers", &talker.numLayers)
        int("num_attention_heads", &talker.numHeads)
        int("num_key_value_heads", &talker.numKVHeads)
        int("head_dim", &talker.headDim)
        int("intermediate_size", &talker.intermediateSize)
        int("text_hidden_size", &talker.textHiddenSize)
        int("text_vocab_size", &talker.textVocabSize)
        int("vocab_size", &talker.codecVocabSize)
        if talker.hiddenSize != 1024 {
            throw VoiceError.unsupported("talker hidden size \(talker.hiddenSize); only the 0.6B model is supported")
        }
        if let ids = t["spk_id"] as? [String: Any] {
            for (k, v) in ids {
                // Official configs store either an int or a one-element list.
                if let i = v as? Int { speakers[k.lowercased()] = i }
                else if let a = v as? [Int], let i = a.first { speakers[k.lowercased()] = i }
                else { throw VoiceError.invalidBundle(dir, "spk_id[\(k)] is not an integer") }
            }
        }
        if let d = t["spk_is_dialect"] as? [String: Any] {
            for (k, v) in d { if let s = v as? String { speakerDialects[k.lowercased()] = s.lowercased() } }
        }
        if let l = t["codec_language_id"] as? [String: Int] {
            languages = Dictionary(uniqueKeysWithValues: l.map { ($0.key.lowercased(), $0.value) })
        }
    }
}

enum BundleResolver {
    static let talkerFiles = ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json"]

    static func resolve(_ source: ModelSource, required: [String], options: LoadOptions) async throws -> URL {
        switch source {
        case .directory(let url):
            try check(url, required: required)
            return url
        case .hub(let id):
            let dir = options.cacheRoot.appendingPathComponent("models").appendingPathComponent(id)
            if (try? check(dir, required: required)) != nil { return dir }
            guard options.allowDownload else {
                throw VoiceError.missingFile(dir)
            }
            do {
                let hub = HubApi(downloadBase: options.cacheRoot)
                _ = try await hub.snapshot(
                    from: Hub.Repo(id: id), matching: required + ["*.safetensors", "*.safetensors.index.json"])
            } catch {
                throw VoiceError.download(id, "\(error)")
            }
            try check(dir, required: required)
            return dir
        }
    }

    static func check(_ dir: URL, required: [String]) throws {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir) else {
            throw VoiceError.missingFile(dir)
        }
        guard isDir.boolValue else { throw VoiceError.notADirectory(dir) }
        for f in required where !FileManager.default.fileExists(atPath: dir.appendingPathComponent(f).path) {
            throw VoiceError.missingFile(dir.appendingPathComponent(f))
        }
        _ = try safetensorFiles(in: dir)
    }

    /// The weight files of a bundle: the shards named by `model.safetensors.index.json`
    /// when present, else every `*.safetensors`.
    static func safetensorFiles(in dir: URL) throws -> [URL] {
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: index.path) {
            guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any],
                let map = json["weight_map"] as? [String: String]
            else { throw VoiceError.invalidBundle(dir, "bad model.safetensors.index.json") }
            let files = Set(map.values).sorted().map { dir.appendingPathComponent($0) }
            for f in files where !FileManager.default.fileExists(atPath: f.path) { throw VoiceError.missingFile(f) }
            return files
        }
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        guard !files.isEmpty else { throw VoiceError.missingFile(dir.appendingPathComponent("model.safetensors")) }
        return files
    }

    static func loadWeights(_ dir: URL, dtype: DType) throws -> [String: MLXArray] {
        var all: [String: MLXArray] = [:]
        for f in try safetensorFiles(in: dir) {
            for (k, v) in try MLX.loadArrays(url: f) {
                all[k] = v.dtype.isFloatingPoint && v.dtype != dtype ? v.asType(dtype) : v
            }
        }
        return all
    }
}

/// Runs speech-swift weight-application code and verifies that it set every parameter of
/// each module with a tensor of the expected shape. speech-swift's appliers skip absent
/// keys silently (which would leave initial values behind), and `update(parameters:)`
/// writes in place, so every parameter is first replaced by a lazy sentinel (NaN for
/// floats, all-ones bits for integers) that must be gone afterwards.
func applyStrict(_ modules: [(String, Module)], _ apply: () -> Void) throws {
    var shapes: [[String: [Int]]] = []
    for (_, m) in modules {
        let params = m.parameters().flattened()
        shapes.append(Dictionary(uniqueKeysWithValues: params.map { ($0.0, $0.1.shape) }))
        let sentinels = params.map { (k, v) -> (String, MLXArray) in
            let fill = v.dtype.isFloatingPoint ? MLXArray(Float.nan).asType(v.dtype) : MLXArray(UInt32.max).asType(v.dtype)
            return (k, broadcast(fill, to: v.shape))
        }
        m.update(parameters: ModuleParameters.unflattened(sentinels))
    }
    apply()
    for ((name, module), shapes) in zip(modules, shapes) {
        var missing: [String] = []
        for (k, v) in module.parameters().flattened() {
            if let s = shapes[k], s != v.shape {
                throw VoiceError.weightMismatch("\(name).\(k): expected shape \(s), got \(v.shape)")
            }
            let unset =
                v.dtype.isFloatingPoint
                ? isNaN(v).all().item(Bool.self)
                : (v .== MLXArray(UInt32.max).asType(v.dtype)).all().item(Bool.self)
            if unset { missing.append(k) }
        }
        if !missing.isEmpty {
            throw VoiceError.weightMismatch(
                "\(name): \(missing.count) parameters not in the checkpoint, e.g. \(missing.sorted().prefix(4).joined(separator: ", "))")
        }
    }
}
