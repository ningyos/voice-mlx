import Foundation
import MLX
import MLXRandom
import VoiceAudio
import XCTest

@testable import VoiceMLX

private let cache = ModelSource.defaultCacheRoot.appendingPathComponent("models")
private let baseDir = cache.appendingPathComponent("aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit")
private let customDir = cache.appendingPathComponent("aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-bf16")
private let codecDir = cache.appendingPathComponent("Qwen/Qwen3-TTS-Tokenizer-12Hz")
private let refWav = URL(fileURLWithPath: "/tmp/omnivoice/ref2.wav")

private func exists(_ u: URL) -> Bool { FileManager.default.fileExists(atPath: u.path) }

/// Shared models (loading takes a few seconds; MLX is not re-entrant across tests).
private enum Shared {
    static let base: Qwen3Voice? = {
        guard exists(baseDir), exists(codecDir) else { return nil }
        MLX.Memory.cacheLimit = 256 << 20
        return try? Qwen3Voice(modelDirectory: baseDir, codecDirectory: codecDir)
    }()
    static let reference: VoiceReference? = {
        guard let base, exists(refWav) else { return nil }
        // First 6 s of the real recording keeps the tests fast.
        guard let clip = try? AudioIO.read(refWav) else { return nil }
        let six = Array(clip.samples.prefix(clip.sampleRate * 6))
        return try? base.encodeReference(
            audio: six, sampleRate: clip.sampleRate,
            transcript: "<exactly what is said in the reference recording>")
    }()
}

final class TokenizerTests: XCTestCase {
    func testMatchesQwen2TokenizerFast() throws {
        guard exists(baseDir) else { throw XCTSkip("Base bundle not cached") }
        let tok = try TextTokenizer(directory: baseDir)
        // Expected ids from transformers' Qwen2TokenizerFast loaded from the official
        // Qwen/Qwen3-TTS-12Hz-0.6B-Base snapshot.
        let cases: [(String, [Int])] = [
            ("<|im_start|>assistant\nHello, world!<|im_end|>\n",
             [151644, 77091, 198, 9707, 11, 1879, 0, 151645, 198]),
            ("I'm 9mm   ok\n\n  x", [40, 2776, 220, 24, 3821, 256, 5394, 271, 220, 856]),
            ("這個 bug 你昨天也遇過，記得嗎？",
             [104446, 9876, 220, 56568, 104373, 74763, 99688, 99751, 3837, 113983, 105793, 11319]),
            ("Café naïve — “quotes” 123456",
             [34, 2577, 963, 94880, 586, 1959, 1036, 53282, 854, 220, 16, 17, 18, 19, 20, 21]),
        ]
        for (text, ids) in cases {
            XCTAssertEqual(try tok.encode(text), ids, text)
        }
        // Special tokens split out, plain text round-trips through the template slicing.
        let ids = try tok.encode("<|im_start|>assistant\n好，我看一下。<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(Array(ids.prefix(3)), [151644, 77091, 198])
        XCTAssertEqual(Array(ids.suffix(5)), [151645, 198, 151644, 77091, 198])
    }
}

final class BundleLoadingTests: XCTestCase {
    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("vmlx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    func testMissingDirectoryThrows() {
        let nowhere = URL(fileURLWithPath: "/nonexistent/voice-mlx-bundle")
        XCTAssertThrowsError(try Qwen3Voice(modelDirectory: nowhere, codecDirectory: nowhere)) { e in
            XCTAssertEqual(e as? VoiceError, .missingFile(nowhere))
        }
    }

    func testMissingRequiredFileThrows() throws {
        guard exists(baseDir), exists(codecDir) else { throw XCTSkip("bundles not cached") }
        let d = try tempDir()
        for f in ["config.json", "merges.txt", "tokenizer_config.json"] {
            try FileManager.default.copyItem(at: baseDir.appendingPathComponent(f), to: d.appendingPathComponent(f))
        }
        XCTAssertThrowsError(try Qwen3Voice(modelDirectory: d, codecDirectory: codecDir)) { e in
            XCTAssertEqual(e as? VoiceError, .missingFile(d.appendingPathComponent("vocab.json")))
        }
    }

    func testIncompleteWeightsThrow() throws {
        guard exists(baseDir), exists(codecDir) else { throw XCTSkip("bundles not cached") }
        let d = try tempDir()
        for f in BundleResolver.talkerFiles {
            try FileManager.default.copyItem(at: baseDir.appendingPathComponent(f), to: d.appendingPathComponent(f))
        }
        try MLX.save(arrays: ["talker.model.norm.weight": MLXArray.ones([1024])], url: d.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try Qwen3Voice(modelDirectory: d, codecDirectory: codecDir)) { e in
            guard case .weightMismatch = e as? VoiceError else { return XCTFail("\(e)") }
        }
    }

    func testQuantizedBundleRejectsBF16() throws {
        guard exists(baseDir), exists(codecDir) else { throw XCTSkip("bundles not cached") }
        var o = LoadOptions()
        o.precision = .bf16
        XCTAssertThrowsError(try Qwen3Voice(modelDirectory: baseDir, codecDirectory: codecDir, options: o)) { e in
            guard case .precisionUnavailable = e as? VoiceError else { return XCTFail("\(e)") }
        }
    }
}

final class GenerationTests: XCTestCase {
    private func model() throws -> (Qwen3Voice, VoiceReference) {
        guard let m = Shared.base, let r = Shared.reference else { throw XCTSkip("Base bundle / reference audio missing") }
        return (m, r)
    }

    func testUnknownLanguageThrows() throws {
        let (m, r) = try model()
        XCTAssertThrowsError(try m.generate(text: "Hi.", language: "klingon", reference: r, seed: 0)) { e in
            guard case .unknownLanguage("klingon", _) = e as? VoiceError else { return XCTFail("\(e)") }
        }
        // ISO codes, names and auto are accepted.
        for l in ["en", "English", "auto", "zh"] { XCTAssertNoThrow(try m.languageId(l)) }
    }

    func testSpeakerPathNeedsSpeakerBundle() throws {
        let (m, _) = try model()
        XCTAssertThrowsError(try m.generate(text: "Hi.", language: "auto", speaker: "nobody", seed: 0)) { e in
            XCTAssertEqual(e as? VoiceError, .unknownSpeaker("nobody", known: []))
        }
    }

    func testSeedReproducibility() throws {
        let (m, r) = try model()
        var s = SamplingOptions()
        s.maxNewTokens = 40
        let a = try m.generate(text: "好，我看一下。", language: "zh", reference: r, sampling: s, seed: 7)
        let b = try m.generate(text: "好，我看一下。", language: "zh", reference: r, sampling: s, seed: 7)
        let c = try m.generate(text: "好，我看一下。", language: "zh", reference: r, sampling: s, seed: 8)
        XCTAssertEqual(a.codes, b.codes)
        XCTAssertEqual(a.samples, b.samples)
        XCTAssertNotEqual(a.codes, c.codes)
        // An unseeded run reports a seed that replays it.
        let u = try m.generate(text: "好，我看一下。", language: "zh", reference: r, sampling: s, seed: nil)
        let replay = try m.generate(text: "好，我看一下。", language: "zh", reference: r, sampling: s, seed: u.seed)
        XCTAssertEqual(u.codes, replay.codes)
    }

    func testReferenceReuseAndCodable() throws {
        let (m, r) = try model()
        let decoded = try JSONDecoder().decode(VoiceReference.self, from: JSONEncoder().encode(r))
        XCTAssertEqual(decoded, r)
        var s = SamplingOptions.greedy
        s.maxNewTokens = 30
        let a = try m.generate(text: "Okay.", language: "en", reference: r, sampling: s, seed: 0)
        let b = try m.generate(text: "Okay.", language: "en", reference: decoded, sampling: s, seed: 0)
        XCTAssertEqual(a.codes, b.codes)
        XCTAssertGreaterThan(a.frames, 0)
        XCTAssertEqual(a.samples.count, a.frames * 1920)
    }

    func testStreamingCoversUtterance() throws {
        let (m, r) = try model()
        var o = GenerationOptions()
        o.sampling.maxNewTokens = 40
        o.seed = 3
        var chunks: [AudioChunk] = []
        let s = try m.generateStreaming(text: "你今天吃飯了沒？", language: "zh", voice: .reference(r), options: o) {
            chunks.append($0)
        }
        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.last!.isFinal)
        XCTAssertEqual(chunks.map(\.samples.count).reduce(0, +), s.frames * 1920)
        XCTAssertEqual(chunks.last!.frames.upperBound, s.frames)
        // Same seed without streaming yields the same tokens, and (reference + target is
        // under the reference's 300-frame decode chunk) the same audio up to float rounding.
        let full = try m.generate(text: "你今天吃飯了沒？", language: "zh", voice: .reference(r), options: o)
        XCTAssertEqual(full.codes, s.codes)
        XCTAssertGreaterThan(snrDB(full.samples, chunks.flatMap(\.samples)), 30)
    }
}

/// SNR of `x` against `ref` in dB.
private func snrDB(_ ref: [Float], _ x: [Float]) -> Double {
    precondition(ref.count == x.count)
    var sig = 0.0, err = 0.0
    for (a, b) in zip(ref, x) {
        sig += Double(a) * Double(a)
        err += Double(a - b) * Double(a - b)
    }
    return 10 * log10(sig / max(err, 1e-30))
}

final class CodePredictorTests: XCTestCase {
    /// The compiled 15-group loop (production) gives the same codebook tokens as the same
    /// loop run eagerly, sampled with fixed keys and greedy.
    func testCompiledFrameMatchesEagerLoop() throws {
        guard let m = Shared.base else { throw XCTSkip("Base bundle missing") }
        let groups = m.cpConfig.numCodeGroups - 1
        for (i, sampling) in [SamplingOptions(), SamplingOptions.greedy].enumerated() {
            let greedy = !sampling.subtalkerDoSample
            for f in 0 ..< 8 {
                let hidden = MLXRandom.normal([1, 1, m.cpConfig.hiddenSize], key: MLXRandom.key(UInt64(100 * i + f)))
                    .asType(m.computeDType)
                let code0 = MLXArray(Int32(37 * f + 5))
                let key = MLXRandom.key(UInt64(7 + f))
                let compiled = m.predictFrame(hidden: hidden, code0: code0, sampling: sampling, key: key)
                let gumbels = greedy
                    ? MLXArray.zeros([groups, m.cpConfig.vocabSize])
                    : MLXRandom.gumbel([groups, m.cpConfig.vocabSize], key: key)
                let input = concatenated([hidden, m.talker.embedCodec(code0.reshaped(1, 1))], axis: 1)
                let eager = m.codePredictor.predictFrame(
                    inputsEmbeds: input, temperature: MLXArray(sampling.subtalkerTemperature), gumbels: gumbels,
                    topK: greedy ? 0 : sampling.subtalkerTopK, greedy: greedy)
                XCTAssertEqual(compiled.asArray(Int32.self), eager.asArray(Int32.self), "sampling \(i), frame \(f)")
            }
        }
    }
}

final class CodecDecoderStreamingTests: XCTestCase {
    /// A small randomly initialised decoder (deterministic): window 8 so the attention
    /// caches evict within a short fixture; every conv / transposed-conv shape kind is kept.
    private func smallDecoder() -> SpeechTokenizerDecoder {
        var c = SpeechTokenizerDecoderConfig()
        c.latentDim = 32
        c.decoderDim = 64
        c.hiddenSize = 32
        c.numHeads = 4
        c.numKVHeads = 4
        c.headDim = 8
        c.numLayers = 2
        c.upsampleRates = [4, 3]
        c.semanticCodebookSize = 64
        c.acousticCodebookSize = 64
        c.codebookDim = 16
        c.slidingWindow = 8
        MLXRandom.seed(0)
        let d = SpeechTokenizerDecoder(config: c)
        eval(d)
        return d
    }

    // On the CPU stream: float32 matmuls are exact there, while the M5 GPU runs them as TF32
    // (~60 dB), which would hide a state bug of similar size.
    func testChunkedStatefulDecodeEqualsWholeDecode() { Device.withDefaultDevice(.cpu) { chunkedEqualsWhole() } }

    func testStateIsChunkingInvariant() { Device.withDefaultDevice(.cpu) { chunkingInvariant() } }

    private func chunkedEqualsWhole() {
        let d = smallDecoder()
        let frames = 41
        let codes = MLXRandom.randInt(0 ..< 64, [1, 16, frames], key: MLXRandom.key(1)).asType(.int32)
        let whole = d(codes).reshaped(-1).asArray(Float.self)
        // Uneven chunks, incl. single frames and chunks longer than the window.
        var state = d.freshState()
        var streamed: [Float] = []
        var at = 0
        for n in [1, 4, 12, 1, 9, 3, 11] {
            let end = min(at + n, frames)
            streamed += d.decodeStep(codes[0..., 0..., at ..< end], state: &state).reshaped(-1).asArray(Float.self)
            at = end
        }
        XCTAssertEqual(at, frames)
        XCTAssertEqual(state.frames, frames)
        XCTAssertEqual(streamed.count, whole.count)
        XCTAssertGreaterThan(snrDB(whole, streamed), 90)
    }

    private func chunkingInvariant() {
        // A state built in one chunk continues exactly like one built frame by frame.
        let d = smallDecoder()
        let codes = MLXRandom.randInt(0 ..< 64, [1, 16, 30], key: MLXRandom.key(2)).asType(.int32)
        var a = d.freshState()
        _ = d.decodeStep(codes[0..., 0..., 0 ..< 20], state: &a)
        var b = d.freshState()
        for i in 0 ..< 20 { _ = d.decodeStep(codes[0..., 0..., i ..< (i + 1)], state: &b) }
        let tailA = d.decodeStep(codes[0..., 0..., 20...], state: &a).reshaped(-1).asArray(Float.self)
        let tailB = d.decodeStep(codes[0..., 0..., 20...], state: &b).reshaped(-1).asArray(Float.self)
        XCTAssertGreaterThan(snrDB(tailA, tailB), 90)
    }
}
