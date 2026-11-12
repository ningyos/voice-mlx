import Foundation
import MLX
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
        // Same seed without streaming yields the same tokens.
        let full = try m.generate(text: "你今天吃飯了沒？", language: "zh", voice: .reference(r), options: o)
        XCTAssertEqual(full.codes, s.codes)
    }
}
