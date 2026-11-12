import Foundation
import MLX
import XCTest

@testable import OmniVoiceMLX

/// Model-backed tests use the int8 bundle from the local cache (never download)
/// and skip when it is absent. Override the location with OMNIVOICE_BUNDLE_DIR.
final class OmniVoiceMLXTests: XCTestCase {
    static let referenceAudio = URL(fileURLWithPath:
        ProcessInfo.processInfo.environment["OMNIVOICE_TEST_REF"] ?? "/tmp/omnivoice/ref_plus_zh.wav")
    static let referenceText =
        "<exactly what is said in the reference recording> Let's talk about the 9mm Parabellum. Designed in 1901, it is the most popular handgun cartridge today. With its low recoil and high reliability, it's a perfect choice for target practice and self-defense across the world. 好，我看一下。"

    nonisolated(unsafe) static var shared: OmniVoiceTTS?
    nonisolated(unsafe) static var sharedRef: VoiceReference?

    func model() async throws -> (OmniVoiceTTS, VoiceReference) {
        if let m = Self.shared, let r = Self.sharedRef { return (m, r) }
        let dir = ProcessInfo.processInfo.environment["OMNIVOICE_BUNDLE_DIR"].map { URL(fileURLWithPath: $0) }
            ?? OmniVoiceBundle.defaultCacheRoot.appendingPathComponent("models/aufklarer/OmniVoice-MLX-int8")
        let present = OmniVoiceBundle.requiredFiles.allSatisfy {
            FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
        try XCTSkipUnless(present, "int8 bundle not in cache: \(dir.path)")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.referenceAudio.path),
                          "reference audio missing: \(Self.referenceAudio.path)")
        MLX.Memory.cacheLimit = 512 * 1024 * 1024
        let m = try await OmniVoiceTTS(directory: dir)
        let r = try m.encodeReference(audio: Self.referenceAudio, transcript: Self.referenceText)
        Self.shared = m
        Self.sharedRef = r
        return (m, r)
    }

    func fastOptions(seed: UInt64?) -> GenerationOptions {
        var o = GenerationOptions()
        o.diffusion.steps = 4
        o.seed = seed
        return o
    }

    // MARK: Pure logic (no model)

    func testScheduleCoversAllTokens() {
        for (total, steps) in [(8, 1), (800, 16), (1234, 32), (8 * 37, 7)] {
            let s = OmniVoiceModel.unmaskSchedule(totalMask: total, steps: steps, tShift: 0.1)
            XCTAssertEqual(s.count, steps)
            XCTAssertEqual(s.reduce(0, +), total)
            XCTAssertTrue(s.allSatisfy { $0 >= 0 })
        }
    }

    func testLanguageResolution() throws {
        XCTAssertEqual(try OmniVoiceTTS.resolveLanguage("en"), "en")
        XCTAssertEqual(try OmniVoiceTTS.resolveLanguage("English"), "en")
        XCTAssertEqual(try OmniVoiceTTS.resolveLanguage("zh"), "zh")
        XCTAssertNil(try OmniVoiceTTS.resolveLanguage(nil))
        XCTAssertNil(try OmniVoiceTTS.resolveLanguage("None"))
        XCTAssertThrowsError(try OmniVoiceTTS.resolveLanguage("klingon-xx")) {
            XCTAssertEqual($0 as? OmniVoiceError, .unknownLanguage("klingon-xx"))
        }
    }

    func testVoiceReferenceValidatesShape() {
        XCTAssertThrowsError(try VoiceReference(codes: [1, 2, 3], codebooks: 8, frames: 1, transcript: "a", rms: 0.1))
        XCTAssertNoThrow(try VoiceReference(codes: Array(repeating: 1, count: 16), codebooks: 8, frames: 2,
                                            transcript: "a", rms: 0.1))
    }

    func testMissingBundleFileThrows() async {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ov-empty-\(UUID())")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        do {
            _ = try await OmniVoiceTTS(directory: dir)
            XCTFail("expected missingBundleFile")
        } catch let e as OmniVoiceError {
            guard case .missingBundleFile = e else { return XCTFail("wrong error \(e)") }
        } catch {
            XCTFail("unexpected \(error)")
        }
    }

    func testOfflineResolveOfMissingBundleThrows() async {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ov-cache-\(UUID())")
        do {
            _ = try await OmniVoiceTTS.resolveBundle(.repo("nobody/nothing"), cacheRoot: root, offline: true)
            XCTFail("expected missingBundleFile")
        } catch let e as OmniVoiceError {
            guard case .missingBundleFile = e else { return XCTFail("wrong error \(e)") }
        } catch {
            XCTFail("unexpected \(error)")
        }
    }

    // MARK: Model-backed

    func testReferenceReuseAndCodableRoundTrip() async throws {
        let (m, ref) = try await model()
        XCTAssertEqual(ref.codebooks, 8)
        XCTAssertGreaterThan(ref.frames, 100)
        XCTAssertTrue(ref.transcript.hasSuffix("。"))
        let data = try JSONEncoder().encode(ref)
        let back = try JSONDecoder().decode(VoiceReference.self, from: data)
        XCTAssertEqual(back, ref)
        // The same reference object drives several sentences unchanged.
        let a = try m.generateCodes(text: "你好。", language: "zh", reference: ref, options: fastOptions(seed: 1))
        let b = try m.generateCodes(text: "Hello there.", language: "en", reference: back, options: fastOptions(seed: 1))
        XCTAssertGreaterThan(a.frames, 0)
        XCTAssertGreaterThan(b.frames, 0)
        XCTAssertEqual(back, ref)
    }

    func testSeedReproducibility() async throws {
        let (m, ref) = try await model()
        let t = "等一下，我還在想。"
        let a = try m.generateCodes(text: t, language: "zh", reference: ref, options: fastOptions(seed: 42))
        let b = try m.generateCodes(text: t, language: "zh", reference: ref, options: fastOptions(seed: 42))
        let c = try m.generateCodes(text: t, language: "zh", reference: ref, options: fastOptions(seed: 7))
        XCTAssertEqual(a.seed, 42)
        XCTAssertEqual(a.codes, b.codes, "same seed must give identical tokens")
        XCTAssertNotEqual(a.codes, c.codes, "different seeds should differ")
        let unseeded = try m.generateCodes(text: t, language: "zh", reference: ref, options: fastOptions(seed: nil))
        let replay = try m.generateCodes(text: t, language: "zh", reference: ref,
                                         options: fastOptions(seed: unseeded.seed!))
        XCTAssertEqual(unseeded.codes, replay.codes, "reported seed must replay the run")
    }

    func testGreedyIsDeterministic() async throws {
        let (m, ref) = try await model()
        var o = fastOptions(seed: nil)
        o.diffusion = .greedy
        o.diffusion.steps = 4
        let a = try m.generateCodes(text: "Okay.", language: "en", reference: ref, options: o)
        let b = try m.generateCodes(text: "Okay.", language: "en", reference: ref, options: o)
        XCTAssertNil(a.seed)
        XCTAssertEqual(a.codes, b.codes)
    }

    func testSynthesizeProducesAudio() async throws {
        let (m, ref) = try await model()
        let s = try m.synthesize(text: "好，我看一下。", language: "zh", reference: ref, options: fastOptions(seed: 3))
        XCTAssertEqual(s.sampleRate, 24000)
        XCTAssertGreaterThan(s.samples.count, 24000 / 4)
        XCTAssertTrue(s.samples.allSatisfy { $0.isFinite })
    }

    func testBadInputsSurfaceErrors() async throws {
        let (m, ref) = try await model()
        XCTAssertThrowsError(try m.generateCodes(text: "  ", language: "en", reference: ref)) {
            XCTAssertEqual($0 as? OmniVoiceError, .emptyText)
        }
        XCTAssertThrowsError(try m.generateCodes(text: "hi", language: "xx-unknown", reference: ref)) {
            XCTAssertEqual($0 as? OmniVoiceError, .unknownLanguage("xx-unknown"))
        }
        var o = GenerationOptions()
        o.diffusion.steps = 0
        XCTAssertThrowsError(try m.generateCodes(text: "hi", language: "en", reference: ref, options: o))
        o = GenerationOptions()
        o.duration = -1
        XCTAssertThrowsError(try m.generateCodes(text: "hi", language: "en", reference: ref, options: o))
        XCTAssertThrowsError(try m.encodeReference(samples: [], sampleRate: 24000, transcript: "x"))
        XCTAssertThrowsError(try m.encodeReference(samples: [0.1, 0.2], sampleRate: 24000, transcript: "  ")) {
            XCTAssertEqual($0 as? OmniVoiceError, .emptyTranscript)
        }
        XCTAssertThrowsError(try m.encodeReference(
            audio: URL(fileURLWithPath: "/nonexistent/ref.wav"), transcript: "x"))
        XCTAssertThrowsError(try m.decode(codes: [1024, 0, 0, 0, 0, 0, 0, 0], frames: 1))
        XCTAssertThrowsError(try m.decode(codes: [1, 2], frames: 1))
    }
}
