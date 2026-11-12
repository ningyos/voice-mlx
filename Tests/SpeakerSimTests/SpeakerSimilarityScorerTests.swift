import Foundation
import VoiceAudio
import XCTest

@testable import SpeakerSim

final class SpeakerSimilarityScorerTests: XCTestCase {
    func testCosineSanity() {
        let a: [Float] = [1, 2, 3, -4]
        XCTAssertEqual(SpeakerSimilarityScorer.cosine(a, a), 1, accuracy: 1e-6)
        XCTAssertEqual(SpeakerSimilarityScorer.cosine(a, a.map { -2 * $0 }), -1, accuracy: 1e-6)
        XCTAssertEqual(SpeakerSimilarityScorer.cosine([1, 0], [0, 3]), 0, accuracy: 1e-7)
        XCTAssertEqual(SpeakerSimilarityScorer.cosine([1, 1], [1, 0]), Float(0.5.squareRoot()), accuracy: 1e-6)
        // torch clamps each norm at 1e-8, so a zero vector scores 0 instead of NaN.
        XCTAssertEqual(SpeakerSimilarityScorer.cosine([0, 0], [1, 2]), 0)
    }

    func testMinimumSamplesMatchesCNNReceptiveField() {
        XCTAssertEqual(SpeakerSimilarityScorer.minimumSamples, 400)
        XCTAssertEqual(SimModel.frameCount(samples: 399), 0)
        XCTAssertEqual(SimModel.frameCount(samples: 400), 1)
        XCTAssertEqual(SimModel.frameCount(samples: 16_000), 49)
    }

    func testMissingWeightsFileThrows() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnivoice-sim-missing-\(UUID().uuidString).safetensors")
        XCTAssertThrowsError(try SpeakerSimilarityScorer(weightsURL: url)) { error in
            XCTAssertEqual(error as? SpeakerSimilarityError, .weightsNotFound(path: url.path))
        }
    }

    func testIncompleteWeightsFileThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnivoice-sim-partial-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.writeSafetensors(["feature_weight": ([25], [Float](repeating: 0, count: 25))], to: url)
        XCTAssertThrowsError(try SpeakerSimilarityScorer(weightsURL: url)) { error in
            guard case .missingWeights(let keys)? = error as? SpeakerSimilarityError else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertFalse(keys.contains("feature_weight"))
            XCTAssertTrue(keys.contains("linear.weight"))
        }
    }

    func testMismatchedWeightShapeThrows() throws {
        // Shape errors on present tensors are reported before missing tensors.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnivoice-sim-shape-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.writeSafetensors(["feature_weight": ([24], [Float](repeating: 0, count: 24))], to: url)
        XCTAssertThrowsError(try SpeakerSimilarityScorer(weightsURL: url)) { error in
            XCTAssertEqual(
                error as? SpeakerSimilarityError,
                .shapeMismatch(key: "feature_weight", expected: [25], actual: [24]))
        }
    }

    // MARK: - Weight-dependent tests (skipped without exported weights / audio)

    private func scorerOrSkip() throws -> SpeakerSimilarityScorer {
        let url = SpeakerSimilarityScorer.defaultWeightsURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("SIM weights not found at \(url.path); run scripts/export_sim_weights.py or set OMNIVOICE_SIM_WEIGHTS")
        }
        return try SpeakerSimilarityScorer(weightsURL: url)
    }

    func testInvalidAudioThrows() throws {
        let scorer = try scorerOrSkip()
        XCTAssertThrowsError(try scorer.embed([], sampleRate: 16_000)) { error in
            XCTAssertEqual(error as? SpeakerSimilarityError, .emptyAudio)
        }
        XCTAssertThrowsError(try scorer.embed([Float](repeating: 0.1, count: 399), sampleRate: 16_000)) { error in
            XCTAssertEqual(error as? SpeakerSimilarityError, .audioTooShort(samples: 399, minimumSamples: 400))
        }
        XCTAssertThrowsError(try scorer.embed([0.1, 0.2], sampleRate: 0)) { error in
            XCTAssertEqual(error as? SpeakerSimilarityError, .invalidSampleRate(0))
        }
        XCTAssertThrowsError(try scorer.similarity([1, 2], [0.1], sampleRate: 16_000)) { error in
            XCTAssertEqual(error as? SpeakerSimilarityError, .embeddingDimensionMismatch(expected: 256, actual: 2))
        }
    }

    struct Fixture: Decodable {
        struct File: Decodable {
            var path: String
            var score: Float
            var embedding: [Float]
        }
        var ref: String
        var refEmbedding: [Float]
        var files: [File]
    }

    /// Compares against PyTorch embeddings/scores recorded by `scripts/sim_parity.py fixtures`.
    func testParityWithPythonFixtures() throws {
        let fixtureURL = try XCTUnwrap(
            Bundle.module.url(forResource: "sim_parity_fixture", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
        let audio = [fixture.ref] + fixture.files.map(\.path)
        for path in audio where !FileManager.default.fileExists(atPath: path) {
            throw XCTSkip("parity audio not found: \(path)")
        }
        let scorer = try scorerOrSkip()

        let ref = try scorer.embed(contentsOf: URL(fileURLWithPath: fixture.ref))
        XCTAssertEqual(ref.count, 256)
        XCTAssertGreaterThan(SpeakerSimilarityScorer.cosine(ref, fixture.refEmbedding), 0.999)

        for file in fixture.files {
            let clip = try AudioIO.read(URL(fileURLWithPath: file.path))
            let emb = try scorer.embed(clip.samples, sampleRate: clip.sampleRate)
            XCTAssertGreaterThan(SpeakerSimilarityScorer.cosine(emb, file.embedding), 0.999, file.path)
            let score = try scorer.similarity(ref, clip.samples, sampleRate: clip.sampleRate)
            XCTAssertEqual(score, file.score, accuracy: 0.01, file.path)
        }
    }

    // MARK: - Helpers

    /// Minimal safetensors writer (F32 only) so the tests need no MLX import.
    static func writeSafetensors(_ tensors: [String: ([Int], [Float])], to url: URL) throws {
        var header: [String: Any] = [:]
        var payload = Data()
        for (key, (shape, values)) in tensors.sorted(by: { $0.key < $1.key }) {
            precondition(values.count == shape.reduce(1, *))
            let start = payload.count
            values.withUnsafeBufferPointer { payload.append(Data(buffer: $0)) }
            header[key] = ["dtype": "F32", "shape": shape, "data_offsets": [start, payload.count]]
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var length = UInt64(json.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        file.append(payload)
        try file.write(to: url)
    }
}
