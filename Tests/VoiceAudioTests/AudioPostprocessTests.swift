import Foundation
import XCTest
@testable import VoiceAudio

final class AudioPostprocessTests: XCTestCase {
    private func tone(_ count: Int, sampleRate: Int = 24000, amplitude: Float = 0.5) -> [Float] {
        (0 ..< count).map { Float(Double(amplitude) * sin(2 * Double.pi * 440 * Double($0) / Double(sampleRate))) }
    }

    private func quantized(_ s: [Float]) -> [Float] {
        AudioPostprocess.quantizeInt16(s).map { Float($0) / 32768 }
    }

    private func zeros(_ n: Int) -> [Float] { [Float](repeating: 0, count: n) }

    func testQuantizeMatchesNumpyCast() {
        XCTAssertEqual(AudioPostprocess.quantizeInt16([0, 1, -1, 2, -2, 0.5, -0.50001, 1e-5, -1e-5]),
                       [0, 32767, -32768, 32767, -32768, 16384, -16384, 0, 0])
    }

    /// 500 ms silence, 1000 ms tone, 500 ms silence at 24 kHz (expected values cross-checked
    /// against omnivoice.utils.audio.remove_silence).
    func testRemoveSilenceSilenceToneSilence() {
        let t = tone(24000)
        let x = zeros(12000) + t + zeros(12000)

        // Reference-prompt parameters: split keeps 200 ms around the tone, edges keep 100 / 200 ms.
        let ref = AudioPostprocess.removeSilence(x, sampleRate: 24000, midSilenceMs: 200,
                                                 leadSilenceMs: 100, trailSilenceMs: 200)
        XCTAssertEqual(ref, zeros(2400) + quantized(t) + zeros(4800))

        // Generated-audio parameters: nothing split, edges keep 100 / 100 ms.
        let gen = AudioPostprocess.removeSilence(x, sampleRate: 24000, midSilenceMs: 500,
                                                 leadSilenceMs: 100, trailSilenceMs: 100)
        XCTAssertEqual(gen, zeros(2400) + quantized(t) + zeros(2400))
    }

    func testRemoveSilenceCutsLongGapKeepingSilenceOnBothSides() {
        let t = tone(7200)
        let y = AudioPostprocess.removeSilence(t + zeros(24000) + t, sampleRate: 24000, midSilenceMs: 200,
                                               leadSilenceMs: 100, trailSilenceMs: 200)
        XCTAssertEqual(y, quantized(t) + zeros(9600) + quantized(t))
    }

    func testRemoveSilenceShortGapOverlapIsMergedLosslessly() {
        // 300 ms gap < 2 * keep_silence: pydub splits it at the midpoint, so the joined result
        // is the whole (quantized) signal.
        let x = tone(7200) + zeros(7200) + tone(7200)
        let y = AudioPostprocess.removeSilence(x, sampleRate: 24000, midSilenceMs: 200,
                                               leadSilenceMs: 100, trailSilenceMs: 200)
        XCTAssertEqual(y, quantized(x))
    }

    func testRemoveSilenceAllSilentIsEmpty() {
        XCTAssertEqual(AudioPostprocess.removeSilence(zeros(24000), sampleRate: 24000, midSilenceMs: 200,
                                                      leadSilenceMs: 100, trailSilenceMs: 200), [])
        XCTAssertThrowsError(try AudioPostprocess.prepareReference(
            zeros(24000), sampleRate: 24000, hopLength: 960, preprocess: true)) { error in
            XCTAssertEqual(error as? AudioPostprocessError, .emptyAfterSilenceRemoval)
        }
    }

    func testRemoveSilenceReproducesPydubMillisecondPadding() {
        // 24013 samples = 1000.54 ms -> len 1001 ms -> slicing to the end yields 24024 frames,
        // the last 11 zero-filled (pydub's "missing frames" padding).
        let x = tone(24013)
        let y = AudioPostprocess.removeSilence(x, sampleRate: 24000, midSilenceMs: 0,
                                               leadSilenceMs: 100, trailSilenceMs: 300)
        XCTAssertEqual(y, quantized(x) + zeros(11))
    }

    func testFadeAndPadShape() {
        let y = AudioPostprocess.fadeAndPad([Float](repeating: 1, count: 10), sampleRate: 100,
                                            padSeconds: 0.02, fadeSeconds: 0.05)
        XCTAssertEqual(y, [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25, 0, 0, 0])
        // Fade length is capped at half the signal; empty input is not padded.
        XCTAssertEqual(AudioPostprocess.fadeAndPad([1, 1, 1], sampleRate: 100, padSeconds: 0, fadeSeconds: 1),
                       [0, 1, 1])
        XCTAssertEqual(AudioPostprocess.fadeAndPad([], sampleRate: 100, padSeconds: 1, fadeSeconds: 1), [])
    }

    func testRMSMatchesNumpyPairwiseFloat32() {
        XCTAssertEqual(AudioPostprocess.rms([3, 4]), (12.5 as Float).squareRoot())
        XCTAssertTrue(AudioPostprocess.rms([]).isNaN)  // np.mean of an empty array
        // 0.1 summed 1000 times differs between naive and pairwise float32 accumulation;
        // numpy (pairwise) gives exactly this value.
        let ones = [Float](repeating: 0.1, count: 1000)
        XCTAssertEqual(AudioPostprocess.rms(ones), 0.1)
    }

    func testPrepareReferenceNormalizesAndClipsToHop() throws {
        let x = tone(2500, amplitude: 0.01)
        let r = try AudioPostprocess.prepareReference(x, sampleRate: 24000, hopLength: 960, preprocess: false)
        XCTAssertEqual(r.rms, AudioPostprocess.rms(x))
        XCTAssertEqual(r.samples.count, 1920)
        XCTAssertEqual(r.samples[100], x[100] * Float(0.1) / r.rms)
    }

    func testPostprocessVolumeBranches() {
        let x: [Float] = [0.1, -0.2]
        XCTAssertEqual(AudioPostprocess.postprocessGenerated(
            x, sampleRate: 100, referenceRMS: nil, removeSilence: false, padSeconds: 0, fadeSeconds: 0),
            [0.25, -0.5])
        XCTAssertEqual(AudioPostprocess.postprocessGenerated(
            x, sampleRate: 100, referenceRMS: 0.05, removeSilence: false, padSeconds: 0, fadeSeconds: 0),
            x.map { $0 * Float(0.05) / Float(0.1) })
        XCTAssertEqual(AudioPostprocess.postprocessGenerated(
            x, sampleRate: 100, referenceRMS: 0.2, removeSilence: false, padSeconds: 0.01, fadeSeconds: 0),
            [0, 0.1, -0.2, 0])
    }

    func testAddPunctuation() {
        XCTAssertEqual(TextPrep.addPunctuation("hello world"), "hello world.")
        XCTAssertEqual(TextPrep.addPunctuation("  hello world!  "), "hello world!")
        XCTAssertEqual(TextPrep.addPunctuation("你好世界"), "你好世界。")
        XCTAssertEqual(TextPrep.addPunctuation("你好世界，"), "你好世界，")
        XCTAssertEqual(TextPrep.addPunctuation("mixed 中文 text"), "mixed 中文 text。")
        XCTAssertEqual(TextPrep.addPunctuation("\u{3000}全角空白\u{3000}"), "全角空白。")
        XCTAssertEqual(TextPrep.addPunctuation("quoted\u{201D}"), "quoted\u{201D}")
        XCTAssertEqual(TextPrep.addPunctuation("ellipsis\u{2026}"), "ellipsis\u{2026}")
        XCTAssertEqual(TextPrep.addPunctuation("カタカナ"), "カタカナ.")  // no CJK unified ideograph
        XCTAssertEqual(TextPrep.addPunctuation("a\u{200B}"), "a\u{200B}.")  // ZWSP is not whitespace
        XCTAssertEqual(TextPrep.addPunctuation(" \t\n"), "")
        XCTAssertEqual(TextPrep.addPunctuation(""), "")
    }
}

final class ResamplerFixtureTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Case: Decodable {
            let orig_freq: Int
            let new_freq: Int
            let input: [Float]
            let output: [Float]
        }
        let cases: [Case]
    }

    /// `Fixtures/resampler_torchaudio.json` holds torchaudio.functional.resample outputs
    /// (default arguments, CPU float32) for short deterministic signals.
    func testMatchesTorchaudioFixture() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/resampler_torchaudio.json")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertFalse(fixture.cases.isEmpty)
        for c in fixture.cases {
            let y = Resampler.resample(c.input, from: c.orig_freq, to: c.new_freq)
            XCTAssertEqual(y.count, c.output.count, "\(c.orig_freq)->\(c.new_freq) length")
            var maxDiff: Float = 0
            for (a, b) in zip(y, c.output) { maxDiff = max(maxDiff, abs(a - b)) }
            XCTAssertLessThanOrEqual(maxDiff, 5e-6, "\(c.orig_freq)->\(c.new_freq)")
        }
    }
}
