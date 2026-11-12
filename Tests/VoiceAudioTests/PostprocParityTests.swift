import Foundation
import XCTest
@testable import VoiceAudio

/// Compares the Swift port against Python reference outputs written by
/// `scripts/postproc_parity.py`. Skipped unless `OMNIVOICE_POSTPROC_PARITY_DIR` points at
/// that script's `--out` directory.
final class PostprocParityTests: XCTestCase {
    private static let tolerance: Float = 1e-6

    private func loadF32(_ dir: URL, _ name: String) throws -> [Float] {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        return data.withUnsafeBytes { raw in Array(raw.bindMemory(to: Float.self)) }
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0 ..< min(a.count, b.count) { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    func testPythonParity() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["OMNIVOICE_POSTPROC_PARITY_DIR"] else {
            throw XCTSkip("set OMNIVOICE_POSTPROC_PARITY_DIR (see scripts/postproc_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let manifestData = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        let sr = try XCTUnwrap(manifest["sample_rate"] as? Int)
        let hop = try XCTUnwrap(manifest["hop_length"] as? Int)

        for p in try XCTUnwrap(manifest["punctuation"] as? [[String: String]]) {
            let got = TextPrep.addPunctuation(p["input"]!)
            XCTAssertEqual(got, p["output"]!, "addPunctuation(\(p["input"]!.debugDescription))")
        }
        print("addPunctuation: \(manifest["punctuation"].map { ($0 as! [Any]).count } ?? 0) cases checked")

        for entry in try XCTUnwrap(manifest["inputs"] as? [[String: Any]]) {
            let name = entry["name"] as! String
            let rate = entry["rate"] as! Int
            let input = try loadF32(dir, (entry["input"] as! [String: Any])["file"] as! String)

            if let orig = entry["orig"] as? [String: Any] {
                let origSamples = try loadF32(dir, orig["file"] as! String)
                let origRate = entry["orig_sample_rate"] as! Int
                let swiftResampled = Resampler.resample(origSamples, from: origRate, to: sr)
                XCTAssertEqual(swiftResampled.count, input.count, "\(name) resample length")
                let d = maxAbsDiff(swiftResampled, input)
                print("\(name) resample \(origRate)->\(sr): len \(swiftResampled.count)/\(input.count) maxAbs \(d)")
                // End-to-end reference prep on the Swift-resampled signal, compared with the
                // Python prep (which resampled with torchaudio).
                if let pyPrep = (entry["cases"] as! [[String: Any]]).first(where: {
                    $0["kind"] as? String == "prepare_reference"
                        && ($0["params"] as! [String: Any])["preprocess"] as? Bool == true && $0["file"] != nil
                }) {
                    let want = try loadF32(dir, pyPrep["file"] as! String)
                    let got = try AudioPostprocess.prepareReference(
                        swiftResampled, sampleRate: sr, hopLength: hop, preprocess: true)
                    print("\(name) e2e prepareReference(swift resample): len \(got.samples.count)/\(want.count) maxAbs \(maxAbsDiff(got.samples, want))")
                }
            }

            let rms = AudioPostprocess.rms(input)
            let pyRMS = entry["rms"] as! Double
            XCTAssertEqual(Double(rms), pyRMS, "\(name) rms")
            print("\(name) rms swift \(rms) python \(pyRMS)")

            for c in entry["cases"] as! [[String: Any]] {
                let kind = c["kind"] as! String
                let params = c["params"] as! [String: Any]
                let label = "\(name) \(kind) \(params.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ","))"
                if let error = c["error"] as? String {
                    XCTAssertThrowsError(try AudioPostprocess.prepareReference(
                        input, sampleRate: rate, hopLength: hop, preprocess: params["preprocess"] as! Bool),
                        "\(label) expected Python error: \(error)")
                    continue
                }
                let want = try loadF32(dir, c["file"] as! String)
                let got: [Float]
                switch kind {
                case "remove_silence":
                    got = AudioPostprocess.removeSilence(
                        input, sampleRate: rate, midSilenceMs: params["mid"] as! Int,
                        leadSilenceMs: params["lead"] as! Int, trailSilenceMs: params["trail"] as! Int)
                case "fade_and_pad":
                    got = AudioPostprocess.fadeAndPad(
                        input, sampleRate: rate, padSeconds: params["pad"] as! Double,
                        fadeSeconds: params["fade"] as! Double)
                case "prepare_reference":
                    let r = try AudioPostprocess.prepareReference(
                        input, sampleRate: rate, hopLength: hop, preprocess: params["preprocess"] as! Bool)
                    XCTAssertEqual(Double(r.rms), params["ref_rms"] as! Double, "\(label) ref_rms")
                    got = r.samples
                case "postprocess":
                    let refRMS = (params["ref_rms"] as? Double).map { Float($0) }
                    got = AudioPostprocess.postprocessGenerated(
                        input, sampleRate: rate, referenceRMS: refRMS,
                        removeSilence: params["remove_silence"] as! Bool,
                        padSeconds: params["pad"] as! Double, fadeSeconds: params["fade"] as! Double)
                default:
                    XCTFail("unknown case kind \(kind)")
                    continue
                }
                let d = maxAbsDiff(got, want)
                print("\(label): len \(got.count)/\(want.count) maxAbs \(d)")
                XCTAssertEqual(got.count, want.count, label)
                XCTAssertLessThanOrEqual(d, Self.tolerance, label)
            }
        }
    }
}
