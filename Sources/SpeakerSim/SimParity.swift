import Foundation

/// Swift side of the SIM-o parity check (`scripts/sim_parity.py`).
///
/// The list file has one `path [score]` entry per line. Paths are tried as given, then
/// relative to the list's directory, then as a bare file name inside the list's
/// directory (so `/tmp/voice-mlx2/sim.txt`, whose entries read `mlx2/<name>.wav`,
/// resolves to `/tmp/voice-mlx2/<name>.wav`). When a score column is present it is the
/// Python reference score and the absolute difference is reported.
public enum SimParity {
    public struct Entry: Sendable {
        public var listedPath: String
        public var resolvedPath: String
        public var pythonScore: Float?
        public var swiftScore: Float
        public var seconds: Double
        public var absDiff: Float? { pythonScore.map { abs($0 - swiftScore) } }
    }

    public struct Report: Sendable {
        public var entries: [Entry]
        public var refEmbedding: [Float]
        public var loadSeconds: Double
        public var refSeconds: Double

        public var maxAbsDiff: Float? { entries.compactMap(\.absDiff).max() }
        public var meanAbsDiff: Float? {
            let d = entries.compactMap(\.absDiff)
            return d.isEmpty ? nil : d.reduce(0, +) / Float(d.count)
        }
        public var meanSecondsPerFile: Double {
            entries.isEmpty ? 0 : entries.map(\.seconds).reduce(0, +) / Double(entries.count)
        }

        public func failures(tolerance: Float) -> [Entry] {
            entries.filter { ($0.absDiff ?? 0) > tolerance }
        }

        public var summary: String {
            var s = String(
                format: "files=%d load=%.2fs ref=%.3fs mean/file=%.3fs", entries.count, loadSeconds,
                refSeconds, meanSecondsPerFile)
            if let mx = maxAbsDiff, let mean = meanAbsDiff {
                s += String(format: " max|diff|=%.6f mean|diff|=%.6f", mx, mean)
            }
            return s
        }
    }

    /// Scores every listed file against `refURL` and compares with the listed scores.
    /// - Parameters:
    ///   - scoresOut: optional TSV (`resolved_path score` with 6 decimals).
    ///   - embeddingOut: optional JSON array with the Swift reference embedding.
    ///   - log: per-file progress line sink.
    public static func run(
        weightsURL: URL, refURL: URL, listURL: URL, scoresOut: URL? = nil,
        embeddingOut: URL? = nil, log: (String) -> Void = { print($0) }
    ) throws -> Report {
        let items = try parseList(listURL)
        let clock = ContinuousClock()

        var t0 = clock.now
        let scorer = try SpeakerSimilarityScorer(weightsURL: weightsURL)
        let loadSeconds = seconds(clock.now - t0)

        t0 = clock.now
        let ref = try scorer.embed(contentsOf: refURL)
        let refSeconds = seconds(clock.now - t0)

        var entries: [Entry] = []
        for item in items {
            t0 = clock.now
            let emb = try scorer.embed(contentsOf: URL(fileURLWithPath: item.resolved))
            let score = SpeakerSimilarityScorer.cosine(ref, emb)
            let entry = Entry(
                listedPath: item.listed, resolvedPath: item.resolved, pythonScore: item.score,
                swiftScore: score, seconds: seconds(clock.now - t0))
            entries.append(entry)
            var line = String(format: "%@ swift=%.6f", item.resolved, score)
            if let py = item.score, let d = entry.absDiff {
                line += String(format: " python=%.6f diff=%.6f", py, d)
            }
            log(line)
        }

        if let scoresOut {
            let text = entries.map { String(format: "%@ %.6f", $0.resolvedPath, $0.swiftScore) }
                .joined(separator: "\n") + "\n"
            try text.write(to: scoresOut, atomically: true, encoding: .utf8)
        }
        if let embeddingOut {
            let data = try JSONSerialization.data(withJSONObject: ref.map { Double($0) })
            try data.write(to: embeddingOut)
        }
        return Report(entries: entries, refEmbedding: ref, loadSeconds: loadSeconds, refSeconds: refSeconds)
    }

    struct ListItem {
        var listed: String
        var resolved: String
        var score: Float?
    }

    static func parseList(_ url: URL) throws -> [ListItem] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let dir = url.deletingLastPathComponent()
        let fm = FileManager.default
        var items: [ListItem] = []
        for (n, raw) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            // `path score` splits at the last whitespace; a line without a numeric last
            // field is a bare path (which may contain spaces).
            var listed = line
            var score: Float?
            if let cut = line.lastIndex(where: { $0 == " " || $0 == "\t" }),
                let s = Float(line[line.index(after: cut)...])
            {
                score = s
                listed = line[..<cut].trimmingCharacters(in: .whitespaces)
            }
            let candidates = [
                listed,
                dir.appendingPathComponent(listed).path,
                dir.appendingPathComponent((listed as NSString).lastPathComponent).path,
            ]
            guard let resolved = candidates.first(where: { fm.fileExists(atPath: $0) }) else {
                throw SpeakerSimilarityError.invalidParityList(
                    path: url.path, line: n + 1, reason: "audio not found: \(listed)")
            }
            items.append(ListItem(listed: listed, resolved: resolved, score: score))
        }
        return items
    }

    private static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}
