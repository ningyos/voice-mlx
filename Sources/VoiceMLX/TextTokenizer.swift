import Foundation

/// Byte-level BPE tokenizer equivalent to the `Qwen2TokenizerFast` that `qwen_tts`
/// loads from a Qwen3-TTS bundle (`vocab.json`, `merges.txt`, `tokenizer_config.json`):
/// NFC normalisation, added/special tokens split out first, the Qwen2 pre-tokenizer
/// regex, GPT-2 byte-to-unicode mapping, rank-ordered merges.
///
/// speech-swift's tokenizer split words on whitespace only and silently dropped
/// unknown pieces; this one follows the reference pre-tokenizer and throws instead.
public final class TextTokenizer: @unchecked Sendable {
    private let vocab: [String: Int]
    private let mergeRanks: [String: Int]
    private let specialTokens: [(String, Int)]  // longest first
    private let pretokenizer: NSRegularExpression
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()

    static let pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    public init(directory: URL) throws {
        func file(_ name: String) throws -> Data {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw VoiceError.missingFile(url)
            }
            return try Data(contentsOf: url)
        }
        guard let vocab = try JSONSerialization.jsonObject(with: try file("vocab.json")) as? [String: Int] else {
            throw VoiceError.invalidBundle(directory, "vocab.json is not a {token: id} object")
        }
        guard let merges = String(data: try file("merges.txt"), encoding: .utf8) else {
            throw VoiceError.invalidBundle(directory, "merges.txt is not UTF-8")
        }
        var ranks: [String: Int] = [:]
        var rank = 0
        for line in merges.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#version") { continue }
            let parts = line.split(separator: " ")
            guard parts.count == 2 else {
                throw VoiceError.invalidBundle(directory, "bad merges.txt line: \(line)")
            }
            ranks[line.trimmingCharacters(in: .whitespaces)] = rank
            rank += 1
        }
        guard let config = try JSONSerialization.jsonObject(with: try file("tokenizer_config.json")) as? [String: Any],
            let added = config["added_tokens_decoder"] as? [String: [String: Any]]
        else {
            throw VoiceError.invalidBundle(directory, "tokenizer_config.json has no added_tokens_decoder")
        }
        var specials: [(String, Int)] = []
        var fullVocab = vocab
        for (idString, info) in added {
            guard let id = Int(idString), let content = info["content"] as? String else { continue }
            specials.append((content, id))
            fullVocab[content] = id
        }
        self.vocab = fullVocab
        self.mergeRanks = ranks
        self.specialTokens = specials.sorted { $0.0.count > $1.0.count }
        self.pretokenizer = try NSRegularExpression(pattern: Self.pattern)
    }

    /// Encode text; added tokens such as `<|im_start|>` in `text` map to their ids.
    public func encode(_ text: String) throws -> [Int] {
        let nfc = text.precomposedStringWithCanonicalMapping
        var ids: [Int] = []
        var rest = Substring(nfc)
        while !rest.isEmpty {
            // Earliest special-token occurrence (longest wins at equal position).
            var best: (Range<Substring.Index>, Int)?
            for (tok, id) in specialTokens {
                if let r = rest.range(of: tok), best == nil || r.lowerBound < best!.0.lowerBound {
                    best = (r, id)
                }
            }
            let plainEnd = best?.0.lowerBound ?? rest.endIndex
            if plainEnd > rest.startIndex {
                ids += try encodePlain(String(rest[rest.startIndex ..< plainEnd]))
            }
            guard let (range, id) = best else { break }
            ids.append(id)
            rest = rest[range.upperBound...]
        }
        return ids
    }

    private func encodePlain(_ text: String) throws -> [Int] {
        let ns = text as NSString
        var ids: [Int] = []
        for m in pretokenizer.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let word = ns.substring(with: m.range)
            ids += try bpe(word)
        }
        return ids
    }

    private func bpe(_ word: String) throws -> [Int] {
        lock.lock()
        if let hit = cache[word] { lock.unlock(); return hit }
        lock.unlock()
        var pieces = word.utf8.map { String(Self.byteToUnicode[Int($0)]) }
        while pieces.count > 1 {
            var bestRank = Int.max, bestIndex = -1
            for i in 0 ..< pieces.count - 1 {
                if let r = mergeRanks[pieces[i] + " " + pieces[i + 1]], r < bestRank {
                    bestRank = r
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            let (a, b) = (pieces[bestIndex], pieces[bestIndex + 1])
            var merged: [String] = []
            merged.reserveCapacity(pieces.count)
            var i = 0
            while i < pieces.count {
                if i < pieces.count - 1, pieces[i] == a, pieces[i + 1] == b {
                    merged.append(a + b)
                    i += 2
                } else {
                    merged.append(pieces[i])
                    i += 1
                }
            }
            pieces = merged
        }
        let ids = try pieces.map { p -> Int in
            guard let id = vocab[p] else { throw VoiceError.tokenizer("no vocab entry for BPE piece \(p.debugDescription)") }
            return id
        }
        lock.lock()
        cache[word] = ids
        lock.unlock()
        return ids
    }

    /// GPT-2 `bytes_to_unicode`.
    static let byteToUnicode: [Character] = {
        var bs = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(UnicodeScalar(c)!) }
        return table
    }()
}
