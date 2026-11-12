import Foundation

public enum VoiceError: Error, CustomStringConvertible, Equatable {
    case missingFile(URL)
    case notADirectory(URL)
    case invalidBundle(URL, String)
    case weightMismatch(String)
    case precisionUnavailable(String)
    case unknownLanguage(String, known: [String])
    case unknownSpeaker(String, known: [String])
    case unsupported(String)
    case invalidArgument(String)
    case tokenizer(String)
    case download(String, String)

    public var description: String {
        switch self {
        case .missingFile(let u): return "required file missing: \(u.path)"
        case .notADirectory(let u): return "not a directory: \(u.path)"
        case .invalidBundle(let u, let why): return "invalid model bundle \(u.path): \(why)"
        case .weightMismatch(let why): return "weights do not match the model: \(why)"
        case .precisionUnavailable(let why): return "precision unavailable: \(why)"
        case .unknownLanguage(let l, let known):
            return "unknown language \(l.debugDescription); known: \(known.joined(separator: ", "))"
        case .unknownSpeaker(let s, let known):
            return "unknown speaker \(s.debugDescription); this bundle has: \(known.isEmpty ? "none" : known.joined(separator: ", "))"
        case .unsupported(let why): return "unsupported: \(why)"
        case .invalidArgument(let why): return "invalid argument: \(why)"
        case .tokenizer(let why): return "tokenizer: \(why)"
        case .download(let id, let why): return "download of \(id) failed: \(why)"
        }
    }
}
