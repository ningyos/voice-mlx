import Darwin
import Foundation
import MLX

/// Minimal `--key value` / `--flag` parser; unknown keys are an error.
struct Args {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []
    private var consumed: Set<String> = []

    init(_ argv: ArraySlice<String>, flags known: Set<String>) throws {
        var it = argv.makeIterator()
        while let a = it.next() {
            guard a.hasPrefix("--") else { throw BenchError.usage("unexpected argument \(a)") }
            let k = String(a.dropFirst(2))
            if known.contains(k) {
                flags.insert(k)
            } else {
                guard let v = it.next() else { throw BenchError.usage("--\(k) needs a value") }
                values[k] = v
            }
        }
    }

    mutating func string(_ k: String, _ d: String? = nil) throws -> String {
        consumed.insert(k)
        if let v = values[k] { return v }
        if let d { return d }
        throw BenchError.usage("missing --\(k)")
    }

    mutating func optional(_ k: String) -> String? {
        consumed.insert(k)
        return values[k]
    }

    mutating func int(_ k: String, _ d: Int) throws -> Int {
        consumed.insert(k)
        guard let v = values[k] else { return d }
        guard let i = Int(v) else { throw BenchError.usage("--\(k) expects an integer, got \(v)") }
        return i
    }

    mutating func double(_ k: String, _ d: Double?) throws -> Double? {
        consumed.insert(k)
        guard let v = values[k] else { return d }
        guard let x = Double(v) else { throw BenchError.usage("--\(k) expects a number, got \(v)") }
        return x
    }

    func flag(_ k: String) -> Bool { flags.contains(k) }

    func checkAllConsumed() throws {
        let extra = Set(values.keys).subtracting(consumed)
        if !extra.isEmpty { throw BenchError.usage("unknown option(s): \(extra.sorted().map { "--" + $0 })") }
    }
}

enum BenchError: Error, CustomStringConvertible {
    case usage(String)
    case failed(String)
    var description: String {
        switch self {
        case .usage(let m): return "usage error: \(m)"
        case .failed(let m): return m
        }
    }
}

enum ProcessMemory {
    /// Current and lifetime-peak physical footprint (bytes) of this process —
    /// the number Activity Monitor shows as "Memory"; includes MLX's Metal buffers
    /// and its buffer cache.
    static func footprint() -> (current: UInt64, peak: UInt64) {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard rc == 0 else { return (0, 0) }
        return (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
    }
}

func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1_073_741_824) }
func gb(_ bytes: Int) -> String { gb(UInt64(max(0, bytes))) }

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

struct Stats {
    let mean: Double, median: Double, max: Double, min: Double
    init(_ xs: [Double]) {
        precondition(!xs.isEmpty)
        let s = xs.sorted()
        mean = xs.reduce(0, +) / Double(xs.count)
        median = s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        max = s.last!
        min = s.first!
    }
}

func readF32(_ path: String) throws -> [Float] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

/// Pearson correlation over the common prefix, and SNR of b against a (dB).
func compareSignals(_ a: [Float], _ b: [Float]) -> (corr: Double, snrDB: Double, maxAbs: Float, n: Int) {
    let n = min(a.count, b.count)
    guard n > 0 else { return (0, 0, 0, 0) }
    var ma = 0.0, mb = 0.0
    for i in 0 ..< n { ma += Double(a[i]); mb += Double(b[i]) }
    ma /= Double(n); mb /= Double(n)
    var sab = 0.0, saa = 0.0, sbb = 0.0, sig = 0.0, err = 0.0
    var maxAbs: Float = 0
    for i in 0 ..< n {
        let x = Double(a[i]) - ma, y = Double(b[i]) - mb
        sab += x * y; saa += x * x; sbb += y * y
        sig += Double(a[i]) * Double(a[i])
        let e = Double(a[i] - b[i]); err += e * e
        maxAbs = Swift.max(maxAbs, abs(a[i] - b[i]))
    }
    let corr = sab / ((saa * sbb).squareRoot() + 1e-30)
    let snr = 10 * log10(sig / (err + 1e-30))
    return (corr, snr, maxAbs, n)
}

func setCacheLimit(mb: Int) {
    if mb >= 0 { MLX.Memory.cacheLimit = mb * 1024 * 1024 }
}
