import Foundation
import MLX
import MLXRandom
import SpeakerSim
import VoiceAudio
import VoiceMLX

let usage = """
    voice-bench <command> [options]

    Commands:
      synth       Synthesize one line to a WAV file.
      bench       Synthesize a lines file; per-line latency, time-to-first-audio,
                  mean/median/max, per-stage profile, peak memory, optional SIM-o.
      parity      Compare against dumps from scripts/qwen_parity.py.
      sim-parity  Swift scorer vs Python scores (scripts/sim_parity.py).
      matmul-check  fp32 matmul precision probe (env MM_M/MM_K/MM_N).

    Model options (synth/bench/parity):
      --model base|custom|<dir>|<hf-id>   base = aufklarer 0.6B Base 8-bit (default),
                                          custom = aufklarer 0.6B CustomVoice bf16
      --precision int8|bf16      linear-layer weights (default int8; bf16 bundles are
                                 quantized at load for int8)
      --dtype bf16|f16|f32       activations/scales (default bf16)
      --cache-limit-mb N         MLX buffer cache limit (default 256; -1 = MLX default)
      --no-tf32                  exact fp32 matmuls (MLX_ENABLE_TF32=0; parity mode)

    Voice (synth/bench): --speaker NAME (single-speaker bundle) or an ICL reference:
      --ref WAV (default /tmp/omnivoice/ref_plus_zh.wav)
      --ref-text TEXT | --ref-text-file PATH (default ref2.txt + " 好，我看一下。")

    Sampling: --greedy | --seed N (bench: --seeds 0,1,2) --temperature T --top-k K
              --repetition-penalty P --max-new-tokens N --text-feed interleaved|upfront
              --ref-decode-context N (ICL: decode only the last N reference frames)

    synth:  --text TEXT --lang L --out FILE.wav [--stream]
    bench:  --lines PATH (JSON [[text, lang, tag?], ...]; default lines20.json)
            [--indices 0,3,5] --out DIR --warmup N (default 1) --profile
            --stream [--first-chunk N --chunk N --left-context N]
            --score [--sim-weights F --sim-anchor WAV]
    parity: --dump DIR (from scripts/qwen_parity.py)
    """

// MARK: - Model / voice / options

func loadModel(_ a: inout Args) async throws -> Qwen3Voice {
    var o = LoadOptions()
    switch try a.string("precision", "int8") {
    case "int8": o.precision = .int8
    case "bf16": o.precision = .bf16
    case let x: throw BenchError.usage("--precision \(x)")
    }
    switch try a.string("dtype", "bf16") {
    case "bf16": o.computeDType = .bfloat16
    case "f16": o.computeDType = .float16
    case "f32": o.computeDType = .float32
    case let x: throw BenchError.usage("--dtype \(x)")
    }
    setCacheLimit(mb: try a.int("cache-limit-mb", 256))
    let m = try a.string("model", "base")
    let source: ModelSource
    switch m {
    case "base": source = .base8bit
    case "custom": source = .customVoiceBF16
    default:
        source = FileManager.default.fileExists(atPath: m) ? .directory(URL(fileURLWithPath: m)) : .hub(m)
    }
    let t0 = now()
    let tts = try await Qwen3Voice.load(model: source, options: o)
    print(String(format: "load %@ (%@, %@): %.2fs, footprint %@, speakers [%@]", m, o.precision.rawValue,
                 "\(o.computeDType)", now() - t0, gb(ProcessMemory.footprint().current),
                 tts.speakers.joined(separator: ",")))
    return tts
}

func defaultRefText() throws -> String {
    try String(contentsOfFile: "/tmp/omnivoice/ref2.txt", encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines) + " 好，我看一下。"
}

func loadVoice(_ tts: Qwen3Voice, _ a: inout Args) throws -> (Voice, Double) {
    if let s = a.optional("speaker") {
        _ = a.optional("ref"); _ = a.optional("ref-text"); _ = a.optional("ref-text-file")
        return (.speaker(s), 0)
    }
    let refPath = try a.string("ref", "/tmp/omnivoice/ref_plus_zh.wav")
    let text: String
    if let t = a.optional("ref-text") { text = t }
    else if let f = a.optional("ref-text-file") { text = try String(contentsOfFile: f, encoding: .utf8) }
    else { text = try defaultRefText() }
    let t0 = now()
    let ref = try tts.encodeReference(contentsOf: URL(fileURLWithPath: refPath), transcript: text)
    let dt = now() - t0
    print(String(format: "reference: %d frames (%.2fs), encode %.3fs (file read + resample + codec + speaker)",
                 ref.frames, ref.duration, dt))
    return (.reference(ref), dt)
}

func generationOptions(_ a: inout Args) throws -> GenerationOptions {
    var o = GenerationOptions()
    if a.flag("greedy") { o.sampling = .greedy }
    if let t = try a.double("temperature", nil) { o.sampling.temperature = Float(t); o.sampling.subtalkerTemperature = Float(t) }
    if let k = a.optional("top-k") { o.sampling.topK = Int(k)!; o.sampling.subtalkerTopK = Int(k)! }
    if let p = try a.double("repetition-penalty", nil) { o.sampling.repetitionPenalty = Float(p) }
    o.sampling.maxNewTokens = try a.int("max-new-tokens", o.sampling.maxNewTokens)
    if let f = a.optional("text-feed") {
        switch f {
        case "interleaved": o.textFeed = .interleaved
        case "upfront": o.textFeed = .upfront
        default: throw BenchError.usage("--text-feed \(f)")
        }
    }
    if let c = a.optional("ref-decode-context") { o.referenceDecodeContextFrames = Int(c)! }
    if let s = a.optional("seed") {
        guard let v = UInt64(s) else { throw BenchError.usage("--seed expects UInt64") }
        o.seed = v
    }
    o.profile = a.flag("profile")
    return o
}

func streamingOptions(_ a: inout Args) throws -> StreamingOptions {
    var s = StreamingOptions()
    s.firstChunkFrames = try a.int("first-chunk", s.firstChunkFrames)
    s.chunkFrames = try a.int("chunk", s.chunkFrames)
    s.leftContextFrames = try a.int("left-context", s.leftContextFrames)
    return s
}

struct LineResult {
    var samples: [Float]
    var synthesis: Synthesis
    var latency: Double
    var ttfa: Double?
}

func synthesize(_ tts: Qwen3Voice, _ text: String, _ lang: String, _ voice: Voice, _ o: GenerationOptions,
                stream: StreamingOptions?) throws -> LineResult {
    let t0 = now()
    if let stream {
        var samples: [Float] = []
        var first: Double?
        let s = try tts.generateStreaming(text: text, language: lang, voice: voice, options: o, streaming: stream) { c in
            if first == nil { first = now() - t0 }
            samples += c.samples
        }
        return LineResult(samples: samples, synthesis: s, latency: now() - t0, ttfa: first)
    }
    let s = try tts.generate(text: text, language: lang, voice: voice, options: o)
    let dt = now() - t0
    return LineResult(samples: s.samples, synthesis: s, latency: dt, ttfa: nil)
}

// MARK: - synth

func runSynth(_ argv: ArraySlice<String>) async throws {
    var a = try Args(argv, flags: ["greedy", "stream", "profile"])
    let tts = try await loadModel(&a)
    let (voice, _) = try loadVoice(tts, &a)
    let o = try generationOptions(&a)
    let stream = a.flag("stream") ? try streamingOptions(&a) : nil
    let text = try a.string("text")
    let lang = try a.string("lang")
    let out = try a.string("out")
    try a.checkAllConsumed()
    let r = try synthesize(tts, text, lang, voice, o, stream: stream)
    try AudioIO.writeWAV(r.samples, sampleRate: 24000, to: URL(fileURLWithPath: out))
    print(String(format: "%.3fs%@ -> %@ (%.2fs audio, %d frames, eos %@, seed %llu)", r.latency,
                 r.ttfa.map { String(format: " (first audio %.3fs)", $0) } ?? "", out,
                 Double(r.samples.count) / 24000, r.synthesis.frames, r.synthesis.finished ? "yes" : "NO",
                 r.synthesis.seed))
}

// MARK: - bench

func runBench(_ argv: ArraySlice<String>) async throws {
    var a = try Args(argv, flags: ["greedy", "profile", "score", "stream"])
    let tts = try await loadModel(&a)
    let (voice, refEncode) = try loadVoice(tts, &a)
    var o = try generationOptions(&a)
    let stream = a.flag("stream") ? try streamingOptions(&a) : nil
    let linesPath = try a.string("lines", "/tmp/omnivoice/lines20.json")
    let indices = a.optional("indices").map { $0.split(separator: ",").map { Int($0)! } }
    let seeds = try a.string("seeds", o.seed.map { "\($0)" } ?? "0").split(separator: ",").map { UInt64($0)! }
    let outDir = a.optional("out")
    let warmup = try a.int("warmup", 1)
    let score = a.flag("score")
    let simWeights = try a.string("sim-weights", SpeakerSimilarityScorer.defaultWeightsURL.path)
    let simAnchor = try a.string("sim-anchor", "/tmp/omnivoice/ref2.wav")
    let langOverride = a.optional("lang")
    try a.checkAllConsumed()

    let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: linesPath)))
    guard var lines = raw as? [[String]], lines.allSatisfy({ $0.count >= 2 }) else {
        throw BenchError.failed("\(linesPath): expected [[text, lang, tag?], ...]")
    }
    if let indices { lines = indices.map { lines[$0] } }
    if let outDir { try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true) }

    for _ in 0 ..< warmup {
        var w = o
        w.seed = 12345
        _ = try synthesize(tts, "Warm up.", "en", voice, w, stream: stream)
    }
    var lat: [Double] = [], ttfa: [Double] = [], audioSec: [Double] = []
    var prefill: [Double] = [], talkerStep: [Double] = [], cpFrame: [Double] = [], decode: [Double] = []
    var outputs: [(String, [Float])] = []
    var unfinished = 0
    for seed in seeds {
        o.seed = seed
        for (i, line) in lines.enumerated() {
            let r = try synthesize(tts, line[0], langOverride ?? line[1], voice, o, stream: stream)
            let s = r.synthesis
            lat.append(r.latency)
            if let t = r.ttfa { ttfa.append(t) }
            let dur = Double(r.samples.count) / 24000
            audioSec.append(dur)
            prefill.append(s.timings.prefill)
            if s.frames > 1 { talkerStep.append(s.timings.talker / Double(s.frames - 1)) }
            if s.frames > 0 && o.profile { cpFrame.append(s.timings.codePredictor / Double(s.frames)) }
            decode.append(s.timings.decode)
            if !s.finished { unfinished += 1 }
            let tag = line.count > 2 ? line[2] : String(format: "%02d", i)
            let name = "s\(seed)_\(tag)"
            if score { outputs.append((name, r.samples)) }
            if let outDir {
                try AudioIO.writeWAV(r.samples, sampleRate: 24000,
                                     to: URL(fileURLWithPath: outDir).appendingPathComponent("\(name).wav"))
            }
            print(String(format: "[%@] %.3fs%@ audio %.2fs frames %d%@  %@", name, r.latency,
                         r.ttfa.map { String(format: " ttfa %.3fs", $0) } ?? "", dur, s.frames,
                         s.finished ? "" : " (no EOS)", line[0]))
        }
    }
    let st = Stats(lat)
    let mem = ProcessMemory.footprint()
    print(String(format: "latency: mean %.3fs  median %.3fs  max %.3fs  min %.3fs  (n=%d)",
                 st.mean, st.median, st.max, st.min, lat.count))
    if !ttfa.isEmpty {
        let t = Stats(ttfa)
        print(String(format: "time-to-first-audio: mean %.3fs  median %.3fs  max %.3fs", t.mean, t.median, t.max))
    }
    print(String(format: "RTF (sum latency / sum audio): %.3f; lines without EOS: %d",
                 lat.reduce(0, +) / max(audioSec.reduce(0, +), 1e-9), unfinished))
    print(String(format: "stages (mean per line): reference encode %.3fs (once), prefill+first token %.3fs, "
                 + "talker step %.2f ms/frame, decode %.3fs%@",
                 refEncode, Stats(prefill).mean, talkerStep.isEmpty ? 0 : Stats(talkerStep).mean * 1000,
                 Stats(decode).mean,
                 cpFrame.isEmpty ? "" : String(format: ", code predictor %.2f ms/frame (profile: stage evals forced)",
                                               Stats(cpFrame).mean * 1000)))
    print("memory: footprint now \(gb(mem.current)), lifetime peak \(gb(mem.peak)), "
          + "MLX peak active \(gb(MLX.Memory.peakMemory)), MLX cache limit \(gb(MLX.Memory.cacheLimit))")
    if score {
        let t0 = now()
        let scorer = try SpeakerSimilarityScorer(weightsURL: URL(fileURLWithPath: simWeights))
        let anchor = try scorer.embed(contentsOf: URL(fileURLWithPath: simAnchor))
        var sims: [Double] = []
        for (name, o) in outputs {
            let s = Double(try scorer.similarity(anchor, o, sampleRate: 24000))
            sims.append(s)
            print(String(format: "  sim[%@] %.3f  (%.2fs)", name, s, Double(o.count) / 24000))
        }
        let st = Stats(sims)
        print(String(format: "SIM-o vs %@: mean %.3f  median %.3f  min %.3f  (%.2fs incl. scorer load)",
                     simAnchor, st.mean, st.median, st.min, now() - t0))
    }
}

// MARK: - sim-parity

func runSimParity(_ argv: ArraySlice<String>) throws {
    var a = try Args(argv, flags: [])
    let weights = try a.string("weights", SpeakerSimilarityScorer.defaultWeightsURL.path)
    let ref = try a.string("ref", "/tmp/omnivoice/ref2.wav")
    let list = try a.string("list", "/tmp/voice-mlx2/sim.txt")
    let scoresOut = a.optional("scores-out").map { URL(fileURLWithPath: $0) }
    let embOut = a.optional("emb-out").map { URL(fileURLWithPath: $0) }
    let tol = try a.double("tol", 0.01)!
    setCacheLimit(mb: try a.int("cache-limit-mb", 256))
    try a.checkAllConsumed()
    let report = try SimParity.run(
        weightsURL: URL(fileURLWithPath: weights), refURL: URL(fileURLWithPath: ref),
        listURL: URL(fileURLWithPath: list), scoresOut: scoresOut, embeddingOut: embOut)
    print(report.summary)
    print("memory: lifetime peak \(gb(ProcessMemory.footprint().peak))")
    let bad = report.failures(tolerance: Float(tol))
    if !bad.isEmpty {
        throw BenchError.failed("\(bad.count) file(s) exceed |diff| \(tol)")
    }
}

// MARK: - parity

func readI32(_ path: String) throws -> [Int32] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
}

func matchRate(_ a: [Int32], _ b: [Int32]) -> Double {
    let n = min(a.count, b.count)
    guard n > 0 else { return 0 }
    return Double(zip(a.prefix(n), b.prefix(n)).filter { $0 == $1 }.count) / Double(n)
}

func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var ab = 0.0, aa = 0.0, bb = 0.0
    for (x, y) in zip(a, b) { ab += Double(x * y); aa += Double(x * x); bb += Double(y * y) }
    return ab / ((aa * bb).squareRoot() + 1e-30)
}

func runParity(_ argv: ArraySlice<String>) async throws {
    var a = try Args(argv, flags: [])
    let dump = try a.string("dump")
    let tts = try await loadModel(&a)
    try a.checkAllConsumed()
    let j = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: dump + "/parity.json"))) as! [String: Any]
    let kind = j["kind"] as! String
    let voice: Voice
    var refFrames = 0
    if kind == "base" {
        let r = j["ref"] as! [String: Any]
        let text = r["text"] as! String
        let pyCodes = try readI32(dump + "/ref_codes.i32")
        let pySpk = try readF32(dump + "/ref_spk.f32")
        refFrames = r["frames"] as! Int
        let wav = try readF32(dump + "/ref24k.f32")
        let t0 = now()
        let sw = try tts.encodeReference(audio: wav, sampleRate: 24000, transcript: text)
        print(String(format: "A reference encode (%.2fs): frames %d vs %d, codec token match %.4f (codebook 0: %.4f), "
                     + "speaker embedding cosine %.6f", now() - t0, sw.frames, refFrames,
                     matchRate(sw.codes, pyCodes),
                     matchRate(stride(from: 0, to: sw.codes.count, by: 16).map { sw.codes[$0] },
                               stride(from: 0, to: pyCodes.count, by: 16).map { pyCodes[$0] }),
                     cosine(sw.speakerEmbedding, pySpk)))
        let pyRefIds = (r["ref_ids"] as! [Int])
        let swRefIds = try tts.tokenize("<|im_start|>assistant\n\(text)<|im_end|>\n")
        print("A reference transcript token ids equal: \(pyRefIds == swRefIds) (\(swRefIds.count) vs \(pyRefIds.count))")
        // Continue with Python's reference so the generation check is isolated.
        voice = .reference(VoiceReference(codes: pyCodes, frames: refFrames, transcript: text, speakerEmbedding: pySpk))
    } else {
        voice = .speaker(j["speaker"] as! String)
    }
    var greedy = GenerationOptions()
    greedy.sampling = .greedy
    greedy.sampling.maxNewTokens = 600
    greedy.seed = 0
    var tokTotal = 0, tokMatch = 0, exactLines = 0
    let lines = j["lines"] as! [[String: Any]]
    for line in lines {
        let i = line["index"] as! Int
        let text = line["text"] as! String, lang = line["lang"] as! String
        print("== line \(i) [\(lang)] \(text)")
        let pyIds = line["input_ids"] as! [Int]
        let swIds = try tts.tokenize("<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n")
        print("B text token ids equal: \(pyIds == swIds)")
        let pyStep0 = try readF32(dump + "/line\(i)_step0.f32")
        let swStep0 = try tts.firstStepLogits(text: text, language: lang, voice: voice)
        let c0 = compareSignals(pyStep0, swStep0)
        let am = { (x: [Float]) in x.indices.max(by: { x[$0] < x[$1] })! }
        print(String(format: "C step-0 logits: SNR %.1f dB, max|d| %.4f, argmax %d vs %d",
                     c0.snrDB, c0.maxAbs, am(swStep0), am(pyStep0)))
        let pyCodes = try readI32(dump + "/line\(i)_codes.i32")
        let pyFrames = pyCodes.count / 16
        let s = try tts.generate(text: text, language: lang, voice: voice, options: greedy)
        let n = min(s.codes.count, pyCodes.count)
        var firstDiff = -1
        for k in 0 ..< n where s.codes[k] != pyCodes[k] { firstDiff = k / 16; break }
        let m = matchRate(s.codes, pyCodes)
        tokTotal += max(s.codes.count, pyCodes.count)
        tokMatch += zip(s.codes.prefix(n), pyCodes.prefix(n)).filter { $0 == $1 }.count
        if s.codes == pyCodes { exactLines += 1 }
        print(String(format: "D greedy tokens: frames %d vs %d, match %.4f over common prefix, first differing frame %@",
                     s.frames, pyFrames, m, firstDiff < 0 ? "none" : "\(firstDiff)"))
        let pyWav = try readF32(dump + "/line\(i)_wav.f32")
        var codes = pyCodes
        var total = pyFrames
        if case .reference(let r) = voice { codes = r.codes + pyCodes; total += r.frames }
        let full = tts.decode(codes: codes, frames: total)
        let cut = Int(Double(refFrames) / Double(total) * Double(full.count))
        let swWav = Array(full[cut...])
        let c = compareSignals(pyWav, swWav)
        print(String(format: "E decode(python tokens): len %d vs %d, corr %.5f, SNR %.1f dB, max|d| %.4f",
                     swWav.count, pyWav.count, c.corr, c.snrDB, c.maxAbs))
    }
    print(String(format: "== overall: %d/%d lines token-identical; token match %.4f (%d/%d, counting length differences as mismatches)",
                 exactLines, lines.count, Double(tokMatch) / Double(max(tokTotal, 1)), tokMatch, tokTotal))
}

/// Relative error of MLX fp32 matmul (GPU and CPU streams) against Float64.
func runMatmulCheck() {
    let env = ProcessInfo.processInfo.environment
    let (m, k, n) = (Int(env["MM_M"] ?? "") ?? 64, Int(env["MM_K"] ?? "") ?? 1024, Int(env["MM_N"] ?? "") ?? 1024)
    let a = MLXRandom.normal([m, k], key: MLXRandom.key(1))
    let b = MLXRandom.normal([k, n], key: MLXRandom.key(2))
    let gpu = MLX.matmul(a, b).asArray(Float.self)
    let cpuOut = MLX.matmul(a, b, stream: .cpu).asArray(Float.self)
    let av = a.asArray(Float.self), bv = b.asArray(Float.self)
    var errG = 0.0, errC = 0.0, sig = 0.0
    for i in 0 ..< 16 {
        for j in 0 ..< n {
            var s = 0.0
            for t in 0 ..< k { s += Double(av[i * k + t]) * Double(bv[t * n + j]) }
            sig += s * s
            errG += pow(Double(gpu[i * n + j]) - s, 2)
            errC += pow(Double(cpuOut[i * n + j]) - s, 2)
        }
    }
    print(String(format: "fp32 matmul SNR vs float64: GPU %.1f dB, CPU %.1f dB",
                 10 * log10(sig / errG), 10 * log10(sig / errC)))
}

// `--no-tf32` must take effect before MLX's first matmul: MLX reads MLX_ENABLE_TF32
// once, and on GPUs with neural accelerators (M5) float32 matmuls default to TF32.
var argv = CommandLine.arguments
if let i = argv.firstIndex(of: "--no-tf32") {
    setenv("MLX_ENABLE_TF32", "0", 1)
    argv.remove(at: i)
}
print("MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset (default 1)")")
guard argv.count >= 2 else {
    print(usage)
    exit(2)
}
do {
    switch argv[1] {
    case "bench": try await runBench(argv.dropFirst(2))
    case "synth": try await runSynth(argv.dropFirst(2))
    case "parity": try await runParity(argv.dropFirst(2))
    case "matmul-check": runMatmulCheck()
    case "sim-parity": try runSimParity(argv.dropFirst(2))
    case "-h", "--help", "help": print(usage)
    default:
        print(usage)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
