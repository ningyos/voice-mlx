import Foundation
import MLX
import OmniVoiceAudio
import OmniVoiceMLX
import OmniVoiceSim

let usage = """
    omnivoice-bench <command> [options]

    Commands:
      bench    Synthesize a list of lines with a cached reference; print per-line
               latency, mean/median/max, per-stage profile and peak memory.
      synth    Synthesize one text to a WAV file.
      parity   Compare against dumps from scripts/tts_parity.py.
      matmul-check  fp32 matmul precision probe (env MM_M/MM_K/MM_N).

    Common options:
      --no-tf32                  exact fp32 matmuls (sets MLX_ENABLE_TF32=0; parity mode)
      --bundle int8|fp16|<dir>   model bundle (default int8)
      --dtype f32|f16|bf16       backbone compute dtype (default f16)
      --cache-limit-mb N         MLX buffer-cache limit (default 256; -1 = MLX default = unlimited)
      --ref PATH                 reference audio (default /tmp/omnivoice/ref_plus_zh.wav)
      --ref-text TEXT | --ref-text-file PATH   (default: ref2.txt + " 好，我看一下。")
      --ref-max-seconds S        keep only the first S seconds of the reference
      --no-preprocess-ref        skip reference silence removal / punctuation

    bench:  --lines PATH (JSON [[text, lang], ...]) --steps N --guidance G --seed N
            --greedy  --profile  --out DIR (write WAVs)  --warmup N (default 1)
            --position-temperature T --class-temperature T
            --score [--sim-weights F --sim-anchor WAV]  (SIM-o of each output vs anchor)
    sim-parity: --weights F --ref WAV --list FILE ('path [python_score]' lines)
            [--scores-out F --emb-out F --tol 0.01 --cache-limit-mb 256]
    synth:  --text TEXT --lang L --out FILE.wav [--steps --seed --duration --instruct --greedy]
    parity: --dump DIR (default /tmp/ovparity) [--bundle fp16]
    """

func loadModel(_ a: inout Args, defaultBundle: String = "int8") async throws -> OmniVoiceTTS {
    let bundle = try a.string("bundle", defaultBundle)
    let dtype: DType
    switch try a.string("dtype", "f16") {
    case "f32": dtype = .float32
    case "f16": dtype = .float16
    case "bf16": dtype = .bfloat16
    case let x: throw BenchError.usage("--dtype \(x)")
    }
    setCacheLimit(mb: try a.int("cache-limit-mb", 256))
    let t0 = now()
    let tts: OmniVoiceTTS
    switch bundle {
    case "int8": tts = try await OmniVoiceTTS.load(.int8, computeDType: dtype)
    case "fp16": tts = try await OmniVoiceTTS.load(.fp16, computeDType: dtype)
    default: tts = try await OmniVoiceTTS(directory: URL(fileURLWithPath: bundle), computeDType: dtype)
    }
    print(String(format: "load %@ (%@): %.2fs, footprint %@", bundle, "\(dtype)", now() - t0,
                 gb(ProcessMemory.footprint().current)))
    return tts
}

func loadReference(_ tts: OmniVoiceTTS, _ a: inout Args, profiler: StageProfiler? = nil) throws -> VoiceReference {
    let refPath = try a.string("ref", "/tmp/omnivoice/ref_plus_zh.wav")
    let refText: String
    if let t = a.optional("ref-text") {
        refText = t
    } else if let f = a.optional("ref-text-file") {
        refText = try String(contentsOfFile: f, encoding: .utf8)
    } else {
        refText = try String(contentsOfFile: "/tmp/omnivoice/ref2.txt", encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines) + " 好，我看一下。"
    }
    var ro = ReferenceOptions()
    ro.preprocess = !a.flag("no-preprocess-ref")
    ro.maxSeconds = try a.double("ref-max-seconds", nil)
    let t0 = now()
    let clip = try AudioIO.read(URL(fileURLWithPath: refPath))
    let ref = try tts.encodeReference(
        samples: clip.samples, sampleRate: clip.sampleRate, transcript: refText, options: ro,
        profiler: profiler)
    print(String(format: "reference: %d frames (%.2fs), encode total %.2fs", ref.frames,
                 Double(ref.frames) / tts.frameRate, now() - t0))
    return ref
}

func generationOptions(_ a: inout Args) throws -> GenerationOptions {
    var o = GenerationOptions()
    o.diffusion.steps = try a.int("steps", 16)
    o.diffusion.guidance = Float(try a.double("guidance", 2.0)!)
    if a.flag("greedy") {
        o.diffusion.positionTemperature = 0
        o.diffusion.classTemperature = 0
    }
    if let t = try a.double("position-temperature", nil) { o.diffusion.positionTemperature = Float(t) }
    if let t = try a.double("class-temperature", nil) { o.diffusion.classTemperature = Float(t) }
    if let s = a.optional("seed") {
        guard let v = UInt64(s) else { throw BenchError.usage("--seed expects UInt64") }
        o.seed = v
    }
    o.duration = try a.double("duration", nil)
    o.instruct = a.optional("instruct")
    return o
}

func runBench(_ argv: ArraySlice<String>) async throws {
    var a = try Args(argv, flags: ["greedy", "profile", "no-preprocess-ref", "score"])
    let tts = try await loadModel(&a)
    let profiler = a.flag("profile") ? StageProfiler() : nil
    let ref = try loadReference(tts, &a, profiler: profiler)
    var opts = try generationOptions(&a)
    let linesPath = try a.string("lines", "/tmp/omnivoice/lines20.json")
    let outDir = a.optional("out")
    let warmup = try a.int("warmup", 1)
    let score = a.flag("score")
    let simWeights = try a.string("sim-weights", SpeakerSimilarityScorer.defaultWeightsURL.path)
    let simAnchor = try a.string("sim-anchor", "/tmp/omnivoice/ref2.wav")
    try a.checkAllConsumed()
    if opts.seed == nil { opts.seed = 0 }  // fixed default so bench runs are repeatable

    let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: linesPath)))
    guard let pairs = raw as? [[String]], pairs.allSatisfy({ $0.count == 2 }) else {
        throw BenchError.failed("\(linesPath): expected [[text, lang], ...]")
    }
    if let outDir { try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true) }

    for _ in 0 ..< warmup {
        _ = try tts.synthesize(text: "Warm up.", language: "en", reference: ref, options: opts)
    }
    profiler?.reset()
    MLX.Memory.peakMemory = 0

    var lat: [Double] = []
    var audioSec: [Double] = []
    var outputs: [[Float]] = []
    for (i, p) in pairs.enumerated() {
        let t0 = now()
        let s = try tts.synthesize(text: p[0], language: p[1], reference: ref, options: opts, profiler: profiler)
        let dt = now() - t0
        lat.append(dt)
        let dur = Double(s.samples.count) / Double(s.sampleRate)
        audioSec.append(dur)
        if score { outputs.append(s.samples) }
        if let outDir {
            try AudioIO.writeWAV(s.samples, sampleRate: s.sampleRate,
                                 to: URL(fileURLWithPath: outDir).appendingPathComponent("\(i).wav"))
        }
        print(String(format: "[%2d] %.3fs  audio %.2fs  T=%d  %@", i, dt, dur, s.frames, p[0]))
    }
    let st = Stats(lat)
    let mem = ProcessMemory.footprint()
    print(String(format: "latency: mean %.3fs  median %.3fs  max %.3fs  min %.3fs  (n=%d)",
                 st.mean, st.median, st.max, st.min, lat.count))
    print(String(format: "RTF (sum latency / sum audio): %.3f", lat.reduce(0, +) / audioSec.reduce(0, +)))
    print("memory: footprint now \(gb(mem.current)), lifetime peak \(gb(mem.peak)), "
          + "MLX peak active \(gb(MLX.Memory.peakMemory)), MLX cache limit \(gb(MLX.Memory.cacheLimit))")
    if let profiler {
        print("profile (stage boundaries forced; totals over \(pairs.count) lines, per-line mean):")
        for name in profiler.order {
            let t = profiler.totals[name]!, c = profiler.counts[name]!
            print(String(format: "  %-24@ total %7.3fs  calls %4d  per-line %.4fs", name, t, c,
                         t / Double(name.hasPrefix("reference") ? 1 : pairs.count)))
        }
    }
    if score {
        let t0 = now()
        let scorer = try SpeakerSimilarityScorer(weightsURL: URL(fileURLWithPath: simWeights))
        let anchor = try scorer.embed(contentsOf: URL(fileURLWithPath: simAnchor))
        let t1 = now()
        var sims: [Double] = []
        for (i, o) in outputs.enumerated() {
            let s = Double(try scorer.similarity(anchor, o, sampleRate: tts.sampleRate))
            sims.append(s)
            print(String(format: "  sim[%2d] %.3f", i, s))
        }
        let st = Stats(sims)
        print(String(format: "SIM-o vs %@: mean %.3f  median %.3f  min %.3f  (scorer load+anchor %.2fs, %.3fs/line)",
                     simAnchor, st.mean, st.median, st.min, t1 - t0, (now() - t1) / Double(outputs.count)))
        print("memory after scoring: lifetime peak \(gb(ProcessMemory.footprint().peak))")
    }
}

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

func runSynth(_ argv: ArraySlice<String>) async throws {
    var a = try Args(argv, flags: ["greedy", "no-preprocess-ref"])
    let tts = try await loadModel(&a)
    let ref = try loadReference(tts, &a)
    let opts = try generationOptions(&a)
    let text = try a.string("text")
    let lang = a.optional("lang")
    let out = try a.string("out")
    try a.checkAllConsumed()
    let t0 = now()
    let s = try tts.synthesize(text: text, language: lang, reference: ref, options: opts)
    try AudioIO.writeWAV(s.samples, sampleRate: s.sampleRate, to: URL(fileURLWithPath: out))
    print(String(format: "%.3fs -> %@ (%.2fs audio, seed %@)", now() - t0, out,
                 Double(s.samples.count) / Double(s.sampleRate), s.seed.map { "\($0)" } ?? "n/a"))
}

func matchRate(_ a: [Int32], _ b: [Int32]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    return Double(zip(a, b).filter { $0 == $1 }.count) / Double(a.count)
}

func runParity(_ argv: ArraySlice<String>) async throws {
    var a = try Args(argv, flags: ["no-preprocess-ref", "step0-cpu"])
    let step0CPU = a.flag("step0-cpu")
    let dump = try a.string("dump", "/tmp/ovparity")
    let tts = try await loadModel(&a, defaultBundle: "fp16")
    let refPath = try a.string("ref", "/tmp/omnivoice/ref_plus_zh.wav")
    try a.checkAllConsumed()

    let j = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: dump + "/parity.json"))) as! [String: Any]
    let steps = j["steps"] as! Int
    let pyRef = j["ref"] as! [String: Any]
    let pyRefCodes2D = pyRef["codes"] as! [[Int]]
    let pyRefCodes = pyRefCodes2D.flatMap { $0.map(Int32.init) }
    let pyRefText = pyRef["text"] as! String
    let pyRMS = Float(pyRef["rms"] as! Double)
    let pyFrames = pyRefCodes2D[0].count
    print("== reference (python \(pyFrames) frames, rms \(pyRMS), round_fp16 \(j["round_fp16"]!))")

    // A1: Swift end-to-end reference encode from the file.
    let swRef = try tts.encodeReference(
        audio: URL(fileURLWithPath: refPath),
        transcript: try String(contentsOfFile: "/tmp/omnivoice/ref2.txt", encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines) + " 好，我看一下。")
    print("A1 swift file->tokens: frames \(swRef.frames) vs \(pyFrames); "
          + "match \(swRef.frames == pyFrames ? String(format: "%.4f", matchRate(swRef.codes, pyRefCodes)) : "n/a (length differs)"); "
          + "rms \(swRef.rms) vs \(pyRMS); transcript equal: \(swRef.transcript == pyRefText)")
    // A2: encoder only, on python's preprocessed waveform.
    let pyWav = try readF32(dump + "/ref_wav24.f32")
    var noPre = ReferenceOptions()
    noPre.preprocess = false
    let encOnly = try tts.encodeReference(samples: pyWav, sampleRate: 24000, transcript: pyRefText, options: noPre)
    print("A2 swift encoder on python waveform: frames \(encOnly.frames) vs \(pyFrames); match "
          + String(format: "%.4f", encOnly.frames == pyFrames ? matchRate(encOnly.codes, pyRefCodes) : 0))

    let ref = try VoiceReference(codes: pyRefCodes, codebooks: 8, frames: pyFrames, transcript: pyRefText, rms: pyRMS)
    var greedy = GenerationOptions()
    greedy.diffusion = .greedy
    greedy.diffusion.steps = steps
    var tokTotal = 0, tokMatch = 0
    for line in j["lines"] as! [[String: Any]] {
        let i = line["index"] as! Int
        let text = line["text"] as! String, lang = line["lang"] as! String
        let pyIds = (line["input_ids"] as! [[Int]]).flatMap { $0.map(Int32.init) }
        let pyMask = (line["audio_mask"] as! [Int]).map(Int32.init)
        let pyT = line["target_len"] as! Int
        let pyCodes = (line["codes"] as! [[Int]]).flatMap { $0.map(Int32.init) }
        print("== line \(i) [\(lang)] \(text)")
        let inp = try tts.inspectInputs(text: text, language: lang, reference: ref, options: greedy)
        print("B inputs: target_len \(inp.targetLen) vs \(pyT); ids equal \(inp.inputIds == pyIds); mask equal \(inp.audioMask == pyMask) (L=\(inp.length))")
        let s0 = dump + "/line\(i)_step0_cond.f32"
        if FileManager.default.fileExists(atPath: s0) {
            let pyC = try readF32(s0), pyU = try readF32(dump + "/line\(i)_step0_uncond.f32")
            let sw = try Device.withDefaultDevice(step0CPU ? Device.cpu : Device.defaultDevice()) {
                try tts.firstStepLogits(text: text, language: lang, reference: ref, options: greedy)
            }
            for (name, a, b) in [("cond", pyC, sw.cond), ("uncond", pyU, sw.uncond)] {
                let c = compareSignals(a, b)
                let V = 1025
                var agree = 0
                for r in 0 ..< a.count / V {
                    let ra = a[r * V ..< (r + 1) * V], rb = b[r * V ..< (r + 1) * V]
                    if ra.indices.max(by: { ra[$0] < ra[$1] }) == rb.indices.max(by: { rb[$0] < rb[$1] }) { agree += 1 }
                }
                print(String(format: "S0 %@ logits: max|d| %.4f, SNR %.1f dB, argmax agree %d/%d",
                             name, c.maxAbs, c.snrDB, agree, a.count / V))
            }
        }
        let t0 = now()
        let gen = try tts.generateCodes(text: text, language: lang, reference: ref, options: greedy)
        let dt = now() - t0
        let m = gen.frames == pyT ? matchRate(gen.codes, pyCodes) : 0
        if gen.frames == pyT {
            tokTotal += pyCodes.count
            tokMatch += zip(gen.codes, pyCodes).filter { $0 == $1 }.count
        }
        var perCB: [String] = []
        if gen.frames == pyT {
            for c in 0 ..< 8 {
                let r = c * pyT ..< (c + 1) * pyT
                perCB.append(String(format: "%.2f", matchRate(Array(gen.codes[r]), Array(pyCodes[r]))))
            }
        }
        print(String(format: "C tokens (greedy, %d steps, %.2fs): exact match %.4f  per-codebook [%@]",
                     steps, dt, m, perCB.joined(separator: " ")))
        let pyRaw = try readF32(dump + "/line\(i)_raw.f32")
        let swRaw = try tts.decode(codes: pyCodes, frames: pyT)
        let c = compareSignals(pyRaw, swRaw)
        print(String(format: "D decode(python tokens): len %d vs %d, corr %.5f, SNR %.1f dB, max|d| %.4f",
                     swRaw.count, pyRaw.count, c.corr, c.snrDB, c.maxAbs))
        let swOwn = try tts.decode(codes: gen.codes, frames: gen.frames)
        let c2 = compareSignals(pyRaw, swOwn)
        print(String(format: "E decode(swift tokens) vs python raw audio: corr %.5f, SNR %.1f dB", c2.corr, c2.snrDB))
    }
    if tokTotal > 0 {
        print(String(format: "== overall token exact match: %.4f (%d/%d)", Double(tokMatch) / Double(tokTotal), tokMatch, tokTotal))
    }
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

// `--no-tf32` must take effect before MLX's first matmul: MLX reads
// MLX_ENABLE_TF32 once, and on GPUs with neural accelerators (M5) it defaults
// to TF32 for float32 matmuls (~62 dB SNR instead of ~125 dB).
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
