import Foundation
import MLX
import MLXNN
import MLXRandom

/// Talker prompt: prefill embeddings, per-step trailing text, `tts_pad` embedding.
struct Prompt {
    var prefill: MLXArray  // [1, L, D]
    var trailing: MLXArray  // [1, N, D]
    var ttsPad: MLXArray  // [1, 1, D]
}

extension Qwen3Voice {
    // MARK: Prompt construction (port of `Qwen3TTSForConditionalGeneration.generate`)

    func embedText(_ ids: [Int]) -> MLXArray {
        talker.embedText(MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0))
    }

    func embedCodec(_ ids: [Int]) -> MLXArray {
        talker.embedCodec(MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0))
    }

    /// `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` tokenized as a
    /// whole, like `_build_assistant_text` + the HF tokenizer.
    func assistantIds(_ text: String) throws -> [Int] {
        let ids = try tokenizer.encode("<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n")
        guard ids.count > 8 else { throw VoiceError.invalidArgument("text is empty") }
        return ids
    }

    func buildPrompt(text: String, language: String, voice: Voice, textFeed: TextFeed?) throws -> Prompt {
        let inputIds = try assistantIds(text)
        var langId = try languageId(language)
        let speakerEmbed: MLXArray
        switch voice {
        case .reference(let ref):
            guard supportsReference else {
                throw VoiceError.unsupported("ICL references need a Base bundle (this one has no speaker encoder)")
            }
            speakerEmbed = MLXArray(ref.speakerEmbedding, [1, 1, ref.speakerEmbedding.count]).asType(computeDType)
        case .speaker(let name):
            let key = name.lowercased()
            guard let id = speakerIds[key] else { throw VoiceError.unknownSpeaker(name, known: speakers) }
            let l = language.lowercased()
            if ["chinese", "zh", "auto"].contains(l), let dialect = speakerDialects[key] {
                guard let d = languageIds[dialect] else {
                    throw VoiceError.unknownLanguage(dialect, known: languages)
                }
                langId = d
            }
            speakerEmbed = embedCodec([id])
        }
        let special = embedText([SpecialTokens.ttsBos, SpecialTokens.ttsEos, SpecialTokens.ttsPad])
        let ttsBos = special[0..., 0 ..< 1, 0...]
        let ttsEos = special[0..., 1 ..< 2, 0...]
        let ttsPad = special[0..., 2 ..< 3, 0...]

        let think: [Int] =
            langId.map { [SpecialTokens.codecThink, SpecialTokens.codecThinkBos, $0, SpecialTokens.codecThinkEos] }
            ?? [SpecialTokens.codecNothink, SpecialTokens.codecThinkBos, SpecialTokens.codecThinkEos]
        let codecPrefix = concatenated(
            [embedCodec(think), speakerEmbed, embedCodec([SpecialTokens.codecPad, SpecialTokens.codecBos])], axis: 1)
        let n = codecPrefix.dim(1)
        let D = talkerConfig.hiddenSize
        let role = embedText(Array(inputIds[0 ..< 3]))
        let overlay =
            concatenated([broadcast(ttsPad, to: [1, n - 2, D]), ttsBos], axis: 1) + codecPrefix[0..., 0 ..< (n - 1), 0...]
        var prefill = concatenated([role, overlay], axis: 1)
        let textIds = Array(inputIds[3 ..< (inputIds.count - 5)])

        switch voice {
        case .reference(let ref):
            // generate_icl_prompt
            let refIds = try tokenizer.encode("<|im_start|>assistant\n\(ref.transcript)<|im_end|>\n")
            let textEmbed = concatenated([embedText(Array(refIds[3 ..< (refIds.count - 2)]) + textIds), ttsEos], axis: 1)
            let codes = MLXArray(ref.codes, [ref.frames, 16])
            var codecEmbed = talker.embedCodec(codes[0..., 0].expandedDimensions(axis: 0))
            for g in 1 ..< 16 {
                codecEmbed = codecEmbed + codePredictor.codecEmbeddings[g - 1](codes[0..., g].expandedDimensions(axis: 0))
            }
            codecEmbed = concatenated([embedCodec([SpecialTokens.codecBos]), codecEmbed], axis: 1)
            let tl = textEmbed.dim(1), cl = codecEmbed.dim(1)
            let feed = textFeed ?? .interleaved
            if feed == .upfront {
                let icl = concatenated(
                    [
                        textEmbed + embedCodec([Int](repeating: SpecialTokens.codecPad, count: tl)),
                        codecEmbed + ttsPad,
                    ], axis: 1)
                return Prompt(prefill: concatenated([prefill, icl], axis: 1), trailing: ttsPad, ttsPad: ttsPad)
            }
            if tl > cl {
                let icl = textEmbed[0..., 0 ..< cl, 0...] + codecEmbed
                return Prompt(
                    prefill: concatenated([prefill, icl], axis: 1), trailing: textEmbed[0..., cl..., 0...], ttsPad: ttsPad)
            }
            let padded = concatenated([textEmbed, broadcast(ttsPad, to: [1, cl - tl, D])], axis: 1)
            return Prompt(prefill: concatenated([prefill, padded + codecEmbed], axis: 1), trailing: ttsPad, ttsPad: ttsPad)
        case .speaker:
            let feed = textFeed ?? .upfront
            let codecBos = codecPrefix[0..., (n - 1)..., 0...]
            if feed == .upfront {
                let body = concatenated([embedText(textIds), ttsEos], axis: 1)
                    + embedCodec([Int](repeating: SpecialTokens.codecPad, count: textIds.count + 1))
                prefill = concatenated([prefill, body, ttsPad + codecBos], axis: 1)
                return Prompt(prefill: prefill, trailing: ttsPad, ttsPad: ttsPad)
            }
            prefill = concatenated([prefill, embedText([inputIds[3]]) + codecBos], axis: 1)
            let trailing = concatenated([embedText(Array(inputIds[4 ..< (inputIds.count - 5)])), ttsEos], axis: 1)
            return Prompt(prefill: prefill, trailing: trailing, ttsPad: ttsPad)
        }
    }

    // MARK: Compilation

    func setupCompilation() {
        let t = talker
        let layers = talkerConfig.numLayers
        compiledTalkerStep = compile(inputs: [t], outputs: [t], shapeless: true) { inputs in
            var cache: [(MLXArray, MLXArray)] = []
            for i in 0 ..< layers { cache.append((inputs[2 + 2 * i], inputs[3 + 2 * i])) }
            let (logits, hidden, newCache) = t(inputsEmbeds: inputs[0], offset: inputs[1], cache: cache)
            var out = [logits, hidden]
            for (k, v) in newCache { out += [k, v] }
            return out
        }
    }

    func talkerStep(_ embeds: MLXArray, offset: Int, cache: [(MLXArray, MLXArray)]) -> (
        MLXArray, MLXArray, [(MLXArray, MLXArray)]
    ) {
        var inputs = [embeds, MLXArray(Int32(offset))]
        for (k, v) in cache { inputs += [k, v] }
        let out = compiledTalkerStep!(inputs)
        var newCache: [(MLXArray, MLXArray)] = []
        for i in 0 ..< talkerConfig.numLayers { newCache.append((out[2 + 2 * i], out[3 + 2 * i])) }
        return (out[0], out[1], newCache)
    }

    /// Codebooks 1–15 for one frame as one compiled graph (`[15]` int32).
    func predictFrame(hidden: MLXArray, code0: MLXArray, sampling: SamplingOptions, key: MLXArray) -> MLXArray {
        let greedy = !sampling.subtalkerDoSample
        let topK = greedy ? 0 : sampling.subtalkerTopK
        let cacheKey = "\(greedy)-\(topK)"
        let fn: ([MLXArray]) -> [MLXArray]
        if let f = compiledFrames[cacheKey] {
            fn = f
        } else {
            let cp = codePredictor
            fn = compile(inputs: [cp], outputs: [cp], shapeless: false) { inputs in
                [cp.predictFrame(inputsEmbeds: inputs[0], temperature: inputs[1], gumbels: inputs[2], topK: topK, greedy: greedy)]
            }
            compiledFrames[cacheKey] = fn
        }
        let groups = cpConfig.numCodeGroups - 1
        let gumbels = greedy
            ? MLXArray.zeros([groups, cpConfig.vocabSize])
            : MLXRandom.gumbel([groups, cpConfig.vocabSize], key: key)
        let input = concatenated([hidden, talker.embedCodec(code0.reshaped(1, 1))], axis: 1)
        return fn([input, MLXArray(sampling.subtalkerTemperature), gumbels])[0]
    }

    // MARK: Talker sampling (HF logits processors + warpers)

    func sampleFirstCode(
        logits: MLXArray, generated: MLXArray?, step: Int, sampling: SamplingOptions, key: MLXArray
    ) -> MLXArray {
        var s = logits.reshaped(-1).asType(.float32)
        if sampling.repetitionPenalty != 1, let generated {
            let p = MLXArray(sampling.repetitionPenalty)
            s = MLX.where(generated, MLX.where(s .< 0, s * p, s / p), s)
        }
        let negInf = MLXArray(-Float.infinity)
        s = MLX.where(suppressMask, negInf, s)
        if step < sampling.minNewTokens {
            let v = s.dim(0)
            s = MLX.where(MLXArray(Int32(0) ..< Int32(v)) .== Int32(SpecialTokens.codecEos), negInf, s)
        }
        if !sampling.doSample { return argMax(s).asType(.int32) }
        s = s / MLXArray(sampling.temperature)
        let v = s.dim(0)
        if sampling.topK > 0 && sampling.topK < v {
            let kth = MLX.sorted(s)[v - sampling.topK]
            s = MLX.where(s .< kth, negInf, s)
        }
        if sampling.topP < 1 {
            // HF TopPLogitsWarper: drop the low tail whose cumulative mass <= 1 - top_p.
            let order = argSort(s)
            let sortedS = s[order]
            let cum = cumsum(softmax(sortedS))
            var drop = cum .<= MLXArray(1 - sampling.topP)
            drop = MLX.where(MLXArray(Int32(0) ..< Int32(v)) .== Int32(v - 1), MLXArray(false), drop)
            let filtered = MLX.where(drop, negInf, sortedS)
            s = filtered[argSort(order)]
        }
        return argMax(s + MLXRandom.gumbel([v], key: key)).asType(.int32)
    }

    // MARK: Run

    func run(
        text: String, language: String, voice: Voice, options: GenerationOptions,
        streaming: StreamingOptions?, onChunk: ((AudioChunk) throws -> Void)?
    ) throws -> Synthesis {
        try options.sampling.validate()
        let start = now()
        var timings = StageTimings()
        let seed = options.seed ?? UInt64.random(in: 0 ... UInt64(Int64.max))
        var key = MLXRandom.key(seed)
        let sampling = options.sampling
        let profile = options.profile

        let prompt = try buildPrompt(text: text, language: language, voice: voice, textFeed: options.textFeed)
        var refCodes: [Int32] = []
        var refFrames = 0
        if case .reference(let ref) = voice {
            refCodes = ref.codes
            refFrames = ref.frames
        }

        // Prefill
        let L = prompt.prefill.dim(1)
        var (logits, hidden, cache) = talker(inputsEmbeds: prompt.prefill, offset: MLXArray(Int32(0)), cache: nil)
        logits = logits[0..., (L - 1) ..< L, 0...]
        hidden = hidden[0..., (L - 1) ..< L, 0...]

        let vocab = talkerConfig.codecVocabSize
        let vocabIds = MLXArray(Int32(0) ..< Int32(vocab))
        var penalty = MLXArray.zeros([vocab], dtype: .bool)  // tokens generated so far
        var frameCodes: [MLXArray] = []  // per frame: [16] int32 on device
        var codes: [Int32] = []
        var frames = 0
        var finished = false
        var emittedFrames = 0
        var chunkIndex = 0

        func split3() -> (MLXArray, MLXArray) {
            let ks = MLXRandom.split(key: key, into: 3)
            key = ks[0]
            return (ks[1], ks[2])
        }

        /// Host copies of the codes of frames `codes.count / 16 ..< upTo`.
        func syncCodes(upTo: Int) {
            let have = codes.count / 16
            guard upTo > have else { return }
            codes += concatenated(Array(frameCodes[have ..< upTo]), axis: 0).asArray(Int32.self)
        }

        func emit(upTo end: Int, final: Bool) throws {
            guard let onChunk, let streaming else { return }
            let t0 = now()
            syncCodes(upTo: end)
            let startF = emittedFrames
            // Left context from earlier generated frames, then reference frames.
            let ctx = min(streaming.leftContextFrames, startF + refFrames)
            let fromRef = max(0, ctx - startF)
            var ctxCodes: [Int32] = []
            if fromRef > 0 { ctxCodes += refCodes[((refFrames - fromRef) * 16)...] }
            ctxCodes += codes[((startF - (ctx - fromRef)) * 16) ..< (end * 16)]
            let n = ctx + (end - startF)
            let arr = MLXArray(ctxCodes, [n, 16]).transposed(1, 0).expandedDimensions(axis: 0)
            let wav = decoder.executeDecoder(arr).reshaped(-1)
            let kept = wav[(ctx * codecConfig.samplesPerFrame)...]
            let samples = kept.asArray(Float.self)
            timings.decode += now() - t0
            emittedFrames = end
            chunkIndex += 1
            try onChunk(AudioChunk(samples: samples, frames: startF ..< end, isFinal: final, elapsed: now() - start))
        }

        // The loop runs one frame ahead of the host: the graph for frame s+1 (code
        // predictor of s, talker step, sampling of s+1) is built and queued with
        // asyncEval before the host reads frame s's first code for the EOS check, so
        // graph construction overlaps GPU work. Numerics are unchanged; when frame s
        // is EOS the queued speculative frame is discarded.
        let (kT0, kC0) = split3()
        var code0 = sampleFirstCode(logits: logits, generated: nil, step: 0, sampling: sampling, key: kT0)
        var kCP = kC0
        asyncEval(code0)
        var tStep = now()
        if profile { eval(code0); timings.prefill = now() - start; tStep = now() }
        for step in 0 ..< sampling.maxNewTokens {
            let tCP = now()
            let rest = predictFrame(hidden: hidden, code0: code0, sampling: sampling, key: kCP)
            if profile { eval(rest); timings.codePredictor += now() - tCP }
            let frame = concatenated([code0.reshaped(1), rest])
            var next: MLXArray?
            if step + 1 < sampling.maxNewTokens {
                penalty = logicalOr(penalty, vocabIds .== code0)
                // Next talker input: sum of the 16 codebook embeddings + trailing text.
                var embed = talker.embedCodec(code0.reshaped(1, 1))
                for g in 1 ..< 16 {
                    embed = embed + codePredictor.codecEmbeddings[g - 1](rest[(g - 1) ..< g].reshaped(1, 1))
                }
                let t = prompt.trailing.dim(1)
                embed = embed + (step < t ? prompt.trailing[0..., step ..< (step + 1), 0...] : prompt.ttsPad)
                let tT = now()
                (logits, hidden, cache) = talkerStep(embed, offset: L + step, cache: cache)
                if profile { eval(logits, hidden); timings.talker += now() - tT }
                let (kT, kC) = split3()
                kCP = kC
                next = sampleFirstCode(
                    logits: logits, generated: penalty, step: step + 1, sampling: sampling, key: kT)
                asyncEval(frame, next!)
            } else {
                asyncEval(frame)
            }
            // EOS check on frame `step` (already computed or in flight).
            let c0 = code0.item(Int32.self)
            if step == 0 && !profile { timings.prefill = now() - start; tStep = now() }
            if c0 == Int32(SpecialTokens.codecEos) {
                finished = true
                break
            }
            frameCodes.append(frame)
            frames += 1
            if let streaming, onChunk != nil {
                let due = emittedFrames == 0 ? streaming.firstChunkFrames : emittedFrames + streaming.chunkFrames
                if frames >= due { try emit(upTo: frames, final: false) }
            }
            guard let next else { break }
            code0 = next
        }
        syncCodes(upTo: frames)
        if !profile { timings.talker = now() - tStep }

        if let streaming, onChunk != nil {
            _ = streaming
            if frames > emittedFrames || chunkIndex == 0 { try emit(upTo: frames, final: true) }
            timings.total = now() - start
            return Synthesis(samples: [], codes: codes, frames: frames, seed: seed, finished: finished, timings: timings)
        }

        // Decode (reference: prepend the reference codes, decode, cut the reference share).
        let tD = now()
        var samples: [Float] = []
        if frames > 0 {
            let ctx = refFrames > 0 ? min(refFrames, options.referenceDecodeContextFrames ?? refFrames) : 0
            let all = Array(refCodes[((refFrames - ctx) * 16)...]) + codes
            let total = ctx + frames
            let wav = decode(
                codes: all, frames: total, chunkFrames: options.decodeChunkFrames,
                leftContextFrames: options.decodeLeftContextFrames)
            let cut = Int(Double(ctx) / Double(total) * Double(wav.count))
            samples = Array(wav[cut...])
        }
        timings.decode = now() - tD
        timings.total = now() - start
        return Synthesis(samples: samples, codes: codes, frames: frames, seed: seed, finished: finished, timings: timings)
    }
}

@inline(__always) func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
