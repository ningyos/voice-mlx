# voice-mlx

Lean, inference-only Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12 Hz **0.6B** — ICL voice cloning with the
Base model and the single-speaker (CustomVoice-style, speaker name) path used by fine-tuned
models — plus an MLX port of the SIM-o speaker-similarity scorer (WavLM-large + ECAPA-TDNN).
Built for an Apple Silicon Mac (M5, 16 GB) running next to other workloads.

No PyTorch at runtime, no training, no weight conversion (the `voice-train` repo converts
fine-tuned PyTorch checkpoints into the bundle layout loaded here).

## Highlights

Compared to the code it was extracted from (soniqo/speech-swift `Sources/Qwen3TTS`,
commit `468db5f`):

- **Token-exact with the official `qwen-tts`** in parity mode (greedy, float32, TF32 off):
  Base ICL and CustomVoice speaker path, 9/9 lines identical (see Parity). Fixes found on
  the way: the text tokenizer (speech-swift split on whitespace only and dropped unknown
  pieces; now the Qwen2 pre-tokenizer regex + NFC + byte-level BPE), the speaker-encoder
  mel (slaney mel + `(n_fft-hop)/2` reflect pad instead of HTK mel + `n_fft/2`), the
  speaker encoder itself (reflect-padded TDNN convs, missing ReLU in attentive pooling),
  the codec decoder (sliding-window attention of 72 frames, RMSNorm eps 1e-5 instead of
  1e-8) and the reference `chunked_decode` (300-frame chunks, 25 frames left context).
- **Official prompt and sampling semantics.** `generate_voice_clone` (ICL, text interleaved)
  and `generate_custom_voice` (speaker token, text up front, no instruct) prompt layouts;
  HF repetition penalty, `suppress_tokens`, `min_new_tokens=2`, top-k/top-p, separate
  sub-talker sampling. speech-swift bumped the repetition penalty to 1.5 for ICL, injected
  a default instruct and a language id on the speaker path; none of that here.
- **Encode the reference once.** `encodeReference` returns a `Codable` `VoiceReference`
  (codec tokens + x-vector + transcript); every `generate` call reuses it.
- **Seedable, reproducible sampling.** All randomness comes from an MLX PRNG key derived
  from a `UInt64` seed; `seed: nil` draws one and reports it so any run can be replayed.
- **Streaming.** `generateStreaming` / `stream` emit audio chunks while codec frames are
  produced (first chunk after 4 frames by default).
- **Errors surface.** Unknown languages throw (speech-swift silently fell back to `auto`/
  English), unknown speakers throw, missing files throw, and the loader verifies that
  every parameter was set from the checkpoint with the right shape (speech-swift's
  appliers skip absent tensors silently).
- **Only what Qwen3-TTS 0.6B needs** from speech-swift: no AudioCommon/MLXCommon, no
  other models.

## Layout

| Target | What |
|---|---|
| `VoiceMLX` | TTS: talker + code predictor, 12 Hz codec encoder/decoder, speaker encoder, BPE tokenizer, bundle loading, public API |
| `SpeakerSim` | SIM-o scorer: WavLM-large + ECAPA-TDNN on MLX |
| `VoiceAudio` | Audio I/O, torchaudio-compatible resampler, pydub-style silence removal, text helpers |
| `voice-bench` | Synthesis / benchmark / parity CLI |
| `scripts/` | Python reference dumpers (`qwen_parity.py`, scorer parity), scorer weight export |

## Bundles

A model bundle is a directory with `config.json`, `model.safetensors` (+ optional
`model.safetensors.index.json`), `vocab.json`, `merges.txt`, `tokenizer_config.json`; the
codec is `Qwen/Qwen3-TTS-Tokenizer-12Hz`. Hugging Face ids are cached under
`~/Library/Caches/qwen3-speech/models/<id>` (the speech-swift layout) and downloaded on
first use.

| Bundle | Path | Notes |
|---|---|---|
| Base 8-bit (default) | `aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit` | ICL; has `speaker_encoder.*` |
| CustomVoice bf16 | `aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-bf16` | official built-in speakers; float16 on disk |
| fine-tuned (voice-train) | `…/bundle-8bit`, `…/bundle-bf16` | `spk_id {"<speaker>": 3000}`, no speaker encoder |

`LoadOptions.precision`: `.int8` (default) loads 8-bit bundles as stored and quantizes
unquantized bundles at load time (`mx.quantize` affine, group 64, from bf16 — the recipe of
the aufklarer / voice-train 8-bit bundles); `.bf16` needs an unquantized bundle.
`computeDType` (default bf16) is the activation / scale dtype; float32 is the parity mode.

## Usage

```swift
import VoiceMLX

MLX.Memory.cacheLimit = 256 << 20   // host process decides; see Memory below

// Base, ICL voice cloning
let tts = try await Qwen3Voice.load(model: .base8bit)
let voice = try tts.encodeReference(contentsOf: URL(fileURLWithPath: "reference.wav"),
                                    transcript: "<exactly what is said in reference.wav>")   // once
let out = try tts.generate(text: "好，我看一下。", language: "zh", reference: voice, seed: 42)
// out.samples: 24 kHz mono; out.codes, out.seed, out.finished, out.timings

// Single-speaker bundle (a fine-tuned speaker) — language "auto" matches how sft_12hz trains
let spk = try await Qwen3Voice.load(model: .directory(URL(fileURLWithPath: "/path/bundle-8bit")))
let line = try spk.generate(text: "好，我看一下。", language: "auto", speaker: "<speaker>", seed: 0)

// Streaming
for try await chunk in tts.stream(text: "你今天吃飯了沒？", language: "zh", voice: .reference(voice)) {
    play(chunk.samples)
}
```

Languages: `auto`, the names in the bundle's `codec_language_id` (chinese, english, german,
italian, portuguese, spanish, japanese, korean, french, russian; dialects on CustomVoice) or
their ISO codes (`zh`, `en`, …). Anything else throws.

CLI:

```bash
swift build -c release
.build/release/voice-bench synth --text "好，我看一下。" --lang zh --out /tmp/a.wav          # Base ICL, default reference
.build/release/voice-bench synth --model /path/bundle-8bit --speaker <speaker> --lang auto --text "..." --out /tmp/b.wav
.build/release/voice-bench bench --lines /tmp/omnivoice/lines20.json --seeds 0,1 --out /tmp/w --score [--stream] [--profile]
```

## Parity with the official `qwen-tts` (measured)

Reference: `qwen-tts` 0.1.1 from PyPI (torch 2.15, CPU, float32) in a uv venv,
greedy decoding (`do_sample=False, subtalker_dosample=False`; repetition penalty 1.05
still applies, as in HF `generate`). Base: lines 1, 3, 10 (zh), 14, 15 (en) of
`lines20.json`, 19 s reference resampled once to 24 kHz (torchaudio) and fed to both sides;
PyTorch runs the aufklarer 8-bit weights dequantized to float32 (`--mlx-weights`), so both
sides use identical weights. CustomVoice: official `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
vs the aufklarer bf16 bundle (float16 on disk), speaker `ryan`, lines 1, 10, 14, 15
(line 3 is excluded: greedy never emits EOS there in PyTorch either).

| Check | Swift parity mode (f32, `--no-tf32`) |
|---|---|
| Text token ids (target + reference transcript) | identical, all lines |
| Reference codec tokens (239 × 16) | 100 % |
| Reference x-vector | cosine 1.000000 |
| Step-0 talker logits, Base | 116–121 dB SNR, argmax equal |
| **Generated tokens, Base ICL (5 lines, 1824 tokens)** | **100 % identical, 5/5 lines** |
| Step-0 logits, CustomVoice (bf16 weights) | 116–118 dB |
| **Generated tokens, CustomVoice (4 lines, 2880 tokens)** | **100 % identical, 4/4 lines** |
| Codec decode of identical tokens | corr 1.00000, 101–111 dB SNR, equal lengths |
| Quantize-on-load (voice-train Base bf16 → int8) vs aufklarer 8-bit | 5/5 lines token-identical |

Production numerics diverge, as expected for an autoregressive sampler fed different
rounding (one flipped token changes everything after it), so step-0 logit SNR is the
metric there:

| Swift config | Base vs dequantized-8-bit PyTorch | CustomVoice vs PyTorch bf16 weights |
|---|---|---|
| f32 + TF32 (MLX default on M5) | 61–63 dB, 0/5 lines identical (66 % tokens) | — |
| bf16 activations (default) | 41–45 dB | bf16 weights: 40–43 dB |
| 8-bit weights, f32 activations | (same weights) | 37–39 dB |
| 8-bit weights, bf16 activations (default) | 41–45 dB | 35–37 dB |

Divergence causes: (1) on M5 MLX runs float32 matmuls as TF32 (`MLX_ENABLE_TF32` defaults
to 1, ~62 dB; `voice-bench matmul-check`); with it off the port is token-exact. (2) bf16
activations cost about as much as 8-bit weights (≈ 40 dB each). The codec decoder always
runs float32; with TF32 on it is 43–51 dB vs PyTorch on identical tokens. f16 activations
were not measured to completion (the f16 run did not finish within its time box; speech-swift
reports f16 talker decode degenerating) — do not use `--dtype f16`.

Reproduce:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python qwen-tts
PY=.venv/bin/python
$PY scripts/qwen_parity.py --kind base --out /tmp/vmlx/parity_base \
    --mlx-weights ~/Library/Caches/qwen3-speech/models/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit   # ~1.5 min
$PY scripts/qwen_parity.py --kind custom --speaker ryan --indices 1,10,14,15 --out /tmp/vmlx/parity_custom
swift build -c release
.build/release/voice-bench parity --dump /tmp/vmlx/parity_base --dtype f32 --no-tf32
.build/release/voice-bench parity --dump /tmp/vmlx/parity_custom --model custom --precision bf16 --dtype f32 --no-tf32
.build/release/voice-bench parity --dump /tmp/vmlx/parity_base                       # production numerics
```

## Measured on Apple M5, 16 GB

macOS 27.0.1, mlx-swift 0.31.6, release build, MLX cache limit 256 MB, bf16 activations,
1 warm-up line, seed 0 (sampling: official defaults). Latency = text in → 24 kHz samples
out, whole utterance. Peak = lifetime-max `phys_footprint` of the bench process (weights +
Metal buffers + MLX cache), before the scorer loads. A `dollcore` daemon and other jobs were
running, so expect ±5 % noise. Measured 2026-11-13.

| Config | Lines | Mean | Median | Max | First audio (stream) | Peak |
|---|---|---|---|---|---|---|
| **Base 8-bit, ICL 19 s ref** | lines20 (14 zh, 6 en) | **1.30 s** | 1.32 s | 1.86 s | — | 7.3 GB |
| Base 8-bit, ICL, streaming | lines20 | 1.08 s (to last chunk) | 1.00 s | 2.14 s | **0.22 s** (max 0.23) | 3.8 GB |
| Base 8-bit, ICL (before the kept optimizations) | 8 short lines (`qw_lines` 0–7) | 1.09 s | 1.07 s | 1.19 s | — | 5.6 GB |
| Speaker path 8-bit (voice-train smoke bundle, one speaker, `auto`) | lines20 × seeds 0,1 | 0.67 s | 0.62 s | 1.17 s | — | see note |
| Speaker path bf16 (same bundle) | lines20 × seeds 0,1 | 1.07 s | 1.00 s | 1.85 s | — | 4.6 GB |
| CustomVoice 8-bit (`ryan`, quantized at load) | lines20 × seeds 0,1 | 1.26 s | 1.00 s | 6.47 s* | — | 7.2 GB |
| CustomVoice bf16 (`ryan`) | lines20 × seeds 0,1 | 1.86 s | 1.79 s | 4.11 s | — | 5.7 GB |

\* one sampled line ran long (RTF 0.26 overall); latency scales with output length.
Note: speaker-path peaks of 7–8 GB come from the codec decoder's working set on the
longest lines (≈ 2 GB for 56 frames, ≈ 4–5 GB for the ~250 frames of reference + target in
ICL); a single 18-frame line peaks at 2.9 GB. Streaming decodes small chunks and stays
under 4 GB.

Per stage (Base 8-bit, ICL, lines20; `--profile` forces an eval per stage):

| Stage | Time |
|---|---|
| Reference encode (once per voice; file read, resample, codec encoder, x-vector) | 0.10–0.14 s |
| Prompt + prefill (~249 positions) + first token | 35 ms |
| Talker step (28 layers, 8-bit) | 8.6 ms/frame serialized |
| Code predictor (15 groups × 5 layers) | 14.4 ms/frame serialized |
| Talker + code predictor, pipelined (production) | 18.4 ms/frame (12.5 frames = 1 s audio) |
| Codec decode (reference + target, reference chunking) | 0.75–0.78 s/line |
| Model load from local cache | 0.6–1.0 s (bundle), 1.8–2.3 GB footprint after load |

Not measured: cold first load incl. download, other Macs, 1.7B.

### Optimization log

Rule: keep a change only if it is faster and parity is unchanged (outputs byte-identical
WAVs for the same seed on lines20). A/B on lines20, Base 8-bit, seed 0.

| Attempt | Before → after | Parity | Decision |
|---|---|---|---|
| Baseline (speech-swift loop: 2 host syncs per frame, sorted top-k, unfused codec) | mean 1.58 s, 22.0 ms/frame, decode 0.97 s, peak 6.1 GB | — | — |
| Pipeline talker loop one frame ahead (`asyncEval`, penalty mask on device, EOS check lags one frame) | 1.58 → 1.51 s; 22.0 → 18.5 ms/frame | identical | **kept** |
| Fuse SnakeBeta (shapeless `compile`) in the codec decoder | 1.51 → 1.30 s; decode 0.98 → 0.77 s; peak 6.1 → 7.3 GB | identical | **kept** (peak noted) |
| `eval` after each decoder block (to bound memory) | 1.30 → 1.45 s; peak 7.3 → 6.1 GB | identical | rejected (slower) |
| `partitioned` instead of full sort for top-k threshold | 18.48 → 18.36 ms/frame, 1.30 → 1.28 s | identical | rejected (inside noise) |
| Decode only the last N reference frames (`referenceDecodeContextFrames`) | N=72: 1.58 → 0.98 s; N=25: → 0.80 s (measured on the baseline) | audio changes: mean \|Δlog-mel\| 0.04 (N=72), 0.33 (N=25) vs full-reference decode | option only, default off |

## 8-bit vs bf16 on the speaker path (user decision: 8-bit ships; evidence)

- Logits (CustomVoice, vs PyTorch with the official weights): 8-bit weights 35–39 dB,
  bf16 weights with bf16 activations 40–43 dB, bf16 weights with f32 activations 116–118 dB.
  The 8-bit error is of the same order as the bf16-activation error that every non-parity
  config already has; greedy tokens diverge from PyTorch for both.
- SIM-o (sampled, lines20 × 2 seeds): voice-train smoke single-speaker bundle vs
  `ref2.wav`: 8-bit 0.488 mean / bf16 0.487 (smoke bundle ≈ Base + one speaker's x-vector, not a
  quality model). CustomVoice `ryan` vs a separate bf16 `ryan` anchor utterance (seed 99):
  8-bit 0.462 mean / 0.467 median, bf16 0.456 / 0.457 (n=40 each). No measurable
  similarity loss from 8-bit; listening is still the deciding test.
- Speed: 8-bit 17.5–18.3 ms/frame vs bf16 31.5 ms/frame; mean latency 0.67 vs 1.07 s.
- WAVs to listen to: `/tmp/vmlx/listen/{smoke_8bit,smoke_bf16,ryan_int8,ryan_bf16}/`
  (same names = same text and seed), anchor `/tmp/vmlx/listen/ryan_anchor_bf16.wav`.

## Speaker-similarity scorer (SIM-o)

Port of OmniVoice's `omnivoice/eval/speaker_similarity` (WavLM-large + ECAPA-TDNN, 256-d)
loading `k2-fsa/TTS_eval_models` exported to safetensors (fp32, 1.3 GB, not committed;
default path `~/.cache/omnivoice-mlx/sim/wavlm_ecapa.safetensors`, kept so existing exports
keep working). The Python reference stays the `omnivoice` package (needs `s3prl`), used
only for the scorer: `scripts/export_sim_weights.py` and `scripts/sim_parity.py` run in
`/tmp/omnivoice/.venv`. Parity (measured earlier, unchanged code): 80 files max |diff|
0.00066, mean 0.00017; anchor embedding cosine 0.9999977; ~0.05 s per short file.

```bash
$PY scripts/export_sim_weights.py
.build/release/voice-bench sim-parity --list /tmp/speaker-mlx2/sim.txt
```

## Open issues

- Codec decoder working set: up to ~7 GB process peak when decoding the full 19 s reference
  plus target (reference behaviour). Use streaming (< 4 GB) or `referenceDecodeContextFrames`
  (changes audio slightly) if memory matters more than exact reference decoding.
- Greedy decoding can run to `maxNewTokens` without EOS (PyTorch too); sampling is the
  default. Keep `maxNewTokens` capped (default 2048 frames ≈ 164 s); voice-train saw a
  fine-tune run away to max tokens.
- Streaming chunks are decoded with 25 frames of left context, so the streamed waveform
  differs slightly from `generate` (same tokens).
- Not ported: instruct (1.7B only), x-vector-only cloning, batching, text chunking for long
  inputs, 1.7B.

## Tests

`swift test`: tokenizer vs `Qwen2TokenizerFast` ids; bundle loading errors (missing
directory / file, incomplete weights, 8-bit bundle requested as bf16); unknown language and
unknown speaker throw; seed reproducibility (same seed ⇒ identical tokens and audio,
different seed differs, reported seed replays an unseeded run); reference reuse + `Codable`
round trip; streaming covers the utterance with the same tokens as `generate`; pydub /
torchaudio ports; scorer parity on two committed fixtures. Model-backed tests skip
(XCTSkip) when the cached bundles, `/tmp/omnivoice/speaker_ref2.wav` or the scorer weights are
absent.

## Provenance and license

Apache-2.0 (see `LICENSE`, `NOTICE`). Model code extracted from soniqo/speech-swift
`468db5f` (`Sources/Qwen3TTS`, `Sources/MLXCommon`), checked against QwenLM `qwen-tts`
0.1.1. Weights are downloaded separately: `aufklarer/Qwen3-TTS-12Hz-0.6B-*-MLX-*`,
`Qwen/Qwen3-TTS-Tokenizer-12Hz`, scorer `k2-fsa/TTS_eval_models`.
