# omnivoice-mlx

Lean, inference-only Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for
[k2-fsa OmniVoice](https://github.com/k2-fsa/OmniVoice) — zero-shot, voice-cloning,
600+-language TTS — plus an MLX port of OmniVoice's own SIM-o speaker-similarity scorer
(WavLM-large + ECAPA-TDNN). Built to run on an Apple Silicon Mac (M5, 16 GB) next to
other workloads, with a reference voice encoded once and reused for every sentence.

No PyTorch at runtime, no training code, no weight conversion: the backbone/codec weights
are the published MLX bundles `aufklarer/OmniVoice-MLX-int8` (default) and
`aufklarer/OmniVoice-MLX-fp16`.

## Highlights

Compared to the upstream Swift code it was extracted from (soniqo/speech-swift
`Sources/OmniVoiceTTS`, commit `468db5f`) this package:

- **Encodes the reference once.** `encodeReference` returns a `Codable` `VoiceReference`
  (codec tokens + transcript + loudness); every `generate` call reuses it.
- **Seedable sampling.** The reference algorithm samples positions with Gumbel noise
  (`position_temperature = 5`) and optionally tokens (`class_temperature`); upstream Swift
  was greedy-only. Here both are ported and driven by an MLX PRNG key derived from a
  `UInt64` seed; `seed: nil` draws a seed and reports it so any run can be replayed.
- **Python-faithful pre/post-processing.** Reference RMS normalisation, pydub-style silence
  removal, hop clipping, `add_punctuation`, output silence removal / loudness match /
  linear fade + pad — ported from `omnivoice` and checked against it (see Parity).
  Upstream Swift used its own energy-threshold trim and raised-cosine fades instead.
- **torchaudio-compatible resampling** everywhere audio is loaded (reference at 24 kHz,
  scorer at 16 kHz), so the Swift inputs match the Python ones.
- **Parameters, not constants.** Steps, guidance, t-shift, layer penalty, temperatures,
  duration/speed, padding/fade, compute dtype and MLX cache limit are all options.
- **Errors surface.** Unknown languages throw (Python silently falls back to no language),
  missing bundle files throw, invalid options throw.
- **Only what OmniVoice needs** from speech-swift: no AudioCommon/MLXCommon, no other models.

## Layout

| Target | What |
|---|---|
| `OmniVoiceMLX` | TTS: Qwen3 bidirectional diffusion backbone, Higgs-audio v2 codec encoder + decoder, Qwen2 BPE front-end, duration estimator, public API |
| `OmniVoiceSim` | SIM-o scorer: WavLM-large + ECAPA-TDNN on MLX |
| `OmniVoiceAudio` | Audio I/O, torchaudio resampler port, pydub silence-removal port, text prep |
| `omnivoice-bench` | Benchmark / parity / synthesis CLI |
| `scripts/` | Python (reference env) parity dumpers, scorer weight export, language-map generator |

## Usage

```swift
import OmniVoiceMLX

// Downloads aufklarer/OmniVoice-MLX-int8 into ~/Library/Caches/qwen3-speech on first use.
let tts = try await OmniVoiceTTS.load(.int8)            // or .fp16, or OmniVoiceTTS(directory:)

// Encode the voice once (any AVFoundation-readable file; resampled to 24 kHz).
let voice = try tts.encodeReference(
    audio: URL(fileURLWithPath: "reference.wav"),
    transcript: "<exactly what is said in reference.wav>")
try JSONEncoder().encode(voice).write(to: cacheURL)    // optional: persist, skip the encoder next time

// Synthesize: 24 kHz mono Float samples.
let pcm = try tts.generate(text: "好，我看一下。", language: "zh", reference: voice,
                           steps: 16, guidance: 2.0, seed: 42)

// Full control + generated tokens + seed actually used:
var o = GenerationOptions()
o.diffusion.steps = 16
o.diffusion.positionTemperature = 5      // 0 + classTemperature 0 = fully greedy
o.seed = nil                             // fresh seed, reported in result.seed
let result = try tts.synthesize(text: "Hello!", language: "en", reference: voice, options: o)
```

Speaker similarity:

```swift
import OmniVoiceSim

let scorer = try SpeakerSimilarityScorer(weightsURL: weights)   // from scripts/export_sim_weights.py
let refEmb = try scorer.embed(contentsOf: URL(fileURLWithPath: "speaker_ref2.wav"))
let sim = try scorer.similarity(refEmb, pcm, sampleRate: 24000)
```

CLI:

```bash
swift build -c release
.build/release/omnivoice-bench bench --bundle int8 --steps 16 --profile
.build/release/omnivoice-bench synth --text "好，我看一下。" --lang zh --out /tmp/a.wav
```

Note: MLX's buffer cache is process-global and this library does not touch it. Set
`MLX.Memory.cacheLimit` in the host process (the bench uses 256 MB; see the cache table).

## Measured on Apple M5, 16 GB

macOS 27.0.1, mlx-swift 0.31.6, release build, `omnivoice-bench bench`, 20 lines from
`lines20.json` (14 zh, 6 en), reference `ref_plus_zh.wav` (19.1 s → 465 codec frames after
silence removal) encoded once, 1 warm-up sentence, seed 0, default sampling
(`position_temperature 5`), MLX cache limit 256 MB, f16 compute. Latency is per sentence
(text in → post-processed 24 kHz samples out). Peak = lifetime-max `phys_footprint`
of the process (Activity Monitor "Memory"; includes model weights, Metal buffers and
the MLX cache) measured before the scorer is loaded. SIM-o = mean similarity of the 20
outputs to the real recording `ref2.wav`, from the ported scorer. Measured
2026-11-13 with a `dollcore` daemon and other jobs running, so expect ±5–8 % run-to-run
noise (repeated runs of the same config ranged 1.30–1.46 s for int8/16).

| Bundle | Steps | Mean | Median | Max | Peak footprint | SIM-o |
|---|---|---|---|---|---|---|
| **int8 (default)** | **16** | **1.46 s** | **1.41 s** | **1.70 s** | **2.95 GB** | **0.557** |
| int8 | 32 | 3.02 s | 2.91 s | 3.43 s | 2.92 GB | 0.547 |
| fp16 | 16 | 1.63 s | 1.55 s | 1.94 s | 3.41 GB | 0.534 |
| fp16 | 32 | 3.22 s | 3.08 s | 3.81 s | 3.50 GB | 0.530 |
| int8, 8.5 s reference clip* | 16 | 0.95 s | 0.85 s | 1.15 s | 2.59 GB | 0.601 |
| fp16, f32 + `--no-tf32` (parity mode) | 16 | 4.62 s | 4.46 s | 5.20 s | 4.72 GB | 0.547 |

\* `/tmp/ovref/short_en_zh.wav`: the first two English sentences (0–6.42 s) plus the
trailing "好，我看一下。" (2.05 s) of the same reference, with the matching transcript.
This is a data choice, not a code change: it shortens the conditioning sequence
(~350 instead of ~600 positions). The SIM-o anchor contains that same audio, so the
higher score may partly reflect overlap; listen before adopting it.

Other one-off numbers (same setup): model load 0.35–0.8 s from the local cache
(warm disk); reference encode (19 s clip, incl. file read, resample, silence removal)
0.53–0.64 s; for comparison the upstream speech-swift code measured ≈2.4 s/sentence for
int8/16 and ≈4.6 GB peak earlier on this Mac, and PyTorch MPS ≈13 s/sentence.
Not measured: cold first load including the Hugging Face download; other Macs.

### Per-stage profile (int8, 16 steps)

`omnivoice-bench bench --profile`: each stage is forced to evaluate and timed, which
serialises GPU work (mean latency 1.48 s in this mode vs 1.46 s without).

| Stage | Per sentence | Share |
|---|---|---|
| Reference encode (once per voice, not per sentence) | 0.53 s | — |
| Input build (tokenize, duration estimate, ids) | 1.3 ms | 0.1 % |
| Diffusion cond forward (16 × ~600 positions) | 1.170 s | 79 % |
| Diffusion uncond forward (16 × T positions) | 0.221 s | 15 % |
| CFG + log-softmax + top-k unmask (16 steps) | 0.016 s | 1 % |
| Codec decode | 0.043 s | 3 % |
| Post-processing (silence removal, fade, pad) | 0.3 ms | 0 % |

The cost is the conditioned backbone pass, which is linear-ish in the
conditioning length, so the reference length is the main remaining lever.

### Optimization log

Rule: keep a change only if it is faster and parity is unchanged (token-exact for
exact-math changes; for numeric changes, step-0 logit SNR vs PyTorch fp32 and SIM-o
must not get worse). Each A/B ran the 20-line int8/16 bench back to back, alternating.

| Attempt | Before → after (mean) | Parity / quality | Decision |
|---|---|---|---|
| Baseline = speech-swift path (int8 weights, fp32 activations, TF32 matmuls, 1 GB cache) | 1.94 / 2.07 s, peak 3.7–3.8 GB | SIM 0.548 | — |
| Audio head only on the T target positions (skip 8200-wide head on prefix) | 1.94 / 2.11 → 2.05 / 2.10 s | exact | **rejected** (inside noise) |
| **f16 backbone compute** (scales/norms/activations f16) | 1.94 / 2.07 → 1.37 / 1.37 s | step-0 SNR 58.4 → 60.6 dB; SIM 0.548 → 0.557 | **kept** (default) |
| bf16 backbone compute | → 1.38 s | step-0 SNR 50.4 dB; SIM 0.482 | **rejected** (quality) |
| `mx.compile` of the whole forward (f16) | 1.30 / 1.36 → 1.30 / 1.35 s | exact | **rejected** (no gain; adds a per-shape compile cache) |
| MLX cache limit 1 GB → **256 MB** | 1.30–1.36 → 1.37 s; peak 3.7–3.9 → 3.11 GB | exact | **kept** (bench default) |
| MLX cache limit 0 | → 1.48 s; peak 2.45 GB | exact | available, slower |
| No cache limit (MLX default) | → 2.22 s (max 4.3 s); peak 11.4 GB | exact | **never use** on 16 GB |
| Disable TF32 (`--no-tf32`, f32) | → 4.41 s | token-exact | parity mode only |
| Shorter reference clip (8.5 s) | 1.46 → 0.95 s | SIM 0.557 → 0.601 | option (data choice), not default |
| Batch cond + uncond in one forward | not tried | — | the uncond row (T ≈ 40–100) would have to be padded to the cond length (~600), ~doubling work; the reference itself masks the padding |

## Parity with the official PyTorch implementation

Reference: `omnivoice` from PyPI (torch 2.8) loading `k2-fsa/OmniVoice` in fp32 with all
weights rounded to fp16 (`--round-fp16`, identical bit-for-bit to the fp16 MLX bundle —
checked), deterministic decoding (`position_temperature=0`, `class_temperature=0`),
16 steps, lines 0, 1, 3 (zh), 14, 15 (en). PyTorch CPU and MPS agree with each other
token-for-token and to 136 dB on step-0 logits.

| Stage | Swift (fp16 bundle, f32, `--no-tf32`) | Swift default numerics (f32 + TF32) |
|---|---|---|
| Reference preprocessing (RMS, silence removal, hop clip, punctuation) | identical length, RMS, transcript | same |
| Reference codec tokens (465 × 8) | 99.78 % equal | 98.71 % |
| Conditioning ids + audio mask | identical | identical |
| Step-0 logits (cond / uncond) | 131–136 dB SNR, argmax 100 % | 67–69 dB, argmax 98.9–99.6 % |
| **Generated tokens, 5 lines (1944 tokens)** | **100.00 % exact** | 83.6 % (2 lines exact, 3 diverge after a near-tie) |
| Codec decode of identical tokens | corr 1.00000, 70–77 dB SNR | corr 1.00000, 51–54 dB |
| Post-processing (pydub port, 288 cases) | max abs diff 0.0, equal lengths | — |

The one real divergence found: on M5 GPUs MLX sends float32 matmuls through the
neural accelerators in TF32 (`MLX_ENABLE_TF32` defaults to 1), which is ~62 dB SNR
against float64 instead of ~125 dB (`omnivoice-bench matmul-check`). Greedy unmasking
amplifies that into different token choices. With `MLX_ENABLE_TF32=0` (bench:
`--no-tf32`) the port is token-exact. Residual drift in the reference codec tokens
(0.2 %) is float32 noise in the codec encoder that flips residual-VQ near-ties.

For the production numerics the cascade makes token match a poor metric (one flip
changes everything after it), so step-0 logits are reported instead
(`parity --dump /tmp/ovparity`, lines 1, 3, 14, mean, argmax agreement in brackets):
fp16 bundle f16 71.7 dB (95 %); int8 f32 58.0 dB (86 %); **int8 f16 60.6 dB (85 %)**;
int8 bf16 50.5 dB (61 %). Int8 weight quantization dominates the deviation.

Reproduce:

```bash
PY=/tmp/omnivoice/.venv/bin/python
$PY scripts/tts_parity.py --out /tmp/ovparity --round-fp16 --device mps          # ~2 min
swift build -c release
.build/release/omnivoice-bench parity --dump /tmp/ovparity --bundle fp16 --dtype f32 --no-tf32
.build/release/omnivoice-bench parity --dump /tmp/ovparity --bundle int8 --dtype f16
$PY scripts/postproc_parity.py --out /tmp/postproc_parity --synthetic 12 \
    /tmp/omnivoice/ref_plus_zh.wav /tmp/omnivoice/speaker_ref2.wav /tmp/speaker-mlx2/fp16_s16_{0,1,7,13}.wav
OMNIVOICE_POSTPROC_PARITY_DIR=/tmp/postproc_parity swift test --filter PostprocParityTests
```

## Speaker-similarity scorer (SIM-o) parity

Port of `omnivoice/eval/speaker_similarity` (WavLM-large, 24 layers with gated relative
position bias, weighted layer sum, ECAPA-TDNN, 256-d embedding) loading the
`k2-fsa/TTS_eval_models` checkpoint exported to safetensors (fp32, 1.3 GB, not committed).

| Check | Result |
|---|---|
| 80 files `/tmp/speaker-mlx2/*.wav` vs Python with 6-decimal scores | max \|diff\| 0.00066, mean 0.00017 |
| Same 80 files vs the 3-decimal `sim.txt` (includes ±0.0005 rounding) | max \|diff\| 0.00103, mean 0.00031 |
| Reference embedding (`ref2.wav`) | cosine 0.9999977 to Python |
| Tolerance used | \|diff\| ≤ 0.01 (all pass) |
| Speed | 0.054 s per 0.4–3.7 s file, 0.44 s for the 17 s anchor, ~0.08 s for 5 s audio (warm) |
| Memory | 1.60 GB peak footprint for the scorer process (weights 1.3 GB fp32) |

```bash
$PY scripts/export_sim_weights.py          # -> ~/.cache/omnivoice-mlx/sim/wavlm_ecapa.safetensors
.build/release/omnivoice-bench sim-parity --list /tmp/speaker-mlx2/sim.txt
# 6-decimal Python scores: scripts/sim_parity.py python|compare (see --help)
```

## Tests

`swift test`: schedule/language/error-path unit tests; reference reuse + `Codable`
round trip; seed reproducibility (same seed ⇒ identical tokens, reported seed replays an
unseeded run); greedy determinism; pydub/torchaudio ports on synthetic signals and a
committed torchaudio fixture; scorer parity on two committed fixtures. Model- and
weight-backed tests skip (XCTSkip) when the cached int8 bundle, `/tmp/omnivoice/ref_plus_zh.wav`
or the scorer weights are absent.

## Open issues

- `instruct` strings are passed through verbatim; Python's `_resolve_instruct`
  normalisation/validation is not ported.
- Long-text chunking (`audio_chunk_duration` / `_generate_chunked`, > 30 s of output) and
  ASR auto-transcription / `trim_long_audio` (only used without a transcript) are not ported.
- `normalize_text` (WeTextProcessing) is not ported.
- Unknown languages throw; Python silently falls back to language-agnostic mode.
- Production numerics (int8 + f16 + TF32) are not token-identical to PyTorch; see Parity.

## Provenance and license

Apache-2.0 (see `LICENSE`, `NOTICE`). Extracted from soniqo/speech-swift `468db5f`
(`Sources/OmniVoiceTTS`), algorithm and evaluation code from k2-fsa/OmniVoice. Weights are
downloaded separately: `aufklarer/OmniVoice-MLX-int8` / `-fp16`, `k2-fsa/TTS_eval_models`.
