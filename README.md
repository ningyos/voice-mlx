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

SPEAKER_AND_MEASUREMENTS

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
