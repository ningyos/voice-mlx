#!/usr/bin/env python3
"""Greedy reference dumps from the official `qwen-tts` package for voice-mlx's parity check.

Runs Qwen3-TTS-12Hz-0.6B (PyTorch, float32) with deterministic decoding
(do_sample=False, subtalker_dosample=False; repetition penalty 1.05 still applies, as in
HF generate) and writes, per line: prompt token ids, prefill (step-0) logits, generated
codec tokens and the decoded waveform. `voice-bench parity --dump DIR` compares them.

  base:   ICL voice cloning (`generate_voice_clone`, non_streaming_mode=False) with a
          reference; the reference is resampled to 24 kHz once here (torchaudio) and saved
          as ref24k.f32 so both sides see identical samples.
  custom: CustomVoice speaker path (`generate_custom_voice`, non_streaming_mode=True).

--mlx-weights DIR replaces the talker / code-predictor / speaker-encoder weights with
the tensors of an MLX bundle (8-bit tensors dequantized in float32), so PyTorch runs the
exact weights the Swift port loads.

    PY=.venv/bin/python
    $PY scripts/qwen_parity.py --kind base --out /tmp/vmlx/parity_base \
        --mlx-weights ~/Library/Caches/qwen3-speech/models/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit
    $PY scripts/qwen_parity.py --kind custom --speaker ryan --out /tmp/vmlx/parity_custom
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import load_file

from qwen_tts import Qwen3TTSModel

LANG = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german",
        "fr": "french", "es": "spanish", "it": "italian", "pt": "portuguese", "ru": "russian"}
REPOS = {"base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "custom": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}


def snapshot(repo):
    d = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))
    if not d:
        raise SystemExit(f"{repo} not in the HF cache; run: hf download {repo}")
    return d[0]


def dequant(w, scales, biases, group=64):
    """MLX affine 8-bit: uint32 [out, in/4] little-endian bytes -> float32 [out, in]."""
    w = w.to(torch.int64)
    q = torch.stack([(w >> (8 * j)) & 0xFF for j in range(4)], dim=-1).reshape(w.shape[0], -1).float()
    s = scales.float().repeat_interleave(group, dim=1)
    b = biases.float().repeat_interleave(group, dim=1)
    return q * s + b


def load_mlx_weights(model, bundle):
    files = sorted(glob.glob(os.path.join(bundle, "*.safetensors")))
    t = {}
    for f in files:
        t.update(load_file(f))
    sd = model.model.state_dict()
    new, used = {}, set()
    for k in sd:
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and f"{base}.scales" in t:
            new[k] = dequant(t[k], t[f"{base}.scales"], t[f"{base}.biases"])
            used |= {k, f"{base}.scales", f"{base}.biases"}
        elif k in t:
            v = t[k].float()
            if k.startswith("speaker_encoder.") and v.ndim == 3 and tuple(v.shape) != tuple(sd[k].shape):
                v = v.transpose(1, 2)  # MLX conv [out, k, in] -> torch [out, in, k]
            new[k] = v
            used.add(k)
    missing = [k for k in sd if k not in new and (k.startswith("talker.") or k.startswith("speaker_encoder."))]
    if missing:
        raise SystemExit(f"MLX bundle lacks {len(missing)} tensors, e.g. {missing[:3]}")
    for k, v in new.items():
        if tuple(v.shape) != tuple(sd[k].shape):
            raise SystemExit(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}")
    model.model.load_state_dict(new, strict=False)
    print(f"loaded {len(new)} tensors from {bundle} ({len(set(t) - used)} unused)")


def dump(path, arr, dtype):
    np.ascontiguousarray(np.asarray(arr), dtype=dtype).tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["base", "custom"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lines", default="/tmp/omnivoice/lines20.json")
    ap.add_argument("--indices", default="1,3,10,14,15", help="line indices to run")
    ap.add_argument("--ref", default="/tmp/omnivoice/ref_plus_zh.wav")
    ap.add_argument("--ref-text", default=None, help="default: ref2.txt + ' 好，我看一下。'")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--mlx-weights", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    tts = Qwen3TTSModel.from_pretrained(snapshot(REPOS[a.kind]), dtype=torch.float32, device_map=a.device)
    tts.model.eval()
    if a.mlx_weights:
        load_mlx_weights(tts, os.path.expanduser(a.mlx_weights))
    m = tts.model

    logits_log = []
    m.talker.codec_head.register_forward_hook(lambda mod, inp, out: logits_log.append(out.detach()))

    lines = json.load(open(a.lines))
    idx = [int(i) for i in a.indices.split(",")]
    meta = {"kind": a.kind, "mlx_weights": a.mlx_weights, "speaker": a.speaker, "lines": []}
    gen = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=a.max_new_tokens)

    prompt = None
    if a.kind == "base":
        wav, sr = sf.read(a.ref, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav24 = torchaudio.functional.resample(torch.from_numpy(wav), sr, 24000).numpy().astype(np.float32)
        dump(f"{a.out}/ref24k.f32", wav24, np.float32)
        ref_text = a.ref_text or (open("/tmp/omnivoice/ref2.txt").read().strip() + " 好，我看一下。")
        t0 = time.time()
        prompt = tts.create_voice_clone_prompt(ref_audio=(wav24, 24000), ref_text=ref_text)
        code = prompt[0].ref_code.cpu().numpy().astype(np.int32)  # [T, 16]
        dump(f"{a.out}/ref_codes.i32", code, np.int32)
        dump(f"{a.out}/ref_spk.f32", prompt[0].ref_spk_embedding.float().cpu().numpy(), np.float32)
        meta["ref"] = {"frames": int(code.shape[0]), "text": ref_text, "samples": int(len(wav24)),
                       "ref_ids": tts._tokenize_texts([tts._build_ref_text(ref_text)])[0][0].tolist(),
                       "seconds": time.time() - t0}

    for i in idx:
        text, lang = lines[i][0], lines[i][1]
        language = LANG.get(lang, lang)
        logits_log.clear()
        ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
        t0 = time.time()
        with torch.no_grad():
            if a.kind == "base":
                ref_ids = [tts._tokenize_texts([tts._build_ref_text(prompt[0].ref_text)])[0]]
                codes, _ = m.generate(input_ids=[ids], ref_ids=ref_ids,
                                      voice_clone_prompt=tts._prompt_items_to_voice_clone_prompt(prompt),
                                      languages=[language], non_streaming_mode=False, **gen)
                codes = codes[0]
                full = torch.cat([prompt[0].ref_code.to(codes.device), codes], dim=0)
                wavs, fs = m.speech_tokenizer.decode([{"audio_codes": full}])
                w = wavs[0]
                cut = int(prompt[0].ref_code.shape[0] / max(full.shape[0], 1) * w.shape[0])
                w = w[cut:]
            else:
                codes, _ = m.generate(input_ids=[ids], instruct_ids=[None], languages=[language],
                                      speakers=[a.speaker], non_streaming_mode=True, **gen)
                codes = codes[0]
                wavs, fs = m.speech_tokenizer.decode([{"audio_codes": codes}])
                w = wavs[0]
        dt = time.time() - t0
        c = codes.cpu().numpy().astype(np.int32)
        dump(f"{a.out}/line{i}_codes.i32", c, np.int32)
        dump(f"{a.out}/line{i}_wav.f32", np.asarray(w, dtype=np.float32), np.float32)
        step0 = logits_log[0][0, -1].float().cpu().numpy()
        dump(f"{a.out}/line{i}_step0.f32", step0, np.float32)
        meta["lines"].append({"index": i, "text": text, "lang": lang, "input_ids": ids[0].tolist(),
                              "frames": int(c.shape[0]), "prefill_len": int(logits_log[0].shape[1]),
                              "seconds": dt})
        print(f"[{i}] {lang} frames={c.shape[0]} prefill={logits_log[0].shape[1]} {dt:.1f}s  {text}")
    json.dump(meta, open(f"{a.out}/parity.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
