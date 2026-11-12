#!/usr/bin/env python3
"""Write Python reference outputs for the Swift port of OmniVoice audio/text pre/post-processing.

Runs the real OmniVoice code paths (``remove_silence``, ``fade_and_pad_audio``,
``OmniVoice.create_voice_clone_prompt`` and ``OmniVoice._post_process_audio`` with a stub
audio tokenizer, ``add_punctuation``) and writes raw little-endian float32 files plus a
``manifest.json`` into ``--out``. Inputs at other rates are resampled to 24 kHz with
``torchaudio.functional.resample`` (the original is saved too, so the Swift side also reports
``Resampler`` parity). ``--synthetic N`` adds N deterministic burst/gap stress signals at
24/44.1/22.05/16 kHz. The Swift side (``PostprocParityTests``) recomputes every case.

Usage:
    /tmp/omnivoice/.venv/bin/python scripts/postproc_parity.py --out /tmp/postproc_parity \
        --synthetic 12 /tmp/omnivoice/ref_plus_zh.wav /tmp/omnivoice/ref2.wav \
        /tmp/voice-mlx2/fp16_s16_0.wav /tmp/voice-mlx2/fp16_s16_1.wav \
        /tmp/voice-mlx2/fp16_s16_7.wav /tmp/voice-mlx2/fp16_s16_13.wav
    OMNIVOICE_POSTPROC_PARITY_DIR=/tmp/postproc_parity \
        swift test -c release --filter PostprocParityTests
"""

import argparse
import json
import os
import types

import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.audio import fade_and_pad_audio, remove_silence
from omnivoice.utils.text import add_punctuation

SR = 24000
HOP = 960

PUNCT_CASES = [
    "hello world",
    "  hello world.  ",
    "你好世界",
    "你好世界。",
    "mixed 中文 text",
    "ends with quote\u201d",
    "ends with paren）",
    "trailing ellipsis\u2026",
    "\u3000全角空白\u3000",
    "tab\t\n",
    "",
    "   ",
    "x\u001f",
    "a\u200b",
    "Ünïcödé",
    "數字123",
    "comma,",
    "question？",
]


class _StubTokenizer:
    """Captures the waveform handed to ``audio_tokenizer.encode``."""

    def __init__(self):
        self.config = types.SimpleNamespace(hop_length=HOP)
        self.device = "cpu"
        self.captured = None

    def encode(self, wav):
        self.captured = wav.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return types.SimpleNamespace(audio_codes=torch.zeros(1, 8, 1, dtype=torch.long))


def _fake_model(rate):
    return types.SimpleNamespace(audio_tokenizer=_StubTokenizer(), sampling_rate=rate, _asr_pipe=None)


def _synthetic(seed):
    """Tone bursts separated by quiet gaps of random length; odd lengths and several rates
    exercise pydub's ms rounding, end padding, seek-step tail windows and keep-silence overlap."""
    rng = np.random.default_rng(seed)
    rate = [24000, 44100, 22050, 16000][seed % 4]
    parts = []
    for _ in range(int(rng.integers(3, 9))):
        gap = int(rng.integers(0, int(0.9 * rate)))
        level = [0.0, 0.001, 0.003, 0.0031][int(rng.integers(0, 4))]  # around the -50 dBFS line
        parts.append(rng.standard_normal(gap) * level)
        n = int(rng.integers(1, int(0.6 * rate)))
        t = np.arange(n) / rate
        amp = float(rng.uniform(0.002, 0.6))
        parts.append(amp * np.sin(2 * np.pi * float(rng.uniform(80, 3000)) * t))
    parts.append(rng.standard_normal(int(rng.integers(0, rate))) * 0.001)
    x = np.concatenate(parts).astype(np.float32)[np.newaxis, :]
    return f"synthetic{seed}", rate, x


def _save(out_dir, name, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f4").reshape(-1))
    path = os.path.join(out_dir, name + ".f32")
    arr.tofile(path)
    return {"file": os.path.basename(path), "length": int(arr.shape[0])}


def write_resampler_fixture(path):
    """Regenerate Tests/OmniVoiceAudioTests/Fixtures/resampler_torchaudio.json."""
    cases = []
    for orig, new, n in [(44100, 24000, 441), (16000, 24000, 160), (24000, 16000, 240)]:
        i = np.arange(n)
        x = (0.5 * np.sin(2 * np.pi * 440 * i / orig) + 0.25 * np.sin(2 * np.pi * 3100.5 * i / orig)
             + 0.05 * ((i * 7919) % 101 - 50) / 50).astype(np.float32)
        y = torchaudio.functional.resample(torch.from_numpy(x), orig_freq=orig, new_freq=new).numpy()
        cases.append({"orig_freq": orig, "new_freq": new,
                      "input": [float(v) for v in x], "output": [float(v) for v in y]})
    with open(path, "w") as f:
        json.dump({"generator": "torchaudio.functional.resample defaults, torch "
                   + torch.__version__ + " CPU, float32", "cases": cases}, f)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/postproc_parity")
    ap.add_argument("--synthetic", type=int, default=8,
                    help="number of synthetic burst/gap stress signals to add")
    ap.add_argument("--resampler-fixture", default=None,
                    help="also (re)write the committed resampler JSON fixture to this path")
    ap.add_argument("inputs", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.resampler_fixture:
        write_resampler_fixture(args.resampler_fixture)

    gen_cfg = OmniVoiceGenerationConfig()
    manifest = {
        "sample_rate": SR,
        "hop_length": HOP,
        "gen_config": {
            "postprocess_output": gen_cfg.postprocess_output,
            "pad_duration": gen_cfg.pad_duration,
            "fade_duration": gen_cfg.fade_duration,
        },
        "inputs": [],
        "punctuation": [{"input": s, "output": add_punctuation(s)} for s in PUNCT_CASES],
    }

    sources = [("file", p) for p in args.inputs] + [("synthetic", i) for i in range(args.synthetic)]
    for source_kind, source in sources:
        if source_kind == "file":
            path = source
            stem = os.path.splitext(os.path.basename(path))[0]
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            data = data.T
            if data.shape[0] > 1:
                data = np.mean(data, axis=0, keepdims=True)
            entry = {"name": stem, "path": path, "orig_sample_rate": int(sr), "rate": SR, "cases": []}
            if sr != SR:
                entry["orig"] = _save(args.out, f"{stem}.orig", data)
                data = torchaudio.functional.resample(
                    torch.from_numpy(data), orig_freq=sr, new_freq=SR
                ).numpy()
            rate = SR
        else:
            stem, rate, data = _synthetic(source)
            sr = rate
            entry = {"name": stem, "path": None, "orig_sample_rate": rate, "rate": rate, "cases": []}
        x = data.astype(np.float32)  # (1, T) at `rate`
        entry["input"] = _save(args.out, f"{stem}.in", x)
        entry["rms"] = float(np.sqrt(np.mean(x**2)))

        def case(kind, params, y):
            entry["cases"].append({"kind": kind, "params": params, **_save(args.out, f"{stem}.{kind}.{len(entry['cases'])}", y)})

        for mid, lead, trail in [(200, 100, 200), (500, 100, 100), (0, 100, 300)]:
            case("remove_silence", {"mid": mid, "lead": lead, "trail": trail},
                 remove_silence(x, rate, mid_sil=mid, lead_sil=lead, trail_sil=trail))
        for pad, fade in [(0.1, 0.1), (0.0, 0.05), (0.25, 0.0)]:
            case("fade_and_pad", {"pad": pad, "fade": fade},
                 fade_and_pad_audio(x, pad_duration=pad, fade_duration=fade, sample_rate=rate))

        for preprocess in (True, False):
            model = _fake_model(rate)
            try:
                prompt = OmniVoice.create_voice_clone_prompt(
                    model, (torch.from_numpy(x[0].copy()), rate), ref_text="stub", preprocess_prompt=preprocess
                )
                case("prepare_reference", {"preprocess": preprocess, "ref_rms": prompt.ref_rms},
                     model.audio_tokenizer.captured)
            except ValueError as e:
                entry["cases"].append({"kind": "prepare_reference", "params": {"preprocess": preprocess}, "error": str(e)})

        for ref_rms in (entry["rms"], 0.05, 0.3, None):
            for post in (True, False):
                cfg = OmniVoiceGenerationConfig(postprocess_output=post)
                model = _fake_model(rate)
                y = OmniVoice._post_process_audio(model, x.copy(), ref_rms=ref_rms, gen_config=cfg)
                case("postprocess", {"ref_rms": ref_rms, "remove_silence": post,
                                     "pad": cfg.pad_duration, "fade": cfg.fade_duration}, y)
        manifest["inputs"].append(entry)
        print(f"{stem}: sr={sr} in_len={x.shape[-1]} rms={entry['rms']:.6f} cases={len(entry['cases'])}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("wrote", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
