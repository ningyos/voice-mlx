"""Dump reference (official PyTorch OmniVoice) intermediates for TTS parity.

Writes <out>/parity.json plus raw little-endian float32 files:
  ref_wav24.f32         preprocessed 24 kHz reference waveform (what the codec encodes)
  line<i>_raw.f32       codec decode of the generated tokens (before post-processing)
  line<i>_final.f32     fully post-processed output

Decoding is deterministic (position_temperature=0, class_temperature=0) so
token ids are directly comparable with the Swift port
(`omnivoice-bench parity --dump <out>`).

Usage (reference env, CPU):
  /tmp/omnivoice/.venv/bin/python scripts/tts_parity.py --out /tmp/ovparity \
      --lines 0,1,3,14,15 --steps 16 --round-fp16
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
from omnivoice.utils.audio import load_audio, remove_silence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-audio", default="/tmp/omnivoice/ref_plus_zh.wav")
    ap.add_argument("--ref-text-file", default="/tmp/omnivoice/ref2.txt")
    ap.add_argument("--ref-text-suffix", default=" 好，我看一下。")
    ap.add_argument("--lines-json", default="/tmp/omnivoice/lines20.json")
    ap.add_argument("--lines", default="0,1,3,14,15", help="comma-separated indices")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--round-fp16",
        action="store_true",
        help="round all weights to fp16 (then compute in fp32) to match the fp16 MLX bundle",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    logging.disable(logging.WARNING)

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float32)
    if args.round_fp16:
        with torch.no_grad():
            for m in (model, model.audio_tokenizer):
                for p in m.parameters():
                    p.copy_(p.half().float())
    model.eval()
    sr = model.sampling_rate
    ref_text = open(args.ref_text_file, encoding="utf-8").read().strip() + args.ref_text_suffix

    # Reference prompt via the public API (preprocess_prompt=True).
    prompt = model.create_voice_clone_prompt(ref_audio=args.ref_audio, ref_text=ref_text)
    # Re-derive the exact waveform the codec saw (same steps as create_voice_clone_prompt).
    wav = load_audio(args.ref_audio, sr)
    rms = float(np.sqrt(np.mean(wav**2)))
    if 0 < rms < 0.1:
        wav = wav * 0.1 / rms
    wav = remove_silence(wav, sr, mid_sil=200, lead_sil=100, trail_sil=200)
    hop = model.audio_tokenizer.config.hop_length
    clip = wav.shape[-1] % hop
    if clip:
        wav = wav[:, :-clip]
    wav.astype("<f4").tofile(os.path.join(args.out, "ref_wav24.f32"))

    gen_config = OmniVoiceGenerationConfig(
        num_step=args.steps, position_temperature=0.0, class_temperature=0.0
    )
    lines = json.load(open(args.lines_json, encoding="utf-8"))
    out = {
        "steps": args.steps,
        "round_fp16": args.round_fp16,
        "ref": {
            "text": prompt.ref_text,
            "rms": prompt.ref_rms,
            "codes": prompt.ref_audio_tokens.cpu().tolist(),
            "wav24_samples": int(wav.shape[-1]),
        },
        "lines": [],
    }
    for i in [int(x) for x in args.lines.split(",")]:
        text, lang = lines[i]
        t0 = time.time()
        task = model._preprocess_all(text=text, language=lang, voice_clone_prompt=prompt)
        inputs = model._prepare_inference_inputs(
            task.texts[0], task.target_lens[0], task.ref_texts[0], task.ref_audio_tokens[0],
            task.langs[0], task.instructs[0], gen_config.denoise,
        )
        with torch.inference_mode():
            tokens = model._generate_iterative(task, gen_config)[0]
            dev = model.audio_tokenizer.device
            raw = model.audio_tokenizer.decode(tokens.to(dev).unsqueeze(0)).audio_values[0].cpu().numpy()
            final = model._decode_and_post_process(tokens, task.ref_rms[0], gen_config)
        raw.reshape(-1).astype("<f4").tofile(os.path.join(args.out, f"line{i}_raw.f32"))
        final.reshape(-1).astype("<f4").tofile(os.path.join(args.out, f"line{i}_final.f32"))
        out["lines"].append({
            "index": i,
            "text": text,
            "lang": lang,
            "target_len": task.target_lens[0],
            "input_ids": inputs["input_ids"][0].cpu().tolist(),
            "audio_mask": inputs["audio_mask"][0].int().cpu().tolist(),
            "codes": tokens.cpu().tolist(),
        })
        print(f"line {i}: T={task.target_lens[0]} {time.time() - t0:.1f}s", flush=True)
    with open(os.path.join(args.out, "parity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {args.out}/parity.json")


if __name__ == "__main__":
    main()
