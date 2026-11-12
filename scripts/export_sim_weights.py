#!/usr/bin/env python3
"""Export the OmniVoice SIM-o speaker-similarity model (WavLM-large + ECAPA-TDNN)
to a single safetensors file for the Swift `OmniVoiceSim` port.

The model is built exactly like OmniVoice's evaluation code: the s3prl WavLM-large
upstream is loaded from the k2-fsa/TTS_eval_models snapshot, then the speaker
fine-tune checkpoint is applied with `strict=False`. The exported tensors are the
in-memory eval-mode `state_dict()` *after* loading, so any parameter the fine-tune
checkpoint does not cover keeps exactly the value the Python scorer uses.

Usage (from any directory):
    /tmp/omnivoice/.venv/bin/python scripts/export_sim_weights.py \
        [--output ~/.cache/omnivoice-mlx/sim/wavlm_ecapa.safetensors] [--dtype float32]
"""

import argparse
import logging
import os
import sys

import torch


def build_model():
    """Build the eval-mode ECAPA_TDNN_WAVLM scorer exactly as OmniVoice does."""
    from huggingface_hub import snapshot_download
    from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

    root = snapshot_download(
        "k2-fsa/TTS_eval_models", allow_patterns=["speaker_similarity/*"]
    )
    sim_dir = os.path.join(root, "speaker_similarity")
    logging.disable(logging.INFO)
    model = ECAPA_TDNN_WAVLM(
        feat_dim=1024,
        channels=512,
        emb_dim=256,
        sr=16000,
        ssl_model_path=os.path.join(sim_dir, "wavlm_large") + os.sep,
    )
    state = torch.load(
        os.path.join(sim_dir, "wavlm_large_finetune.pth"), map_location="cpu"
    )
    result = model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/.cache/omnivoice-mlx/sim/wavlm_ecapa.safetensors"),
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    from safetensors.torch import save_file

    model, result = build_model()
    print(f"missing keys (kept at in-memory value): {len(result.missing_keys)}")
    for k in result.missing_keys:
        print(f"  missing: {k}")
    print(f"unexpected keys (ignored): {len(result.unexpected_keys)}")
    for k in result.unexpected_keys:
        print(f"  unexpected: {k}")

    dtype = getattr(torch, args.dtype)
    tensors = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu()
        if t.is_floating_point():
            t = t.to(dtype)
        tensors[name] = t.contiguous()

    out = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"format": "pt", "model": "ecapa_tdnn_wavlm_large"})
    size = os.path.getsize(out)
    print(f"wrote {len(tensors)} tensors, {size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    sys.exit(main())
