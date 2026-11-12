#!/usr/bin/env python3
"""SIM-o parity between OmniVoice's PyTorch scorer and the Swift MLX port.

Subcommands
-----------
python    Score audio files against a reference with the PyTorch model (6 decimals).
          Writes `path score` lines (usable directly as the Swift parity list) and the
          reference embedding as JSON.
compare   Compare Python and Swift score files / reference embeddings.
fixtures  Write the small JSON fixture used by Tests/OmniVoiceSimTests.

Typical run (from the repo root):
    PY=/tmp/omnivoice/.venv/bin/python
    $PY scripts/sim_parity.py python --ref /tmp/omnivoice/ref2.wav \
        --glob '/tmp/voice-mlx2/*.wav' --out /tmp/sim_py.txt --emb-out /tmp/sim_py_ref.json
    omnivoice-bench sim-parity --weights ~/.cache/omnivoice-mlx/sim/wavlm_ecapa.safetensors \
        --ref /tmp/omnivoice/ref2.wav --list /tmp/sim_py.txt \
        --scores-out /tmp/sim_swift.txt --emb-out /tmp/sim_swift_ref.json
    $PY scripts/sim_parity.py compare --py /tmp/sim_py.txt --swift /tmp/sim_swift.txt \
        --py-emb /tmp/sim_py_ref.json --swift-emb /tmp/sim_swift_ref.json
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _scorer():
    import torch
    from omnivoice.eval.utils import load_eval_waveform

    from export_sim_weights import build_model

    model, _ = build_model()

    def embed(path):
        wav = load_eval_waveform(path, 16000, device="cpu", max_seconds=120)
        with torch.inference_mode():
            return model([wav])

    def cosine(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()

    return embed, cosine


def cmd_python(args):
    embed, cosine = _scorer()
    ref = embed(args.ref)
    paths = sorted(p for pattern in args.glob for p in glob.glob(pattern))
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(args.ref)]
    lines = []
    for path in paths:
        score = cosine(ref, embed(path))
        lines.append(f"{os.path.abspath(path)} {score:.6f}")
        print(lines[-1], flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if args.emb_out:
        with open(args.emb_out, "w") as f:
            json.dump([float(x) for x in ref[0].tolist()], f)
    print(f"wrote {len(lines)} scores -> {args.out}")


def _read_scores(path):
    scores = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, s = line.rsplit(None, 1)
            scores[os.path.basename(p)] = float(s)
    return scores


def cmd_compare(args):
    py = _read_scores(args.py)
    sw = _read_scores(args.swift)
    missing = sorted(set(py) ^ set(sw))
    if missing:
        print(f"files present in only one side: {missing}")
        return 1
    diffs = {k: abs(py[k] - sw[k]) for k in py}
    worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
    mx = max(diffs.values())
    mean = sum(diffs.values()) / len(diffs)
    print(f"files={len(diffs)} max|diff|={mx:.6f} mean|diff|={mean:.6f}")
    for k, d in worst:
        print(f"  {k}: python={py[k]:.6f} swift={sw[k]:.6f} diff={d:.6f}")
    if args.py_emb and args.swift_emb:
        with open(args.py_emb) as f:
            a = json.load(f)
        with open(args.swift_emb) as f:
            b = json.load(f)
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        maxabs = max(abs(x - y) for x, y in zip(a, b))
        print(
            f"ref embedding: cosine(swift, python)={dot / (na * nb):.8f} "
            f"max|diff|={maxabs:.6f} max|python|={max(abs(x) for x in a):.4f}"
        )
    failed = [k for k, d in diffs.items() if d > args.tol]
    if failed:
        print(f"FAIL: {len(failed)} file(s) exceed {args.tol}")
        return 1
    print(f"PASS: all files within {args.tol}")
    return 0


def cmd_fixtures(args):
    embed, cosine = _scorer()
    ref = embed(args.ref)
    files = []
    for path in args.files:
        e = embed(path)
        files.append(
            {
                "path": os.path.abspath(path),
                "score": cosine(ref, e),
                "embedding": [round(float(x), 6) for x in e[0].tolist()],
            }
        )
    fixture = {
        "ref": os.path.abspath(args.ref),
        "refEmbedding": [round(float(x), 6) for x in ref[0].tolist()],
        "files": files,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("python")
    p.add_argument("--ref", required=True)
    p.add_argument("--glob", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--emb-out")
    p.set_defaults(func=cmd_python)

    p = sub.add_parser("compare")
    p.add_argument("--py", required=True)
    p.add_argument("--swift", required=True)
    p.add_argument("--py-emb")
    p.add_argument("--swift-emb")
    p.add_argument("--tol", type=float, default=0.01)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("fixtures")
    p.add_argument("--ref", required=True)
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fixtures)

    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
