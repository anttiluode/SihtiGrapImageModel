"""Frozen v2 benchmark: compressed phase delays and matched phase-aware readers.

Usage:
  python train_speedzones.py --seed 41 --steps 500 --device auto
This does not overwrite results of v1.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from speedzones import (ARMS, IMAGE_SIZE, GRID_SIZE, SpeedZoneModel,
                        PAYLOAD_BYTES, MAX_PHASE, phase_bytes, decode_phase,
                        physics_receipt)


def make_examples(n, seed):
    """Four neutral objects with moving 2x2 red/blue marks.

    One of four quadrants is queried. Colors and locations are sampled
    independently on each scene. Train/val/test use distinct RNG streams.
    """
    rng = np.random.default_rng(seed)
    imgs = np.empty((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    questions = rng.integers(0, 4, n, dtype=np.int64)
    targets = np.empty((n,), dtype=np.int64)
    for i in range(n):
        background = float(rng.uniform(.07, .26))
        frame = np.clip(background + rng.normal(0, .035, (3, 32, 32)), 0, 1)
        answer = []
        for quadrant in range(4):
            y0, x0 = (quadrant // 2) * 16, (quadrant % 2) * 16
            neutral = float(rng.uniform(.36, .69))
            frame[:, y0+2:y0+14, x0+2:x0+14] = np.clip(
                neutral + rng.normal(0, .045, (3, 12, 12)), 0, 1)
            y = y0 + int(rng.integers(4, 11))
            x = x0 + int(rng.integers(4, 11))
            mark_color = int(rng.integers(0, 2))
            color = (.96, .08, .08) if mark_color == 0 else (.08, .13, .96)
            frame[:, y:y+2, x:x+2] = np.asarray(color)[:, None, None]
            answer.append(mark_color)
        imgs[i] = np.clip(frame, 0, 1).astype(np.float32)
        targets[i] = answer[questions[i]]
    return torch.from_numpy(imgs), torch.from_numpy(questions), torch.from_numpy(targets)


@torch.no_grad()
def evaluate(net, batch, device, shuffle=False, corrupt=0.0):
    net.eval()
    x, questions, answers = batch
    totals = {arm: [0, 0.0] for arm in ARMS}
    for offset in range(0, len(x), 64):
        im = x[offset:offset+64].to(device)
        q = questions[offset:offset+64].to(device)
        lab = answers[offset:offset+64].to(device)
        tau = net.write(im)
        if shuffle and len(tau) > 1:
            tau = torch.roll(tau, shifts=1, dims=0)
        if corrupt:
            # Deterministic perturbation in quantized *phase* units.
            # A corrupt=0.2 means 20% of byte steps are perturbed
            # with Gaussian sigma about 51 levels, not 20% dropout.
            rng = torch.Generator(device=device).manual_seed(1200+offset)
            noise = torch.randn(tau.shape, generator=rng, device=device)
            tau = decode_phase(torch.round(
                phase_bytes(tau).float() + (255 * corrupt) * noise
            ).clamp(0, 255).to(torch.uint8))
        out = net.read(tau, q)
        for arm, logits in out.items():
            totals[arm][0] += (logits.argmax(-1) == lab).sum().item()
            totals[arm][1] += F.cross_entropy(logits, lab, reduction="sum").item()
    return {arm: {"accuracy": c/len(x), "cross_entropy": loss/len(x)}
            for arm, (c, loss) in totals.items()}


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    train = make_examples(args.train_count, args.seed * 100 + 1)
    valid = make_examples(args.val_count, args.seed * 100 + 2)
    test = make_examples(args.test_count, args.seed * 100 + 3)
    net = SpeedZoneModel().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    for step in range(args.steps):
        net.train()
        ix = torch.randint(len(train[0]), (args.batch_size,))
        outputs = net(train[0][ix].to(device), train[1][ix].to(device))
        y = train[2][ix].to(device)
        loss = sum(F.cross_entropy(v, y) for v in outputs.values()) / len(outputs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if args.verbose and (step+1) % 100 == 0:
            print("step", step+1, "loss", round(loss.item(), 5))

    test_metrics = evaluate(net, test, device)
    param_counts = {k: sum(p.numel() for p in v.parameters()) for k, v in net.readers.items()}
    assert len(set(param_counts.values())) == 1
    with torch.no_grad():
        codes = phase_bytes(net.write(test[0][:2].to(device)))
        assert tuple(codes.shape) == (2, 1, GRID_SIZE, GRID_SIZE)
        assert codes.element_size() * codes[0].numel() == PAYLOAD_BYTES
    result = {
        "protocol": "phase-wake-v2 — shared quantized phase writer, matched decoder parameters",
        "seed": args.seed, "device": device, "steps": args.steps,
        "train_count": args.train_count, "val_count": args.val_count,
        "test_count": args.test_count,
        "input_rgb_u8_bytes": 3072, "image_size": [32, 32],
        "phase_map_size": [8, 8], "phase_payload_bytes": PAYLOAD_BYTES,
        "phase_range": [-MAX_PHASE, MAX_PHASE],
        "writer_parameters": sum(p.numel() for p in net.writer.parameters()),
        "reader_parameters": param_counts,
        "validation": evaluate(net, valid, device),
        "test": test_metrics,
        "shuffled_memory": evaluate(net, test, device, shuffle=True),
        "corruption_sigma_20pct_bytes": evaluate(net, test, device, corrupt=.2),
        "physics_checks": physics_receipt(),
        "interpretation": ("intensity_only is blind by construction; the real "
                           "comparisons are commutator versus direct_phase, "
                           "phase_gradient, coherent_mean and amplitude_code. "
                           "All share a co-trained writer. No semantic geometry "
                           "or myelin dynamics established.")
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    if args.checkpoint:
        check = Path(args.checkpoint)
        check.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "seed": args.seed,
                    "phase_payload_bytes": PAYLOAD_BYTES}, check)
    print(json.dumps(result, indent=2))
    return result


def args_from_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--val-count", type=int, default=384)
    parser.add_argument("--test-count", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", default="results/phase_wake_seed41.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(args_from_cli())
