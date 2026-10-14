"""Reproducible multi-arm synthetic visual memory benchmark.

Train: python train.py --steps 450 --seed 41 --device auto
No scores are claimed by the project until this script is run.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import ARMS, H, W, MemoryModel, bytes_for_one_memory


def examples(n, seed):
    """Two textured objects; question = color of left/right tiny mark.

    The foreground color itself is NOT given to the reader. Once the
    image writes W the original pixels are discarded.
    """
    rng = np.random.default_rng(seed)
    imgs = np.empty((n, 3, H, W), dtype=np.float32)
    sides = rng.integers(0, 2, n, dtype=np.int64)
    answers = np.empty(n, dtype=np.int64)
    for i in range(n):
        gray = rng.uniform(.12, .30)
        img = np.clip(gray + rng.normal(0, .045, (H, W)), 0, 1)
        rgb = np.repeat(img[None], 3, axis=0).astype(np.float32)
        colors = rng.integers(0, 2, 2)
        for side in (0, 1):
            x0 = 1 if side == 0 else 9
            shade = rng.uniform(.39, .68)
            rgb[:, 2:12, x0:x0+6] = shade + rng.normal(
                0, .035, (3, 10, 6))
            # A small two-pixel mark is deliberately difficult to
            # preserve under coarse or isotropic image readout.
            y = int(rng.integers(4, 9))
            x = int(rng.integers(x0+1, x0+4))
            if colors[side] == 0:
                rgb[:, y:y+2, x:x+2] = np.array(
                    [.96, .13, .13])[:, None, None]
            else:
                rgb[:, y:y+2, x:x+2] = np.array(
                    [.13, .18, .96])[:, None, None]
        imgs[i] = np.clip(rgb, 0, 1)
        answers[i] = colors[sides[i]]
    return (torch.from_numpy(imgs), torch.from_numpy(sides),
            torch.from_numpy(answers))


@torch.no_grad()
def evaluate(model, samples, batch_size, device, shuffled=False):
    model.eval()
    correct = {arm: 0 for arm in ARMS}
    losses = {arm: 0. for arm in ARMS}
    total = 0
    images, sides, labels = samples
    for start in range(0, len(images), batch_size):
        x = images[start:start+batch_size].to(device)
        s = sides[start:start+batch_size].to(device)
        y = labels[start:start+batch_size].to(device)
        w = model.write(x)
        if shuffled:
            # Shift graph memories across different scenes while leaving
            # query and answer fixed (a negative causal control).
            w = torch.roll(w, shifts=1, dims=0)
        outputs = model.read(w, s)
        for arm, logits in outputs.items():
            correct[arm] += (logits.argmax(1) == y).sum().item()
            losses[arm] += F.cross_entropy(logits, y, reduction="sum").item()
        total += len(x)
    return {arm: {"accuracy": correct[arm] / total,
                  "cross_entropy": losses[arm] / total}
            for arm in ARMS}


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if (
        args.device == "auto") else args.device
    train = examples(args.train_count, args.seed * 100 + 1)
    val = examples(args.val_count, args.seed * 100 + 2)
    test = examples(args.test_count, args.seed * 100 + 3)
    net = MemoryModel().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    # Separate readout heads; one shared graph writer jointly optimized
    # across all arms. No arm gets privileged exposure during writing.
    net.train()
    for step in range(args.steps):
        idx = torch.randint(0, len(train[0]), (args.batch_size,))
        x = train[0][idx].to(device)
        side = train[1][idx].to(device)
        label = train[2][idx].to(device)
        outputs = net(x, side)
        loss = sum(F.cross_entropy(outputs[a], label) for a in ARMS) / len(ARMS)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if args.verbose and (step + 1) % 100 == 0:
            print("step", step + 1, "loss", round(loss.item(), 4))
    val_metrics = evaluate(net, val, args.batch_size, device)
    test_metrics = evaluate(net, test, args.batch_size, device)
    shuffled = evaluate(net, test, args.batch_size, device, shuffled=True)
    with torch.no_grad():
        x = test[0][:1].to(device)
        payload = bytes_for_one_memory(net.write(x))
        assert payload.shape == (1, 4, H, W)
        assert payload.dtype == torch.uint8
    reader_params = {a: sum(p.numel() for p in net.readers[a].parameters())
                     for a in ARMS}
    assert len(set(reader_params.values())) == 1
    result = {
        "protocol": "v1; same graph writer, equal-size reader heads",
        "seed": args.seed, "steps": args.steps, "device": device,
        "train_count": args.train_count,
        "val_count": args.val_count, "test_count": args.test_count,
        "image_size": [H, W], "graph_payload_bytes": 896,
        "writer_parameters": sum(p.numel() for p in net.writer.parameters()),
        "reader_parameters": reader_params,
        "validation": val_metrics, "test": test_metrics,
        "shuffled_graph_control": shuffled,
        "limitations": ("Synthetic tiny-object color question; shared-writer "
                        "training; two moving marks; no natural images, "
                        "unobserved geometry, or biological interpretation.")
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.checkpoint:
        torch.save({"model": net.state_dict(), "protocol": result["protocol"],
                    "image_size": [H, W]}, args.checkpoint)
    print(json.dumps(result, indent=2))
    return result


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-count", type=int, default=2048)
    p.add_argument("--val-count", type=int, default=384)
    p.add_argument("--test-count", type=int, default=512)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--out", default="results/seed41.json")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
