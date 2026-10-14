"""Operator-coded visual memory: a byte-quantized, image-written graph.

Every sample retains four 14x16 uint8 conductance planes (896 bytes).
Image activations are discarded before any query. The query-dependent
ordered read uses two distinct, generally noncommuting transport rules.
This is a research prototype, not a biological membrane simulation.
"""
import torch
from torch import nn
from torch.nn import functional as F

H, W = 14, 16
ARMS = ("symmetric", "commutator", "ordered", "isotropic", "direct_graph")


def edge_mask(device):
    mask = torch.ones((1, 4, H, W), device=device)
    mask[:, 0, :, -1] = 0                     # east
    mask[:, 1, -1, :] = 0                     # south
    mask[:, 2, -1, :] = 0                     # south-east
    mask[:, 2, :, -1] = 0
    mask[:, 3, -1, :] = 0                     # south-west
    mask[:, 3, :, 0] = 0
    return mask


def shift_east(q):
    return F.pad(q[..., 1:], (0, 1, 0, 0))


def shift_south(q):
    return F.pad(q[:, :, 1:, :], (0, 0, 0, 1))


def shift_south_east(q):
    return F.pad(q[:, :, 1:, 1:], (0, 1, 0, 1))


def shift_south_west(q):
    return F.pad(q[:, :, 1:, :-1], (1, 0, 0, 1))


def transport(w, q, axis):
    """A and B use directed, non-periodic edges: no wraparound."""
    if axis == "A":
        return .5 * (w[:, 0:1] * shift_east(q) +
                     w[:, 2:3] * shift_south_east(q))
    if axis == "B":
        return .5 * (w[:, 1:2] * shift_south(q) +
                     w[:, 3:4] * shift_south_west(q))
    raise ValueError(axis)


def query_for(side, batch, device):
    """Two query channels; side 0 asks left, side 1 asks right."""
    yy = torch.linspace(-1, 1, H, device=device)[None, None, :, None]
    xx = torch.arange(W, device=device)[None, None, None, :]
    target = torch.where(side[:, None, None, None] == 0,
                         xx < W // 2, xx >= W // 2).float()
    target = target.expand(batch, 1, H, W)
    return torch.cat([target, target * yy.expand_as(target)], dim=1)


def all_reads(w, side):
    """Equal four-transport-call ordered, commutator and symmetric controls."""
    q = query_for(side, w.shape[0], w.device)
    ab = transport(w, transport(w, q, "B"), "A")
    ba = transport(w, transport(w, q, "A"), "B")
    mean = .5 * (ab + ba)
    diff = ab - ba
    # The isotropic two-step reader invokes A and B at each step, four
    # transports total: ((A+B)/2)^2 q.
    once = .5 * (transport(w, q, "A") + transport(w, q, "B"))
    iso = .5 * (transport(w, once, "A") + transport(w, once, "B"))
    zero = torch.zeros_like(mean)
    marker = q[:, :1]
    return {
        "symmetric": torch.cat([mean, zero, marker], dim=1),
        "commutator": torch.cat([diff, zero, marker], dim=1),
        "ordered": torch.cat([ab, ba, marker], dim=1),
        "isotropic": torch.cat([iso, zero, marker], dim=1),
        "direct_graph": torch.cat([w, marker], dim=1),
    }


class Writer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1), nn.GELU(),
            nn.Conv2d(20, 20, 3, padding=1), nn.GELU(),
            nn.Conv2d(20, 4, 1),
        )

    def forward(self, image):
        w = torch.sigmoid(self.net(image)) * edge_mask(image.device)
        byte_roundtrip = (255 * w).round() / 255
        # Straight-through estimator in training; genuine byte values
        # in both forward and evaluation paths.
        return w + (byte_roundtrip - w).detach()


class Reader(nn.Module):
    """All five arms have exactly the same learned parameter budget."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Linear(16 * 3 * 3, 2),
        )

    def forward(self, readings):
        return self.net(readings)


class MemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.writer = Writer()
        self.readers = nn.ModuleDict({arm: Reader() for arm in ARMS})

    def write(self, image):
        return self.writer(image)

    def read(self, w, side):
        feats = all_reads(w, side)
        return {arm: self.readers[arm](feats[arm]) for arm in ARMS}

    def forward(self, image, side):
        return self.read(self.write(image), side)


def bytes_for_one_memory(w):
    if w.shape[1:] != (4, H, W):
        raise ValueError("Expected (batch,4,14,16)")
    payload = (w.detach().clamp(0, 1) * 255).round().to(torch.uint8)
    assert payload[0].numel() == 896
    return payload
