"""V2: 64-byte image-written PHASE delay memory, not an amplitude diffuser.

The mathematical probes operate on unit-modulus complex fields using
periodic shifts, so phase transport itself preserves L2 energy. The
commutator is a measured DIFFERENCE, and need not preserve energy.

tau is a phase delay, not a direct numerical solution of the eikonal
equation. x/y shifts are abstract unitary operators, not ray tracing.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = 32
GRID_SIZE = 8
PAYLOAD_BYTES = GRID_SIZE * GRID_SIZE
MAX_PHASE = 1.2
ARMS = (
    "commutator", "ordered_pair", "phase_gradient",
    "coherent_mean", "direct_phase", "amplitude_code", "intensity_only",
)


def phase_bytes(tau):
    """8-bit map; no per-sample side information."""
    return torch.round(((tau / MAX_PHASE + 1) * 127.5)).clamp(0, 255).to(torch.uint8)


def decode_phase(code):
    return (code.to(torch.float32) / 255 * 2 - 1) * MAX_PHASE


class SpeedWriter(nn.Module):
    """Learns the medium, subject to a strict 8x8x1 byte bottleneck."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(20, 20, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(20, 1, 3, padding=1),
        )

    def forward(self, images):
        if images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Expected (batch,3,32,32) images")
        continuous = MAX_PHASE * torch.tanh(self.net(images))
        encoded = ((continuous / MAX_PHASE + 1) * 127.5).round().clamp(0, 255)
        quantized = (encoded / 255 * 2 - 1) * MAX_PHASE
        # The forward pass really uses quantized 8-bit phases.
        return continuous + (quantized - continuous).detach()


def complex_phase(tau):
    return torch.polar(torch.ones_like(tau), tau)


def roll_x(z):
    return torch.roll(z, shifts=-1, dims=-1)


def roll_y(z):
    return torch.roll(z, shifts=-1, dims=-2)


def commutator_fields(tau):
    """[Sx,U]1 and [Sy,U]1, exactly neighboring phase differences."""
    u = complex_phase(tau)
    x = roll_x(u) - u
    y = roll_y(u) - u
    return x, y


def query_mask(target, device):
    """Quadrant query is visible to all readers."""
    row = torch.arange(GRID_SIZE, device=device)[None, :, None]
    col = torch.arange(GRID_SIZE, device=device)[None, None, :]
    ty = target // 2
    tx = target % 2
    region = ((row >= 4) == (ty[:, None, None] == 1)) & (
        (col >= 4) == (tx[:, None, None] == 1)
    )
    return region[:, None].to(torch.float32)


def features(tau, target):
    """All heads share the same writer and identically sized readout."""
    u = complex_phase(tau)
    ux, uy = roll_x(u), roll_y(u)
    cx, cy = ux - u, uy - u
    gx = roll_x(tau) - tau
    gy = roll_y(tau) - tau
    mx = .5 * (ux + u)
    my = .5 * (uy + u)
    zero = torch.zeros_like(tau)
    mask = query_mask(target, tau.device)
    assert mask.shape == tau.shape

    def pack(*channels):
        if len(channels) > 8:
            raise ValueError("At most eight feature channels")
        return torch.cat([*channels, *([zero] * (8 - len(channels))), mask], dim=1)

    return {
        "commutator": pack(cx.real, cx.imag, cy.real, cy.imag),
        # Both orderings of Sx and U, and both orderings of Sy and U;
        # never discard the raw responses before the paired baseline.
        "ordered_pair": pack(ux.real, ux.imag, u.real, u.imag,
                             uy.real, uy.imag, u.real, u.imag),
        "phase_gradient": pack(gx, gy),
        "coherent_mean": pack(mx.real, mx.imag, my.real, my.imag),
        "direct_phase": pack(u.real, u.imag),
        "amplitude_code": pack((tau / MAX_PHASE + 1) / 2),
        "intensity_only": pack(u.abs().square()),
    }


class Classifier(nn.Module):
    """Equal parameter budget across all seven reader arms."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(16 * GRID_SIZE * GRID_SIZE, 2),
        )

    def forward(self, x):
        return self.net(x)


class SpeedZoneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.writer = SpeedWriter()
        self.readers = nn.ModuleDict({k: Classifier() for k in ARMS})

    def write(self, image):
        return self.writer(image)

    def read(self, tau, target):
        f = features(tau, target)
        return {arm: reader(f[arm]) for arm, reader in self.readers.items()}

    def forward(self, image, target):
        return self.read(self.write(image), target)


def phase_step(field, tau, axis="x"):
    """Two unitary operations: multiplication by e^{i tau}, then roll."""
    if axis not in ("x", "y"):
        raise ValueError(axis)
    advanced = field * complex_phase(tau)
    return roll_x(advanced) if axis == "x" else roll_y(advanced)


def attenuation_step(field, tau, axis="x"):
    """Explicit lossy foil, NOT a competitive trained amplitude baseline."""
    atten = .7 + .15 * torch.sigmoid(tau)
    advanced = field * atten
    return roll_x(advanced) if axis == "x" else roll_y(advanced)


@torch.no_grad()
def physics_receipt():
    """Strict numeric invariants and localized wake (not trained results)."""
    uniform = torch.full((1, 1, 8, 8), .35)
    slow = uniform.clone()
    slow[:, :, 2:5, 3:6] += .7
    flat_comm = commutator_fields(uniform)
    wake_x, wake_y = commutator_fields(slow)
    initial = torch.ones_like(uniform, dtype=torch.complex64) / math.sqrt(64)
    p, damp = initial, initial
    for k in range(20):
        axis = "x" if k % 2 == 0 else "y"
        p = phase_step(p, slow, axis)
        damp = attenuation_step(damp, slow, axis)
    exact = roll_x(complex_phase(slow)) - complex_phase(slow)
    out = {
        "scope": "analytic controlled 8x8 phase field, not a trained image",
        "uniform_commutator_abs_max": max(f.abs().max().item() for f in flat_comm),
        "localized_wake_rms": torch.sqrt((wake_x.abs().square()
                                          + wake_y.abs().square()).mean()).item(),
        "commutator_identity_abs_max": (wake_x - exact).abs().max().item(),
        "phase_energy_after_20": p.abs().square().sum().item(),
        "lossy_foil_energy_after_20": damp.abs().square().sum().item(),
        "payload_bytes": PAYLOAD_BYTES,
        "raw_rgb_u8_bytes": 32 * 32 * 3,
        "caveat": "Unitary phase step is not the commutator itself; speed-zone physics is analogical.",
    }
    return out
