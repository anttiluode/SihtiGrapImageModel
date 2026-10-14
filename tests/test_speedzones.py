import math
import torch

from speedzones import (ARMS, GRID_SIZE, PAYLOAD_BYTES, SpeedZoneModel,
                        phase_bytes, decode_phase, commutator_fields,
                        complex_phase, phase_step, attenuation_step,
                        features, physics_receipt)
from train_speedzones import make_examples, evaluate


def test_phase_byte_roundtrip_and_payload():
    net = SpeedZoneModel().eval()
    img, sides, _ = make_examples(3, 42)
    with torch.no_grad():
        tau = net.write(img)
        assert tau.shape == (3, 1, 8, 8)
        codes = phase_bytes(tau)
        assert codes.dtype == torch.uint8
        assert codes[0].numel() == PAYLOAD_BYTES == 64
        assert torch.allclose(decode_phase(codes), tau, atol=3e-7)
        result = net.read(tau, sides)
        assert set(result) == set(ARMS)
        assert all(logits.shape == (3, 2) for logits in result.values())


def test_gradient_commutator_identity_and_uniform_null():
    flat = torch.zeros((2, 1, 8, 8)) + .6
    cx, cy = commutator_fields(flat)
    assert cx.abs().max().item() < 1e-6
    assert cy.abs().max().item() < 1e-6
    defect = flat.clone()
    defect[:, :, 3:5, 4:6] = -.4
    x, y = commutator_fields(defect)
    U = complex_phase(defect)
    assert torch.allclose(x, torch.roll(U, -1, -1)-U)
    assert torch.allclose(y, torch.roll(U, -1, -2)-U)
    assert x.abs().max() > .5


def test_unitary_propagation_not_unitary_commutator():
    t = torch.linspace(-1.0, 1.0, 64).view(1, 1, 8, 8)
    initial = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
    x = initial
    y = initial
    for k in range(20):
        axis = ("x" if k % 2 else "y")
        x = phase_step(x, t, axis=axis)
        y = attenuation_step(y, t, axis=axis)
    assert torch.allclose(x.abs().square().sum(), initial.abs().square().sum(), rtol=1e-5)
    assert y.abs().square().sum() < initial.abs().square().sum() / 10


def test_no_label_leakage_to_phase_intensity_and_reader_size():
    imgs, target, labels = make_examples(5, 131)
    net = SpeedZoneModel()
    f = features(net.write(imgs), target)
    assert torch.allclose(f["intensity_only"][:, 0], torch.ones_like(f["intensity_only"][:, 0]), atol=1e-5)
    assert len({sum(p.numel() for p in v.parameters()) for v in net.readers.values()}) == 1
    loss = sum(torch.nn.functional.cross_entropy(o, labels)
               for o in net(imgs, target).values())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.writer.parameters())


def test_smoke_eval_and_physics_receipt():
    images = make_examples(8, 1)
    net = SpeedZoneModel().eval()
    m = evaluate(net, images, "cpu")
    assert set(m) == set(ARMS)
    p = physics_receipt()
    assert p["uniform_commutator_abs_max"] < 1e-6
    assert p["localized_wake_rms"] > .01
    assert abs(p["phase_energy_after_20"] - 1.0) < 1e-5
    assert p["lossy_foil_energy_after_20"] < .1
