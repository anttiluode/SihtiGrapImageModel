import torch

from model import H, W, ARMS, MemoryModel, all_reads, edge_mask, transport
from train import examples, evaluate


def test_payload_and_reader_budget():
    net = MemoryModel().eval()
    x, sides, _ = examples(3, 123)
    with torch.no_grad():
        w = net.write(x)
        assert w.shape == (3, 4, H, W)
        assert w[0].numel() == 896
        assert torch.allclose(w * 255, (w * 255).round(), atol=1e-5)
        assert (w * (1 - edge_mask(w.device))).abs().max() == 0
        outputs = net.read(w, sides)
    assert set(outputs) == set(ARMS)
    assert all(y.shape == (3, 2) for y in outputs.values())
    sizes = [sum(p.numel() for p in h.parameters())
             for h in net.readers.values()]
    assert len(set(sizes)) == 1


def test_noncommutativity_and_algebra():
    w = edge_mask("cpu").repeat(1, 1, 1, 1) * .65
    # Induce a local mismatch between the two transport paths.
    w[:, 0, 6, 6] = .10
    q = torch.ones(1, 2, H, W)
    ab = transport(w, transport(w, q, "B"), "A")
    ba = transport(w, transport(w, q, "A"), "B")
    assert torch.max((ab - ba).abs()).item() > .01
    reads = all_reads(w, torch.zeros(1, dtype=torch.long))
    assert reads["commutator"].shape == (1, 5, H, W)
    # On a homogeneous infinite-grid interior, these shifts commute.
    homogeneous = edge_mask("cpu") * .65
    u = transport(homogeneous, transport(homogeneous, q, "B"), "A")
    v = transport(homogeneous, transport(homogeneous, q, "A"), "B")
    assert torch.allclose(u[:, :, 3:-3, 3:-3], v[:, :, 3:-3, 3:-3])


def test_ordered_contains_symmetric_and_difference():
    w = edge_mask("cpu") * .6
    side = torch.tensor([0])
    r = all_reads(w, side)
    ab, ba = r["ordered"][:, :2], r["ordered"][:, 2:4]
    assert torch.allclose((ab + ba) / 2, r["symmetric"][:, :2])
    assert torch.allclose(ab - ba, r["commutator"][:, :2])


def test_reproducible_fixture_and_smoke_learning():
    a = examples(12, 77)
    b = examples(12, 77)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    net = MemoryModel()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    outputs = net(a[0], a[1])
    loss = sum(torch.nn.functional.cross_entropy(z, a[2])
               for z in outputs.values())
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.writer.parameters())
    opt.step()
    metrics = evaluate(net, a, 6, "cpu")
    assert set(metrics) == set(ARMS)
    assert all(0 <= m["accuracy"] <= 1 for m in metrics.values())
