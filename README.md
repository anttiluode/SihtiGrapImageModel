## Phase-wake v2 — speed-zone experiment (new)

**[Open the phase-wake interactive laboratory](https://anttiluode.github.io/SihtiGrapImageModel/speedzones.html)** · **[Read the frozen v2 protocol](SPEEDZONE_PROTOCOL.md)**

Gemini's proposed speed-zone hypothesis is now an explicitly testable
**separate v2**; the original v1 and its two seed receipts remain intact.

- Write each **32×32 RGB** image (3,072 uncompressed uint8 bytes)
  into a single **8×8 uint8 phase-delay map (64 bytes)**.
- A unitary phase multiplication `Uτ` and periodic spatial shifts
  `Sx, Sy` define a wake: `[Sx,Uτ]1 = Sx(e^{iτ})−e^{iτ}`.
- Compare a commutator reader to a **strong paired-order** reader,
  a signed **phase-gradient** reader, a complex coherent mean,
  a direct-phase reader, and a monotone **amplitude-code** baseline.
  Intensity-only is an intentionally blind negative control.
- Keep one jointly optimized writer and **identically sized classifiers**.
  Report held-out accuracy, cross-entropy, shuffled-graph and phase-byte
  corruption controls.
- An analytic test verifies zero wake for a constant phase map,
  a nonzero localized wake, and 20-step energy conservation of the
  unitary transport. It does not demonstrate physical ray focusing,
  persistent phase wakes in tissue, or myelin memory.

```bash
pip install -r requirements.txt
pytest -q
python train_speedzones.py --steps 500 --seed 41 --device auto --out results/phase_wake_seed41.json --checkpoint checkpoints/phase_wake_seed41.pt
python train_speedzones.py --steps 500 --seed 97 --device auto --out results/phase_wake_seed97.json --checkpoint checkpoints/phase_wake_seed97.pt
```

**No trained v2 performance is claimed here.** Better performance than
intensity-only is tautological for a pure phase signal and is **not**
evidence that a commutator helps. The meaningful comparisons are
direct-phase, finite-difference phase-gradient, amplitude-coded memory
and raw ordered-response readers. See [SPEEDZONE_PROTOCOL.md](SPEEDZONE_PROTOCOL.md).

---

# SihtiGrapImageModel — probing a visual memory

**Interactive experiment:** https://anttiluode.github.io/SihtiGrapImageModel/

A separate research branch from [SihtiResponseGraph](https://github.com/anttiluode/SihtiResponseGraph). This repository asks a different question:

> Can *ordered queries* on a learned, image-written conductance graph recover visual identity information more effectively than order-blind queries at the same stored-byte budget?

The key test is the readout difference

```math
\Delta(q;W) = A(W)B(W)q - B(W)A(W)q.
```

This is a **commutator** of distinct transport operators. Even when both
operators are linear in the query, their image-specific conductance values
can make them fail to commute. Noncommutativity is not automatically
nonlinearity or meaningful object detection.

## The model

- **Write:** a small CNN maps a 14×16 RGB scene into four directional
  uint8 conductance planes: east, south, southeast, southwest.
  There are 14×16×4 = **896 bytes per scene**, including zero padding on
  invalid boundary edges. The four planes are quantized in both training
  forward passes and evaluation; the straight-through gradient
  approximates the rounding step during training.
- **Erase pixels:** future readers receive W and a new left/right query,
  never original image activations.
- **Interrogate:** apply A and B in both orders to the same two-channel
  query field. The ordering is deliberately fixed, reported and testable.
- **Answer:** separately trained but identically sized readout heads
  classify the color of the 2×2 mark on the queried object. One graph
  writer is jointly trained across all heads.

No learned checkpoint is bundled. The static site is an interactive
**analytical illustration**, not a visualization of trained weights.

## Five comparison arms

| Arm | Observations | Operator calls | Reader weights |
| --- | --- | ---: | --- |
| Symmetric | (ABq + BAq)/2 | 4 | matched |
| Commutator | ABq − BAq | 4 | matched |
| Ordered | [ABq, BAq] | 4 | matched |
| Isotropic | ((A+B)/2)^2 q | 4 | matched |
| Direct graph | [W, query-side mask] | 0 | matched |

A paired-response reader has **strictly more available information** than
its subtraction; any performance reversal would be attributable to the
limited decoder/training regime, not information creation. A direct-W
head is included to test whether the ordered operations help at all,
but is **not** a compute-matched transport baseline.

## Run it

Requires Python 3.10+ and PyTorch. Both CPU and an RTX 3060 are usable.

```bash
python -m pip install -r requirements.txt
pytest -q
python train.py --steps 450 --seed 41 --device auto --out results/seed41.json
python train.py --steps 450 --seed 97 --device auto --out results/seed97.json
```

For a very fast plumbing check:

```bash
python train.py --steps 1 --train-count 24 --val-count 16 --test-count 16 --batch-size 8 --device cpu --out results/smoke.json
```

Each JSON receipt includes the held-out accuracies and cross entropies of
all five readers, validation scores, byte budget, parameter counts, and
a **shuffled-across-scenes graph** negative control. The test data are
independent seeded synthetic draws. Train/validation/test are separate.

**There are no claimed winning scores yet.** Read [PROTOCOL.md](PROTOCOL.md)
for comparison rules and limitations. More seeds and deliberately harder
tasks are needed before drawing comparative conclusions.

## Web lab

The root [index.html](index.html) runs entirely client-side, with no
installation. Generate a two-object scene or upload your own image.
The site constructs image-conditioned conductances from local RGB
differences, quantizes each value to 8 bits, then displays AB, BA,
the signed difference and their mean. You can change:

- the two colored marks and which object's identity to interrogate;
- a region, global, or localized query;
- sensitivity of image → conductance conversion;
- conductance corruption (a deterministic mixture with noise).

The display rescales response maps separately to make structure legible;
do not read brightness differences as calibrated classification scores.
The web and Python graph writers are **different implementations**,
connected by the same operator question, not equivalent learned weights.

## Scientific context and boundaries

**Sihti** measured what survives repeated purification.
**ResearchQuestion** tested what information a compressed state must keep
for future questions.
**SihtiResponseGraph** explored storing image-specific conductances rather
than only feature vectors.
Here, **interrogation order** is the independent variable.

Biological dendrites can exhibit order-sensitive and state-dependent
integration, and membrane/myelin chemistry can influence signal
propagation. The reported 2026 lipid atlas is motivation for considering
nonuniform media, **not** evidence that a lipid patch is a digital
conductance memory, or that this trained graph reproduces brain tissue.
The four edge directions are computational operators, not reconstructed
axon or dendrite geometries.

Not a diffusion image generator, not an outpainting model, not a claim of
unseen 3D geometry. The immediate target is a *queryable visual-memory
mechanism* and a result that could fail.

MIT licensed. Research code and browser experiment by the project author
with AI-assisted implementation.
