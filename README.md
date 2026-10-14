## v3b: Common-path phase cancellation — verified mechanism, conditional value (14 Oct 2026)

**[Interferometry dashboard: real frozen checkpoint results](https://anttiluode.github.io/SihtiGrapImageModel/interferometry.html)** · [Frozen v3b protocol](INTERFEROMETRY_PROTOCOL.md)

We independently ran both existing frozen 64-byte writers with a new
two-shot interferometric readout:

```text
z1 = exp(i*(tau(source) + sum(tau(intermediate))))
z2 = exp(i*(sum(tau(intermediate)) + tau(port)))
z1*conj(z2) = exp(i*(tau(source) - tau(port)))
```

The **common-path phase cancels exactly**, checked numerically to
<1.1e-6 on actual frozen learned delay maps, and confirmed invariant
when intermediate phase values are deliberately changed. Ordinary
subtraction `z1-z2` preserves the common unknown rotation; the
demodulated subtraction `(z1-z2)*conj(z2)` restores the ratio
(up to subtracting 1).

However, the stronger **classification prediction failed**:
the ratio does not reach direct local performance on both writers.
All results below are held-out accuracy at K=2 actual complex
detector measurements per question, 2,048 questions/checkpoint.

| K=2 readout | Seed 41 | Seed 97 |
| --- | ---: | ---: |
| Both full raw orders | 78.37% | 98.73% |
| Explicit conjugate ratio | **59.23%** | **98.58%** |
| Uncalibrated subtraction | 71.78% | 72.95% |
| Direct phase from **same single source coordinate** | 62.74% | 99.95% |
| Direct phase from **two** addressed source coordinates | 78.91% | 100.00% |

**Interpretation:** seed 97's interrogated source site already stores a
near-perfect label signal; seed 41's same probe source site is much
less informative. The ratio cannot create source signal. Moreover,
it returns *source minus variable port phase* (port variability is
nonzero), not absolute source phase. The complete raw pair can retain
task-relevant intermediate phases that the ratio cancels.

This is an exact and useful **relative-phase interferometric
measurement identity**, not a novel mathematical theorem or proof of
myelin/dendrite wave physics. The finite-path model is ideal unitary
propagation, and the trained writer is a supervised convolutional
image code. A single ratio uses **two sensor readings per one source
position**, whereas the two-sample direct-access control reads two
source positions; the new matched-source control fixes that
important source-count mismatch but uses privileged addressed access.

Original v3 results stay intact at
[results/restricted_ports/](results/restricted_ports/).
New [seed 41](results/interferometry/seed41.json) and
[seed 97](results/interferometry/seed97.json) numerical receipts,
including K=2,4,8, are preserved separately.

---

## Frozen v3 restricted-port test — measured results (October 14, 2026)

**[Interactive results dashboard: question-directed probes against stronger controls](https://anttiluode.github.io/SihtiGrapImageModel/restricted_ports.html)** · [Frozen test specification](FROZEN_PORT_PROTOCOL.md)

Both v2 phase writers (seeds 41 and 97) were **frozen** and evaluated on
the same scene-disjoint 32×32 synthetic data splits, asking **all four
quadrant questions for each scene** (2,048 held-out queries/seed).
The 8×8 quantized writer state remains **64 bytes**. New learned
**reader heads only** use K=2,4,8 complex sensor measurements per
question. Each active shot is exactly 12 unitary phase-and-grid-shift
steps, ending at one of **two fixed detector ports**.

| Four complex measurements | Seed 41 | Seed 97 | Mean |
| --- | ---: | ---: | ---: |
| Question-directed wave (phase→shift) | 91.26% | 95.90% | **93.58%** |
| Reversed phase/shift order | 91.21% | 95.41% | 93.31% |
| Both raw operator orders, two paired paths | 87.16% | 98.93% | 93.04% |
| Difference of order pairs | 74.85% | 81.49% | 78.17% |
| Question-independent routes | 58.84% | 71.78% | 65.31% |
| Addressable local phase (stronger interface) | 98.05% | 100.00% | **99.02%** |
| Addressable local amplitude | 97.07% | 100.00% | 98.54% |
| Passive phase at fixed ports | 61.13% | 59.52% | 60.33% |

**Decision:** question-conditional spatial access wins against question-
independent fixed-port routes, but the *predeclared stronger test fails*:
active waves do not beat direct local phase sampling on both checkpoints
at the same sensor-reading count. Nor is an operator-order advantage
replicated across both writers. The order-*difference* generally loses.

At eight shots, target-directed wave accuracy rises to 98.46%,
question-independent wave routes to 80.03%, while local phase stays
99.02%. An unusual result deserves follow-up: seed 97's paired-order
head scores 98.73% from just two readings, compared with 78.12% for
seed 41. Neither paired performance nor simple commutator alone
licenses an information-creation claim.

**Important fairness limits:** the active probe visits 12 phase locations
per sensor reading, whereas the direct-local control reads one phase
location; information-access and computation budgets are therefore
not identical, even though the final number of complex-valued
measurements is matched. Question-conditioned paths were constructed
using target **quadrant geometry** without examining image labels;
this is not a feedback-adaptive controller. Fixed ports cannot be
relocated. No moving objects, histories, or physical speed PDE were
tested. Details, failure conditions, and run configurations are in
[FROZEN_PORT_PROTOCOL.md](FROZEN_PORT_PROTOCOL.md).

[Real seed 41 measurements](results/restricted_ports/seed41.json) ·
[Real seed 97 measurements](results/restricted_ports/seed97.json) ·
[Benchmark implementation](train_probe_gate.py) ·
[Physical path simulator and unit tests](probe_gate.py)

---

## Actual checkpoint inspection: learned 64-byte maps (October 14, 2026)

The **saved** `phase_wake_seed41.pt` and `phase_wake_seed97.pt`
models have now been loaded and inspected, separately from the website's
hand-built analytic phase generator. We ran a previously unused, fixed
set of 256 synthetic 32×32 images through each frozen writer. For every
image, we changed exactly the four RGB pixels of one quadrant's 2×2
colored mark and kept all other pixels identical, repeated for all
four quadrants.

| Measured quantity | Seed 41 | Seed 97 |
| --- | ---: | ---: |
| Phase bytes changed per mark flip (average) | 12.72 / 64 | 12.75 / 64 |
| Magnitude of byte changes **inside changed quadrant** | 98.74% | 99.01% |
| Counterfactual queried mark accuracy, all 6 informative heads | 100% | 100% |
| Other quadrant predictions unchanged, direct-phase reader | 99.967% | 100% |
| Distinct byte values seen over 256 scenes | 243 / 256 | 239 / 256 |
| Pooled byte-value entropy (NOT image-memory capacity) | 7.79 bits | 7.67 bits |

The same-image spatial correlation **between the two trained phase
maps** averaged −0.208, showing their exact coding patterns differ;
it is **not** proof of an intrinsic geometric coordinate system.

**[View seed 41 learned maps and per-quadrant interventions](checkpoint_inspection/seed41.html)** ·
**[View seed 97 maps](checkpoint_inspection/seed97.html)** ·
**[Full report JSON](checkpoint_inspection/summary.json)** ·
**[Reproducible inspector](inspect_checkpoint.py)**

**Limits:** the writer is convolutional, so local counterfactual
responses are encouraged by its finite receptive field; the observation
does **not** prove that a phase-speed medium, myelin, or an actively
chosen operator path was discovered. All six heads can decode these
patterns. Two models' differences need additional tests before
interpreting their phase coordinate choices physically.

---

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
