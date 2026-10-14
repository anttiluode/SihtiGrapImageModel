# Phase-wake v2 — frozen experiment specification (October 2026)

## Why this follows v1

The 896-byte v1 memory used 14×16 images (672 B if RGB is stored
as uint8), and all five trained readers achieved 100% held-out accuracy
on the two 450-step synthetic runs (seeds 41, 97). V1 therefore
established that the scene-specific graph carries the answer, **not**
that ordered probes are uniquely helpful or that any image compression
occurred. Do not replace or reinterpret that result.

## What is stored?

A 32×32×3 uint8 RGB image is 3,072 B before optional compression.
The v2 learned writer converts it to one 8×8 quantized phase-delay map,
**64 bytes per scene, 48× smaller than the raw RGB-byte representation**.
The byte grid is retained at read time; original pixels are discarded.
The writer and readers have separate shared parameter costs, so 48× is
only a **per-scene payload ratio**, not whole-system compression.

The phase range is [-1.2,1.2] radians. Each site stores one 8-bit phase
delay tau. The model explicitly **does not solve the eikonal equation**:
tau is a learned phase mask, not a physically calibrated slowness,
arrival-time map, myelin conductivity, or lipid concentration.

## Exact operators

Define U_tau(q)(i,j)=exp(i*tau(i,j))*q(i,j). Let Sx/Sy be periodic
horizontal/vertical shifts, each preserving the L2 norm. Then

    [Sx,U_tau] 1 = shift_x(exp(i*tau)) - exp(i*tau)
    [Sy,U_tau] 1 = shift_y(exp(i*tau)) - exp(i*tau)

Those are exact complex phase differences. For small phase differences,
[Sx,U]1 approx i*exp(i*tau)*Delta_x(tau). The phase-gradient control
receives the simpler finite differences of tau directly.

Unitary phase multiplication and periodic shift preserve L2 energy
under repeated composition; the commutator difference is not unitary.
A lossy attenuation foil is included **only as a physics sanity check**.
No speed, eikonal wavefront travel or wake persistence over time is
simulated. Constant tau implies zero commutator; local differences yield
nonzero responses. Periodic roll entails wraparound at borders.

## Seven tested readers

Each has the same learned CNN readout architecture and parameters,
with input channels zero-padded to nine total:

| Arm | What it observes | Status |
|---|---|---|
| commutator | [Sx,U]1 and [Sy,U]1 (real/imag) | hypothesis |
| ordered_pair | both raw orderings SxU1,USx1,SyU1,USy1 | strictly stronger information baseline |
| phase_gradient | signed spatial finite differences in tau | simple phase-aware baseline |
| coherent_mean | complex sums (SxU1+USx1)/2 etc. | order-blind, phase-aware |
| direct_phase | cos(tau),sin(tau) | all-phase baseline |
| amplitude_code | monotone amplitude re-encoding of same tau | no-phase-computation baseline |
| intensity_only | abs(exp(i*tau))^2 = 1 | **deliberately blind negative control**, not competitive |

All receive identical explicit quadrant question masks.
The same writer is jointly trained using all seven losses, avoiding
writer privilege for the proposed commutator but not proving each
independently optimized architecture's performance.

**Important:** the commutator is a deterministic function of the raw
ordered responses and of the full tau map. Its merit can only be
efficiency, invariance, learnability, hardware access or robustness —
never magical creation of information absent from the input.

## Task & data

Each 32×32 image contains four textured neutral rectangles (one per
quadrant) with independently colored 2×2 red or blue marks. Marks
move within each quadrant and nuisance intensities vary. The question
selects a quadrant; the answer is its binary mark color. The labels
are balanced in expectation by seeded independent draws. The marks
are small enough for 4× reduction to discard detail unless the writer
learns to preserve them; this is still a **controlled synthetic task**.

Default train/val/test: 2,048 / 384 / 512 independently generated scenes
using seeds 100*s+1, +2, +3, respectively; batch size 64; AdamW
learning rate 0.001, 500 optimizer updates.

    python train_speedzones.py --seed 41 --steps 500 --device auto --out results/phase_wake_seed41.json --checkpoint checkpoints/phase_wake_seed41.pt
    python train_speedzones.py --seed 97 --steps 500 --device auto --out results/phase_wake_seed97.json --checkpoint checkpoints/phase_wake_seed97.pt

Predeclaration: retain all reader accuracies and cross-entropy scores,
ablate with across-scene graph shuffling, and corrupt phase bytes with
Gaussian noise at 20% of the 255-level byte range. Do not select the
best seed or relabel a tied accuracy ceiling as a win. A proper
multi-seed comparison needs at least eight seeds and adjusted confidence
intervals for multiple arms; 41 and 97 are pilots.

## Validation and failure modes

The numeric invariants tested separately:
1. uniform delay yields zero [S,U]1 on a periodic grid;
2. local variation yields a nonzero response and agrees with the
   exact complex finite difference;
3. 20 steps of composed phase multiplication and periodic shifts
   preserve total complex-signal energy to numerical tolerance;
4. the attenuation foil dissipates energy.

Possible failures include: phase-aware direct_tau or amplitude_code
beating commutators, phase gradient carrying all useful signal, learned
writer hiding the answer as an arbitrary code, and all readers reaching
another ceiling. None of those should be described as confirming a
speed-zone neural mechanism.

## Interpretation

This is **not** a demonstrated neural, myelin or 539-lipizone model.
The lipid-atlas and dendritic examples motivate heterogeneous
propagation media, but do not verify the memory or optical assumptions.
An eikonal arrival-time gradient is not generally a persistent wake
stored in the medium after a pulse has passed; a sustained phase front
requires a maintained/propagating wave state. Multiplication by
exp(i*tau) preserves modulus; the follow-on phase difference reads
spatial variation, without demonstrating lensing, focusing, or
polychronization.

Real image recognition, dynamic history-written speed zones, and
query-trained wave scheduling remain untested follow-ups.
