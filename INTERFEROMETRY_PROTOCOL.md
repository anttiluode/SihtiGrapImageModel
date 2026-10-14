# Frozen interferometry v3b: exact endpoint phase cancellation

**Frozen specification, October 14, 2026.** This runs the original
v3 experiment on both frozen `phase_wake_seed41.pt` and
`phase_wake_seed97.pt`. No changes to the image-writer weights, train
/ validation / test scene seeds, question-conditioned route bank,
number of readout ports, propagation steps, or reader architecture.

## Hypothesis

For a sequence of visited nodes `p0,p1,...,p12`:

    z1 = exp(i*(tau[p0]+...+tau[p11]))
    z2 = exp(i*(tau[p1]+...+tau[p12]))
    z1 * conj(z2) = exp(i*(tau[p0]-tau[p12]))

The identity holds even if nodes repeat and the intermediate phase
values change, as long as both probes follow precisely the same path
and all operators remain unit modulus. It is **relative** phase: a
variable detector-phase offset remains. It does **not** recover absolute
source phase without a calibrated/otherwise known port phase.

Subtracting raw complex results gives

    z1 - z2 = exp(i * common_path_phase) *
              (exp(i*tau_source)-exp(i*tau_port))

and retains an unknown common rotation. A demodulated subtraction
`(z1-z2)*conj(z2)` is exactly `z1*conj(z2)-1`,
equivalent to the ratio up to a translation.

## Four diagnostic readouts at K=2/4/8

- `paired_orders`: Both raw responses, K physical complex readings,
  `K/2` source locations.
- `order_difference`: Uncorrected subtraction, K readings,
  `K/2` source locations.
- `interferometric_ratio`: `z1*conj(z2)`, K readings,
  `K/2` source locations. The remaining decoder channels are zeros
  rather than fabricated additional observations.
- `demodulated_difference`: `(z1-z2)*conj(z2)`,
  identical information to the ratio. Not an additional sensor.

Unmodified route arms and strong direct-phase control are rerun under
the same training protocol to avoid accidental reweighting bias. All
heads receive identically sized small neural readouts, but their
**nonzero information content differs**. There are two truly physical
sensor ports and they do not move. The question can select the launch
coordinate and a precommitted route.

## What constitutes evidence?

1. The analytical check must satisfy complex endpoint identity with
   maximum error under 1e-5 on actual saved checkpoint phase maps.
2. A synthetic intermediate-cell perturbation must leave the ratio
   invariant and rotate/change the raw subtraction.
3. Compare test accuracies and cross-entropy for ratio, demodulated
   subtraction, raw subtraction, paired responses and target-addressed
   direct phase for both **independent writers**, K=2/4/8.
4. If ratio improves only seed 97, report that and examine the sampled
   local source positions. If ratio surpasses raw pair, attribute that
   to inductive bias or optimization: it is a deterministic function of
   the same paired measurements, so cannot contain more information.
5. A ratio classifier **need not reach 99%** if the source phase carries
   insufficient label information or the port phase varies.
6. No causal claim about soma/dendrites, myelin, true propagating waves
   or a new mathematical theorem. The unitary toy model is not lossy
   membrane physics; interferometric common-mode rejection is known.

**Controlled constraints:** No v2 checkpoint retraining, no hidden
labels in path generation, identical scene splits, same 480-step
head-training schedule with a shared validation checkpoint.
The original v3 receipts stay intact in `results/restricted_ports/`.
All new receipts go to `results/interferometry/`.
