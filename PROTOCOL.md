# Frozen v1 design — ordered probes of byte-valued visual memory

This protocol describes *planned* comparisons. It does not assert a win.

## Task

14×16 RGB synthetic scenes contain two textured rectangular objects, each
with a 2×2 red or blue mark. The query asks whether the mark on the left or
right object is red or blue. Noise, texture, mark locations and colors vary.
A new scene supplies only its byte-valued conductance graph at inference;
the image activation itself is not available to the readers.

## Representation

The convolutional writer Eθ maps the RGB image to a directed conductance
field W with 4 planes of 14×16 uint8 values, totaling exactly 896
payload bytes per image, including padding at forbidden boundary edges.
All forward evaluations use rounded byte conductances; training uses a
straight-through estimator for those rounding decisions.

A(W) transports east and southeast; B(W) transports south and southwest.
Both operators are weighted by W. Four individual transport calls are
used to obtain AB(q) and BA(q). Single-shot contrast is AB-BA and paired
contrast is [AB,BA]. The order-blind controls are (AB+BA)/2 and
((A+B)/2)^2, each measured on the same graph. A direct-W head is included
as an upper-capacity query readout, with no transport cost.

All heads have identical learned architectures and parameter counts;
the paired head gets more input information, intentionally. Their
feature channels are padded with zeros to equalize learned parameter
counts. All arms share a jointly trained writer; this avoids fitting
the writer only for the preferred arm but does not isolate optimal
architecture performance. No arm is allowed to see uncompressed pixels
at read time. The query side is supplied explicitly to all arms.

## Data and split

For each apparatus seed, training, validation and held-out test sets
are generated from disjoint seeded draws (seed×100+1/+2/+3).
Default: 2048 train, 384 validation, 512 test scenes; 450 AdamW
updates, batch size 64, lr .001. Main reproducible commands:

    python train.py --steps 450 --seed 41 --out results/seed41.json
    python train.py --steps 450 --seed 97 --out results/seed97.json

No manual threshold tuning on test labels. Report per-arm accuracy,
cross entropy, per-seed differences, reader parameter sizes, runtime
and shuffled-across-scene conductance ablation. A score after one seed is
a pilot, not confirmation.

## Interpretation boundary

Commutator may isolate order-sensitive path differences; it cannot
carry more information than both original responses. A raw W reader
may beat every transport variant. Noncommutativity is not in itself
nonlinearity, image segmentation, an intrinsic 3D representation,
neural plasticity, lipid coding or general intelligence.

A positive first result is a held-out advantage over order-blind
equal-budget readers, paired with loss of that advantage when the
conductances are shuffled. Follow-ups must test graph corruption,
object translations, harder backgrounds, natural-image tasks, and
repetition over at least eight random seeds. Do not infer a
competitive image architecture from this synthetic proof-of-concept.
