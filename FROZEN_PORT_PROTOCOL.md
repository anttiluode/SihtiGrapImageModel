# Frozen Writer Restricted-Port Gate (v3 pilot)

Experiment date: 2026-10-14. **No writer updates**, no retuning paths to observed labels. This is the first test after the frozen v2 checkpoint visual inspection.

## Hypothesis and pre-declared pass gate

Does an *explicit question-selected probe launch/path* collect information more efficiently from a frozen 64-byte visual memory than passive alternatives when only two fixed readout ports are available?

- **Strong gain** requires `active_phase_then_shift` beating *both* `fixed_routes` and `direct_local_phase` on both independently trained writer seeds at the same measurement budget. Direct local is stronger than a fixed-port observer because it reads target-addressed sites directly.
- **Order-specific gain** further requires a reproducible gain over reversed order `active_shift_then_phase` and over the raw `paired_orders` read at the same physical reading budget. Paired outputs contain the differences, so stronger information claims are impossible.
- If active beats only fixed routes or blind ports, call it an *access/routing advantage*, **not** evidence of noncommutativity.
- If all arms tie or differences swap between seeds, the result is inconclusive; retain all numbers.

## Frozen resources

Checkpoints: `checkpoints/phase_wake_seed41.pt` and `phase_wake_seed97.pt`, trained v2 writers. The 32×32 synthetic generator is unchanged. Each 64-byte phase map is quantized on the original 8×8 grid, fixed through all queries. A test scene has four independently colored marks; *every* quadrant is queried per scene. All train, validation and test scenes are disjoint, with shared scene seeds across both independent writers. Default split is 1024/256/512 scenes, resulting in 4096/1024/2048 questions.

## Wave-path instrumentation

A **single shot** is a unit-amplitude complex wave injected at a question-dependent grid site; it undergoes exactly twelve unitary local phase multiplications and cardinal periodic shifts, arrives at either (0,0) or (0,4), and produces one complex scalar measurement there. The path is selected by the question and a fixed geometric schedule chosen without seeing images or labels. It is not adapted online to previous measurements.

- Fixed physical ports: `(0,0)`, `(0,4)`
- Complex readings per question: `K ∈ {2,4,8}`
- Every classifier receives exactly `2K + 4` real floats (K complex slots plus a four-element explicit question encoding), and has identical architecture and weights count at a given K. Zero padding is not an extra measurement.
- The analytic phase-sum along the path is numerically verified against explicit full 12-step complex-wave evolution in unit tests.
- **Allowed privilege:** only the question-selected source and unitary shift path can vary with the question. Detector ports do not.
- Paths are periodic 8×8 grid walks, not a numerical ray tracer, eikonal solver, or biological conduction model. Wave energy is conserved trivially by construction.

## Eight arms, fixed budget accounting

| Arm | Physical readings | Source access |
| --- | ---: | --- |
| active_phase_then_shift | K end-port waves | question-selected route, phase-then-shift |
| active_shift_then_phase | K end-port waves | same route, reverse local order |
| paired_orders | K/2 route pairs (K port readings) | both orders, retains all raw data |
| order_difference | K/2 route pairs (K port readings) | subtraction of those same port readings |
| fixed_routes | K end-port waves | one bank of routes **independent of question** |
| direct_local_phase | K direct phase samples | addressable **target** source; stronger nonphysical interface |
| direct_local_amplitude | K direct scalar samples | same access, phase represented as invertible real amplitude |
| passive_ports | K static phases at repeated two fixed ports | no transport; intentionally blind |

The direct-addressable controls violate strict fixed-port-only measurement access; they serve as an explicit **stronger interface** to challenge the hypothesis. The fixed-route arm uses the same wave apparatus and observation budget as active.

Readouts are learned; writers remain frozen. Independent classifiers are trained with the same optimizer/schedule, and joint validation loss selects a common checkpoint epoch within each K. No results are selected to favor a preferred arm.

## Interpretations that would be overclaims

- The medium already learned a *local supervised code* in v2; reaching it by a question-dependent path could be useful even if no new information is created by order.
- The learned phase writer has not been shown to encode propagation speed, causal history, natural-image geometry or cortical lipids.
- The paths are precomputed from question identity and grid geometry. This is **question-conditioned** probing, not feedback-adaptive closed-loop experiment design.
- A gain here does not demonstrate active sensing under equal sensor-placement freedoms without further controls.
- Higher accuracy for an ordered probe cannot mean it contains more information than the same-budget raw paired observations. If the raw pair fails, consider optimization/decoder limitations rather than information generation.

Run the actual saved .pt weights on GitHub Actions in `.github/workflows/frozen-port-gate.yml`, save unchanged JSON receipts in `results/restricted_ports/`.
