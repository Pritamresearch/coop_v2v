# Pre-registration: final-round training schedule and checkpoint rule

**Written 2026-11-16T14:44:20Z (UTC). Git HEAD at writing: `43cb1086213e5fe6a815852ff3607e33831b0cbe`.**
**Not launched yet. The final round waits on the image-augmentation readout.**

## Why the schedule changes

`ego_armb60` (60-epoch schedule) reached val AP50 0.018; `no_fusion_lidardepth`
(120-epoch schedule) reached 0.036. A matched-epoch comparison
(`scripts/compare_val_curves.py`) shows the two are indistinguishable through epoch 22
(mean AP50 difference +0.0012, the 60-epoch run marginally ahead) and separate from
epoch 26 as their cosine schedules diverge: by epoch 38 the 60-epoch run is at 33% of
base LR against 81%, and by epoch 58 at 1.3% against 57%.

The 60-epoch budget was justified by the 120-epoch run plateauing around epoch 58. But
it was implemented by **rescaling** the cosine into 60 epochs, which is not the same as
**stopping** at 60. The plateau was measured in the 120-epoch LR regime; compressing the
anneal leaves a different, worse optimum. The final round reproduces the regime the
plateau was actually measured in.

## The schedule, fixed now

| setting | value | why |
|---|---|---|
| `epochs` | 120 | the cosine LR horizon (T_max), NOT the training length |
| `max_epoch` | 59 | training stops after epoch 59 -> exactly 60 epochs, 0..59 |
| `lr_warmup_epochs` | 5 | as in the 120-epoch reference run |
| `gumbel_epochs` | 60 | anneal T0->T1 across the TRAINED epochs |
| `warmup_epochs` (lambda) | 20 | bandwidth term active for epochs 20-59 |
| `save_every` | 10 | checkpoints at 0,10,...,50 plus last |

At the final trained epoch the LR is **55.2% of base** (against 1.3% under the compressed
60-epoch schedule), Gumbel T is at **T1 = 0.50**, and lambda has been active for 40 epochs.

**`gumbel_epochs` is new.** `gumbel_T` annealed over `cfg.epochs`, so `epochs=120`
would have ended training at **T = 2.77**, never reaching the final temperature, and
selection would have been evaluated in a regime it was never annealed to. The anneal
horizon is now separate from the LR horizon. Default `-1` keeps the old behaviour, so
no existing run's configuration changes meaning.

Verified by `scripts/check_final_schedule.py` against the production schedule functions:
7 checks pass, and 4 negative controls each fail as they must -- Gumbel tied to
`cfg.epochs`, the compressed 60-epoch schedule, lambda warmup past the last trained
epoch, and no early stop.

## Checkpoint rule

**The final-epoch checkpoint (epoch 59) for every method. No best-on-validation
selection.** Validation has been used for decisions throughout, so picking the best
val epoch would select on the same data twice and inflate every method unequally --
by an amount that depends on how noisy each method's val curve is.

## Applies identically to every method

`ego`, `ours`, `where2comm_cam` (Where2comm-faithful), `coca3d_cam` (CoCa3D-style),
`dense`. Same schedule, same seed policy, same checkpoint rule, same augmentation
configuration (whichever the image-aug readout selects). Any method that deviates is
reported as a deviation, not folded into the table.

## Sanity floor, declared in advance

The final ego-only run must land in **val AP50 0.025-0.036**. That is the range the
120-epoch reference run occupied over epochs 50-59 (0.027, 0.021, 0.020, 0.027, 0.036,
0.036 at epochs 50,52,54,56,58,59). **Below 0.025 means the regime was not reproduced
and the round is void** -- the fault is investigated before any result is reported, and
no number from a void round is quoted, including comparisons between methods within it.

This floor is a reproduction check on the schedule, not a hypothesis about the methods.
It applies only to the ego-only arm, where the bandwidth term is inert.

## What this does NOT change

H1 and H2 in `PREREG_COOPERATION.md` stand exactly as written. They compare
`ours_nodepth60` against `ego_armb60`, which share a schedule, a seed and a depth loss;
that comparison is internally matched and is unaffected by the schedule being wrong in
absolute terms. The final round produces the paper's numbers; it does not re-decide H1/H2.

The held-out test rule also stands: one evaluation, after every choice is frozen,
`COOP_TEST_FREEZE=1`. The final round is trained and selected without touching test.

## Open item, stated rather than hidden

`warmup_epochs=20` for the lambda bandwidth term is NOT what the 120-epoch reference
used (40). It is scaled to the 60 trained epochs instead, giving 40 epochs of bandwidth
pressure rather than 20. This cannot affect the sanity floor, which is measured on the
ego-only arm where lambda is inert, but it is a deliberate deviation from the reference
regime and is recorded as one.

---

# AMENDMENT 1 — sanity floor and checkpoint reporting

**Written 2026-11-16T14:51:08Z (UTC). Git HEAD at writing: `623b4618d1e93165c14f34c1af1070c735bf6e12`.**
**Legitimate: no final-round run has been launched and no final-round result exists.**
The original text above is left unedited; this section supersedes the two items named.

## A1.1 Sanity floor: 0.013, not 0.025

The original floor of 0.025 was wrong, and in the direction that matters: the
reference run's own epochs 50-59 include 0.020 and 0.021, so a final run statistically
identical to the reference could have been declared void.

Recomputed from the reference log. **`val_every=2`, so epochs 50-59 contain only five
distinct validation events** — the odd epochs are carried-forward duplicates, not
measurements, and averaging all ten would understate the spread:

| epoch | 50 | 52 | 54 | 56 | 58 |
|---|---|---|---|---|---|
| val AP50 | 0.027 | 0.021 | 0.020 | 0.027 | 0.036 |

    mean = 0.0262    sd (sample, ddof=1) = 0.0064    sd (population) = 0.0057
    mean - 2 sd = 0.0134 (sample)  /  0.0148 (population)

**Registered floor: val AP50 >= 0.013** for the final ego-only run — mean minus two
sample standard deviations, rounded down. The sample sd is used because five points is
a small sample and it is the honest estimator of the underlying spread; rounding down
errs toward not voiding a sound run.

(The 0.0056 sd in the proposal matches the population sd exactly. The 0.017 figure
implies a mean of 0.0282; the measured mean is 0.0262, which is why the floor lands at
0.013 rather than 0.017.)

**The void-round rule is unchanged.** Below 0.013 the regime was not reproduced, the
round is void, the cause is investigated before anything is reported, and no number from
a void round is quoted — including comparisons between methods inside it.

## A1.2 Reporting: mean +/- sd over epochs 55-59, not one checkpoint

Single-epoch AP is noisy at this LR. In the reference window the five distinct
validations span 0.020 to 0.036 — a half-range of +/-0.008, matching the proposal's
estimate. Reporting one final checkpoint would make every method's headline number a
draw from that spread.

**Registered rule, for every method:**

- Train with `save_all_from=55`, so epochs 55, 56, 57, 58 and 59 are all checkpointed
  (verified: the production save rule keeps 10, 20, 30, 40, 50, 55, 56, 57, 58, 59).
- Report **mean +/- sd over those five checkpoints**, for validation and for the single
  frozen test evaluation.
- **This is not selection.** All five are reported; none is chosen. The window is fixed
  in advance at the last five trained epochs and does not depend on any observed value.
- Evaluating five checkpoints on test is still one test event: the five passes happen
  together, after the freeze, with no choice made between them and no return visit. The
  test rule (one evaluation, `COOP_TEST_FREEZE=1`) is unchanged.
- The same code path computes val and test numbers, so the two are comparable.
- sd is the sample sd over the five checkpoints, reported alongside every mean. Two
  methods whose intervals overlap are not claimed to differ on that metric.

`save_all_from` is new in `src/config.py`; the default `-1` preserves existing
behaviour. `scripts/check_final_schedule.py` now asserts all five reporting epochs are
checkpointed, with "save_all_from off" added as a fifth negative control. 8 checks pass,
5 controls fail as required.

Still to implement before launch: one evaluation script taking a checkpoint list and a
split, so the val and test numbers come from the identical path. It will be gated before
use, and the sanity floor above is checked on its validation output.

## Unchanged by this amendment

H1 and H2 in `PREREG_COOPERATION.md`; the schedule in the original section above
(epochs=120, max_epoch=59, gumbel_epochs=60, warmup_epochs=20, lr_warmup_epochs=5);
the held-out test protocol; the requirement that the final round not launch until the
image-augmentation readout has decided the augmentation configuration.

## A1.3 Clarification — how the floor is to be read (2026-11-16T14:56:16Z)

The registered value 0.013 is unchanged. It is a **broken-run detector**, not a quality
bar: below it, the training regime was not reproduced and the round is void. It is set
two sample sd below the reference mean precisely so that a run statistically identical
to the reference cannot trip it.

**Quality is judged against the reference band, mean ~0.026** (reference epochs 50-58:
0.027, 0.021, 0.020, 0.027, 0.036). A final ego-only run that clears 0.013 but sits far
below 0.026 has not failed the void check and must not be reported as sound on that
basis — it is a run whose regime reproduced badly, and the gap is reported as such.

So the two numbers answer different questions: 0.013 asks "did this round run at all?",
and the reference band asks "did it run as well as the regime it was copied from?".

---

# AMENDMENT 2 — cooperative methods train under the cell budget

**Written 2026-11-16T17:29:12Z (UTC). Git HEAD at writing: `49543add337043efba157c963447c031280fc619`.**
**Not launched.** The final round still waits on the augmentation decision, which the
epoch-20 readout has now made: no image augmentation.

## The mismatch this fixes

Measured on real frames (`scripts/check_train_mask.py`): the training path applies
top-k over the AGENT axis only. The cell budget lives solely in `hard_topk`, which runs
only at `hard=True`. So training transmits `k*H*W` pairs — every cell, for its top-k
agents — against an inference budget of `n_keep*k`:

| frame | agents | train sum(M) | vs n_keep*k = 52428 | eval sum(M) |
|---|---|---|---|---|
| 0 | 5 | 131072 | 2.50x | 52428 (1.00x) |
| 395 | 2 | 65536 | 1.25x | 26214 (0.50x) |

The forward mask is already straight-through (fraction of entries neither 0 nor 1:
0.000000), so features do NOT leak in at fractional weights. The problem is the missing
budget, not a soft mask.

## Registered

Every cooperative method in the final round — ego and `dense` excepted, the first
because it transmits nothing and the second because transmitting everything is its
definition — trains with the cell budget enforced in the forward pass:

- `train_budget=True`: exactly `n_keep` cells forward, gradient through a
  temperature-sigmoid of each cell's margin to the cut, so cells just inside and just
  outside the budget both receive signal.
- `train_budget_range=(0.1, 0.6)`: `budget_frac` sampled per iteration from the range
  we report, so **one model per method yields the whole rate-accuracy curve** instead of
  one model per operating point.
- Ties broken by ascending cell index, as in the inference path.

## Gate

`scripts/check_train_budget.py`, on real `ours_nodepth60` val frames:

    budget_frac=0.1   n_keep= 6553  cells sent  6553- 6553  non-binary 0.000000  EXACT
    budget_frac=0.25  n_keep=16384  cells sent 16384-16384  non-binary 0.000000  EXACT
    budget_frac=0.4   n_keep=26214  cells sent 26214-26214  non-binary 0.000000  EXACT
    budget_frac=0.6   n_keep=39321  cells sent 39321-39321  non-binary 0.000000  EXACT
    gradients: 92 parameters receive non-zero gradient

Negative control, the current no-budget path: 65536 cells sent regardless of the
requested fraction — MISMATCH at every fraction, as required.

## Note for the write-up

`ours_nodepth60` was trained WITHOUT the budget and evaluated WITH it. H1 and H2 are
therefore tested under a train/test mismatch that disadvantages the cooperative arm: the
model never saw the sparsity it is scored at. The passing H1 result is a lower bound in
that respect, and the final round removes the mismatch rather than inheriting it.

This does not retroactively change H1/H2, which are decided on the checkpoints as they
exist under `PREREG_COOPERATION.md` Amendment 2.
