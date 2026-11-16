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
