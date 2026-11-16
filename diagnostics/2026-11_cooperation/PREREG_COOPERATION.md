# Pre-registration: does cooperation help, and does it help *where it should*?

**Written 2026-11-16T14:19:42Z (UTC).**
**Git HEAD at writing: `2bfb67996bcd02125626789015cc03c40c0113be`.**

## What had and had not been seen when this was written

**Seen:** the training-log validation curves for both 60-epoch runs. `ours_nodepth60`
best val AP@0.5 = 0.052 (epochs 28-59 range 0.037-0.052); `ego_armb60` best = 0.018
(range 0.008-0.018). Those two ranges do not overlap. This is why the claim below is
worth testing, and it is also why the primary test is NOT overall AP: overall AP is
already known to favour the cooperative model, so re-testing it proves nothing.

**NOT seen:** any per-band result, any recoverable/not-recoverable subset result, and
any byte measurement from the fixed accounting. At the time of writing
`fc60_shard*.pkl` did not exist (0 files), no merge had been run, and
`bytes_ours_nodepth60.json` did not exist. Jobs 11022580 (bands) and 11023525 (shards)
were still running; 11022579 (image-aug readout) was still running and is governed by
its own earlier pre-registration in `scripts/imgaug_readout.py`.

## Models compared

Both at their final epoch (`last.pth`, epoch 59 of 60), identical seed, LR schedule,
heatmap loss, depth supervision and depth-loss weighting. The only difference is
cooperation.

| | run | fusion |
|---|---|---|
| ours | `ours_nodepth60` | cooperative, budget_frac 0.4, utility gate on |
| ego | `ego_armb60` | ego-only |

Evaluated on the same val frames, unaugmented, via `scripts/fusion_checks.py` (6 shards,
300 frames) merged by `scripts/fusion_checks_merge.py`.

## Definitions, fixed now

- **far**: GT vehicle range from ego > 20 m.
- **recoverable**: within 20 m of a collaborator AND projecting into one of that
  collaborator's four cameras (in image, forward depth within [d_min, d_max]).
  Occlusion is ignored, so "recoverable" is an upper bound on what cooperation could
  recover. Computed from the agents' world poses in their yaml files, independently of
  `rel_pose`.
- **centre AP@2m**: all-points AP with a detection matched to a GT centre within 2 m,
  greedy by descending score. Chosen over IoU-based AP because at these AP levels IoU
  matching is dominated by box-regression noise; centre distance isolates "did it find
  the vehicle".
- **CI**: 95% percentile interval, 1000 bootstrap resamples **over frames**, paired —
  the same resampled frame indices are used for every quantity in a given resample.

## Hypotheses

**H1 (primary).** On far-recoverable vehicles, `ours_nodepth60` beats `ego_armb60` on
centre AP@2m, and the frame-bootstrap CI of the difference excludes 0.

**H2 (specificity).** The cooperative gain on far-recoverable vehicles exceeds the gain
on far-not-recoverable vehicles:

    (ours_rec - ego_rec) - (ours_nonrec - ego_nonrec) > 0,  CI excluding 0.

H2 is the load-bearing test. H1 alone is consistent with the cooperative model simply
being a better detector everywhere — more capacity, more training signal, anything. Only
H2 says the gain is located where a collaborator can actually see the vehicle, which is
the mechanism we claim.

Note this supersedes the criterion hardcoded in `fusion_checks_merge.py` on 2026-11-05,
which compared `full_rec` against `full_nonrec` **within the cooperative model only**.
That is not a cooperation test: recoverable vehicles may be easier for both models (they
are near *some* agent), so that comparison can pass with no cooperative benefit at all.
Its second clause also thresholded against 0.012 from an epoch-20 ego-only run, which is
not the comparison we are now making. The old numbers are still printed for continuity
and are explicitly marked superseded.

## Decision rule

| H1 | H2 | Conclusion |
|---|---|---|
| met | met | **PASS.** Cooperation helps, and it helps where collaborators can see. |
| met | not met | **FAIL as a cooperation claim.** Report as a general detector difference, not a cooperation mechanism. Do not claim the mechanism. |
| not met | — | **FAIL.** No cooperative benefit on the subset where it was predicted. |

## Reported but NOT deciding

Per-band AP50 / AP30 / centre AP@2m (0-20, 20-35, 35-51.2, >51.2 m) and measured bytes
under the fixed accounting are **descriptive**. They do not enter the decision, in either
direction. In particular a favourable per-band table does not rescue a failed H2, and the
byte figure is a cost to report alongside the result, not evidence for it.

Negative controls already available: `ego_armb60` transmits 0 bytes by construction, and
the byte cross-check (production `bytes_kb_from_out` vs the model's own `soft_B`) must
agree on every frame or the measurement is void.

## Known limitations, stated in advance

- One seed per arm. We cannot separate the cooperative effect from seed variance. The
  non-overlapping val AP ranges across ~16 validation points make pure seed noise
  unlikely, but a second seed is the honest fix and is not being claimed here.
- Occlusion is ignored in the recoverable definition, so H2's effect size is diluted
  toward zero: some "recoverable" vehicles are not actually visible. This biases against
  H2, not toward it.
- AP is low in absolute terms and our evaluation range is not comparable to published
  OPV2V camera numbers. Only within-experiment contrasts are claimed.

## Standing rule, adopted with this pre-registration

**Validation has been used for decisions** — the 60-epoch budget came from a val plateau,
the image-aug A/B is judged on val, the utility threshold will be calibrated on val, and
the depth A/B arm was chosen on val. Validation is therefore no longer an unbiased
estimate of generalisation for any of these methods.

**Final paper numbers for every method come from a single evaluation on the held-out test
split, run once, after every choice is frozen.** Nothing is tuned, selected or
early-stopped on test. No test evaluation happens before the freeze: not for a sanity
check, not for one method, not "just to see". Until then test stays untouched.
