# Pre-registration: purely geometric reliability (G1, G2)

**Committed 2026-11-10 07:50:42 UTC, before any G1/G2 result was computed.** No
cross-agent test of any kind has been run: the earlier cross-agent test was conditional on the
learned head passing, and it did not (commit 1cbe056).

## Motivation

The learned head failed because depth error is predicted better by image POSITION than by content,
i.e. by a fitted, model-specific bias map. A purely geometric signal avoids that: it is computed
from poses and camera geometry alone, fits nothing, and transfers to any model or dataset.

## The two signals

Both are defined so that **higher = predicted more reliable**.

- **G1 = -(sender-to-cell range).** The Euclidean distance from the sending agent's camera origin to
  the BEV cell centre, from poses alone. Nothing is transmitted to compute it. Closer is scored
  higher.
- **G2 = -|predicted depth - flat-ground depth|.** For a pixel, intersect its camera ray with the
  local ground plane and take the depth of that intersection; compare with the model's predicted
  depth. The ground plane z is estimated per agent per frame from that agent's own LiDAR points
  (low percentile of z in the ego frame), which is self-measurable and not model-specific.
  Ray geometry uses this dataset's camera convention (x=depth, y=right, z=up).

**Secondary variant, pre-registered here so it is not post hoc: G2signed = -max(0, predicted depth -
flat-ground depth).** Rationale stated in advance: a pixel below the horizon has a maximum
geometrically possible depth, since anything beyond the ground intersection would be underground.
Predicting *nearer* than the ground plane is legitimate (the pixel is on a vehicle), so the absolute
value in G2 penalises correct above-ground content. G2signed penalises only the impossible direction.
This is reported alongside G2 and judged by the same rule, but G2 as specified is the primary.

## Primary test: cross-agent choice

For BEV cells observed by >= 2 agents, where each of those agents has a LiDAR-labelled pixel whose
TRUE 3D position falls in that cell:

- Cell identity and per-agent depth error both come from LiDAR, so the comparison is grounded in
  measurement, not in the model's own (biased) predicted position.
- For every unordered pair of agents at a cell, the score "wins" if the agent it ranks higher is the
  agent with the smaller |predicted depth - true depth|. Pairs that tie on error are discarded.
- Report **win rate** against 0.5, with 95% CIs bootstrapped over FRAMES.

**Compared against:** entropy-based choice (pick the agent with lower per-pixel depth entropy) and
random (0.5).

**PASS requires, for G1 and separately for G2:** frame-bootstrap CI lower bound **> 0.5** AND above
the entropy baseline's point estimate.

## Data

Val frames **disjoint from the 200 already used** for the learned-head test (which were
`np.linspace(0, 1979, 200)`). Only frames with >= 2 agents that each have their own LiDAR .pcd.

## Outcome

- **PASS** -> a geometric reliability term is viable; it would then be wired into the utility and
  gated as before (Spearman against magnitude_only must drop clearly below 0.977).
- **FAIL** -> option 3 stands: drop the depth term, reframe around receiver-centric conditioning and
  the fixed byte budget.

All queued jobs stay as they are either way.

---

# RESULT (2026-11-10, after the pre-registration above): **PASS for G2; G1 passes but is negligible**

347,033 agent pairs (206,394 ties dropped) from 120 val frames disjoint from the learned-head set.
Frozen `no_fusion_lidardepth` epoch-100 encoder.

## Primary (pre-registered rule: CI lower bound > 0.5 AND above entropy)

| score | win rate | 95% CI (frame bootstrap) | verdict |
|---|---|---|---|
| **G2 (−\|pred − ground\|)** | 0.744 | [0.720, 0.768] | **PASS** |
| G1 (−sender range) | 0.513 | [0.507, 0.519] | PASS, but negligible |
| G2signed (secondary) | 0.568 | [0.561, 0.576] | reported only |
| entropy (baseline) | **0.415** | [0.407, 0.423] | below chance |

## Confound check: is G2 just measuring the road?

Most LiDAR-labelled pixels are road surface, where flat-ground depth ≈ true depth, which would make
G2 a readout of the error rather than a prediction of it — and useless for vehicles. Splitting the
same pairs by height above the plane:

| score | ALL | GROUND-LIKE (<1 m) | **ELEVATED (>2 m, vehicles)** |
|---|---|---|---|
| G2 (LiDAR ground) | 0.759 | 0.867 | **0.791** [0.755, 0.821] |
| **G2p (POSE ground)** | **0.799** | 0.878 | **0.820** [0.800, 0.840] |
| G1 | 0.514 | 0.495 (chance) | 0.540 |
| entropy | 0.413 | 0.388 | 0.405 |

**The confound does not materialise**: G2 holds at 0.79–0.82 on elevated pixels. G1's entire effect
comes from elevated pixels (0.540 vs 0.495), and is too small to use either way.

## Deployability: the ground plane must not come from LiDAR

The first measurement estimated the plane from each agent's own LiDAR points, which a camera-only
system does not have at inference. **G2p** uses only the sensor height from pose/calibration against
a flat world — and scores *higher* (0.820 vs 0.791 on vehicles), probably because a low-percentile
estimate from whatever the LiDAR happened to hit is noisier than the flat-world assumption.

## Why this differs from the failures

- It **fits nothing** — no bias map, no training, no model-specific parameters — so unlike the
  learned head it cannot be a map of where this particular model fails.
- It is a **geometric consistency check**: a prediction beyond what the ground plane permits is
  suspect however confident the network is. That is precisely the measured failure mode, where the
  model is confidently wrong.

## Not yet established

1. The Spearman-vs-`magnitude_only` gate (< 0.977) — requires wiring it into the utility.
2. Tie rate is 37% of raw pairs; where those concentrate is not characterised.
3. Measured on ONE checkpoint. It should hold on the arm-B checkpoint too; a signal that only works
   for one model would be a bias map in disguise.
4. The flat-world assumption suits these CARLA towns. Real-world or hilly data would need a genuine
   ground-plane estimate, and this result should not be assumed to transfer.

---

# STEP 1 (2026-11-10): replication, ties, and the RANGE-PRIOR baseline

The range prior is the baseline the literature demands. CoCam4D (arXiv 2610.11577) weights
observations by an analytic `sigma_z^2 ~ z^4/f^2` -- reliability as a monotone function of predicted
depth alone. A positional prior already beat our learned head, so G2 has to beat a range prior.

Win rate, ties counted as 0.5 / ties dropped, 553,427 pairs, 120 frames:

| score | no_fusion ep100 | arm-B ep20 (replication) |
|---|---|---|
| **G2p (pose ground)** | 0.677 / **0.782** | 0.697 / **0.773** |
| G2 (LiDAR ground) | 0.653 / 0.744 | 0.672 / 0.738 |
| **range prior −z** | 0.329 / **0.228** | 0.400 / **0.361** |
| range prior −z² | 0.329 / 0.228 | 0.400 / 0.361 |
| G1 (−sender range) | 0.508 / 0.513 | 0.506 / 0.509 |
| entropy | 0.447 / 0.415 | 0.499 / 0.498 |

**Replicates:** G2p is 0.782 and 0.773 on two independently trained checkpoints, so it is not
checkpoint-specific and cannot be a bias map in disguise.

**The range prior is beaten decisively, and is itself BELOW chance (0.23-0.36).** Preferring the
agent with the shorter predicted depth is actively wrong here, because this model *under*-predicts
depth at range: a short prediction is often the badly-wrong one. Head-to-head, G2p and the range
prior disagree on 66-68% of pairs, and on those G2p is right **85%** (ep100) and **79%** (arm-B)
of the time. G2 is not a range prior in disguise -- flat-ground depth depends on pixel row and
camera pitch, so two pixels at the same predicted depth can score very differently.

**Ties:** 37.3% (ep100) and 28.0% (arm-B), i.e. decision coverage 62.7% / 72.0%. Scored at 0.5 they
pull G2p to 0.677 / 0.697, still far above chance.

**Ties are NOT the above-horizon artefact.** Pixels whose ray never meets the ground have an
undefined flat-ground depth, but they are only 4.5-4.7% of pairs and are *less* common among tied
pairs (2.5%) than untied ones (6.0%). No tied pair has both errors exactly zero. Ties are simply
both agents landing in the same depth bin, which 128-bin quantisation makes common.
