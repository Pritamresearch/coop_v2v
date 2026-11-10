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
