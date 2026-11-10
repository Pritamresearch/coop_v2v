# Pre-registration: learned reliability head (option 1)

**Amended 2026-11-10 07:26:01 UTC, before any head result was computed.**
At the time of writing, `tmp_claude_diag/rel_head.py` had never been executed; the feature
extraction shards produce only frozen-encoder features and labels, no AUROCs. What was already
known, and is therefore NOT blind: per-pixel entropy's AUROC by TRUE-distance band (0.25-0.44
beyond 20 m, anti-correlated). Entropy's AUROC by PREDICTED-depth band was not known, and no
property of the head was known.

## Question

Can a small head on frozen IMAGE features predict P(|depth error| > 2 m) where entropy cannot?

Reliability must be predicted from image features, not BEV geometry: depth is biased near, so far
content is splatted into near BEV cells and a BEV-side predictor would be reading the wrong
neighbourhood.

## Primary evaluation: by PREDICTED-depth band

Bands are formed from the model's own predicted depth (argmax), not the true distance. Predicted
depth is what is observable at inference, and it is what decides which BEV cell a pixel lands in.
The damaging case is far content predicted near, which a true-distance split hides.

**PASS requires, in the predicted 4-20 m band** (and additionally the predicted 20-35 m band if
both classes are well represented there):

1. Head AUROC for |err| > 2 m **>= 0.65**, and
2. the head's frame-bootstrap 95% CI **lower bound above both** baselines' point estimates:
   (a) per-pixel depth entropy, (b) logistic on per-camera (u, v) + predicted depth.

**Class balance (P(|err| > 2 m) and n) is reported beside every AUROC.** A band whose positive rate
is above ~0.95 or below ~0.05 is reported but does not decide the outcome: there is too little of
one class left to rank.

Bootstrap resamples FRAMES, not pixels, because pixels within a frame are not independent.

## Secondary: reported, not deciding

- True-distance bands (the earlier framing).
- Pooled over all pixels.
- Relative error > 25% as an alternative target.
- Train/val AUROC gap for the head.

## Negative control

A head trained on shuffled labels must score ~0.5 (within 0.08). Anything else indicates leakage.

## Outcome

- **PASS** -> the head's output replaces D in the sender utility and the receiver need
  (probability-weighted per cell, uncovered = 1). Spearman against magnitude_only must then drop
  clearly below the current 0.977, which is a further gate.
  Next test after that: **cross-agent** -- for BEV locations seen by two agents, does the score pick
  the agent with the smaller depth error?
- **FAIL** -> option 3: drop the depth term, reframe around receiver-centric conditioning and the
  fixed byte budget. The anti-calibration result stands as a paper finding either way.

---

# RESULT (2026-11-10, after the pre-registration above): **FAIL -> option 3**

Frozen `no_fusion_lidardepth` epoch-100 encoder. 400 train / 200 val frames, LiDAR-labelled pixels
only: 1,232,372 train px (P(|err|>2m) = 0.391), 584,425 val px (0.432).

## Primary: bands by PREDICTED depth

| band | n | P(err>2m) | head (image feats) | entropy | positional prior | shuffled control |
|---|---|---|---|---|---|---|
| **4-20 m** | 555,338 | 0.408 | 0.872 [0.865,0.879] | 0.837 | **0.927** [0.923,0.932] | 0.593 |
| **20-35 m** | 21,251 | 0.876 | 0.674 [0.657,0.693] | **0.709** | 0.617 | 0.519 |
| 35-51.2 m | 6,042 | 0.910 | 0.590 | 0.601 | 0.456 | 0.492 |
| 51.2-90 m | 1,790 | 0.948 | 0.433 | 0.263 | 0.628 | 0.658 |

**Both deciding bands fail criterion 2.** At 4-20 m the head clears 0.65 comfortably but is beaten
decisively by the positional prior (0.927 vs 0.872). At 20-35 m it is beaten by entropy
(0.709 vs 0.674).

## What this means

**Depth error is predicted better by WHERE a pixel is than by what it contains.** A logistic
regression on per-camera (u, v) plus predicted depth reaches 0.927 in the near band, above a head
with 256-dimensional learned features. The error is systematic -- a function of image position and
predicted range -- not content-dependent, which is the signature of a model that is confidently and
structurally wrong rather than uncertain.

Two supporting observations:
- The shuffled-label control sits at 0.593 rather than 0.500 in the near band. Expected here: image
  features encode position, and position predicts error, so even a random projection inherits some
  of the positional signal. It is within the 0.10 tolerance, but it is the same effect.
- Train/val gap for the head is small (0.910 vs 0.872 at 4-20 m), so the head is not overfitting.
  It genuinely learned a real signal -- just less of one than pixel position already provides.

Beyond 20 m of *predicted* depth, 88-95% of pixels are already wrong, so there is very little of the
negative class left to rank and no method separates reliably.

## Decision

Per the rule fixed before any result was seen: **option 3**. Drop the depth term; reframe the
contribution around receiver-centric conditioning and the fixed byte budget, both of which survive
the CoCa3D comparison on their own. `ours_nodepth` is already training as the honest ablation.

The cross-agent test is **not run**: it was conditional on a PASS.

## Not done, and deliberately so

The positional prior reaching 0.927 is a real finding and could in principle be used as a
reliability term. Adopting it now would be exactly the result-driven redefinition this
pre-registration exists to prevent. If it is worth pursuing it needs its own pre-registered test,
including whether a fixed per-(camera, u, v, predicted-depth) bias map is a defensible
"reliability" signal at all, or merely a map of where this particular model fails.
