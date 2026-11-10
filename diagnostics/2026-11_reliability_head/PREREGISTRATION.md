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
