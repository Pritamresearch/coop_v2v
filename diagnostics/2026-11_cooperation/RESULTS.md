# Results — cooperation pre-registration (H1, H2)

Recorded 2026-11-16T17:32:52Z. Decided on the FIXED `hard_topk` per Amendment 2.
300 val frames, centre AP@2m, frame-bootstrap 95% CIs, paired resampling.

## Verdict: PASS on both hypotheses

| quantity | fixed code | 95% CI | superseded (old code) |
|---|---|---|---|
| ours, far-recoverable | 0.1027 | [0.0850, 0.1247] | 0.1024 [0.0847, 0.1240] |
| ego, far-recoverable | 0.0126 | [0.0089, 0.0171] | 0.0126 [0.0089, 0.0171] |
| ours, far-NOT-recoverable | 0.0359 | [0.0291, 0.0455] | 0.0358 [0.0290, 0.0454] |
| ego, far-NOT-recoverable | 0.0265 | [0.0224, 0.0328] | 0.0265 [0.0224, 0.0328] |
| **H1: ours - ego, recoverable** | **+0.090** | **[+0.0745, +0.1094]** | [+0.0742, +0.1089] |
| ours - ego, not recoverable | +0.009 | [+0.0033, +0.0162] | [+0.0031, +0.0162] |
| **H2: gain_rec - gain_nonrec** | **+0.081** | **[+0.0665, +0.0980]** | [+0.0662, +0.0980] |

**H1 MET** (CI excludes 0). **H2 MET** (CI excludes 0). Both criteria required; both hold.

The cooperative gain is about **ten times larger** where a collaborator can actually see
the vehicle (+0.090 vs +0.009). Ego-only scores WORSE on far-recoverable (0.0126) than on
far-not-recoverable (0.0265): recoverable vehicles are precisely the ones ego fails on,
so cooperation is not harvesting easy targets.

Old and fixed differ in the fourth decimal, as the 40-frame budget check predicted
(0 of 40 frames differed). Cells sent: 45700 fixed vs 47014 old.

## Per-band detection, 200 shared val frames

| band | GT | AP50 ours / ego | AP30 ours / ego | centre AP@2m ours / ego |
|---|---|---|---|---|
| all | 4154 | 0.0444 / 0.0137 | 0.1662 / 0.0902 | 0.2337 / 0.1350 |
| 0-20 m | 934 | 0.1271 / 0.0489 | 0.4290 / 0.3118 | 0.5808 / 0.4338 |
| 20-35 m | 951 | 0.0397 / 0.0098 | 0.1458 / 0.0499 | 0.2041 / 0.0935 |
| 35-51.2 m | 1629 | 0.0127 / 0.0005 | 0.0595 / 0.0051 | 0.0949 / 0.0122 |
| >51.2 m | 640 | 0.0014 / 0.0001 | 0.0106 / 0.0016 | 0.0156 / 0.0041 |

Report ABSOLUTE differences with CIs, never far-range ratios: ego is 0.0005 at 35-51.2 m,
so a "25x" is a division artefact. Depth quality is nearly identical between the two
models (expected-depth ratio 1.372 vs 1.266 over 4-90 m), so the gain is not better ego
depth.

## Bandwidth — NOT fit to report as a system result

6124.7 KB/frame = **501.7 Mbps at 10 Hz**, against a <=2 Mbps design target: **251x over**.

| component | KB/frame | share |
|---|---|---|
| feature payload (45874 pairs x 128 B) | 5734.3 | 93.6% |
| dense control maps x4 senders | 256.0 | 4.2% |
| cell indices | 89.6 | 1.5% |
| per-cell utility | 44.8 | 0.7% |

The control maps alone are 10.5x a 25 KB frame budget before one feature is sent. H1/H2
are detection results and stand; the operating point they were measured at is not a
deployable one, and no bandwidth claim is made from these runs.

## Caveats that must appear in the write-up

1. Both runs used the compressed 60-epoch schedule, which the matched-epoch comparison
   showed costs accuracy. Lower bounds on both arms.
2. `ours_nodepth60` trained WITHOUT the cell budget, evaluated WITH it. The mismatch
   disadvantages the cooperative arm, so H1 passes under a handicap.
3. One seed per arm; seed variance is not separated from the effect.
4. Occlusion is ignored in "recoverable", which dilutes H2 toward zero.
5. The budget-1.0/gate-off arm is NOT evidence for selection: it changes the budget and
   the gate at once. A budget sweep at fixed gating is the honest test.
6. Image augmentation: the pre-registered FAIL stands, but its arms' depth labels were
   misaligned, so it is not evidence about augmentation itself.
