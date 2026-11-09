# Why detection AP is near zero: findings and plan (2026-11-04)

## The problem

After 40+ epochs, `no_fusion_camfix` and `ours_full_camfix` reach validation **AP50 of about 0.001–0.004** and
recall50 of about 4%. The detector predicts about 88 boxes per frame against 10–20 real vehicles, and the
average heatmap value on background cells is about 0.4, where it should be near zero.

Already fixed and verified before this investigation: BEV grid range, decode NMS, the `pos_weight` wiring, the
camera-axis convention in `frustum_to_ego`, and the depth bin count. None of them was enough.

## What we found

All numbers below were measured on real model outputs from the epoch-40/41 checkpoints (raw data in this folder).

### 1. Eval-time changes cannot fix it
- Raising the score threshold from 0.05 to 0.70 leaves AP50 unchanged at 0.0022; above 0.70 it falls.
  AP matches detections greedily by score, so a stricter threshold can only remove detections, never improve AP.
- Heatmap smoothing, a wider peak window and a larger NMS radius reached AP50 0.0045 at best.

### 2. The heatmap is the bottleneck, not box size or heading
- Replacing each detection's centre with the ground-truth centre raises AP50 from 0.002 to 0.046.
  Replacing size and yaw with ground truth only reaches 0.008.
- There are about 2,600 local peaks above 0.2 per frame against about 21 real vehicles. A typical real vehicle
  centre outranks only 81% of background cells (ranking AUC about 0.71 on 400 validation frames).
- Separation is much better on training frames (AUC about 0.94), so the model also generalises poorly.

### 3. The heatmap loss is miscalibrated, but fixing it does not raise AP
- The loss divides the negative term by the number of negative cells (about 65,000), where CenterNet divides by
  positives. A flat heatmap of about 0.6 is then loss-optimal, matching what we saw. `pos_weight=4` pushes the
  wrong way.
- Training also used a near single-cell Gaussian target (sigma 0.13 m) while validation used 0.8 m.
- The class imbalance was 3,200:1 on validation and 6,400:1 on training, not the 285:1 assumed earlier.
- A one-epoch A/B on the same data, standard CenterNet loss versus the original:

  | | Background heat | Peaks above 0.2 per frame | Vehicle-vs-background AUC | AP50 |
  |---|---|---|---|---|
  | Original loss | 0.39 | 2,744 | 0.70 | 0.0014 |
  | CenterNet loss | 0.024 | 2.7 | 0.71 | 0.0032 (noise) |

  The patch fixes the flood of false peaks but not the ranking. Caveat: this was a one-epoch fine-tune from
  weights trained under the old loss, not a from-scratch run.

### 4. The depth supervision is wrong (leading explanation)
- The ZoeDepth labels are about **2.8 times too far** compared with LiDAR (median +12.2 m, under 1% within 1 m).
  Against ground-truth vehicle distances they are +13.1 m off, with only 10% within 3 m. LiDAR labels pass the
  same check.
- They were also written for 64 bins over 4–50 m, while training reads them as 128 bins over 4–90 m.
- The earlier "90.8% in range" check only compared range fractions, so it never measured accuracy.
- The trained model predicts depth about **2.7 times too far** (1% of pixels within 1 m), so the wrong labels
  reached the model. Features are placed at the wrong distance in the BEV map, so vehicles do not stand out.
- Swapping correct LiDAR depth into the already-trained model made detection worse, not better. This is
  expected, because its head learned the wrong geometry. It means this test cannot measure the benefit; only
  training with correct depth can. The test used 30 frames from the training split.

### 5. Data was deleted on 2026-11-02
Most LiDAR files, the download archives and nearly all test-split annotations vanished from scratch at about
15:18 GMT. Not done by Claude, not an automatic purge, cause unknown. Restored on 2026-11-03 from the official
UCLA Box share: test split loads again (2,170 frames), and train/val LiDAR is complete (6,374 of 6,374 train
frames). Full archives are kept in `datasets/OPV2V_archives` as a backup.

## What we are doing now

Training `no_fusion_lidardepth` (jobs 10657729 → 10657730 → 10657731, queued, estimated start 5 Nov about 14:50):
- LiDAR as the depth teacher for the training split only. The final model stays camera-only.
- Standard CenterNet heatmap loss, CenterPoint-style target on both train and validation, evaluation
  threshold 0.05.
- Fresh start from ImageNet weights, 120 epochs, chained 3-day jobs that resume automatically.
- Code is in commits `20fbf06` and `d3bbec0`; old settings remain the defaults.

**Decision rule.** Compare validation AP50, recall50 and the heat at objects versus background against
`no_fusion_camfix` at the same epochs (about 0.001–0.004 AP50, 4% recall50), starting from the first validation
result about 3–4 hours after the job starts.
- **If AP clearly improves:** run `ours_full` with the same settings, then the budget sweep, ablations and
  test-set evaluation (the test annotations are restored).
- **If it does not:** depth was not the main cause. Next candidates are the large train-versus-validation gap,
  the BEV lift and pooling, and augmentation. Start with the overfit test on 8 frames, then examine feature
  quality at vehicle locations.

## Open items
- The key claim, that correct depth lifts AP, is a hypothesis until the run above reports.
- Who deleted the data is unknown. Worth asking the cluster administrators; keep the archive backup.
- Fair-share priority is low after the two 3-day runs, so GPU start times are long.
