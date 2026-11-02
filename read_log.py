#!/usr/bin/env python
import sys, csv

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/logs/ours_full/train_log.csv"
    cols = ["epoch", "loss", "loss_det", "loss_hm",
            "ap50", "ap70", "recall50", "precision50",
            "n_gt", "n_pred", "tp50", "fp50",
            "heat_max", "heat_mean", "heat_pos_mean", "heat_neg_mean",
            "val_loss_det", "n_peaks_mean", "score_max", "score_mean"]

    with open(path) as f:
        r = csv.DictReader(f)
        header = r.fieldnames
        missing = [c for c in cols if c not in header]
        if missing:
            print(f"WARNING: these columns are not in the CSV header: {missing}")
        present = [c for c in cols if c in header]
        rows = list(r)

    print(f"{path}  ({len(rows)} epochs logged)\n")
    print("  ".join(f"{c:>13s}" for c in present))
    idxs = sorted(set(list(range(0, len(rows), 5)) + list(range(max(0, len(rows)-5), len(rows)))))
    for i in idxs:
        row = rows[i]
        vals = []
        for c in present:
            v = row.get(c, "")
            try:
                v = f"{float(v):.5f}"
            except ValueError:
                pass
            vals.append(f"{v:>13s}")
        print("  ".join(vals))

    if "ap70" in present:
        best_i = max(range(len(rows)), key=lambda i: float(rows[i].get("ap70", 0) or 0))
        print(f"\nBEST ap70 = {rows[best_i]['ap70']} at epoch {rows[best_i]['epoch']}")
    if "heat_pos_mean" in present and "heat_neg_mean" in present:
        first, last = rows[0], rows[-1]
        print(f"heat_pos_mean: epoch0={first.get('heat_pos_mean')}  last={last.get('heat_pos_mean')}")
        print(f"heat_neg_mean: epoch0={first.get('heat_neg_mean')}  last={last.get('heat_neg_mean')}")

if __name__ == "__main__":
    main()
