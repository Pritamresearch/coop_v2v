"""
All figures. Reads ONLY from results/ so the whole paper regenerates from
saved CSVs.

Figures:
    fig_accuracy_bandwidth.pdf   -- AP vs KB/frame, ours vs baselines
    fig_oof_recall.pdf           -- out-of-FoV recall vs KB/frame
    fig_ablation_topk_gate.pdf   -- A1 decisive ablation
    fig_ablation_utility.pdf     -- A2 utility variants (bar)
    fig_robustness.pdf           -- three-panel pose / latency / agents
"""
from __future__ import annotations
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


FIG_DIR = "plots/figures"
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({
    "font.size": 11,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.autolayout": False,
})
PAL = plt.cm.tab10.colors


def combine_curves(root="results", out_name="main.csv"):
    """Merge every results/curves/<exp>.csv (with method column) into main.csv."""
    files = glob.glob(f"{root}/curves/*.csv")
    files = [f for f in files if not f.endswith(f"/{out_name}")]
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "method" not in df.columns:
            df["method"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    path = f"{root}/curves/{out_name}"
    out.to_csv(path, index=False)
    return path


def plot_accuracy_bandwidth(root="results"):
    p = f"{root}/curves/main.csv"
    if not os.path.exists(p):
        print(f"skip accuracy-bandwidth: {p} missing"); return
    df = pd.read_csv(p)
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (m, g) in enumerate(df.groupby("method")):
        g = g.sort_values("kb")
        lw = 2.4 if m == "ours_full" else 1.4
        ax.plot(g["kb"].clip(lower=0.1), g["ap70"], marker="o", ms=5,
                label=m, color=PAL[i % len(PAL)], linewidth=lw)
    ax.set_xscale("log")
    ax.set_xlabel("Communication volume (KB/frame, log)")
    ax.set_ylabel("AP@0.7")
    ax.set_title("Accuracy vs. bandwidth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.savefig(f"{FIG_DIR}/fig_accuracy_bandwidth.pdf")
    fig.savefig(f"{FIG_DIR}/fig_accuracy_bandwidth.png")
    plt.close(fig)


def plot_oof_recall(root="results"):
    p = f"{root}/curves/main.csv"
    if not os.path.exists(p):
        print(f"skip OOF recall: {p} missing"); return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (m, g) in enumerate(df.groupby("method")):
        g = g.sort_values("kb")
        lw = 2.4 if m == "ours_full" else 1.4
        ax.plot(g["kb"].clip(lower=0.1), g["oof_recall"], marker="s", ms=5,
                label=m, color=PAL[i % len(PAL)], linewidth=lw)
    ax.set_xscale("log")
    ax.set_xlabel("Communication volume (KB/frame, log)")
    ax.set_ylabel("Out-of-FoV recall")
    ax.set_title("Cooperation payoff: objects the ego cannot see alone")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.savefig(f"{FIG_DIR}/fig_oof_recall.pdf")
    fig.savefig(f"{FIG_DIR}/fig_oof_recall.png")
    plt.close(fig)


def plot_ablation_topk_gate(root="results"):
    p = f"{root}/ablation/A1_topk_gate.csv"
    if not os.path.exists(p):
        print(f"skip A1: {p} missing"); return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for gate, sub in df.groupby("gate"):
        sub = sub.sort_values("k")
        color = "#0E7C86" if gate == 1 else "#D9533B"
        label = "with gating" if gate == 1 else "without gating"
        ax.plot(sub["k"], sub["ap70"], marker="o", ms=7,
                label=label, color=color, linewidth=2.0)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("k (agents per cell)")
    ax.set_ylabel("AP@0.7")
    ax.set_title("A1: does gating reverse 'more-is-worse'?")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(f"{FIG_DIR}/fig_ablation_topk_gate.pdf")
    fig.savefig(f"{FIG_DIR}/fig_ablation_topk_gate.png")
    plt.close(fig)


def plot_ablation_utility(root="results"):
    p = f"{root}/ablation/A2_utility.csv"
    if not os.path.exists(p):
        print(f"skip A2: {p} missing"); return
    df = pd.read_csv(p)
    order = ["magnitude_only", "confidence_only", "learned_fue", "depth_aware"]
    df = df.set_index("utility_type").reindex([u for u in order if u in df["utility_type"].values
                                               or u in df.index]).reset_index()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = range(len(df))
    ax.bar(x, df["ap70"], color=["#8899AA", "#5D8AA8", "#f4a261", "#0E7C86"][:len(df)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["utility_type"], rotation=15)
    ax.set_ylabel("AP@0.7")
    ax.set_title("A2: utility variants at matched bandwidth")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(f"{FIG_DIR}/fig_ablation_utility.pdf")
    fig.savefig(f"{FIG_DIR}/fig_ablation_utility.png")
    plt.close(fig)


def plot_robustness(root="results"):
    files = glob.glob(f"{root}/robustness/*.csv")
    if not files:
        print("skip robustness: no CSVs"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.6))
    for ax, (axis, xl) in zip(axs, [
        ("pose",        "pose noise sigma (m)"),
        ("latency",     "delay (ms)"),
        ("packet_loss", "packet loss probability"),
        ("agents",      "# collaborators"),
    ]):
        sub = df[df["axis"] == axis].sort_values("x")
        if len(sub) == 0:
            ax.set_visible(False); continue
        ax.plot(sub["x"], sub["ap70"], marker="o", color="#0E7C86", linewidth=2, label="AP@0.7")
        ax.plot(sub["x"], sub["oof_recall"], marker="s", color="#D9533B", linewidth=1.5, label="OOF recall")
        ax.set_xlabel(xl); ax.set_ylabel("score")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Robustness sweeps")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_robustness.pdf")
    fig.savefig(f"{FIG_DIR}/fig_robustness.png")
    plt.close(fig)


def make_all(root: str = "results"):
    combine_curves(root)
    plot_accuracy_bandwidth(root)
    plot_oof_recall(root)
    plot_ablation_topk_gate(root)
    plot_ablation_utility(root)
    plot_robustness(root)
    print(f"figures written to {FIG_DIR}/")


if __name__ == "__main__":
    make_all()
