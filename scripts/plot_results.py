"""Regenerate the Test reliability and coverage figures.

Usage (from the repository root):
    uv run --with matplotlib python scripts/plot_results.py \
        runs/evaluation/<split>/classification.json runs/evaluation/<split>/figures
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

c = json.load(open(sys.argv[1]))
out = sys.argv[2]
s = c["selected"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
    }
)


def base(ax, title):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)


# 1. Reliability (five bins, counts annotated)
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
base(ax, f"Reliability on Test (n = {c['test_count']}), served TF-IDF model")
ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (4, 3)), label="Perfect calibration")
for key, color, name, off in (
    ("raw_reliability", ORANGE, "Raw", (8, -16)),
    ("calibrated_reliability", BLUE, "Calibrated (sigmoid, V-cal)", (-30, 11)),
):
    pts = [b for b in s[key] if b["count"]]
    x = [b["mean_confidence"] for b in pts]
    y = [b["accuracy"] for b in pts]
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURF,
        markeredgewidth=2,
        label=name,
        zorder=3,
    )
    for b in pts:
        ax.annotate(
            f"n={b['count']}",
            (b["mean_confidence"], b["accuracy"]),
            textcoords="offset points",
            xytext=off,
            fontsize=8,
            color=INK2,
        )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Mean predicted confidence (top class) per bin")
ax.set_ylabel("Accuracy in bin")
ax.legend(frameon=False, loc="upper left", labelcolor=INK)
ax.text(
    0,
    -0.17,
    "Five equal-width bins; empty bins omitted. Log loss "
    f"{s['raw_log_loss']:.3f} raw -> {s['calibrated_log_loss']:.3f} calibrated.",
    transform=ax.transAxes,
    fontsize=8,
    color=INK2,
)
fig.tight_layout()
fig.savefig(f"{out}/reliability_test.png", facecolor=SURF)
plt.close(fig)

# 2. Accepted accuracy vs coverage
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
base(ax, "Accepted accuracy vs coverage on Test (calibrated thresholds)")
curve = s["coverage_curve"]
x = [r["coverage"] for r in curve]
y = [r["accepted_accuracy"] for r in curve]
ax.plot(
    x,
    y,
    color=BLUE,
    linewidth=2,
    marker="o",
    markersize=8,
    markeredgecolor=SURF,
    markeredgewidth=2,
    zorder=3,
)
for r in curve:
    frozen = r["threshold"] == s["threshold"]
    ax.annotate(
        f"t={r['threshold']:.2f}" + (" (frozen)" if frozen else ""),
        (r["coverage"], r["accepted_accuracy"]),
        textcoords="offset points",
        xytext=(6, -14 if not frozen else 8),
        fontsize=8,
        color=INK if frozen else INK2,
        fontweight="bold" if frozen else "normal",
    )
ax.axhline(0.9, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(0.99, 0.905, "90% target (chosen on validation)", ha="right", fontsize=8, color=INK2)
ax.set_xlim(0, 1)
ax.set_ylim(0.8, 1.0)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Coverage (share of issues given an automatic category)")
ax.set_ylabel("Accuracy among accepted issues")
fig.tight_layout()
fig.savefig(f"{out}/coverage_test.png", facecolor=SURF)
plt.close(fig)
print("ok")
