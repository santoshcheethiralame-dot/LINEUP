"""Fig 1 — the LINEUP framework: pipeline (top) + the 2x2 role grid (bottom). Pure design, no data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures" / "fig1_framework.png"

CULPRIT, MISLEADING, SILENT, INERT = "#2a9d8f", "#e9c46a", "#a8c7e0", "#d9d9d9"
BOX = "#eef2f6"
EDGE = "#33475b"


def box(ax, x, y, w, h, text, fc=BOX, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.4, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#11202e", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=EDGE))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # ---- top: pipeline ----
    ax.text(0.1, 5.95, "a) Pipeline", fontsize=12, fontweight="bold", color=EDGE)
    yb, h = 4.55, 1.05
    stages = [
        (0.15, 1.95, "Question +\n6 retrieved passages\n(gold · near-miss · decoys)"),
        (2.40, 1.55, "LLM answers\n— a wrong answer"),
        (4.30, 1.95, "Leave-one-out oracle\nremove each passage,\nre-ask the model"),
        (6.65, 1.75, "Assign every passage\nits 2×2 role", ),
        (8.75, 2.05, "Score attribution\nmethods against\nthe ground-truth roles"),
    ]
    centers = []
    for s in stages:
        x, w, txt = s[0], s[1], s[2]
        fc = "#fdf6e3" if "oracle" in txt else BOX
        box(ax, x, yb, w, h, txt, fc=fc, fs=9.5, bold=("oracle" in txt))
        centers.append((x, x + w))
    for i in range(len(stages) - 1):
        arrow(ax, centers[i][1] + 0.02, yb + h / 2, centers[i + 1][0] - 0.02, yb + h / 2)

    # ---- bottom: 2x2 grid ----
    ax.text(0.1, 3.75, "b) The two axes define four passage roles", fontsize=12, fontweight="bold", color=EDGE)
    gx, gy, cw, ch = 3.05, 0.55, 2.95, 1.35  # grid origin and cell size
    # x-axis = salient (right), y-axis = causal (up):
    #   top-right = causal+salient = CULPRIT; top-left = causal only = SILENT;
    #   bottom-right = salient only = MISLEADING; bottom-left = neither = INERT
    cells = [
        (1, 1, "CULPRIT", "causal + salient", CULPRIT),
        (0, 1, "SILENT", "causal, not salient", SILENT),
        (1, 0, "MISLEADING", "salient, not causal\n(the salience trap)", MISLEADING),
        (0, 0, "INERT", "neither", INERT),
    ]
    for col, rowy, name, sub, color in cells:
        x = gx + col * cw
        y = gy + rowy * ch
        ax.add_patch(FancyBboxPatch((x, y), cw - 0.08, ch - 0.08, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.4, edgecolor=EDGE, facecolor=color))
        ax.text(x + (cw - 0.08) / 2, y + (ch - 0.08) * 0.62, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#11202e")
        ax.text(x + (cw - 0.08) / 2, y + (ch - 0.08) * 0.26, sub, ha="center", va="center",
                fontsize=9, color="#11202e")

    # axis labels
    ax.text(gx - 0.30, gy + ch, "causal\n(removing it\nchanges the answer)", ha="right", va="center",
            fontsize=9.5, color=EDGE)
    ax.text(gx - 0.30, gy + ch * 0.0 + ch / 2 - ch, "", ha="right", va="center")
    ax.annotate("", xy=(gx - 0.18, gy + 2 * ch - 0.05), xytext=(gx - 0.18, gy + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.4))
    ax.text(gx + cw, gy + 2 * ch + 0.14, "salient (states the wrong value)", ha="center", va="bottom",
            fontsize=9.5, color=EDGE)
    ax.annotate("", xy=(gx + 2 * cw - 0.05, gy + 2 * ch + 0.02), xytext=(gx + 0.05, gy + 2 * ch + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.4))

    # headline callout
    ax.text(gx + 2 * cw + 0.45, gy + ch, "no CULPRIT\n→ ill-posed\n(~1/3 of errors)",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color=CULPRIT)

    fig.tight_layout()
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
