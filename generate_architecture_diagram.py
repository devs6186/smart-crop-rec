"""
Architecture diagram v2 — landscape, glow effects, compact & creative.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis("off")
BG = "#080d18"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Colour palette ──────────────────────────────────────
TEAL   = "#00e5c3"
BLUE   = "#3d9eff"
PINK   = "#ff4fa3"
PURPLE = "#b06fff"
ORANGE = "#ff9f0a"
GREEN  = "#30d158"
GOLD   = "#ffd60a"
GRAY   = "#3a3f4b"
TEXT   = "#dde3ee"
MUTED  = "#5a6478"

# ── Subtle dot-grid background ──────────────────────────
for gx in np.arange(0.5, 22, 0.9):
    for gy in np.arange(0.5, 13, 0.9):
        ax.plot(gx, gy, ".", color="#1a2035", markersize=1.8, alpha=0.6)

# ── Decorative glow ring behind Predictor ───────────────
for r_, a_ in [(3.0, 0.03), (2.5, 0.05), (1.9, 0.07)]:
    c = Circle((13.45, 9.95), r_, color=TEAL, alpha=a_)
    ax.add_patch(c)

# ════════════════════════════════════════════════════════
def glow_box(x, y, w, h, color, title, lines=(), tsz=10.5, lsz=8.5, subtitle=""):
    for exp, alp in [(0.14, 0.03), (0.07, 0.07), (0.03, 0.13)]:
        ax.add_patch(FancyBboxPatch(
            (x - exp, y - exp), w + 2*exp, h + 2*exp,
            boxstyle="round,pad=0.04", facecolor=color, edgecolor="none", alpha=alp))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="none", alpha=0.12))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor="none", edgecolor=color, alpha=0.75, lw=1.6))
    # Top accent bar
    ax.plot([x + 0.12, x + w - 0.12], [y + h, y + h],
            color=color, lw=2.2, alpha=0.9, solid_capstyle="round")
    ax.text(x + w / 2, y + h - 0.33, title, ha="center", va="top",
            fontsize=tsz, fontweight="bold", color=color, family="monospace",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
    if subtitle:
        ax.text(x + w / 2, y + h - 0.68, subtitle, ha="center", va="top",
                fontsize=lsz - 1, color=MUTED, family="monospace")
    start = y + h - (0.95 if subtitle else 0.75)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, start - i * 0.42, ln, ha="center", va="top",
                fontsize=lsz, color=TEXT, family="monospace", alpha=0.88)


def arr(x1, y1, x2, y2, color, label="", dashed=False):
    style = dict(arrowstyle="-|>", color=color, lw=1.7,
                 linestyle=("dashed" if dashed else "solid"))
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=style)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.13, label, ha="center", va="bottom",
                fontsize=7, color=color, family="monospace", style="italic")


def curved_arr(x1, y1, x2, y2, color, label="", rad=0.25):
    style = dict(arrowstyle="-|>", color=color, lw=1.5,
                 connectionstyle=f"arc3,rad={rad}")
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=style)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my, label, ha="left", va="center",
                fontsize=7, color=color, family="monospace", style="italic")


# ════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════
ax.plot([0.4, 21.6], [12.75, 12.75], color=TEAL, lw=0.6, alpha=0.25)
ax.text(11, 12.55, "⚡  SMART AGRICULTURE ADVISORY SYSTEM",
        ha="center", fontsize=20, fontweight="bold", color=TEAL,
        family="monospace",
        path_effects=[pe.withStroke(linewidth=5, foreground="#001520")])
ax.text(11, 12.08, "51 Crops  ·  SVM 96%  ·  Profit-First Ranking  ·  Region-Aware  ·  28 States",
        ha="center", fontsize=9.5, color=MUTED, family="monospace")
ax.plot([0.4, 21.6], [11.75, 11.75], color=TEAL, lw=0.6, alpha=0.25)

# ════════════════════════════════════════════════════════
# LAYER A — USER FLOW  (y = 9.2)
# ════════════════════════════════════════════════════════
glow_box(0.3,  9.2, 3.4, 2.2, GREEN,  "FARMER INPUT",
         lines=["State · District",  "Land (bigha)"])
glow_box(4.6,  9.2, 4.6, 2.2, BLUE,   "STREAMLIT APP",
         lines=["Top-5 crop cards",  "Risk · Economics · CSV export"])
glow_box(10.2, 8.8, 6.5, 2.7, TEAL,   "★  PREDICTOR HUB",
         subtitle="predictor.py — Central Orchestrator",
         lines=["ML predict_proba → candidate pool",
                "Enrich profit+risk → rank Top-5"],
         tsz=11.5)
glow_box(17.8, 9.2, 3.9, 2.2, GOLD,   "TOP 5 CROPS",
         lines=["Ranked by profit+fit", "CSV export ready"])

arr(3.7,  10.3, 4.6,  10.3, GREEN,  label="inputs")
arr(9.2,  10.3, 10.2, 10.3, BLUE,   label="request")
arr(16.7, 10.15, 17.8, 10.15, TEAL,  label="results")

# ════════════════════════════════════════════════════════
# LAYER B — INTELLIGENCE ENGINES  (y = 6.2)
# ════════════════════════════════════════════════════════
glow_box(0.3,  6.2, 5.0, 2.3, PINK,   "PROFIT ENGINE",
         lines=["Revenue = yield × price", "Net profit · ROI %"])
glow_box(6.1,  6.2, 5.0, 2.3, PURPLE, "RISK ENGINE",
         lines=["51 crops · 200+ diseases", "Composite risk score"])
glow_box(11.9, 6.2, 5.0, 2.3, "#a371f7", "REGION DATA",
         lines=["District→State→National", "Yield · Price · Vulnerability"])
glow_box(17.7, 6.2, 4.0, 2.3, ORANGE, "ZONE DEFAULTS",
         lines=["7 agro-climatic zones", "Soil · pH · Climate"])

# Fan arrows: Predictor ↔ Engines
curved_arr(13.45, 8.8, 2.8,  8.5, PINK,   rad=0.2)
curved_arr(13.45, 8.8, 8.6,  8.5, PURPLE, rad=0.08)
curved_arr(13.45, 8.8, 14.4, 8.5, "#a371f7", rad=-0.08)
arr(17.8, 9.2, 19.7, 8.5, ORANGE)

# Return arrows (dashed)
curved_arr(2.8,  8.5, 11.0, 9.0, PINK,   rad=-0.2, label="enriched")
curved_arr(8.6,  8.5, 11.5, 9.0, PURPLE, rad=-0.1)
curved_arr(14.4, 8.5, 13.5, 9.0, "#a371f7", rad=0.12)

# ════════════════════════════════════════════════════════
# LAYER C — ML TRAINING PIPELINE  (y = 3.0)
# ════════════════════════════════════════════════════════
ax.text(11, 5.85, "── ML TRAINING PIPELINE  (run_pipeline.py) ──",
        ha="center", fontsize=10, fontweight="bold", color=ORANGE, family="monospace")

stages = [
    (0.3,  "INGEST DATA",    ["5310 rows · 51 crops",  "Balanced classes"]),
    (5.5,  "PREPROCESS",     ["StandardScaler",         "Stratified 80/20"]),
    (10.7, "TRAIN & SELECT", ["6 models · GridSearchCV","Best: SVM  96% F1"]),
    (15.9, "EVALUATE",       ["Accuracy · F1-macro",    "Confusion matrix"]),
]
bw = 4.8
for i, (sx, title, lines) in enumerate(stages):
    glow_box(sx, 3.0, bw, 2.4, ORANGE, title, lines=lines, tsz=10)
    if i < len(stages) - 1:
        arr(sx + bw, 4.2, sx + bw + 0.4, 4.2, ORANGE)

# ════════════════════════════════════════════════════════
# LAYER D — STORAGE  (y = 0.7)
# ════════════════════════════════════════════════════════
glow_box(0.3,  0.7, 6.4, 1.9, GRAY, "DATASETS",
         lines=["Crop_Recommendation.csv", "market_prices.csv + optional"],
         tsz=9.5, lsz=8.2)
glow_box(7.6,  0.7, 6.8, 1.9, GRAY, "MODEL ARTIFACTS",
         lines=["model.joblib  (SVM)", "scaler · encoder · metadata"],
         tsz=9.5, lsz=8.2)
glow_box(15.3, 0.7, 6.4, 1.9, GRAY, "REPORTS",
         lines=["EDA · Learning curves", "Feature importance · SHAP"],
         tsz=9.5, lsz=8.2)

# Pipeline → Storage (down)
arr(2.7,  3.0, 3.5,  2.6, GRAY)
arr(10.7, 3.0, 11.0, 2.6, GRAY)
arr(18.3, 3.0, 18.5, 2.6, GRAY)

# Artifacts → Predictor (dashed load arrow)
curved_arr(11.0, 2.6, 12.5, 8.8, TEAL, label="load model", rad=-0.35)

# ════════════════════════════════════════════════════════
# STATS BAR
# ════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch(
    (0.3, 0.05), 21.4, 0.55, boxstyle="round,pad=0.04",
    facecolor=TEAL, alpha=0.07, edgecolor=TEAL, lw=0.8))

stats = [("51", "Crops"), ("7", "Features"), ("5 310", "Samples"),
         ("6", "ML Models"), ("96 %", "Accuracy"),
         ("Top 5", "Recs"), ("28", "States"), ("200+", "Diseases")]
for i, (val, lbl) in enumerate(stats):
    sx = 0.85 + i * 2.62
    ax.text(sx, 0.45, val, ha="left", va="center",
            fontsize=9, color=TEAL, family="monospace", fontweight="bold")
    ax.text(sx, 0.18, lbl, ha="left", va="center",
            fontsize=7.2, color=MUTED, family="monospace")
    if i < len(stats) - 1:
        ax.plot([sx + 2.25, sx + 2.25], [0.12, 0.52],
                color=MUTED, lw=0.5, alpha=0.4)

plt.tight_layout(pad=0.4)
plt.savefig(
    r"C:\SMART CROP REC\ARCHITECTURE_DIAGRAM.png",
    dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
print("Saved: ARCHITECTURE_DIAGRAM.png")
plt.close()
