# -*- coding: utf-8 -*-
"""
Figure 1: Model comparison based on MCC and AUC
UPDATED for Nested CV results (unbiased)

Handles MultiIndex header (2 header rows) in the summary CSV
Outputs: PNG + TIFF with mean ± std error bars
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =====================================================
# PATHS
# =====================================================

INPUT_FILE = r"D:\zebfish1\revision1\FIGNet_NestedCV_Results\NestedCV_Summary.csv"
OUTPUT_DIR = r"D:\zebfish1\revision1\Results\updatedfigures1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figure1_NestedCV_Model_Comparison_MCC_AUC.png")
OUTPUT_TIFF = os.path.join(OUTPUT_DIR, "Figure1_NestedCV_Model_Comparison_MCC_AUC.tiff")

# =====================================================
# FONT SETTINGS
# =====================================================

TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 16
VALUE_SIZE = 15

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

# =====================================================
# LOAD DATA (handle 2-row header)
# =====================================================

print("Loading nested CV summary...")
df = pd.read_csv(INPUT_FILE, header=[0, 1])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Flatten MultiIndex columns: metric + stat
df.columns = [
    "method" if "Unnamed" in str(c[0]) else f"{c[0]}_{c[1]}"
    for c in df.columns
]
print("\nFlattened columns:")
print(df.columns.tolist())

# Reset index so method becomes a column
df = df.reset_index(drop=True)

# Drop any NaN rows (the trailing artifact)
df = df.dropna(subset=["method"]).reset_index(drop=True)

# Keep only the 12 real model rows
df = df[df["method"].apply(lambda x: isinstance(x, str) and x != "method")]

# Coerce metric columns to numeric
for metric in ["accuracy", "precision", "recall", "f1", "mcc", "auc"]:
    for stat in ["mean", "std"]:
        col = f"{metric}_{stat}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nCleaned data:")
print(df[["method", "mcc_mean", "mcc_std", "auc_mean", "auc_std"]].round(4).to_string(index=False))

# =====================================================
# SORT DATA FOR PLOTTING
# =====================================================

df = df.sort_values("mcc_mean", ascending=True).reset_index(drop=True)

# =====================================================
# FIGURE
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# =====================================================
# PANEL A: MCC
# =====================================================

axes[0].barh(
    df["method"].astype(str),
    df["mcc_mean"],
    color="steelblue",
    alpha=0.85,
)

axes[0].set_xlabel(
    "Matthews Correlation Coefficient (MCC)",
    fontsize=LABEL_SIZE,
    fontweight="bold",
)

axes[0].set_title(
    "A. MCC comparison (Nested CV)",
    fontsize=TITLE_SIZE,
    fontweight="bold",
)

axes[0].tick_params(axis="both", labelsize=TICK_SIZE)

for label in axes[0].get_yticklabels():
    label.set_fontweight("bold")

for i, (mean, std) in enumerate(zip(df["mcc_mean"], df["mcc_std"])):
    axes[0].text(
        mean + std + 0.008,
        i,
        f"{mean:.3f}",
        va="center",
        fontsize=VALUE_SIZE,
        fontweight="bold",
    )

axes[0].set_xlim(0, (df["mcc_mean"] + df["mcc_std"]).max() + 0.06)

# =====================================================
# PANEL B: AUC
# =====================================================

df_auc = df.sort_values("auc_mean", ascending=True).reset_index(drop=True)

axes[1].barh(
    df_auc["method"].astype(str),
    df_auc["auc_mean"],
    color="darkorange",
    alpha=0.85,
)

axes[1].set_xlabel(
    "Area Under ROC Curve (AUC)",
    fontsize=LABEL_SIZE,
    fontweight="bold",
)

axes[1].set_title(
    "B. AUC comparison (Nested CV)",
    fontsize=TITLE_SIZE,
    fontweight="bold",
)

axes[1].tick_params(axis="both", labelsize=TICK_SIZE)

for label in axes[1].get_yticklabels():
    label.set_fontweight("bold")

for i, (mean, std) in enumerate(zip(df_auc["auc_mean"], df_auc["auc_std"])):
    axes[1].text(
        mean + std + 0.008,
        i,
        f"{mean:.3f}",
        va="center",
        fontsize=VALUE_SIZE,
        fontweight="bold",
    )

axes[1].set_xlim(0, (df_auc["auc_mean"] + df_auc["auc_std"]).max() + 0.02)

# =====================================================
# FINAL FORMATTING
# =====================================================

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
plt.savefig(OUTPUT_TIFF, dpi=600, bbox_inches="tight", format="tiff")

plt.show()

print("\nFigure saved:")
print(OUTPUT_PNG)
print(OUTPUT_TIFF)