# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:52:47 2026

@author: H.A.R
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


# =====================================================
# PATHS
# =====================================================

INPUT_FILE = r"D:\zebfish1\revision1\Results\CV_RESULTS_WITH_MODELS.csv"

OUTPUT_DIR = r"D:\zebfish1\revision1\Results"

OUTPUT_FIG = os.path.join(
    OUTPUT_DIR,
    "Figure1_Model_Comparison_MCC_AUC.png"
)


# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)


print("Shape:")
print(df.shape)



# =====================================================
# FIND BEST CONFIGURATION OF EACH MODEL
# =====================================================

best_models = (
    df
    .groupby(
        [
            "model_name",
            "configuration"
        ]
    )
    .agg(
        Accuracy=("Accuracy","mean"),
        Precision=("Precision","mean"),
        Recall=("Recall","mean"),
        F1=("F1","mean"),
        MCC=("MCC","mean"),
        AUC=("AUC","mean")
    )
    .reset_index()
)



best_models = (
    best_models
    .sort_values(
        "MCC",
        ascending=False
    )
    .groupby("model_name")
    .head(1)
)



# Sort for plotting

best_models = best_models.sort_values(
    "MCC",
    ascending=True
)


print("\nBest configurations:")
print(
    best_models[
        [
            "model_name",
            "configuration",
            "MCC",
            "AUC"
        ]
    ]
    .round(4)
)



# =====================================================
# PLOT
# =====================================================


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14,6)
)



# --------------------------
# MCC PANEL
# --------------------------

axes[0].barh(
    best_models["model_name"],
    best_models["MCC"]
)


axes[0].set_xlabel(
    "Matthews Correlation Coefficient (MCC)"
)

axes[0].set_title(
    "A. MCC comparison"
)



# Add values

for i, value in enumerate(best_models["MCC"]):

    axes[0].text(
        value + 0.005,
        i,
        f"{value:.3f}",
        va="center"
    )



# --------------------------
# AUC PANEL
# --------------------------

best_auc = best_models.sort_values(
    "AUC",
    ascending=True
)


axes[1].barh(
    best_auc["model_name"],
    best_auc["AUC"]
)


axes[1].set_xlabel(
    "Area Under ROC Curve (AUC)"
)

axes[1].set_title(
    "B. AUC comparison"
)



for i, value in enumerate(best_auc["AUC"]):

    axes[1].text(
        value + 0.002,
        i,
        f"{value:.3f}",
        va="center"
    )



plt.tight_layout()



# Save

plt.savefig(
    OUTPUT_FIG,
    dpi=600,
    bbox_inches="tight"
)


plt.show()



print("\nFigure saved:")
print(OUTPUT_FIG)