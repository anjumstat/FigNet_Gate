# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:53:10 2026

@author: H.A.R
"""

import pandas as pd
import os


# ======================================================
# PATH
# ======================================================

INPUT = r"D:\zebfish1\revision1\Results\Feature_Stability_CV\CV_Feature_Stability_Jaccard.csv"


OUTPUT = r"D:\zebfish1\revision1\Results\Feature_Stability_CV"


# ======================================================
# LOAD
# ======================================================

df = pd.read_csv(INPUT)


print("="*70)
print("CV FEATURE STABILITY SUMMARY")
print("="*70)


print(df.head())


# ======================================================
# SUMMARY BY MODEL + CONFIGURATION
# ======================================================


summary = (
    df.groupby(
        [
            "Model",
            "Configuration"
        ]
    )
    .agg(
        Mean_Jaccard=("Mean_Jaccard","mean"),
        SD_Jaccard=("Mean_Jaccard","std"),
        Species_Evaluated=("Species","nunique"),
        Total_Runs=("Number_of_Folds","sum")
    )
    .reset_index()
)



summary = summary.sort_values(
    "Mean_Jaccard",
    ascending=False
)


print("\nTOP STABLE MODELS")
print(summary.head(20))



# ======================================================
# SAVE
# ======================================================


summary.to_excel(
    os.path.join(
        OUTPUT,
        "CV_Feature_Stability_Summary.xlsx"
    ),
    index=False
)


summary.to_csv(
    os.path.join(
        OUTPUT,
        "CV_Feature_Stability_Summary.csv"
    ),
    index=False
)


print("\nSaved:")
print(
    os.path.join(
        OUTPUT,
        "CV_Feature_Stability_Summary.xlsx"
    )
)