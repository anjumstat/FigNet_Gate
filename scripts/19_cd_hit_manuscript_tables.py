# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:33:43 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-

import pandas as pd
import os


INPUT = r"D:\zebfish1\revision1\Results\CDHIT_Analysis\CDHIT_All_Model_Configurations.csv"


OUTPUT = r"D:\zebfish1\revision1\Results\CDHIT_Analysis\CDHIT_Manuscript_Table.xlsx"


df = pd.read_csv(INPUT)


# ----------------------------------------------------
# clean configuration names
# ----------------------------------------------------

df["Configuration"] = (
    df["Configuration"]
    .str.replace(
        "lr0_00010",
        "lr0.0001"
    )
    .str.replace(
        "lr0_00100",
        "lr0.001"
    )
    .str.replace(
        "lr0_01000",
        "lr0.01"
    )
)


# ----------------------------------------------------
# Best configuration per model
# ----------------------------------------------------

best = (
    df
    .sort_values(
        "MCC",
        ascending=False
    )
    .groupby(
        "Model",
        as_index=False
    )
    .first()
)


best = best[
    [
        "Model",
        "Configuration",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "AUC"
    ]
]


best = best.sort_values(
    "MCC",
    ascending=False
)


# ----------------------------------------------------
# Save
# ----------------------------------------------------

best.to_excel(
    OUTPUT,
    index=False
)


print("="*80)
print("CD-HIT MANUSCRIPT TABLE GENERATED")
print("="*80)

print(best.to_string(index=False))


print("\nSaved:")
print(OUTPUT)