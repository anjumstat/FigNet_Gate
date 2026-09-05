# -*- coding: utf-8 -*-
"""
07_loso_model_wise_species_winner_analysis.py

Analysis:
1. Overall best model across all LOSO experiments
2. Best model for each species
3. Species win count
"""

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"D:\zebfish1\revision1\Results\LOSO_Analysis\LOSO_All_Model_Results.csv"


OUTPUT_DIR = r"D:\zebfish1\revision1\Results\LOSO_Analysis\Model_Winner_Analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_EXCEL = os.path.join(
    OUTPUT_DIR,
    "LOSO_Model_Winner_Analysis.xlsx"
)


# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("LOADING LOSO RESULTS")
print("="*70)


df = pd.read_csv(
    INPUT_FILE
)


print("\nDataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())



# ============================================================
# CLEAN MODEL NAME
# ============================================================

# Ensure model names are available

if "Model" in df.columns:

    df["Model_Name"] = (
        df["Model"]
    )


elif "model_name" in df.columns:

    df["Model_Name"] = (
        df["model_name"]
    )


else:

    raise Exception(
        "Model column not found"
    )



# ============================================================
# 1. OVERALL MODEL PERFORMANCE
# ============================================================


print("\n")
print("="*70)
print("OVERALL MODEL PERFORMANCE")
print("="*70)



overall = (
    df.groupby("Model_Name")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "MCC",
            "AUC"
        ]
    ]
    .mean()
    .reset_index()
)



overall = overall.sort_values(
    by="MCC",
    ascending=False
)



print(
    overall.round(4)
    .to_string(index=False)
)



overall.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Overall_Model_Ranking.csv"
    ),
    index=False
)



# ============================================================
# 2. BEST MODEL FOR EACH SPECIES
# ============================================================


print("\n")
print("="*70)
print("BEST MODEL FOR EACH SPECIES")
print("="*70)



species_results = []



for species in sorted(
    df["Species"].unique()
):


    temp = (
        df[
            df["Species"] == species
        ]
        .groupby("Model_Name")[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "MCC",
                "AUC"
            ]
        ]
        .mean()
        .reset_index()
    )


    # Best according to MCC

    best = (
        temp
        .sort_values(
            "MCC",
            ascending=False
        )
        .iloc[0]
    )


    species_results.append(
        {
            "Species": species,
            "Best_Model": best["Model_Name"],
            "Accuracy": best["Accuracy"],
            "Precision": best["Precision"],
            "Recall": best["Recall"],
            "F1": best["F1"],
            "MCC": best["MCC"],
            "AUC": best["AUC"]
        }
    )



species_best = pd.DataFrame(
    species_results
)



print(
    species_best.round(4)
    .to_string(index=False)
)



species_best.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Best_Model_Per_Species.csv"
    ),
    index=False
)



# ============================================================
# 3. MODEL WIN COUNT
# ============================================================


print("\n")
print("="*70)
print("SPECIES WIN COUNT")
print("="*70)



wins = (
    species_best
    ["Best_Model"]
    .value_counts()
    .reset_index()
)



wins.columns = [
    "Model_Name",
    "Species_Wins"
]


wins["Percentage"] = (
    wins["Species_Wins"]
    /
    len(species_best)
    *
    100
)



print(
    wins.round(2)
    .to_string(index=False)
)



wins.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Species_Win_Count.csv"
    ),
    index=False
)



# ============================================================
# 4. EXPORT EXCEL
# ============================================================


with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:


    overall.to_excel(
        writer,
        sheet_name="Overall_Ranking",
        index=False
    )


    species_best.to_excel(
        writer,
        sheet_name="Species_Best_Model",
        index=False
    )


    wins.to_excel(
        writer,
        sheet_name="Species_Win_Count",
        index=False
    )



print("\n")
print("="*70)
print("ANALYSIS COMPLETED")
print("="*70)

print("\nSaved files:")
print(OUTPUT_DIR)

print("\nExcel:")
print(OUTPUT_EXCEL)