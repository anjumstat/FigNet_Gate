# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:56:42 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
08_loso_architecture_level_analysis.py

Architecture-level LOSO analysis

Removes hyperparameter effects:
lr and batch size

Compares only model architectures.
"""


import os
import pandas as pd
import re



# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"D:\zebfish1\revision1\Results\LOSO_Analysis\LOSO_All_Model_Results.csv"
)


OUTPUT_DIR = (
    r"D:\zebfish1\revision1\Results\LOSO_Analysis"
    r"\Architecture_Level_Analysis"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



OUTPUT_EXCEL = os.path.join(
    OUTPUT_DIR,
    "Architecture_Level_LOSO_Analysis.xlsx"
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


print("\nOriginal shape:")
print(df.shape)



# ============================================================
# REMOVE HYPERPARAMETERS
# ============================================================


def extract_architecture(name):

    """
    Convert:

    FIGNet_Gate_Only_lr0.0001_bs128

    into:

    FIGNet_Gate_Only
    """

    name = str(name)

    name = re.sub(
        r"_lr[0-9.]+",
        "",
        name
    )


    name = re.sub(
        r"_bs[0-9]+",
        "",
        name
    )


    return name



df["Architecture"] = (
    df["Model"]
    .apply(extract_architecture)
)



print("\nArchitectures found:")
print(
    df["Architecture"]
    .unique()
)



# ============================================================
# TABLE 1
# OVERALL ARCHITECTURE PERFORMANCE
# ============================================================


print("\n")
print("="*70)
print("TABLE 1: ARCHITECTURE LEVEL PERFORMANCE")
print("="*70)



architecture_summary = (

    df.groupby(
        "Architecture"
    )
    [
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



architecture_summary = (
    architecture_summary
    .sort_values(
        "MCC",
        ascending=False
    )
)



print(
    architecture_summary
    .round(4)
    .to_string(index=False)
)



architecture_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Architecture_Performance_Ranking.csv"
    ),
    index=False
)



# ============================================================
# TABLE 2
# BEST ARCHITECTURE PER SPECIES
# ============================================================


print("\n")
print("="*70)
print("TABLE 2: BEST ARCHITECTURE PER SPECIES")
print("="*70)



species_best = []



for species in sorted(
    df["Species"].unique()
):


    temp = (

        df[
            df["Species"] == species
        ]

        .groupby(
            "Architecture"
        )
        [
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



    best = (

        temp
        .sort_values(
            "MCC",
            ascending=False
        )
        .iloc[0]

    )



    species_best.append(

        {

        "Species":species,

        "Best_Architecture":
            best["Architecture"],

        "Accuracy":
            best["Accuracy"],

        "Precision":
            best["Precision"],

        "Recall":
            best["Recall"],

        "F1":
            best["F1"],

        "MCC":
            best["MCC"],

        "AUC":
            best["AUC"]

        }

    )



species_best = pd.DataFrame(
    species_best
)



print(
    species_best
    .round(4)
    .to_string(index=False)
)



species_best.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Best_Architecture_Per_Species.csv"
    ),

    index=False

)



# ============================================================
# TABLE 3
# ARCHITECTURE WIN COUNT
# ============================================================


print("\n")
print("="*70)
print("TABLE 3: ARCHITECTURE WIN COUNT")
print("="*70)



win_count = (

    species_best
    ["Best_Architecture"]
    .value_counts()
    .reset_index()

)



win_count.columns = [

    "Architecture",
    "Species_Wins"

]


win_count["Percentage"] = (

    win_count["Species_Wins"]
    /
    len(species_best)
    *
    100

)



print(
    win_count
    .round(2)
    .to_string(index=False)
)



win_count.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Architecture_Win_Count.csv"
    ),

    index=False

)



# ============================================================
# EXPORT EXCEL
# ============================================================


with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:


    architecture_summary.to_excel(

        writer,

        sheet_name="Architecture_Ranking",

        index=False

    )


    species_best.to_excel(

        writer,

        sheet_name="Species_Best",

        index=False

    )


    win_count.to_excel(

        writer,

        sheet_name="Architecture_Wins",

        index=False

    )



print("\n")
print("="*70)
print("COMPLETED")
print("="*70)


print("\nSaved folder:")
print(OUTPUT_DIR)


print("\nExcel file:")
print(OUTPUT_EXCEL)