# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:04:50 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
09_generate_final_manuscript_tables.py

Generate final manuscript tables from LOSO results
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
    r"\Manuscript_Tables"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ============================================================
# LOAD DATA
# ============================================================


print("="*80)
print("LOADING LOSO RESULTS")
print("="*80)


df = pd.read_csv(
    INPUT_FILE
)


print("Dataset:")
print(df.shape)



# ============================================================
# Extract architecture name
# ============================================================


def remove_configuration(model):

    model = str(model)

    model = re.sub(
        r"_lr[0-9.]+",
        "",
        model
    )

    model = re.sub(
        r"_bs[0-9]+",
        "",
        model
    )

    return model



df["Architecture"] = (
    df["Model"]
    .apply(remove_configuration)
)



# ============================================================
# TABLE 1
# BEST CONFIGURATION PER MODEL
# ============================================================


print("\n")
print("="*80)
print("TABLE 1: BEST MODEL CONFIGURATION")
print("="*80)



table1 = (

    df.groupby(
        "Model"
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



# select best configuration of each architecture

table1["Architecture"] = (
    table1["Model"]
    .apply(remove_configuration)
)



table1 = (

    table1
    .sort_values(
        "MCC",
        ascending=False
    )
    .groupby(
        "Architecture"
    )
    .head(1)

)



table1 = table1.sort_values(
    "MCC",
    ascending=False
)



print(
    table1.round(4)
    .to_string(index=False)
)



table1.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Table_1_Best_Model_LOSO_Performance.csv"
    ),

    index=False

)



# ============================================================
# TABLE S1
# ALL CONFIGURATIONS
# ============================================================


print("\nGenerating Supplementary Table S1")


table_s1 = (

    df.groupby(
        "Model"
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



table_s1 = table_s1.sort_values(
    "MCC",
    ascending=False
)



table_s1.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Table_S1_All_LOSO_Configurations.csv"
    ),

    index=False

)



# ============================================================
# TABLE S2
# ARCHITECTURE LEVEL PERFORMANCE
# ============================================================


print("Generating Supplementary Table S2")


table_s2 = (

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



table_s2 = table_s2.sort_values(
    "MCC",
    ascending=False
)



table_s2.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Table_S2_Architecture_Level_Performance.csv"
    ),

    index=False

)



# ============================================================
# TABLE S3
# SPECIES-WISE BEST MODEL
# ============================================================


print("Generating Supplementary Table S3")



species_results = []



for species in sorted(
    df["Species"].unique()
):


    temp = (

        df[
            df["Species"] == species
        ]
        .sort_values(
            "MCC",
            ascending=False
        )

    )


    best = temp.iloc[0]


    species_results.append(

        {

        "Species":species,

        "Best_Model":best["Model"],

        "Accuracy":best["Accuracy"],

        "Precision":best["Precision"],

        "Recall":best["Recall"],

        "F1":best["F1"],

        "MCC":best["MCC"],

        "AUC":best["AUC"]

        }

    )



table_s3 = pd.DataFrame(
    species_results
)



table_s3.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Table_S3_Species_Wise_Best_Model.csv"
    ),

    index=False

)



# ============================================================
# EXCEL FILE
# ============================================================


excel_file = os.path.join(
    OUTPUT_DIR,
    "FINAL_MANUSCRIPT_LOSO_TABLES.xlsx"
)



with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:


    table1.to_excel(
        writer,
        sheet_name="Table1_Main_LOSO",
        index=False
    )


    table_s1.to_excel(
        writer,
        sheet_name="TableS1_All_Config",
        index=False
    )


    table_s2.to_excel(
        writer,
        sheet_name="TableS2_Architecture",
        index=False
    )


    table_s3.to_excel(
        writer,
        sheet_name="TableS3_Species",
        index=False
    )



print("\n")
print("="*80)
print("FINAL TABLES GENERATED")
print("="*80)


print("\nSaved folder:")
print(OUTPUT_DIR)


print("\nExcel:")
print(excel_file)