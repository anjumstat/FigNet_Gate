# -*- coding: utf-8 -*-
"""
04_LOSO_Final_Analysis.py

Final LOSO analysis from Independent Test Results

Author: H.A.R
"""

import os
import pandas as pd
import numpy as np


# =====================================================
# PATHS
# =====================================================

BASE_DIR = r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results"

OUTPUT_DIR = r"D:\zebfish1\revision1\Results\LOSO_Analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_EXCEL = os.path.join(
    OUTPUT_DIR,
    "FINAL_LOSO_MANUSCRIPT_TABLES.xlsx"
)


# =====================================================
# FUNCTION TO READ LOSO RESULT FILE
# =====================================================

def extract_metrics(file):

    df = pd.read_csv(file)

    metrics = {
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "MCC": np.nan,
        "AUC": np.nan
    }


    # Case 1:
    # Metric,value format

    if df.shape[1] >= 2:

        for i,row in df.iterrows():

            key = str(row.iloc[0]).lower()
            value = row.iloc[1]


            if "accuracy" in key:
                metrics["Accuracy"] = float(value)

            elif "precision" in key:
                metrics["Precision"] = float(value)

            elif "recall" in key:
                metrics["Recall"] = float(value)

            elif "f1" in key:
                metrics["F1"] = float(value)

            elif "mcc" in key:
                metrics["MCC"] = float(value)

            elif "auc" in key:
                metrics["AUC"] = float(value)



    # Case 2:
    # Column format

    for col in df.columns:

        c = col.lower()


        if "accuracy" in c:
            metrics["Accuracy"] = df[col].iloc[0]

        elif "precision" in c:
            metrics["Precision"] = df[col].iloc[0]

        elif "recall" in c:
            metrics["Recall"] = df[col].iloc[0]

        elif "f1" in c:
            metrics["F1"] = df[col].iloc[0]

        elif "mcc" in c:
            metrics["MCC"] = df[col].iloc[0]

        elif "auc" in c:
            metrics["AUC"] = df[col].iloc[0]


    return metrics



# =====================================================
# EXTRACT ALL LOSO RESULTS
# =====================================================


all_results = []


print("="*80)
print("STARTING LOSO EXTRACTION")
print("="*80)



for species in os.listdir(BASE_DIR):


    species_path = os.path.join(
        BASE_DIR,
        species
    )


    if not os.path.isdir(species_path):
        continue



    print("\n")
    print("="*60)
    print(species)
    print("="*60)



    for model_folder in os.listdir(species_path):


        model_path = os.path.join(
            species_path,
            model_folder
        )


        result_file = os.path.join(
            model_path,
            "csv_files",
            "Final_Independent_Test_Result.csv"
        )


        if os.path.exists(result_file):


            metrics = extract_metrics(
                result_file
            )


            all_results.append(
                {
                    "Species":species,
                    "Model":model_folder,
                    **metrics
                }
            )


            print("✓", model_folder)


        else:

            print(
                "Missing:",
                model_folder
            )



# =====================================================
# CREATE MASTER DATASET
# =====================================================


results = pd.DataFrame(
    all_results
)


print("\n")
print("="*80)
print("EXTRACTION COMPLETE")
print("="*80)

print(
    "Total LOSO results:",
    len(results)
)


print(results.head())



master_file = os.path.join(
    OUTPUT_DIR,
    "LOSO_All_Model_Results.csv"
)


results.to_csv(
    master_file,
    index=False
)



print(
    "Saved:",
    master_file
)



# =====================================================
# CLEAN MODEL NAME
# =====================================================


results["Model_Name"] = (
    results["Model"]
    .str.replace(
        r"_lr.*",
        "",
        regex=True
    )
)



results["Configuration"] = (
    results["Model"]
    .str.extract(
        r"(lr.*)"
    )
)



# =====================================================
# TABLE 1
# OVERALL MODEL PERFORMANCE
# =====================================================


table1 = (

results

.groupby(
    "Model_Name"
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

.sort_values(
    "MCC",
    ascending=False
)

)



table1.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Table1_LOSO_Overall_Performance.csv"
    ),
    index=False
)



# =====================================================
# TABLE 2
# BEST CONFIGURATION
# =====================================================


table2 = (

results

.groupby(
[
"Model_Name",
"Configuration"
]

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

.sort_values(
    "MCC",
    ascending=False
)

)



table2.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Table2_LOSO_Best_Configurations.csv"
    ),
    index=False
)



# =====================================================
# TABLE 3
# SPECIES PERFORMANCE
# =====================================================


table3 = (

results

.groupby(
[
"Species",
"Model_Name"
]

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



table3.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Table3_LOSO_Species_Performance.csv"
    ),
    index=False
)



# =====================================================
# SAVE EXCEL
# =====================================================


with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:


    results.to_excel(
        writer,
        sheet_name="All_LOSO_results",
        index=False
    )


    table1.to_excel(
        writer,
        sheet_name="Table1_Main",
        index=False
    )


    table2.to_excel(
        writer,
        sheet_name="Table2_Config",
        index=False
    )


    table3.to_excel(
        writer,
        sheet_name="Table3_Species",
        index=False
    )



print("\n")
print("="*80)
print("ALL LOSO TABLES GENERATED")
print("="*80)

print(
    OUTPUT_EXCEL
)