# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:31:58 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-

"""
FINAL CD-HIT RESULT ANALYSIS
Extract all configurations and generate manuscript tables
"""

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\zebfish1\revision1\FIGNet_CDHIT_Results\runs"

OUTPUT_DIR = r"D:\zebfish1\revision1\Results\CDHIT_Analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MODEL EXTRACTION
# ============================================================

records = []


print("="*80)
print("EXTRACTING CD-HIT RESULTS")
print("="*80)


for config in sorted(os.listdir(BASE_DIR)):

    config_path = os.path.join(
        BASE_DIR,
        config
    )

    if not os.path.isdir(config_path):
        continue


    print("\nCONFIGURATION:")
    print(config)


    # convert folder name
    # lr_0_00010_bs_128
    configuration = config.replace(
        "lr_",
        "lr"
    ).replace(
        "_bs_",
        "_bs"
    )


    for model in sorted(os.listdir(config_path)):

        model_path = os.path.join(
            config_path,
            model
        )


        if not os.path.isdir(model_path):
            continue


        csv_path = os.path.join(
            model_path,
            "csv_files",
            "Experiment_Summary.csv"
        )


        if not os.path.exists(csv_path):

            print(
                "Missing:",
                csv_path
            )

            continue


        try:

            df = pd.read_csv(csv_path)


            row = df.iloc[0]


            records.append({

                "Model":
                    model,

                "Configuration":
                    configuration,

                "Accuracy":
                    row.get(
                        "test_accuracy",
                        row.get(
                            "accuracy",
                            None
                        )
                    ),

                "Precision":
                    row.get(
                        "test_precision",
                        row.get(
                            "precision",
                            None
                        )
                    ),

                "Recall":
                    row.get(
                        "test_recall",
                        row.get(
                            "recall",
                            None
                        )
                    ),

                "F1":
                    row.get(
                        "test_f1",
                        row.get(
                            "f1",
                            None
                        )
                    ),

                "MCC":
                    row.get(
                        "test_mcc",
                        row.get(
                            "mcc",
                            None
                        )
                    ),

                "AUC":
                    row.get(
                        "test_auc",
                        row.get(
                            "auc",
                            None
                        )
                    )

            })


            print(
                "✓",
                model,
                configuration
            )


        except Exception as e:

            print(
                "ERROR:",
                csv_path,
                e
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results = pd.DataFrame(records)


print("\n")
print("="*80)
print("EXTRACTION COMPLETE")
print("="*80)

print(
    results.shape
)

print(
    results.head()
)


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "CDHIT_All_Model_Configurations.csv"
    ),
    index=False
)


# ============================================================
# TABLE 1
# ALL CONFIGURATIONS RANKING
# ============================================================


table1 = results.sort_values(
    by="MCC",
    ascending=False
)


table1.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Table1_All_CDHIT_Configurations.xlsx"
    ),
    index=False
)


# ============================================================
# TABLE 2
# BEST CONFIGURATION PER MODEL
# ============================================================


best_model = (
    results
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


best_model = best_model.sort_values(
    "MCC",
    ascending=False
)


best_model.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Table2_Best_Configuration_Per_Model.xlsx"
    ),
    index=False
)


# ============================================================
# TABLE 3
# ARCHITECTURE LEVEL PERFORMANCE
# Average all configurations
# ============================================================


architecture = (
    results
    .groupby(
        "Model"
    )[
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


architecture = architecture.sort_values(
    "MCC",
    ascending=False
)


architecture.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Table3_CDHIT_Architecture_Level.xlsx"
    ),
    index=False
)



# ============================================================
# TABLE 4
# BEST CONFIGURATION OVERALL
# ============================================================


overall_best = results.sort_values(
    "MCC",
    ascending=False
).head(20)


overall_best.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "Table4_Top20_CDHIT_Models.xlsx"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================


print("\n")
print("="*80)
print("CD-HIT ANALYSIS COMPLETED")
print("="*80)


print(
    "Total results:",
    len(results)
)


print(
    "\nSaved files:"
)


for f in os.listdir(OUTPUT_DIR):

    print(
        os.path.join(
            OUTPUT_DIR,
            f
        )
    )