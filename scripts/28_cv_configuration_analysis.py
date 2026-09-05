# -*- coding: utf-8 -*-

"""
CV Configuration Performance Analysis
Full Model Names + All Metrics

Author: H.A.R
"""


import pandas as pd
import os



# =====================================================
# PATHS
# =====================================================

INPUT = r"D:\zebfish1\revision1\Results\CV_Performance\CV_All_Configurations_Performance.csv"


OUTPUT = r"D:\zebfish1\revision1\Results\CV_Performance\CV_Final_Analysis.xlsx"



# =====================================================
# LOAD DATA
# =====================================================


df=pd.read_csv(INPUT)



print("="*80)
print("AVAILABLE COLUMNS")
print("="*80)

print(df.columns)



# =====================================================
# RECOVER FULL MODEL NAME
# =====================================================


def extract_model(row):

    text=str(row["Configuration"])+" "+str(row["output_dir"])


    models=[
        "FIGNet_Gate_AdaptiveVD",
        "FIGNet_Gate_RealVD",
        "FIGNet_Gate_Only",
        "FIGNet_Gate_Full",
        "FIGNet_Gate_Sparsity",
        "MLP_Baseline",
        "SVM_RBF",
        "SVM_Linear",
        "ReliefF_MLP",
        "ReliefF_SVM",
        "Logistic_Regression"
    ]


    for m in models:

        if m in text:
            return m


    return "Unknown"



df["Architecture"]=df.apply(
    extract_model,
    axis=1
)



# remove unknown

df=df[
    df["Architecture"]!="Unknown"
]



# =====================================================
# RANK ALL CONFIGURATIONS
# =====================================================


ranked=df.sort_values(
    by=[
        "mean_mcc",
        "mean_auc"
    ],
    ascending=False
)



ranked.insert(
    0,
    "Rank",
    range(
        1,
        len(ranked)+1
    )
)



print("\n")
print("="*80)
print("TOP 20 CONFIGURATIONS")
print("="*80)


print(
ranked[
[
"Rank",
"Architecture",
"learning_rate",
"batch_size",
"mean_accuracy",
"mean_precision",
"mean_recall",
"mean_f1",
"mean_mcc",
"mean_auc"
]
].head(20)
)



# =====================================================
# BEST CONFIGURATION PER MODEL
# =====================================================


best_model = (
    ranked
    .groupby("Architecture")
    .first()
    .reset_index()
)



best_model=best_model[
[
"Architecture",
"learning_rate",
"batch_size",
"mean_accuracy",
"mean_precision",
"mean_recall",
"mean_f1",
"mean_mcc",
"mean_auc"
]
]



print("\n")
print("="*80)
print("BEST CONFIGURATION PER MODEL")
print("="*80)


print(best_model)



# =====================================================
# PERFORMANCE SUMMARY
# =====================================================


summary=(
    df
    .groupby("Architecture")
    [
[
"mean_accuracy",
"mean_precision",
"mean_recall",
"mean_f1",
"mean_mcc",
"mean_auc"
]
]
    .agg(
        [
        "mean",
        "std"
        ]
    )
)



print("\n")
print("="*80)
print("AVERAGE MODEL PERFORMANCE")
print("="*80)


print(summary)



# =====================================================
# COUNT GOOD CONFIGURATIONS
# =====================================================


robust=[]


for model,group in df.groupby(
    "Architecture"
):


    robust.append({

        "Model":model,

        "Total configurations":
        len(group),


        "MCC >=0.85":
        (group["mean_mcc"]>=0.85).sum(),


        "AUC >=0.97":
        (group["mean_auc"]>=0.97).sum(),


        "Accuracy >=0.94":
        (group["mean_accuracy"]>=0.94).sum(),

    })



robust=pd.DataFrame(
    robust
)



print("\n")
print("="*80)
print("ROBUSTNESS COUNT")
print("="*80)


print(robust)



# =====================================================
# SAVE EVERYTHING
# =====================================================


with pd.ExcelWriter(
    OUTPUT
) as writer:


    ranked.to_excel(
        writer,
        sheet_name="All_Config_Ranking",
        index=False
    )


    best_model.to_excel(
        writer,
        sheet_name="Best_Per_Model",
        index=False
    )


    summary.to_excel(
        writer,
        sheet_name="Mean_SD_Performance"
    )


    robust.to_excel(
        writer,
        sheet_name="Robustness_Count",
        index=False
    )



print("\nCompleted:")
print(OUTPUT)