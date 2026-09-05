# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:04:29 2026

@author: H.A.R
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import friedmanchisquare, wilcoxon


# ============================================
# PATHS
# ============================================

INPUT = r"D:\zebfish1\revision1\Results\CV_RESULTS_WITH_MODELS.csv"

OUTPUT_DIR = r"D:\zebfish1\revision1\Results\Statistical_Analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(
    INPUT,
    low_memory=False
)


print("Loaded rows:", len(df))


# ============================================
# SELECT FINAL CONFIGURATIONS
# ============================================


selected = df[
    (
        (df["model_name"]=="FIGNet_Gate_RealVD") &
        (df["configuration"]=="lr_0_00010_bs_128")
    )
    |
    (df["model_name"].isin(
        [
            "FIGNet_Gate_Only",
            "FIGNet_Gate_AdaptiveVD",
            "MLP_Baseline",
            "ReliefF_MLP",
            "ReliefF_SVM",
            "SVM_RBF",
            "SVM_Linear",
            "Logistic_Regression"
        ]
    ))
]


print(
    "\nSelected rows:",
    len(selected)
)


print(
    selected["model_name"]
    .value_counts()
)



# ============================================
# FRIEDMAN TEST
# ============================================


friedman_results=[]


for metric in [
    "MCC",
    "F1",
    "AUC"
]:

    pivot = (
        selected
        .pivot_table(
            index="Fold",
            columns="model_name",
            values=metric,
            aggfunc="mean"
        )
    )


    pivot = pivot.dropna()


    methods = pivot.columns.tolist()


    data = [
        pivot[m].values
        for m in methods
    ]


    stat, p = friedmanchisquare(
        *data
    )


    friedman_results.append(
        {
            "Metric":metric,
            "Chi_square":stat,
            "p_value":p,
            "Significant":p<0.05,
            "Number_of_models":len(methods)
        }
    )


    print("\n")
    print(metric)
    print("p-value:",p)



friedman_df=pd.DataFrame(
    friedman_results
)


friedman_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Friedman_results.csv"
    ),
    index=False
)



# ============================================
# WILCOXON PAIRWISE TESTS
# ============================================


comparisons=[
    
("FIGNet_Gate_RealVD","MLP_Baseline"),

("FIGNet_Gate_RealVD","SVM_RBF"),

("FIGNet_Gate_RealVD","ReliefF_MLP"),

("FIGNet_Gate_RealVD","ReliefF_SVM")

]



wilcoxon_results=[]


for metric in [
    "MCC",
    "F1",
    "AUC"
]:

    for model1,model2 in comparisons:


        a = (
            selected[
                selected.model_name==model1
            ]
            .groupby("Fold")[metric]
            .mean()
        )


        b = (
            selected[
                selected.model_name==model2
            ]
            .groupby("Fold")[metric]
            .mean()
        )


        common = a.index.intersection(
            b.index
        )


        if len(common)>0:

            stat,p = wilcoxon(
                a.loc[common],
                b.loc[common]
            )


            wilcoxon_results.append(
                {
                    "Metric":metric,
                    "Model1":model1,
                    "Model2":model2,
                    "Wilcoxon_stat":stat,
                    "p_value":p,
                    "Significant":p<0.05
                }
            )



wilcoxon_df=pd.DataFrame(
    wilcoxon_results
)



wilcoxon_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Wilcoxon_pairwise_results.csv"
    ),
    index=False
)



print("\n==============================")
print("FRIEDMAN RESULTS")
print("==============================")

print(
    friedman_df
)



print("\n==============================")
print("WILCOXON RESULTS")
print("==============================")

print(
    wilcoxon_df
)



print("\nSaved in:")
print(OUTPUT_DIR)