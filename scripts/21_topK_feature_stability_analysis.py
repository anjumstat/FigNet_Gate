# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:32:49 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-

"""
Top-K Feature Stability Analysis using Jaccard Similarity

Calculates feature selection stability at:
Top-10, Top-25, Top-50, Top-100, Top-200

For:
1. CD-HIT
2. LOSO
"""

import os
import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# PATHS
# ============================================================

CDHIT_DIR = r"D:\zebfish1\revision1\FIGNet_CDHIT_Results\runs"

LOSO_DIR = r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results"


OUTPUT_DIR = r"D:\zebfish1\revision1\Results\Feature_Stability_TopK"

os.makedirs(OUTPUT_DIR, exist_ok=True)


TOP_K_VALUES = [10, 25, 50, 100, 200]


# ============================================================
# JACCARD FUNCTION
# ============================================================

def jaccard(set1, set2):

    if len(set1)==0 or len(set2)==0:
        return 0

    return len(set1.intersection(set2)) / len(set1.union(set2))



# ============================================================
# READ FEATURE FILES
# ============================================================

def extract_features(base_dir, dataset):

    feature_sets = []

    print("\nSearching:", dataset)

    for root, dirs, files in os.walk(base_dir):

        for file in files:

            if dataset=="CDHIT":

                if file == "Top_Features.csv":

                    path=os.path.join(root,file)

                    try:

                        df=pd.read_csv(path)

                        features=list(
                            df.sort_values("Rank")
                            ["Feature_Name"]
                            .values
                        )

                        model=os.path.basename(
                            os.path.dirname(
                                os.path.dirname(path)
                            )
                        )

                        feature_sets.append(
                            {
                            "Model":model,
                            "Path":path,
                            "Features":features
                            }
                        )

                    except Exception as e:
                        print(e)



            elif dataset=="LOSO":

                if file=="Final_Top_Features.csv":

                    path=os.path.join(root,file)

                    try:

                        df=pd.read_csv(path)

                        features=list(
                            df.sort_values("Rank")
                            ["Feature_Name"]
                            .values
                        )

                        model=os.path.basename(
                            os.path.dirname(
                                os.path.dirname(path)
                            )
                        )

                        species=os.path.basename(
                            os.path.dirname(
                                os.path.dirname(
                                    os.path.dirname(path)
                                )
                            )
                        )


                        feature_sets.append(
                            {
                            "Model":model,
                            "Species":species,
                            "Path":path,
                            "Features":features
                            }
                        )


                    except Exception as e:
                        print(e)


    print("Files extracted:",len(feature_sets))

    return feature_sets



# ============================================================
# CALCULATE TOP-K STABILITY
# ============================================================


def calculate_stability(feature_sets, dataset):


    results=[]


    # group by model

    groups={}


    for item in feature_sets:

        model=item["Model"]

        if model not in groups:
            groups[model]=[]

        groups[model].append(
            item["Features"]
        )



    for model, runs in groups.items():


        if len(runs)<2:
            continue


        for k in TOP_K_VALUES:


            jaccards=[]


            for a,b in combinations(runs,2):

                set_a=set(a[:k])
                set_b=set(b[:k])

                j=jaccard(
                    set_a,
                    set_b
                )

                jaccards.append(j)



            results.append(
                {
                "Dataset":dataset,
                "Model":model,
                "Top_K":k,
                "Mean_Jaccard":np.mean(jaccards),
                "SD_Jaccard":np.std(jaccards),
                "Number_of_Comparisons":len(jaccards),
                "Number_of_Runs":len(runs)
                }
            )


    return pd.DataFrame(results)



# ============================================================
# RUN
# ============================================================


print("="*70)
print("TOP-K FEATURE STABILITY ANALYSIS")
print("="*70)



# CDHIT

cdhit_features = extract_features(
    CDHIT_DIR,
    "CDHIT"
)


cdhit_results = calculate_stability(
    cdhit_features,
    "CDHIT"
)



# LOSO

loso_features = extract_features(
    LOSO_DIR,
    "LOSO"
)


loso_results = calculate_stability(
    loso_features,
    "LOSO"
)



# Combine

final=pd.concat(
    [
        cdhit_results,
        loso_results
    ],
    ignore_index=True
)



# Sort

final=final.sort_values(
    [
    "Dataset",
    "Model",
    "Top_K"
    ]
)



print("\n")
print("="*70)
print("RESULTS")
print("="*70)

print(final)



# ============================================================
# SAVE
# ============================================================


csv_file=os.path.join(
    OUTPUT_DIR,
    "TopK_Feature_Stability_Jaccard.csv"
)


excel_file=os.path.join(
    OUTPUT_DIR,
    "TopK_Feature_Stability_Jaccard.xlsx"
)


final.to_csv(
    csv_file,
    index=False
)


final.to_excel(
    excel_file,
    index=False
)


print("\n")
print("="*70)
print("COMPLETED")
print("="*70)

print("Saved:")
print(csv_file)
print(excel_file)