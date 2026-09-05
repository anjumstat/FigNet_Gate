# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:29:05 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-

"""
Feature Selection Stability Analysis
Jaccard similarity using Feature_Index
"""

import os
import pandas as pd
import itertools
import numpy as np


# =====================================================
# PATHS
# =====================================================

CDHIT_PATH = (
r"D:\zebfish1\revision1\FIGNet_CDHIT_Results\runs"
)


LOSO_PATH = (
r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results"
)


OUTPUT = (
r"D:\zebfish1\revision1\Results\Feature_Stability"
)

os.makedirs(
    OUTPUT,
    exist_ok=True
)



# =====================================================
# JACCARD FUNCTION
# =====================================================

def jaccard(set1,set2):

    if len(set1)==0 and len(set2)==0:
        return 1

    return len(set1 & set2) / len(set1 | set2)



def calculate_pairwise_jaccard(feature_sets):

    scores=[]

    for a,b in itertools.combinations(feature_sets,2):

        scores.append(
            jaccard(a,b)
        )

    if len(scores)==0:
        return np.nan,np.nan

    return (
        np.mean(scores),
        np.std(scores)
    )



# =====================================================
# CD-HIT ANALYSIS
# =====================================================


def analyze_cdhit():

    print("\n"+"="*70)
    print("CD-HIT FEATURE STABILITY")
    print("="*70)


    records=[]


    for config in os.listdir(CDHIT_PATH):

        config_path=os.path.join(
            CDHIT_PATH,
            config
        )

        if not os.path.isdir(config_path):
            continue


        for model in os.listdir(config_path):

            feature_file=os.path.join(
                config_path,
                model,
                "csv_files",
                "Top_Features.csv"
            )


            if os.path.exists(feature_file):

                df=pd.read_csv(feature_file)


                features=set(
                    df["Feature_Index"]
                )


                records.append(
                    {
                    "Model":model,
                    "Configuration":config,
                    "Features":features
                    }
                )



    result=[]


    df=pd.DataFrame(records)


    for model,group in df.groupby("Model"):

        feature_sets=list(
            group["Features"]
        )


        mean,std=calculate_pairwise_jaccard(
            feature_sets
        )


        result.append(
            {
            "Model":model,
            "Mean_Jaccard":mean,
            "SD_Jaccard":std,
            "Number_of_Runs":len(feature_sets)
            }
        )


    result=pd.DataFrame(result)

    result=result.sort_values(
        "Mean_Jaccard",
        ascending=False
    )


    print(result)


    result.to_excel(
        os.path.join(
            OUTPUT,
            "CDHIT_Feature_Stability.xlsx"
        ),
        index=False
    )



# =====================================================
# LOSO ANALYSIS
# =====================================================


def analyze_loso():

    print("\n"+"="*70)
    print("LOSO FEATURE STABILITY")
    print("="*70)


    records=[]


    for species in os.listdir(LOSO_PATH):

        species_path=os.path.join(
            LOSO_PATH,
            species
        )


        if not os.path.isdir(species_path):
            continue



        for model_folder in os.listdir(species_path):

            feature_file=os.path.join(
                species_path,
                model_folder,
                "csv_files",
                "Final_Top_Features.csv"
            )


            if os.path.exists(feature_file):

                df=pd.read_csv(feature_file)


                features=set(
                    df["Feature_Index"]
                )


                # remove configuration
                model_name="_".join(
                    model_folder.split("_")[:3]
                )


                records.append(
                    {
                    "Species":species,
                    "Model":model_folder,
                    "Features":features
                    }
                )



    df=pd.DataFrame(records)


    results=[]


    # per model configuration stability

    for model,group in df.groupby("Model"):

        mean,std=calculate_pairwise_jaccard(
            list(group["Features"])
        )


        results.append(
            {
            "Model":model,
            "Mean_Jaccard":mean,
            "SD_Jaccard":std,
            "Runs":len(group)
            }
        )


    results=pd.DataFrame(results)


    results=results.sort_values(
        "Mean_Jaccard",
        ascending=False
    )


    print(results.head(20))


    results.to_excel(
        os.path.join(
            OUTPUT,
            "LOSO_Feature_Stability.xlsx"
        ),
        index=False
    )



# =====================================================
# RUN
# =====================================================


analyze_cdhit()

analyze_loso()


print("\n")
print("="*70)
print("COMPLETED")
print("="*70)

print("Saved:")
print(OUTPUT)
