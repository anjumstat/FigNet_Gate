import os
import pandas as pd
import numpy as np
from itertools import combinations


# ======================================================
# PATH
# ======================================================

BASE_DIR = r"D:\zebfish1\revision1\FIGNet_LOSO_Results1_first\cv_runs"

OUTPUT = r"D:\zebfish1\revision1\Results\Feature_Stability_CV"

os.makedirs(OUTPUT, exist_ok=True)


# ======================================================
# JACCARD FUNCTION
# ======================================================

def jaccard(a,b):

    a=set(a)
    b=set(b)

    return len(a.intersection(b))/len(a.union(b))



# ======================================================
# ANALYSIS
# ======================================================

results=[]


for species in os.listdir(BASE_DIR):

    species_path=os.path.join(BASE_DIR,species)

    if not os.path.isdir(species_path):
        continue


    for config in os.listdir(species_path):

        config_path=os.path.join(species_path,config)


        if not os.path.isdir(config_path):
            continue


        for model in os.listdir(config_path):

            model_path=os.path.join(config_path,model)

            csv_dir=os.path.join(model_path,"csv_files")


            if not os.path.exists(csv_dir):
                continue


            fold_files=[]


            for f in os.listdir(csv_dir):

                if (
                    f.startswith("fold")
                    and "Top_Features" in f
                    and f.endswith(".csv")
                ):

                    fold_files.append(
                        os.path.join(csv_dir,f)
                    )


            if len(fold_files)<2:
                continue



            feature_sets=[]


            for file in fold_files:

                df=pd.read_csv(file)

                features=df["Feature_Index"].tolist()

                feature_sets.append(set(features))



            scores=[]


            for a,b in combinations(feature_sets,2):

                scores.append(
                    jaccard(a,b)
                )



            results.append({

                "Species":species,
                "Model":model,
                "Configuration":config,
                "Mean_Jaccard":np.mean(scores),
                "SD_Jaccard":np.std(scores),
                "Number_of_Folds":len(feature_sets)

            })



# ======================================================
# SAVE
# ======================================================


df=pd.DataFrame(results)


df=df.sort_values(
    "Mean_Jaccard",
    ascending=False
)


print("="*70)
print("CV FEATURE STABILITY")
print("="*70)

print(df.head(20))



df.to_csv(
    os.path.join(
        OUTPUT,
        "CV_Feature_Stability_Jaccard.csv"
    ),
    index=False
)


df.to_excel(
    os.path.join(
        OUTPUT,
        "CV_Feature_Stability_Jaccard.xlsx"
    ),
    index=False
)


print("\nSaved:")
print(OUTPUT)