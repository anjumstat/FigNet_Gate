# -*- coding: utf-8 -*-
"""
Figure 4:
Feature Importance and Stability Analysis of FIGNet

Panels:
A - Top latent embedding features
B - Feature selection stability (LOSO Jaccard)
C - SHAP feature contribution

Author: H.A.R
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



# =====================================================
# PATHS
# =====================================================


TOP_FEATURE_FILE = (
r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results\atlansalmon\FIGNet_Gate_AdaptiveVD_lr0.0001_bs128\csv_files\Final_Top_Features.csv"
)


STABILITY_FILE = (
r"D:\zebfish1\revision1\Results\Feature_Stability_TopK\TopK_Feature_Stability_Jaccard.xlsx"
)


SHAP_FILE = (
r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results\atlansalmon\FIGNet_Gate_AdaptiveVD_lr0.0001_bs128\shap_explanations\shap_summary_plot.png"
)


OUTPUT = (
r"D:\zebfish1\revision1\Results\Figures"
)


os.makedirs(
    OUTPUT,
    exist_ok=True
)



# =====================================================
# CREATE FIGURE
# =====================================================


fig = plt.figure(
    figsize=(16,6)
)


gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1,1,1.2]
)


ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])





# =====================================================
# PANEL A
# TOP LATENT FEATURE IMPORTANCE
# =====================================================


print("="*70)
print("TOP LATENT FEATURE IMPORTANCE")
print("="*70)



df = pd.read_csv(
    TOP_FEATURE_FILE
)



print("\nOriginal feature ranking:")
print(df.head(20))



# -----------------------------------------------------
# REMOVE CONSTANT METADATA FEATURE
# -----------------------------------------------------

df_filtered = df[
    df["Feature_Name"] != "Uncertain_Annotation"
]



# Keep only embedding features

df_filtered = df_filtered[
    df_filtered["Feature_Name"].str.contains(
        "Embedding"
    )
]



top = df_filtered.head(20)



print("\nFinal features shown in Figure 4A:")
print(top[[
    "Rank",
    "Feature_Name",
    "Importance"
]])



ax1.barh(

    top["Feature_Name"][::-1],

    top["Importance"][::-1]

)



ax1.set_title(

    "A. Top 20 Latent Features\n(FIGNet AdaptiveVD)",

    fontsize=12,

    fontweight="bold"

)



ax1.set_xlabel(

    "Feature Importance Score"

)



ax1.tick_params(

    axis="y",

    labelsize=7

)





# =====================================================
# PANEL B
# LOSO FEATURE STABILITY
# =====================================================


print("\n")
print("="*70)
print("LOSO FEATURE STABILITY")
print("="*70)



stab = pd.read_excel(
    STABILITY_FILE
)



print(stab.columns)

print(stab.head())



# Keep LOSO only

if "Dataset" in stab.columns:

    stab = stab[
        stab["Dataset"]=="LOSO"
    ]



# FIGNet models only

stab = stab[
    stab["Model"].str.contains(
        "FIGNet"
    )
]



stab_summary = (

    stab.groupby(
        "Model"
    )["Mean_Jaccard"]

    .mean()

    .sort_values(
        ascending=False
    )

)



print("\nFinal stability:")
print(stab_summary)



ax2.bar(

    stab_summary.index,

    stab_summary.values

)



ax2.set_title(

    "B. Feature Selection Stability\n(LOSO Jaccard Similarity)",

    fontsize=12,

    fontweight="bold"

)



ax2.set_ylabel(

    "Mean Jaccard Similarity"

)



ax2.tick_params(

    axis="x",

    rotation=60,

    labelsize=8

)





# =====================================================
# PANEL C
# SHAP SUMMARY
# =====================================================


print("\n")
print("="*70)
print("SHAP FEATURE CONTRIBUTION")
print("="*70)



if os.path.exists(SHAP_FILE):


    img = Image.open(
        SHAP_FILE
    )


    ax3.imshow(
        img
    )


    ax3.axis(
        "off"
    )


else:


    ax3.text(

        0.5,

        0.5,

        "SHAP file not found",

        ha="center",

        va="center"

    )



ax3.set_title(

    "C. SHAP Feature Contribution\n(FIGNet AdaptiveVD)",

    fontsize=12,

    fontweight="bold"

)





# =====================================================
# SAVE FIGURE
# =====================================================


plt.tight_layout()



png = os.path.join(

    OUTPUT,

    "Figure4_Feature_Importance_Stability_SHAP_UPDATED.png"

)



pdf = os.path.join(

    OUTPUT,

    "Figure4_Feature_Importance_Stability_SHAP_UPDATED.pdf"

)



plt.savefig(

    png,

    dpi=300,

    bbox_inches="tight"

)



plt.savefig(

    pdf,

    dpi=300,

    bbox_inches="tight"

)



plt.show()





print("\n")
print("="*70)
print("FIGURE 4 COMPLETED")
print("="*70)


print("Saved:")
print(png)
print(pdf)