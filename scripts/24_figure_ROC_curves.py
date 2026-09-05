# ==========================================================
# FINAL MANUSCRIPT ROC FIGURE
# Macro-average ROC
# LOSO + CD-HIT
# ==========================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc



# ==========================================================
# PATHS
# ==========================================================


LOSO_PATH = r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results"

CDHIT_PATH = r"D:\zebfish1\revision1\FIGNet_CDHIT_Results\runs"


SAVE_PATH = r"D:\zebfish1\revision1\Results\Figures"

os.makedirs(
    SAVE_PATH,
    exist_ok=True
)



# ==========================================================
# MODELS
# ==========================================================


LOSO_MODELS = {


"FIGNet AdaptiveVD":
"FIGNet_Gate_AdaptiveVD_lr0.0001_bs128",


"FIGNet Gate Only":
"FIGNet_Gate_Only_lr0.0001_bs128",


"FIGNet RealVD":
"FIGNet_Gate_RealVD_lr0.0001_bs64",


"SVM-RBF":
"SVM_RBF_lr0.001_bs32"

}



CDHIT_MODELS = {


"FIGNet AdaptiveVD":
(
"FIGNet_Gate_AdaptiveVD",
"lr_0_00010_bs_128"
),


"FIGNet Gate Only":
(
"FIGNet_Gate_Only",
"lr_0_00010_bs_128"
),


"FIGNet RealVD":
(
"FIGNet_Gate_RealVD",
"lr_0_00010_bs_64"
),


"SVM-RBF":
(
"SVM_RBF",
"lr_0_00100_bs_32_1"
)

}







# ==========================================================
# LOSO LOADER
# ==========================================================


def load_loso_species(model_folder):


    results=[]



    for species in os.listdir(LOSO_PATH):


        path=os.path.join(
            LOSO_PATH,
            species,
            model_folder,
            "npy_files"
        )


        label_file=os.path.join(
            path,
            "test_true_labels.npy"
        )


        prob_file=os.path.join(
            path,
            "test_predicted_proba.npy"
        )



        if os.path.exists(label_file):


            y=np.load(label_file)

            p=np.load(prob_file)



            results.append(
                (
                species,
                y,
                p
                )
            )



    return results







# ==========================================================
# MACRO ROC
# ==========================================================


def macro_average_roc(results):


    mean_fpr=np.linspace(
        0,
        1,
        200
    )


    interpolated_tprs=[]

    auc_values=[]



    for species,y,p in results:


        fpr,tpr,_=roc_curve(
            y,
            p
        )


        roc_auc=auc(
            fpr,
            tpr
        )


        auc_values.append(
            roc_auc
        )


        interp_tpr=np.interp(
            mean_fpr,
            fpr,
            tpr
        )


        interp_tpr[0]=0

        interpolated_tprs.append(
            interp_tpr
        )



    mean_tpr=np.mean(
        interpolated_tprs,
        axis=0
    )


    mean_tpr[-1]=1



    macro_auc=np.mean(
        auc_values
    )



    return mean_fpr,mean_tpr,macro_auc







# ==========================================================
# CD-HIT LOADER
# ==========================================================


def load_cdhit(model,config):


    csv_file=os.path.join(

        CDHIT_PATH,

        config,

        model,

        "csv_files",

        "Test_Predictions.csv"

    )



    if not os.path.exists(csv_file):

        return np.array([]),np.array([])



    df=pd.read_csv(csv_file)



    y=df["true_label"].values


    p=df["predicted_proba_enzyme"].values



    return y,p







# ==========================================================
# FIGURE
# ==========================================================


fig,ax=plt.subplots(
    1,
    2,
    figsize=(14,6)
)





# ==========================================================
# LOSO
# ==========================================================


print("="*70)
print("LOSO MACRO ROC")
print("="*70)



for name,model in LOSO_MODELS.items():


    results=load_loso_species(
        model
    )


    fpr,tpr,score=macro_average_roc(
        results
    )


    print(
        name,
        "Species:",
        len(results),
        "Macro AUC:",
        round(score,4)
    )



    ax[0].plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC={score:.3f})"
    )



ax[0].plot(
    [0,1],
    [0,1],
    linestyle="--"
)



ax[0].set_title(
    "LOSO Evaluation"
)


ax[0].set_xlabel(
    "False Positive Rate"
)


ax[0].set_ylabel(
    "True Positive Rate"
)


ax[0].legend(
    fontsize=9
)






# ==========================================================
# CD-HIT
# ==========================================================


print("\n")
print("="*70)
print("CD-HIT ROC")
print("="*70)



for name,(model,config) in CDHIT_MODELS.items():


    y,p=load_cdhit(
        model,
        config
    )


    if len(y)==0:

        continue



    fpr,tpr,_=roc_curve(
        y,
        p
    )


    score=auc(
        fpr,
        tpr
    )


    print(
        name,
        "AUC:",
        round(score,4)
    )



    ax[1].plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC={score:.3f})"
    )



ax[1].plot(
    [0,1],
    [0,1],
    linestyle="--"
)



ax[1].set_title(
    "CD-HIT Redundancy Controlled Evaluation"
)


ax[1].set_xlabel(
    "False Positive Rate"
)


ax[1].set_ylabel(
    "True Positive Rate"
)


ax[1].legend(
    fontsize=9
)





# ==========================================================
# SAVE
# ==========================================================


plt.tight_layout()



plt.savefig(
    os.path.join(
        SAVE_PATH,
        "Figure_ROC_Macro_LOSO_CDHIT.png"
    ),
    dpi=300,
    bbox_inches="tight"
)



plt.savefig(
    os.path.join(
        SAVE_PATH,
        "Figure_ROC_Macro_LOSO_CDHIT.pdf"
    ),
    dpi=300,
    bbox_inches="tight"
)



plt.show()



print("\nROC FIGURE COMPLETED")