# -*- coding: utf-8 -*-
"""
Training Dynamics Analysis
Six Neural Architectures
LOSO + CD-HIT Redundancy-Controlled Evaluation

Author: H.A.R
"""

import os
import numpy as np
import matplotlib.pyplot as plt



# =====================================================
# PATHS
# =====================================================

LOSO_PATH = r"D:\zebfish1\revision1\FIGNet_LOSO_Results_Priority_All_Models\loso_results"

CDHIT_PATH = r"D:\zebfish1\revision1\FIGNet_CDHIT_Results\runs"

OUTPUT = r"D:\zebfish1\revision1\Results\Figures"

os.makedirs(
    OUTPUT,
    exist_ok=True
)



# =====================================================
# SIX NEURAL MODELS
# =====================================================


models = {


"FIGNet AdaptiveVD":
{
"loso":
"FIGNet_Gate_AdaptiveVD_lr0.0001_bs128",

"cdhit":
("FIGNet_Gate_AdaptiveVD",
"lr_0_00010_bs_128")
},


"FIGNet Full":
{
"loso":
"FIGNet_Gate_Full_lr0.0001_bs128",

"cdhit":
("FIGNet_Gate_Full",
"lr_0_00010_bs_128")
},


"FIGNet Gate Only":
{
"loso":
"FIGNet_Gate_Only_lr0.0001_bs128",

"cdhit":
("FIGNet_Gate_Only",
"lr_0_00010_bs_128")
},


"FIGNet RealVD":
{
"loso":
"FIGNet_Gate_RealVD_lr0.0001_bs64",

"cdhit":
("FIGNet_Gate_RealVD",
"lr_0_00010_bs_64")
},


"FIGNet Sparsity":
{
"loso":
"FIGNet_Gate_Sparsity_lr0.0001_bs64",

"cdhit":
("FIGNet_Gate_Sparsity",
"lr_0_00010_bs_64")
},


"MLP Baseline":
{
"loso":
"MLP_Baseline_lr0.0001_bs128",

"cdhit":
("MLP_Baseline",
"lr_0_00010_bs_128")
}

}



# =====================================================
# LOSO SPECIES
# =====================================================

species = "atlansalmon"



# =====================================================
# CREATE FIGURE
# =====================================================


fig, ax = plt.subplots(
    2,
    2,
    figsize=(15,11)
)





# =====================================================
# LOSO TRAINING CURVES
# =====================================================


print("\n")
print("="*80)
print("LOSO TRAINING DYNAMICS")
print("="*80)



for name,data in models.items():


    folder = os.path.join(

        LOSO_PATH,

        species,

        data["loso"],

        "npy_files"

    )


    print("\nMODEL:",name)
    print("PATH:",folder)



    acc_file=os.path.join(
        folder,
        "final_train_accuracy.npy"
    )


    loss_file=os.path.join(
        folder,
        "final_train_loss.npy"
    )



    if os.path.exists(acc_file):


        acc=np.load(acc_file)


        print(
            "Epochs:",
            len(acc)
        )


        print(
            "Final Accuracy:",
            round(acc[-1],4)
        )


        ax[0,0].plot(
            acc,
            label=name
        )



    if os.path.exists(loss_file):


        loss=np.load(loss_file)


        print(
            "Final Loss:",
            round(loss[-1],4)
        )


        ax[1,0].plot(
            loss,
            label=name
        )







# =====================================================
# CD-HIT REDUNDANCY CONTROLLED CURVES
# =====================================================


print("\n")
print("="*80)
print("CD-HIT REDUNDANCY-CONTROLLED DYNAMICS")
print("="*80)



for name,data in models.items():


    model,config=data["cdhit"]



    folder=os.path.join(

        CDHIT_PATH,

        config,

        model,

        "npy_files"

    )


    print("\nMODEL:",name)
    print("PATH:",folder)



    train_acc=os.path.join(
        folder,
        "train_accuracy.npy"
    )


    val_acc=os.path.join(
        folder,
        "val_accuracy.npy"
    )


    train_loss=os.path.join(
        folder,
        "train_loss.npy"
    )


    val_loss=os.path.join(
        folder,
        "val_loss.npy"
    )



    if os.path.exists(train_acc):


        tr_acc=np.load(train_acc)


        print(
            "Final Train Accuracy:",
            round(tr_acc[-1],4)
        )


        ax[0,1].plot(
            tr_acc,
            label=name+" train"
        )



    else:

        print("Train accuracy missing")




    if os.path.exists(val_acc):


        va_acc=np.load(val_acc)


        print(
            "Final Validation Accuracy:",
            round(va_acc[-1],4)
        )


        ax[0,1].plot(

            va_acc,

            linestyle="--",

            label=name+" validation"

        )



    else:

        print("Validation accuracy missing")






    if os.path.exists(train_loss):


        tr_loss=np.load(train_loss)


        print(
            "Final Train Loss:",
            round(tr_loss[-1],4)
        )


        ax[1,1].plot(
            tr_loss,
            label=name+" train"
        )




    if os.path.exists(val_loss):


        va_loss=np.load(val_loss)


        print(
            "Final Validation Loss:",
            round(va_loss[-1],4)
        )


        ax[1,1].plot(

            va_loss,

            linestyle="--",

            label=name+" validation"

        )








# =====================================================
# MANUSCRIPT TITLES
# =====================================================


titles = [

"LOSO Training Accuracy",

"CD-HIT Redundancy-Controlled Accuracy Dynamics",

"LOSO Training Loss",

"CD-HIT Redundancy-Controlled Loss Dynamics"

]



for axis,title in zip(
    ax.flatten(),
    titles
):


    axis.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )


    axis.set_xlabel(
        "Epoch"
    )


    axis.set_ylabel(
        "Value"
    )


    axis.legend(
        fontsize=8
    )





plt.tight_layout()



# =====================================================
# SAVE FIGURE
# =====================================================


png=os.path.join(
    OUTPUT,
    "Figure_Training_Dynamics_LOSO_CDHIT_6Models.png"
)


pdf=os.path.join(
    OUTPUT,
    "Figure_Training_Dynamics_LOSO_CDHIT_6Models.pdf"
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
print("="*80)
print("TRAINING DYNAMICS FIGURE COMPLETED")
print("="*80)

print("Saved:")
print(png)
print(pdf)