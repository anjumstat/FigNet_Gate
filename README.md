# FigNet: Enzyme Classification Pipeline

# FIGNet: Feature Importance Gate Network for Genome-Wide Enzyme Classification

## Overview

FIGNet (Feature Importance Gate Network) is a deep learning framework for genome-wide enzyme classification from protein embeddings. It introduces a differentiable feature importance gate that learns continuous importance weights for each embedding dimension, enabling intrinsic model interpretability without post-hoc approximations.

This repository contains the complete code, data processing pipeline, and evaluation framework used in the manuscript:

> **"FIGNet: A Deep Learning Framework with Feature Importance Gate for Genome-Wide Enzyme Classification"**

## Key Features

- **5 FIGNet variants** with differentiable feature importance gate
- **Interpretability** via feature gate, SHAP, and LIME
- **Leave-One-Species-Out (LOSO)** cross-validation for cross-species generalization
- **10-fold stratified cross-validation** for hyperparameter selection
- **11 baseline models** including MLP, Logistic Regression, SVM, and ReliefF
- **Comprehensive evaluation** with MCC, F1, AUC, Accuracy, Precision, Recall
- **Three-level checkpoint system** for resuming interrupted runs

---

## Repository Structure
FIGNet/
├── codes/
│ ├── 01_Organize_Fish_Data.py # Organize raw HDF5 and TSV files
│ ├── 02_Unified_Processor.py # Process and merge embeddings with labels
│ ├── 03_Prepare_Binary_Data.py # Convert to binary classification format
│ ├── 04_Create_LOSO_Splits.py # Create Leave-One-Species-Out splits
│ └── 05_fignet_LOSO_training.py # Main training script with checkpointing
├── data/
│ ├── raw/ # Raw HDF5 and TSV files (not included)
│ └── processed/ # Processed datasets (not included)
├── results/ # Results directory (generated)
│ ├── cv_runs/ # CV results per species
│ ├── loso_results/ # LOSO test results per species
│ ├── checkpoint.json # Progress checkpoint
│ ├── LOSO_Results_All_Models.csv # All LOSO results
│ └── LOSO_Method_Summary.csv # Summary by method
├── requirements.txt # Python dependencies
└── README.md # This file

---

## Dataset

### Species Included (12 Fish Species)

| Species | Proteins | Enzymes | Non-Enzymes |
|---------|----------|---------|-------------|
| Zebrafish | 3,303 | 869 | 2,434 |
| Rainbow Trout | 351 | 65 | 286 |
| Atlantic Salmon | 183 | 55 | 128 |
| Fugu | 172 | 34 | 138 |
| Channel Catfish | 103 | 15 | 88 |
| Goldfish | 129 | 26 | 103 |
| Common Carp | 117 | 31 | 86 |
| Tetraodon | 75 | 22 | 53 |
| Medaka | 74 | 25 | 49 |
| Coho Salmon | 27 | 4 | 23 |
| Nile Tilapia | 21 | 3 | 18 |
| Electric Eel | 13 | 2 | 11 |
| **Total** | **4,568** | **1,151** | **3,417** |

### Protein Embeddings

- **Source**: UniProt Knowledgebase (Swiss-Prot)
- **Embedding type**: Pre-computed UniProt protein language model embeddings
- **Dimension**: 1024-dimensional vector representations
- **Format**: `Embedding_0` to `Embedding_1023`

---

## Installation

### Requirements

- Python 3.9+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- SHAP (optional, for interpretability)
- LIME (optional, for interpretability)
- skrebate (optional, for ReliefF)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FIGNet.git
cd FIGNet

# Install dependencies
pip install -r requirements.txt

# Optional: Install for interpretability
pip install shap lime skrebate
requirements.txt
tensorflow>=2.10.0
scikit-learn>=1.1.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
h5py>=3.7.0
joblib>=1.2.0
Usage
Step 1: Organize Raw Data
python codes/01_Organize_Fish_Data.py
Input: Raw HDF5 and TSV files in D:\zebfish\
Output: Organized data in D:\zebfish_organized\

Step 2: Process and Merge Data
python codes/02_Unified_Processor.py
Input: Organized data from Step 1
Output: Processed embeddings and combined dataset

Step 3: Prepare Binary Classification Data
python codes/03_Prepare_Binary_Data.py
Input: Combined dataset from Step 2
Output: Binary classification dataset with species labels

Step 4: Create LOSO Splits
python codes/04_Create_LOSO_Splits.py
Input: Binary dataset with species
Output: Train/test splits for each species

Step 5: Train and Evaluate FIGNet Models
python codes/05_fignet_LOSO_training.py
Input: LOSO splits from Step 4
Output: CV and LOSO results for all models

Models
FIGNet Variants (5)
Model	Description
FIGNet_Gate_Only	Feature Importance Gate only
FIGNet_Gate_RealVD	Gate + Real Variational Dropout
FIGNet_Gate_AdaptiveVD	Gate + Adaptive Variational Dropout
FIGNet_Gate_Sparsity	Gate + Dynamic Sparsity
FIGNet_Gate_Full	All components combined
Baseline Models (6)
Model	Description
Logistic_Regression	Linear classifier
MLP_Baseline	Standard MLP with dropout
SVM_RBF	SVM with RBF kernel
SVM_Linear	SVM with linear kernel
ReliefF_MLP	ReliefF feature selection + MLP
ReliefF_SVM	ReliefF feature selection + SVM
Checkpoint System
The code includes a three-level checkpoint system for resuming interrupted runs:

Level	What It Tracks	Saved In
Level 1	Individual CV runs (method + lr + bs)	checkpoint.json
Level 2	Species CV complete (all models done)	checkpoint.json
Level 3	Individual LOSO model tests	checkpoint.json
Resuming After Interruption
Simply re-run the same command:
python codes/05_fignet_LOSO_training.py
The code will automatically:

Load checkpoint.json

Skip completed runs

Resume from where it stopped

Results Structure
CV Results
results/cv_runs/[species]/lr_[lr]_bs_[bs]/[method]/
├── csv_files/
│   ├── Experiment_Summary.csv       # Mean ± std across folds
│   ├── Fold_Metrics.csv             # Per-fold metrics
│   └── All_CV_Predictions.csv       # All predictions
├── npy_files/                       # All fold .npy files
├── plots/                           # Training history and confusion matrices
└── models/                          # 10 fold models
LOSO Results
results/loso_results/[species]/[method]_lr[lr]_bs[bs]/
├── csv_files/
│   ├── Final_Independent_Test_Result.csv   # Test metrics
│   └── Independent_Test_Predictions.csv    # Per-sample predictions
├── npy_files/                              # Predictions and labels
├── plots/                                  # Confusion matrix
└── models/                                 # Final trained model
Summary Files
results/
├── checkpoint.json                         # Progress checkpoint
├── LOSO_Results_All_Models.csv             # All LOSO results
├── LOSO_Method_Summary.csv                 # Summary by method
└── LOSO_Species_Performance.png            # Species-wise performance plot
Evaluation Metrics
Accuracy: Overall correctness

Precision: Positive predictive value

Recall: Sensitivity / True positive rate

F1 Score: Harmonic mean of precision and recall

MCC: Matthews correlation coefficient (primary metric for imbalanced data)

AUC: Area under ROC curve

Jaccard Stability: Feature selection consistency across CV folds

Citation
If you use this code in your research, please cite:
@article{shahzad2026fignet,
  title={FIGNet: A Deep Learning Framework with Feature Importance Gate for Genome-Wide Enzyme Classification},
  author={Shahzad, Anjum and ...},
  journal={Scientific Reports},
  year={2026},
}

