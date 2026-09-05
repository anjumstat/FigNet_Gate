# 🧬 FIGNet: Feature Importance Gate Network

## Genome-Wide Enzyme Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Research-Scientific%20Computing-purple)

**FIGNet (Feature Importance Gate Network)** is a deep learning framework designed for **genome-wide enzyme classification from protein embeddings**.

The framework introduces a **differentiable Feature Importance Gate**, allowing the model to automatically learn the contribution of each embedding dimension while providing intrinsic interpretability.

This repository contains the complete implementation, preprocessing pipeline, training framework, evaluation scripts, and interpretability analysis used in:

> **"FIGNet: A Deep Learning Framework with Feature Importance Gate for Genome-Wide Enzyme Classification"**

---

# ✨ Key Contributions

## 🔹 Feature Importance Learning

* Differentiable feature importance gate
* Learns continuous importance weights for protein embedding dimensions
* Provides model interpretability without relying only on post-hoc methods

## 🔹 Comprehensive Evaluation Framework

The pipeline includes:

* Leave-One-Species-Out (LOSO) cross-validation
* 10-fold stratified cross-validation
* Homology-aware CD-HIT evaluation
* Multiple baseline comparisons
* Automated checkpoint recovery system

## 🔹 Interpretability Analysis

Supported explanation methods:

* Feature Importance Gate
* SHAP analysis
* LIME explanations

---

# 🚀 Main Features

| Feature                  | Description                                     |
| ------------------------ | ----------------------------------------------- |
| 🧠 FIGNet Models         | 5 variants with feature importance gating       |
| 🔍 Interpretability      | Gate-based explanation + SHAP + LIME            |
| 🧬 Cross-species Testing | Leave-One-Species-Out evaluation                |
| 🔬 Homology Control      | CD-HIT sequence similarity filtering            |
| 📊 Metrics               | MCC, F1-score, AUC, Accuracy, Precision, Recall |
| 💾 Checkpoint System     | Resume interrupted experiments automatically    |
| ⚖️ Baselines             | Comparison with 6 classical ML models           |

---

# 📂 Repository Structure

```text
FIGNet/
│
├── codes/
│   ├── 01_Organize_Fish_Data.py
│   ├── 02_Unified_Processor.py
│   ├── 03_Prepare_Binary_Data.py
│   ├── 04_Create_LOSO_Splits.py
│   └── 05_fignet_LOSO_training.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── cv_runs/
│   ├── loso_results/
│   ├── checkpoint.json
│   ├── LOSO_Results_All_Models.csv
│   └── LOSO_Method_Summary.csv
│
├── requirements.txt
└── README.md
```

---

# 🧬 Dataset

## Species Dataset

The dataset contains **12 fish species** with a total of:

| Category       | Number |
| -------------- | -----: |
| Total Proteins |  4,568 |
| Enzymes        |  1,151 |
| Non-Enzymes    |  3,417 |

## Species Distribution

| Species         | Proteins |  Enzymes | Non-Enzymes |
| --------------- | -------: | -------: | ----------: |
| Zebrafish       |     3303 |      869 |        2434 |
| Rainbow Trout   |      351 |       65 |         286 |
| Atlantic Salmon |      183 |       55 |         128 |
| Fugu            |      172 |       34 |         138 |
| Channel Catfish |      103 |       15 |          88 |
| Goldfish        |      129 |       26 |         103 |
| Common Carp     |      117 |       31 |          86 |
| Tetraodon       |       75 |       22 |          53 |
| Medaka          |       74 |       25 |          49 |
| Coho Salmon     |       27 |        4 |          23 |
| Nile Tilapia    |       21 |        3 |          18 |
| Electric Eel    |       13 |        2 |          11 |
| **Total**       | **4568** | **1151** |    **3417** |

---

# 🧪 Protein Embeddings

| Property  | Details                                        |
| --------- | ---------------------------------------------- |
| Source    | UniProt Knowledgebase (Swiss-Prot)             |
| Type      | Pre-computed protein language model embeddings |
| Dimension | 1024 features                                  |
| Format    | Embedding_0 → Embedding_1023                   |

---

# ⚙️ Installation

## Requirements

* Python ≥ 3.9
* TensorFlow ≥ 2.x
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
* SHAP *(optional)*
* LIME *(optional)*
* skrebate *(optional)*

---

## Setup

```bash
# Clone repository

git clone https://github.com/yourusername/FIGNet.git

cd FIGNet


# Install dependencies

pip install -r requirements.txt


# Install interpretability packages

pip install shap lime skrebate
```
# ▶️ Usage

The complete FIGNet pipeline consists of five major stages:

---

## Step 1 — Organize Raw Data

```bash
python codes/01_Organize_Fish_Data.py
```

**Input**

```text
Raw HDF5 and TSV files
```

**Output**

```text
Organized dataset
```

---

## Step 2 — Process and Merge Data

```bash
python codes/02_Unified_Processor.py
```

**Input**

```text
Organized data from Step 1
```

**Output**

```text
Processed embeddings with labels
```

---

## Step 3 — Prepare Binary Classification Dataset

```bash
python codes/03_Prepare_Binary_Data.py
```

**Input**

```text
Combined dataset from Step 2
```

**Output**

```text
Binary enzyme/non-enzyme classification dataset
```

---

## Step 4 — Create LOSO Splits

```bash
python codes/04_Create_LOSO_Splits.py
```

**Input**

```text
Binary dataset containing species information
```

**Output**

```text
Leave-One-Species-Out training/testing splits
```

---

## Step 5 — Train and Evaluate FIGNet Models

```bash
python codes/05_fignet_LOSO_training.py
```

**Input**

```text
LOSO splits
```

**Output**

```text
Cross-validation and independent LOSO test results
```

---

# 🧠 Models

## FIGNet Variants

The framework contains five FIGNet architectures:

| Model                  | Description                         |
| ---------------------- | ----------------------------------- |
| FIGNet_Gate_Only       | Feature Importance Gate only        |
| FIGNet_Gate_RealVD     | Gate + Real Variational Dropout     |
| FIGNet_Gate_AdaptiveVD | Gate + Adaptive Variational Dropout |
| FIGNet_Gate_Sparsity   | Gate + Dynamic Sparsity             |
| FIGNet_Gate_Full       | Complete model with all components  |

---

## Baseline Models

Six classical machine learning models are implemented for comparison:

| Model               | Description                            |
| ------------------- | -------------------------------------- |
| Logistic Regression | Linear classification model            |
| MLP Baseline        | Standard multilayer perceptron         |
| SVM-RBF             | Support Vector Machine with RBF kernel |
| SVM-Linear          | Linear Support Vector Machine          |
| ReliefF-MLP         | ReliefF feature selection + MLP        |
| ReliefF-SVM         | ReliefF feature selection + SVM        |

---

# 💾 Checkpoint System

FIGNet includes a **three-level checkpoint mechanism** to automatically recover interrupted experiments.

| Level   | Tracks                                                          | Storage         |
| ------- | --------------------------------------------------------------- | --------------- |
| Level 1 | Individual CV experiments (method + learning rate + batch size) | checkpoint.json |
| Level 2 | Species-level CV completion                                     | checkpoint.json |
| Level 3 | Individual LOSO model testing                                   | checkpoint.json |

---

## Resume Interrupted Training

Simply restart the training command:

```bash
python codes/05_fignet_LOSO_training.py
```

The framework automatically:

✅ Loads previous checkpoints <br>
✅ Detects completed experiments <br>
✅ Skips finished runs <br>
✅ Continues from the last unfinished experiment

---

# 📊 Results Organization

## Cross-Validation Results

Results are saved as:

```text
results/
└── cv_runs/
    └── species/
        └── lr_[lr]_bs_[bs]/
            └── method/
```

Structure:

```text
method/
│
├── csv_files/
│   ├── Experiment_Summary.csv
│   ├── Fold_Metrics.csv
│   └── All_CV_Predictions.csv
│
├── npy_files/
│
├── plots/
│
└── models/
```

---

## LOSO Independent Test Results

```text
results/
└── loso_results/
    └── species/
        └── method_lr_[lr]_bs_[bs]/
```

Contains:

```text
csv_files/
│
├── Final_Independent_Test_Result.csv
└── Independent_Test_Predictions.csv


npy_files/

plots/

models/
```

---

# 📁 Summary Result Files

The final generated outputs include:

| File                         | Description                            |
| ---------------------------- | -------------------------------------- |
| checkpoint.json              | Training progress tracking             |
| LOSO_Results_All_Models.csv  | Complete LOSO results                  |
| LOSO_Method_Summary.csv      | Performance summary by model           |
| LOSO_Species_Performance.png | Species-wise performance visualization |

---

# 📈 Evaluation Metrics

FIGNet performance is evaluated using multiple metrics:

| Metric            | Description                                                               |
| ----------------- | ------------------------------------------------------------------------- |
| Accuracy          | Overall prediction correctness                                            |
| Precision         | Positive prediction reliability                                           |
| Recall            | True positive detection ability                                           |
| F1 Score          | Harmonic mean of precision and recall                                     |
| MCC               | Matthews Correlation Coefficient (primary metric for imbalanced datasets) |
| AUC               | Area under ROC curve                                                      |
| Jaccard Stability | Feature selection consistency across folds                                |

---

# 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@article{shahzad2026fignet,

title={FIGNet: A Deep Learning Framework with Feature Importance Gate for Genome-Wide Enzyme Classification},

author={Shahzad, Anjum and ...},

journal={Scientific Reports},

year={2026}

}
```
# 🧬 CD-HIT Homology-Aware Evaluation

## Motivation

To address possible concerns regarding **sequence similarity leakage** between training and testing datasets, FIGNet was additionally evaluated using a **homology-aware data splitting strategy**.

The CD-HIT evaluation ensures that highly similar protein sequences are not distributed across different dataset partitions.

---

# 🔬 CD-HIT Workflow Overview

The homology-aware evaluation pipeline applies:

* CD-HIT clustering
* Sequence identity filtering
* Cluster-based train/validation/test separation
* Independent FIGNet evaluation

A sequence identity threshold of:

```text
60%
```

was used to minimize possible homology leakage.

---

# 📂 CD-HIT Pipeline Scripts

The complete CD-HIT evaluation consists of five additional scripts.

---

# Script 1

## `06_CDHIT_Clustering.py`

### Purpose

Perform CD-HIT clustering on all protein sequences to generate homology-reduced groups.

### Main Steps

1. Load protein sequences from species-specific FASTA files

2. Combine all sequences into a single FASTA file

3. Run CD-HIT clustering:

```text
Sequence identity threshold: 60%
Word size: 4
```

4. Assign cluster IDs to proteins

### Output

```text
filtered_4568_cdhit/

└── sequences.fasta
```

Additional output:

```text
CD-HIT cluster assignments
```

---

# Script 2

## `07_Extract_Protein_IDs.py`

### Purpose

Extract unique protein identifiers from the original dataset.

### Processing

* Load:

```text
all_fish_embeddings_combined.csv
```

* Extract unique:

```text
UniProt_ID
```

* Save reference IDs

### Output

```text
filtered_4568/

└── protein_ids_4568.txt
```

Contains:

```text
4,527 unique protein IDs
```

---

# Script 3

## `08_Filter_New_Datasets.py`

### Purpose

Filter homology-aware datasets to retain only proteins present in the reference dataset.

### Processing

The script:

* Loads:

```text
train.csv
val.csv
test.csv
```

* Filters using:

```text
protein_ids_4568.txt
```

* Generates clean datasets

### Output

```text
filtered_4568_exact/

├── train.csv
├── val.csv
└── test.csv
```

---

# Script 4

## `09_Run_CDHIT_Splits.py`

### Purpose

Generate final homology-aware train/validation/test splits.

### Main Steps

1. Combine filtered datasets

2. Load corresponding FASTA sequences

3. Perform CD-HIT clustering

4. Assign cluster IDs

5. Split clusters into:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        10% |
| Testing    |        20% |

---

# Final CD-HIT Dataset

| Split      |  Samples |  Enzymes | Non-Enzymes |
| ---------- | -------: | -------: | ----------: |
| Training   |     3200 |      791 |        2409 |
| Validation |      481 |      128 |         353 |
| Testing    |      887 |      232 |         655 |
| **Total**  | **4568** | **1151** |    **3417** |

---

# CD-HIT Statistics

| Parameter          |          Value |
| ------------------ | -------------: |
| Total sequences    |           4568 |
| Sequences retained |           4297 |
| CD-HIT clusters    |           3374 |
| Identity threshold |            60% |
| Split strategy     | Homology-aware |

---

# Script 5

## `10_FIGNet_CDHIT_Training.py`

### Purpose

Train FIGNet models and baseline methods using CD-HIT homology-aware splits.

---

## Training Pipeline

For every model configuration:

1. Load CD-HIT datasets

2. Extract:

```text
1024-dimensional protein embeddings
```

3. Train model

4. Validate using validation set

5. Evaluate on independent test set

6. Generate explanations:

* SHAP
* LIME

7. Save results with checkpoint support

---

# Models Evaluated

## FIGNet Models

* FIGNet_Gate_Only
* FIGNet_Gate_RealVD
* FIGNet_Gate_AdaptiveVD
* FIGNet_Gate_Sparsity
* FIGNet_Gate_Full

## Baseline Models

* Logistic Regression
* MLP Baseline
* SVM-RBF
* SVM-Linear
* ReliefF-MLP
* ReliefF-SVM

---

# CD-HIT Output Structure

```text
FIGNet_CDHIT_Results/

├── runs/
│   └── lr_[lr]_bs_[bs]/
│       └── method/
│
├── CDHIT_Results_All_Models.csv
│
├── CDHIT_Method_Summary.csv
│
└── checkpoint.json
```

Individual model results:

```text
method/

├── csv_files/
│
│   ├── Experiment_Summary.csv
│   └── Test_Predictions.csv
│
├── npy_files/
│
├── plots/
│
└── models/
```

---

# 🛡️ How CD-HIT Addresses Reviewer Concerns

| Concern                                      | Solution                                                |
| -------------------------------------------- | ------------------------------------------------------- |
| Random splits may introduce homology leakage | CD-HIT removes high similarity sequences between splits |
| Limited cross-species generalization         | Homology-aware testing provides stricter evaluation     |
| Possible test-set contamination              | Cluster-based separation ensures independent evaluation |

---

# 📜 Script Summary

| Order | Script                        | Purpose                      |
| ----- | ----------------------------- | ---------------------------- |
| 1     | `06_CDHIT_Clustering.py`      | Perform CD-HIT clustering    |
| 2     | `07_Extract_Protein_IDs.py`   | Extract protein IDs          |
| 3     | `08_Filter_New_Datasets.py`   | Filter datasets              |
| 4     | `09_Run_CDHIT_Splits.py`      | Create homology-aware splits |
| 5     | `10_FIGNet_CDHIT_Training.py` | Train FIGNet models          |

---

# 🌟 Reproducibility

This repository provides:

✅ Complete preprocessing pipeline
✅ Training scripts
✅ Baseline comparisons
✅ Interpretability analysis
✅ LOSO evaluation
✅ Homology-aware validation
✅ Checkpoint-based experiment recovery

The complete framework enables reproducible genome-wide enzyme classification experiments using deep learning.

---

# 📬 Contact

For questions regarding the implementation, experiments, or manuscript:

**Corresponding Author:**
Anjum Shahzad

---

⭐ If this repository contributes to your research, please consider citing the associated publication.
