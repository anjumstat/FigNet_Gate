# FigNet: Enzyme Classification Pipeline

This repository contains Python scripts for VARDON (Variational Adaptive Real Dropout Neural Network), a pipeline designed for enzyme classification across 12 fish species.

---

## File Descriptions

### 01_organize_fish_data.py
**Purpose:** Organize raw UniProt downloads into structured format.

**Functionality:**
- Scans raw data folder with species subfolders.
- Identifies HDF5 embedding files and TSV annotation files.
- Copies HDF5 files as `embeddings.h5` and TSV files as `{species}_annotations.tsv`.
- Creates `metadata.csv` and `master_summary.csv`.

**Input:** `D:\zebfish\`  
**Output:** `D:\zebfish_organized\`

---

### 02_uniprot_data_processor.py
**Purpose:** Process embeddings and annotations to create labeled datasets.

**Functionality:**
- Loads HDF5 embeddings and TSV annotations.
- Labels proteins as `Enzyme` or `Non-enzyme`.
- Merges embeddings with labels by UniProt ID.
- Saves clean dataset for deep learning.

**Input:** `D:\zebfish_organized\`  
**Output:** `D:\zebfish_processed_results\combined_data\clean_fish_dataset_for_dl.csv`

---

### 03_prepare_binary_dataset.py
**Purpose:** Convert labeled dataset into binary format.

**Functionality:**
- Removes metadata columns.
- Converts labels to numeric: `Enzyme` → 1, `Non-enzyme` → 0.
- Renames classification column to `Label`.

**Input:** `clean_fish_dataset_for_dl.csv`  
**Output:** `binary_classification_dataset.csv`

---

### 04_Feature_Gate_Experiments.py
**Purpose:** Run FigNet Gate variants and baseline models.

**Functionality:**
- Implements 5 DynamicSparsity variants and 3 baseline models.
- Performs 90/10 train-test split.
- Runs 10-fold cross-validation.
- Evaluates best model on test set.
- Saves training histories, metrics, and trained models.

**Input:** `binary_classification_dataset.csv`  
**Output:** Result directories for each learning rate (0.01, 0.001, 0.0001)

---

### 05_generate_figures.py
**Purpose:** Create publication-ready figures.

**Figures Generated:**
1. Hyperparameter effects
2. Training and validation curves
3. FigNet variant performance
4. Cross-validation boxplots

**Input:** Combined fold metrics + training histories  
**Output:** Figures in PNG and TIFF

---
### 06_generate_roc_curve.py
**Purpose:** Generate ROC curves.

**Functionality:**
- Computes FPR, TPR, and AUC for each method.
- Plots ROC curves for all models.
- Saves figures in PNG and TIFF.

**Input:** Predictions and true labels (`.npy`)  
**Output:** ROC curve figures

---
## Execution Order

1. `01_organize_fish_data.py`
2. `02_uniprot_data_processor.py`
3. `03_prepare_binary_dataset.py`
4. `04_Feature_Gate_Experiments.py`
5. `05_generate_figures`
6. `06_generate_roc_curve.py`

---

## Directory Structure
