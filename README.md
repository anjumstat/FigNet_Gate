# 🧬 FIGNet: Feature Importance Gate Network

## Genome-Wide Enzyme Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Research-Scientific%20Computing-purple)

**FIGNet (Feature Importance Gate Network)** is a deep learning framework for binary enzyme classification using protein language-model embeddings.

The framework introduces a differentiable **Feature Importance Gate** that learns continuous importance values for individual embedding dimensions while performing enzyme/non-enzyme classification.

This repository contains the preprocessing pipeline, FIGNet implementations, baseline models, cross-validation procedures, cross-species evaluation, homology-aware testing, interpretability analyses, statistical analyses, and figure-generation scripts associated with:

> **"FIGNet: A Deep Learning Framework with Feature Importance Gate for Genome-Wide Enzyme Classification"**

---

# ✨ Key Contributions

## 🔹 Feature Importance Learning

FIGNet includes a differentiable Feature Importance Gate that:

- assigns learnable importance values to embedding dimensions;
- allows feature contributions to be examined directly from the trained model;
- complements post-hoc interpretation approaches such as SHAP and LIME.

Gate-derived importance, SHAP, and LIME are treated as complementary explanation approaches and are not assumed to provide identical feature rankings.

---

## 🔹 Comprehensive Evaluation Framework

The repository contains multiple complementary evaluation strategies:

- **Nested cross-validation**
  - 10-fold outer stratified cross-validation for performance estimation;
  - 5-fold inner stratified cross-validation for hyperparameter selection.

- **Leave-One-Species-Out (LOSO) evaluation**
  - evaluates cross-species generalization by holding out one fish species at a time.

- **CD-HIT homology-aware evaluation**
  - separates protein clusters to reduce sequence-similarity leakage across dataset partitions.

- **Baseline comparisons**
  - Logistic Regression;
  - Multilayer Perceptron;
  - Linear SVM;
  - RBF SVM;
  - ReliefF + MLP;
  - ReliefF + SVM.

- **Interpretability analyses**
  - Feature Importance Gate;
  - SHAP;
  - LIME;
  - feature-stability analyses.

- **Statistical and manuscript-level analyses**
  - model summaries;
  - architecture-level analyses;
  - feature-stability summaries;
  - statistical comparisons;
  - manuscript figures.

---

# 🚀 Main Features

| Feature | Description |
|---|---|
| 🧠 FIGNet Models | Five Feature Importance Gate variants |
| 🔁 Nested CV | 5-fold inner model selection + 10-fold outer evaluation |
| 🧬 Cross-Species Testing | Leave-One-Species-Out evaluation |
| 🔬 Homology Control | CD-HIT sequence-cluster-aware evaluation |
| 🔍 Interpretability | Gate importance, SHAP, and LIME |
| 📊 Metrics | MCC, AUC, F1, Accuracy, Precision, Recall |
| 📈 Feature Stability | Jaccard-based feature stability analyses |
| ⚖️ Baselines | Neural-network and classical machine-learning comparisons |
| 💾 Checkpoints | Recovery of interrupted model-training experiments |

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
│   ├── 05_fignet_LOSO_training.py
│   ├── 06_CDHIT_Clustering.py
│   ├── 07_Extract_Protein_IDs.py
│   ├── 08_Filter_New_Datasets.py
│   ├── 09_Run_CDHIT_Splits.py
│   ├── 10_FIGNet_CDHIT_Training.py
│   ├── 11_Extract_CV_Results.py
│   ├── 12_Generate_CV_Tables.py
│   ├── 13_Statistical_Analysis.py
│   ├── 14_LOSO_Final_Analysis.py
│   ├── 15_LOSO_Model_Winner_Analysis.py
│   ├── 16_Architecture_Level_Analysis.py
│   ├── 17_Generate_Final_Manuscript_Tables.py
│   ├── 18_CDHIT_Result_Analysis.py
│   ├── 19_CDHIT_Manuscript_Table.py
│   ├── 20_Feature_Stability_Analysis.py
│   ├── 21_TopK_Feature_Stability.py
│   ├── 22_CV_Feature_Stability.py
│   ├── 23_Feature_Stability_Summary.py
│   ├── 24_Figure_ROC_Curves.py
│   ├── 25_Figure_Training_Dynamics.py
│   ├── 26_Figure_Feature_Importance.py
│   ├── 27_Figure_Model_Comparison.py
│   ├── 28_FIGNet_NestedCV_Training.py
│   └── 29_Figure1_NestedCV_Model_Comparison.py
│
├── data/
🧬 Dataset
Species Dataset
The study dataset contains proteins from 12 fish species.
Category	Number
Total Proteins	4,568
Enzymes	1,151
Non-Enzymes	3,417


Species Distribution
Species	Proteins	Enzymes	Non-Enzymes
Zebrafish	3303	869	2434
Rainbow Trout	351	65	286
Atlantic Salmon	183	55	128
Fugu	172	34	138
Channel Catfish	103	15	88
Goldfish	129	26	103
Common Carp	117	31	86
Tetraodon	75	22	53
Medaka	74	25	49
Coho Salmon	27	4	23
Nile Tilapia	21	3	18
Electric Eel	13	2	11
Total	4568	1151	3417


🧪 Protein Embeddings
Property	Details
Source	UniProt Knowledgebase
Representation	Pre-computed protein language-model embeddings
Dimension	1024
Feature format	Embedding_0 to Embedding_1023


Protein embeddings are used as the input representation for FIGNet and the baseline classification models.
⚙️ Installation
Requirements
The main Python dependencies include:
- Python ≥ 3.9
- TensorFlow 2.x
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn
- SciPy
- joblib
- SHAP (optional interpretability analysis)
- LIME (optional interpretability analysis)
- skrebate (for ReliefF models)
Setup
git clone https://github.com/yourusername/FIGNet.git
cd FIGNet

pip install -r requirements.txt
Optional interpretability and feature-selection packages can also be installed using:
pip install shap lime skrebate
▶️ Main Analysis Workflow
The complete analysis contains several complementary experimental branches rather than a single sequential evaluation.
The major workflow is:
Raw protein/annotation data
        │
        ▼
Data organization and preprocessing
        │
        ▼
Binary enzyme/non-enzyme dataset
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
LOSO evaluation                CD-HIT evaluation
Cross-species                  Homology-aware
generalization                 generalization
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
              Nested Cross-Validation
          Inner model selection + outer
             performance estimation
                       │
                       ▼
           Statistical / stability analyses
                       │
                       ▼
              Manuscript figures
Nested cross-validation provides the primary standard cross-validation framework for separating hyperparameter selection from outer-fold performance estimation.
1️⃣ Data Organization
01_Organize_Fish_Data.py
Organizes the raw fish protein datasets and associated annotation/embedding files for subsequent preprocessing.
python codes/01_Organize_Fish_Data.py
python codes/01_Organize_Fish_Data.py

2️⃣ Data Processing
02_Unified_Processor.py
Processes species-level UniProt information and integrates protein information with the corresponding embedding data.
python codes/02_Unified_Processor.py

3️⃣ Binary Classification Dataset
03_Prepare_Binary_Data.py
Creates the final binary classification dataset:
Enzyme     → 1
Non-enzyme → 0

Species information is retained where required for LOSO analysis.
python codes/03_Prepare_Binary_Data.py

🧬 Leave-One-Species-Out Evaluation
LOSO evaluation is used to assess cross-species generalization.
For each experiment:
1. one fish species is held out;
2. the remaining species form the training dataset;
3. the trained model is evaluated on the unseen species.
04_Create_LOSO_Splits.py
Creates species-specific training/testing datasets.
python codes/04_Create_LOSO_Splits.py

05_fignet_LOSO_training.py
Runs FIGNet and baseline models under the LOSO evaluation framework.
python codes/05_fignet_LOSO_training.py

The LOSO experiments are used specifically to evaluate performance on an unseen species rather than as the primary nested hyperparameter-selection framework.
🔬 CD-HIT Homology-Aware Evaluation
Motivation
Protein sequence similarity can lead to overly optimistic performance estimates when highly related proteins occur in both training and testing partitions.
A CD-HIT-based analysis is therefore included to evaluate performance under a stricter sequence-similarity-controlled setting.
A sequence identity threshold of:
60%

is used in the homology-aware workflow.
CD-HIT Pipeline
06_CDHIT_Clustering.py
Performs CD-HIT clustering of protein sequences.
python codes/06_CDHIT_Clustering.py

07_Extract_Protein_IDs.py
Extracts protein identifiers corresponding to the reference classification dataset.
python codes/07_Extract_Protein_IDs.py

08_Filter_New_Datasets.py
Filters the homology-aware datasets to proteins included in the target reference dataset.
python codes/08_Filter_New_Datasets.py

09_Run_CDHIT_Splits.py
Generates the final homology-aware train, validation, and test partitions.
python codes/09_Run_CDHIT_Splits.py

The intended partition proportions are:
Dataset	Percentage
Training	70%
Validation	10%
Testing	20%


10_FIGNet_CDHIT_Training.py
Runs FIGNet and baseline models on the homology-aware partitions.
python codes/10_FIGNet_CDHIT_Training.py

The CD-HIT evaluation is complementary to LOSO and nested cross-validation and addresses a different generalization question: robustness to sequence similarity and homology structure.
🔁 Nested Cross-Validation
Motivation
Nested cross-validation was introduced to clearly separate:
1. hyperparameter selection, and
2. final performance estimation.
This prevents the outer evaluation folds from being used to select the learning rate or batch size of neural-network models.
The nested-CV implementation is contained in:
28_FIGNet_NestedCV_Training.py

Nested-CV Design
The nested analysis consists of:
Level	Procedure	Purpose
Outer CV	10-fold StratifiedKFold	Performance estimation
Inner CV	5-fold StratifiedKFold	Hyperparameter selection


For each outer fold:
1. the dataset is divided into an outer-training partition and an untouched outer-test partition;
2. 5-fold inner cross-validation is performed only within the outer-training partition;
3. candidate hyperparameter configurations are compared using mean inner-validation MCC;
4. the best configuration is selected;
5. the selected configuration is trained using the outer-training data;
6. the resulting model is evaluated on the corresponding outer-test fold;
7. metrics are aggregated across the 10 outer folds.
The outer-test fold is therefore not used during hyperparameter selection.
Neural-Network Hyperparameter Search
The following search space is evaluated for TensorFlow-based models:
Hyperparameter	Candidate Values
Learning Rate	0.01, 0.001, 0.0001
Batch Size	32, 64, 128


The primary hyperparameter-selection criterion is:
Matthews Correlation Coefficient (MCC)

Important Note for Classical Baselines
Learning rate and batch size apply to the TensorFlow neural-network models.
The scikit-learn models, including:
- Logistic Regression;
- SVM-RBF;
- SVM-Linear;
do not use neural-network learning rate or batch-size hyperparameters.
Within the nested-CV implementation, these classical models are evaluated using their corresponding scikit-learn configurations rather than an LR × batch-size search.
Run Nested CV
python codes/28_FIGNet_NestedCV_Training.py

Nested-CV Outputs
The nested-CV analysis generates fold-level predictions, metrics, model files, training histories, and summaries.
The main output directory is:
FIGNet_NestedCV_Results/
│
├── inner_cv/
│   └── outer_fold_*/
│       └── model/
│
├── outer_cv/
│   └── outer_fold_*/
│       └── model/
│           ├── csv_files/
│           ├── npy_files/
│           ├── plots/
│           └── models/
│
├── NestedCV_All_Results.csv
├── NestedCV_Summary.csv
└── checkpoint.json

NestedCV_All_Results.csv contains the individual outer-fold performance results.
NestedCV_Summary.csv contains the mean and standard deviation of the evaluation metrics across outer folds.
🧠 FIGNet Models
Five main FIGNet variants are implemented.
Model	Description
FIGNet_Gate_Only	Feature Importance Gate
FIGNet_Gate_RealVD	Feature Importance Gate + Real Variational Dropout
FIGNet_Gate_AdaptiveVD	Feature Importance Gate + Adaptive Variational Dropout
FIGNet_Gate_Sparsity	Feature Importance Gate + Dynamic Sparsity
FIGNet_Gate_Full	Combined FIGNet components


The variants are evaluated to examine the contribution of different regularization and gating components.
⚖️ Baseline Models
The repository includes the following comparison models:
Model	Description
Logistic Regression	Linear classification baseline
MLP Baseline	Multilayer perceptron
SVM-RBF	Support Vector Machine with radial-basis-function kernel
SVM-Linear	Linear-kernel Support Vector Machine
ReliefF-MLP	ReliefF feature selection followed by MLP
ReliefF-SVM	ReliefF feature selection followed by SVM


The nested-CV implementation also includes a NoGate_Ablation neural model for assessing performance in the absence of the Feature Importance Gate.
🔍 Interpretability Analysis
The repository contains three forms of model interpretation:
- Feature Importance Gate values;
- SHAP;
- LIME.
These methods provide complementary descriptions of model behavior.
Agreement between the methods should not be interpreted as necessary for validity because they quantify feature contribution using different principles.
Quantitative agreement analyses are therefore reported directly rather than assuming that the explanation methods produce identical rankings.
📈 Feature Stability Analysis
Feature-selection stability is evaluated using the Jaccard similarity between sets of highly ranked features.
The repository contains analyses for:
- LOSO feature rankings;
- CD-HIT feature rankings;
- cross-validation folds;
- multiple values of top-K.
The investigated values include:
K = 10, 25, 50, 100, 200

📊 Evaluation Metrics
Performance is evaluated using:
Metric	Description
Accuracy	Overall classification correctness
Precision	Reliability of positive predictions
Recall	Sensitivity to the positive class
F1 Score	Harmonic mean of precision and recall
MCC	Matthews Correlation Coefficient
AUC	Area under the receiver-operating-characteristic curve


MCC is used as an important evaluation and model-selection metric because the enzyme and non-enzyme classes are imbalanced.
💾 Checkpoint Recovery
The computationally intensive training pipelines include checkpoint-based recovery.
This allows interrupted experiments to be restarted without repeating already completed model runs.
For example:
python codes/28_FIGNet_NestedCV_Training.py

can be restarted after interruption, and completed nested-CV folds/configurations recorded by the checkpoint system can be skipped.
📊 Analysis and Reproducibility Scripts
After model training, additional scripts extract results, perform statistical analyses, evaluate feature stability, and generate manuscript figures.
Result Extraction and Statistical Analysis
Script	Purpose
11_Extract_CV_Results.py	Combine cross-validation results
12_Generate_CV_Tables.py	Generate manuscript-ready CV tables
13_Statistical_Analysis.py	Statistical model comparisons
14_LOSO_Final_Analysis.py	Aggregate final LOSO results
15_LOSO_Model_Winner_Analysis.py	Model performance by held-out species
16_Architecture_Level_Analysis.py	Architecture-level LOSO analysis
17_Generate_Final_Manuscript_Tables.py	Generate LOSO manuscript tables
18_CDHIT_Result_Analysis.py	Aggregate CD-HIT results
19_CDHIT_Manuscript_Table.py	Generate CD-HIT manuscript table


📈 Feature-Stability Scripts
Script	Purpose
20_Feature_Stability_Analysis.py	Jaccard stability for LOSO/CD-HIT experiments
21_TopK_Feature_Stability.py	Top-K feature stability
22_CV_Feature_Stability.py	Feature stability across CV folds
23_Feature_Stability_Summary.py	Aggregate feature-stability results


🖼️ Figure-Generation Scripts
Script	Purpose
24_Figure_ROC_Curves.py	ROC analysis for LOSO and CD-HIT experiments
25_Figure_Training_Dynamics.py	Training-accuracy and loss dynamics
26_Figure_Feature_Importance.py	Feature importance, stability, and SHAP visualization
27_Figure_Model_Comparison.py	Original model-comparison visualization retained for reproducibility
29_Figure1_NestedCV_Model_Comparison.py	Revised primary Figure 1 using nested-CV outer-fold performance


⚠️ Role of Script 27 Versus Script 29
27_Figure_Model_Comparison.py is retained to preserve reproducibility of the earlier analysis pipeline.
However, the revised primary model-comparison figure is generated using:
29_Figure1_NestedCV_Model_Comparison.py

This revised figure is based on the nested cross-validation results generated by:
28_FIGNet_NestedCV_Training.py

The revised workflow is therefore:
28_FIGNet_NestedCV_Training.py
              │
              ▼
      NestedCV_Summary.csv
              │
              ▼
29_Figure1_NestedCV_Model_Comparison.py
              │
              ▼
     Revised manuscript Figure 1

The nested-CV figure should be used for the primary standard-CV model comparison in the revised manuscript.
🖼️ Revised Figure 1
29_Figure1_NestedCV_Model_Comparison.py
This script generates the revised model-comparison figure using nested-CV results.
python codes/29_Figure1_NestedCV_Model_Comparison.py

The figure contains:
- Panel A: mean MCC across the outer nested-CV folds;
- Panel B: mean AUC across the outer nested-CV folds.
The plotted results therefore correspond to outer-fold performance estimates rather than performance used during hyperparameter selection.
Main output:
Figure1_NestedCV_Model_Comparison_MCC_AUC.png
Figure1_NestedCV_Model_Comparison_MCC_AUC.tiff

🔬 Relationship Between the Evaluation Strategies
The different evaluation procedures answer different scientific questions.
Evaluation	Main Question
Nested CV	How well does the model perform when hyperparameter selection is separated from evaluation?
LOSO	How well does the model generalize to an unseen species?
CD-HIT	How well does the model perform when sequence-similarity leakage is reduced?
Feature Stability	How stable are highly ranked embedding dimensions across experiments/folds?
SHAP/LIME/Gate	How do different interpretation approaches characterize feature contributions?


These analyses are complementary and should not be interpreted as interchangeable evaluations.
📁 Main Result Files
Depending on the experimental branch, generated outputs include:
File	Description
LOSO_Results_All_Models.csv	Aggregated LOSO model results
LOSO_Method_Summary.csv	LOSO model-level summary
CDHIT_Results_All_Models.csv	Aggregated CD-HIT results
CDHIT_Method_Summary.csv	CD-HIT model-level summary
NestedCV_All_Results.csv	Outer-fold nested-CV results
NestedCV_Summary.csv	Mean ± SD nested-CV performance
TopK_Feature_Stability_Jaccard.xlsx	Top-K feature-stability results
Friedman_results.csv	Friedman statistical-test results
Wilcoxon_pairwise_results.csv	Pairwise statistical comparisons


📁 Main Figures
Figure	Description
Figure1_NestedCV_Model_Comparison_MCC_AUC.png	Revised MCC/AUC model comparison using nested-CV outer-fold results
Figure2_ROC_Macro_LOSO_CDHIT.png	ROC analyses for LOSO and CD-HIT
Figure3_Training_Dynamics.png	Training-dynamics visualization
Figure4_Feature_Importance_Stability_SHAP.png	Feature importance, stability, and SHAP visualization


🔁 Recommended Reproducibility Workflow
A complete reproduction can be organized as follows.
Data Preparation
python codes/01_Organize_Fish_Data.py
python codes/02_Unified_Processor.py
python codes/03_Prepare_Binary_Data.py

LOSO Evaluation
python codes/04_Create_LOSO_Splits.py
python codes/05_fignet_LOSO_training.py

CD-HIT Evaluation
python codes/06_CDHIT_Clustering.py
python codes/07_Extract_Protein_IDs.py
python codes/08_Filter_New_Datasets.py
python codes/09_Run_CDHIT_Splits.py
python codes/10_FIGNet_CDHIT_Training.py

Nested Cross-Validation
python codes/28_FIGNet_NestedCV_Training.py

Revised Primary Model-Comparison Figure
python codes/29_Figure1_NestedCV_Model_Comparison.py

Other analysis scripts can then be executed as required for LOSO summaries, CD-HIT tables, statistical comparisons, feature-stability analyses, and additional manuscript figures.
🛡️ Reproducibility Considerations
The repository provides:
✅ Complete preprocessing workflow
✅ FIGNet model implementations
✅ Classical and neural-network baselines
✅ Nested cross-validation
✅ Independent outer-fold evaluation
✅ LOSO cross-species evaluation
✅ CD-HIT homology-aware testing
✅ Feature-stability analyses
✅ SHAP and LIME interpretability analyses
✅ Statistical comparison scripts
✅ Manuscript figure-generation scripts
✅ Checkpoint-based experiment recovery  
Random seeds are specified in the principal training scripts to improve computational reproducibility.
Because neural-network training and software/hardware environments may introduce small numerical differences, reproduced values may not always be bitwise identical.
📌 Important Methodological Notes
Nested Model Selection
Hyperparameters for TensorFlow models are selected only within the inner cross-validation loop.
The outer folds are reserved for performance estimation.
SVM Baselines
SVM models are implemented using scikit-learn.
Learning rate and batch size are neural-network parameters and should not be interpreted as SVM hyperparameters.
For interpretation:
- Linear SVM can provide direct linear coefficients;
- RBF SVM does not provide an equivalent direct linear input-feature coefficient vector.
Interpretability
Gate-derived feature rankings, SHAP, and LIME are conceptually different interpretation methods.
Observed agreement or disagreement between them is reported quantitatively and should not automatically be interpreted as biological validation.
Evaluation Scope
LOSO, CD-HIT, and nested CV evaluate different aspects of generalization.
Accordingly, their results should be interpreted separately rather than combined into a single performance estimate.
📜 Script Summary
Order	Script	Main Purpose
1	01_Organize_Fish_Data.py	Organize raw datasets
2	02_Unified_Processor.py	Process UniProt and embedding data
3	03_Prepare_Binary_Data.py	Create binary enzyme dataset
4	04_Create_LOSO_Splits.py	Generate LOSO splits
5	05_fignet_LOSO_training.py	LOSO model training/evaluation
6	06_CDHIT_Clustering.py	CD-HIT clustering
7	07_Extract_Protein_IDs.py	Extract reference protein IDs
8	08_Filter_New_Datasets.py	Filter target datasets
9	09_Run_CDHIT_Splits.py	Generate homology-aware splits
10	10_FIGNet_CDHIT_Training.py	CD-HIT model evaluation
11	11_Extract_CV_Results.py	Extract CV results
12	12_Generate_CV_Tables.py	Generate CV tables
13	13_Statistical_Analysis.py	Statistical comparisons
14	14_LOSO_Final_Analysis.py	Final LOSO analysis
15	15_LOSO_Model_Winner_Analysis.py	Species-level model winners
16	16_Architecture_Level_Analysis.py	Architecture-level comparison
17	17_Generate_Final_Manuscript_Tables.py	LOSO manuscript tables
18	18_CDHIT_Result_Analysis.py	CD-HIT result aggregation
19	19_CDHIT_Manuscript_Table.py	CD-HIT manuscript table
20	20_Feature_Stability_Analysis.py	Feature stability
21	21_TopK_Feature_Stability.py	Top-K stability
22	22_CV_Feature_Stability.py	Fold-level feature stability
23	23_Feature_Stability_Summary.py	Stability summary
24	24_Figure_ROC_Curves.py	ROC figure
25	25_Figure_Training_Dynamics.py	Training-dynamics figure
26	26_Figure_Feature_Importance.py	Feature-importance figure
27	27_Figure_Model_Comparison.py	Original model-comparison figure retained for reproducibility
28	28_FIGNet_NestedCV_Training.py	Nested CV with inner selection and outer evaluation
29	29_Figure1_NestedCV_Model_Comparison.py	Revised Figure 1 using nested-CV results


📚 Citation
If you use this repository or the FIGNet framework in your research, please cite the associated publication.
@article{shahzad2026fignet,
  title   = {FIGNet: A Deep Learning Framework with Feature Importance Gate for Genome-Wide Enzyme Classification},
  author  = {Shahzad, Anjum and others},
  journal = {Scientific Reports},
  year    = {2026}
}

Please update the citation information after publication with the final volume, issue, pages, and DOI.
📬 Contact
For questions regarding the implementation, experiments, or manuscript:
Corresponding Author:
Anjum Shahzad
⭐ Repository Status
This repository contains the analysis and reproducibility code associated with the revised FIGNet study, including the nested cross-validation framework introduced to separate hyperparameter selection from outer-fold performance estimation.

### Before you upload it

Make just **one code change** so your two new scripts connect correctly.

In `29_Figure1_NestedCV_Model_Comparison.py`, you currently have:

```python
INPUT_FILE = r"D:\zebfish1\revision1\FIGNet_NestedCV_Results\NestedCV_Summary_Full.csv"

But script 28 actually saves:
NestedCV_Summary.csv


  10a_FIGNet_NestedCV_Training
So change Figure 1 to:
INPUT_FILE = r"D:\zebfish1\revision1\FIGNet_NestedCV_Results\NestedCV_Summary.csv"


Then the README + script 28 + script 29 will all agree, which is particularly important because the reviewer already criticized internal inconsistencies.
├── results/
├── requirements.txt
└── README.md
