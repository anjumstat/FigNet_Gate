import os
import pandas as pd


# =====================================================
# PATHS
# =====================================================

BASE_DIR = r"D:\zebfish1\revision1\FIGNet_LOSO_Results1_first\cv_runs"

RESULT_DIR = r"D:\zebfish1\revision1\Results"


OUTPUT_DATA = os.path.join(
    RESULT_DIR,
    "CV_RESULTS_WITH_MODELS.csv"
)


OUTPUT_EXCEL = os.path.join(
    RESULT_DIR,
    "FINAL_CV_MANUSCRIPT_TABLES.xlsx"
)



# =====================================================
# STEP 1
# EXTRACT ALL CV RESULTS WITH MODEL NAMES
# =====================================================

all_results = []


for species in os.listdir(BASE_DIR):

    species_path = os.path.join(BASE_DIR, species)

    if not os.path.isdir(species_path):
        continue


    for config in os.listdir(species_path):

        config_path = os.path.join(
            species_path,
            config
        )


        if not os.path.isdir(config_path):
            continue


        for model in os.listdir(config_path):

            model_path = os.path.join(
                config_path,
                model
            )


            metrics_file = os.path.join(
                model_path,
                "csv_files",
                "Fold_Metrics.csv"
            )


            if os.path.exists(metrics_file):

                df = pd.read_csv(metrics_file)

                df["species"] = species
                df["configuration"] = config
                df["model_name"] = model

                all_results.append(df)



results = pd.concat(
    all_results,
    ignore_index=True
)



print("\n======================================")
print("EXTRACTION COMPLETED")
print("======================================")

print("Dataset shape:")
print(results.shape)



# Save complete dataset

results.to_csv(
    OUTPUT_DATA,
    index=False
)


print("\nSaved extracted data:")
print(OUTPUT_DATA)



# =====================================================
# FUNCTION FOR ALL METRIC TABLES
# =====================================================


def metric_summary(group_columns):

    table = (
        results
        .groupby(group_columns)
        .agg(
            Accuracy=("Accuracy","mean"),
            Precision=("Precision","mean"),
            Recall=("Recall","mean"),
            F1=("F1","mean"),
            MCC=("MCC","mean"),
            AUC=("AUC","mean")
        )
        .reset_index()
    )

    return table



# =====================================================
# CHECK MODELS
# =====================================================


print("\n======================================")
print("MODELS FOUND")
print("======================================")

print(
    results["model_name"]
    .value_counts()
)



# =====================================================
# TABLE 1
# OVERALL PERFORMANCE
# =====================================================


table1 = metric_summary(
    ["model_name"]
)


table1 = table1.sort_values(
    "MCC",
    ascending=False
)



print("\n======================================")
print("TABLE 1: OVERALL MODEL PERFORMANCE")
print("======================================")

print(
    table1.round(4)
    .to_string(index=False)
)



# =====================================================
# TABLE 2
# BEST CONFIGURATION PER MODEL
# =====================================================


table2 = metric_summary(
    [
        "model_name",
        "configuration"
    ]
)


table2 = table2.sort_values(
    "MCC",
    ascending=False
)



print("\n======================================")
print("TABLE 2: BEST CONFIGURATIONS")
print("======================================")

print(
    table2.head(40)
    .round(4)
    .to_string(index=False)
)



# =====================================================
# TABLE 3
# SPECIES-WISE PERFORMANCE
# =====================================================


table3 = metric_summary(
    [
        "species",
        "model_name"
    ]
)


print("\n======================================")
print("TABLE 3: SPECIES PERFORMANCE")
print("======================================")

print(
    table3.round(4)
    .to_string(index=False)
)



# =====================================================
# TABLE 4
# CONFIGURATION EFFECT
# =====================================================


table4 = metric_summary(
    [
        "configuration"
    ]
)


table4 = table4.sort_values(
    "MCC",
    ascending=False
)


print("\n======================================")
print("TABLE 4: CONFIGURATION EFFECT")
print("======================================")

print(
    table4.round(4)
    .to_string(index=False)
)



# =====================================================
# TABLE 5
# COMPETITIVE METHODS ONLY
# =====================================================


competitive_models = [
    "MLP_Baseline",
    "Logistic_Regression",
    "ReliefF_MLP",
    "ReliefF_SVM",
    "SVM_Linear",
    "SVM_RBF"
]


table5 = (
    results[
        results["model_name"]
        .isin(competitive_models)
    ]
    .groupby("model_name")
    .agg(
        Accuracy=("Accuracy","mean"),
        Precision=("Precision","mean"),
        Recall=("Recall","mean"),
        F1=("F1","mean"),
        MCC=("MCC","mean"),
        AUC=("AUC","mean")
    )
    .reset_index()
)


table5 = table5.sort_values(
    "MCC",
    ascending=False
)



print("\n======================================")
print("TABLE 5: COMPETITIVE MODELS")
print("======================================")

print(
    table5.round(4)
    .to_string(index=False)
)



# =====================================================
# SAVE INDIVIDUAL CSV TABLES
# =====================================================


tables = {

"Table1_Overall_Performance": table1,

"Table2_Best_Configurations": table2,

"Table3_Species_Performance": table3,

"Table4_Configuration_Effect": table4,

"Table5_Competitive_Models": table5

}



for name, table in tables.items():

    table.to_csv(
        os.path.join(
            RESULT_DIR,
            name + ".csv"
        ),
        index=False
    )



# =====================================================
# SAVE EXCEL WORKBOOK
# =====================================================


with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:


    for name, table in tables.items():

        table.to_excel(
            writer,
            sheet_name=name[:31],
            index=False
        )



print("\n======================================")
print("ALL TABLES SAVED")
print("======================================")

print(OUTPUT_EXCEL)