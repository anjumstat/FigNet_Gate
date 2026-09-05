# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:40:14 2026

@author: H.A.R
"""

import os
import pandas as pd


BASE_DIR = r"D:\zebfish1\revision1\FIGNet_LOSO_Results1_first\cv_runs"


results = []


for species in os.listdir(BASE_DIR):

    species_path = os.path.join(BASE_DIR, species)

    if not os.path.isdir(species_path):
        continue


    for config in os.listdir(species_path):

        config_path = os.path.join(species_path, config)

        if not os.path.isdir(config_path):
            continue


        # search csv files
        for root, dirs, files in os.walk(config_path):

            for file in files:

                if file.endswith(".csv"):

                    file_path = os.path.join(root, file)

                    try:

                        df = pd.read_csv(file_path)

                        df["species"] = species
                        df["configuration"] = config
                        df["source_file"] = file

                        results.append(df)


                    except Exception as e:
                        print("Error reading:", file_path)
                        print(e)



if len(results)==0:
    print("No CSV files found")
else:

    final = pd.concat(results, ignore_index=True)

    output = os.path.join(
        BASE_DIR,
        "ALL_CV_RESULTS_COMBINED.csv"
    )

    final.to_csv(output,index=False)

    print("Saved:")
    print(output)

    print("\nColumns:")
    print(final.columns.tolist())

    print("\nShape:")
    print(final.shape)