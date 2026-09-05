# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:32:31 2026

@author: H.A.R
"""

import pandas as pd
import os

print("=" * 80)
print("STEP 1: EXTRACT 4,568 PROTEIN IDs FROM OLD DATASET")
print("=" * 80)

# Load old dataset
old_path = r"D:\zebfish\revision\zebfish_processed_results\combined_data\all_fish_embeddings_combined.csv"
old_df = pd.read_csv(old_path)

# Get unique protein IDs (remove duplicates if any)
protein_ids = old_df['UniProt_ID'].unique()

print(f"\n📁 Old dataset loaded:")
print(f"   Total samples: {len(old_df)}")
print(f"   Unique protein IDs: {len(protein_ids)}")

# Save the protein IDs to a file
output_dir = r"D:\zebfish\data\evaluation_splits\filtered_4568"
os.makedirs(output_dir, exist_ok=True)

protein_ids_path = os.path.join(output_dir, 'protein_ids_4568.txt')
with open(protein_ids_path, 'w') as f:
    for pid in protein_ids:
        f.write(f"{pid}\n")

print(f"\n✅ Saved {len(protein_ids)} protein IDs to: {protein_ids_path}")

# Show sample IDs
print(f"\n📋 Sample protein IDs:")
for pid in list(protein_ids)[:10]:
    print(f"   - {pid}")