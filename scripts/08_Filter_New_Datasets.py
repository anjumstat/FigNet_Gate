import pandas as pd
import os

print("=" * 80)
print("EXTRACTING DATASET")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================
OLD_PATH = r"D:\zebfish\revision\zebfish_processed_results\combined_data\all_fish_embeddings_combined.csv"
NEW_SPLIT_DIR = r"D:\zebfish\data\evaluation_splits\homology_aware"
OUTPUT_DIR = r"D:\zebfish\data\evaluation_splits\filtered_4568_exact"

# ============================================
# LOAD OLD DATASET AND GET PROTEIN IDs
# ============================================
old_df = pd.read_csv(OLD_PATH)
protein_ids = set(old_df['UniProt_ID'].unique())

# ============================================
# LOAD NEW DATASET
# ============================================
train = pd.read_csv(os.path.join(NEW_SPLIT_DIR, 'train.csv'))
val = pd.read_csv(os.path.join(NEW_SPLIT_DIR, 'val.csv'))
test = pd.read_csv(os.path.join(NEW_SPLIT_DIR, 'test.csv'))

# ============================================
# FILTER TO EXACT PROTEIN IDs
# ============================================
train_filtered = train[train['protein_id'].isin(protein_ids)]
val_filtered = val[val['protein_id'].isin(protein_ids)]
test_filtered = test[test['protein_id'].isin(protein_ids)]

# ============================================
# SAVE
# ============================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_filtered.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_filtered.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_filtered.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print(f"\n✅ Dataset saved to: {OUTPUT_DIR}")
print("\n" + "=" * 80)
print("✅ COMPLETE")
print("=" * 80)