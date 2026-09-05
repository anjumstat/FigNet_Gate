import pandas as pd
import os
import subprocess
import tempfile
import shutil
import re
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 80)
print("RUNNING CD-HIT CLUSTERING")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================
INPUT_DIR = r"D:\zebfish\data\evaluation_splits\filtered_4568_exact"
OUTPUT_DIR = r"D:\zebfish\data\evaluation_splits\filtered_4568_cdhit"
CDHIT_IDENTITY = 0.6
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

print(f"\nConfiguration:")
print(f"  Identity threshold: {CDHIT_IDENTITY*100}%")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Test size: {TEST_SIZE*100}%")
print(f"  Validation size: {VAL_SIZE*100}%")

# ============================================
# LOAD DATA
# ============================================
print("\nLoading dataset...")
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
val = pd.read_csv(os.path.join(INPUT_DIR, 'val.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
df = pd.concat([train, val, test], ignore_index=True)

print(f"  Total samples: {len(df)}")

# ============================================
# LOAD FASTA SEQUENCES
# ============================================
def extract_uniprot_id(header):
    if '.' in header:
        header = header.split('.')[0]
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    return header.split()[0]

def load_sequences():
    all_sequences = {}
    species_folders = {
        'Atlantic_salmon': 'atlansalmon',
        'Channel_catfish': 'channelfish',
        'Coho_salmon': 'cohosalmon',
        'Common_carp': 'comcarp',
        'Electric_eel': 'eletricell',
        'Fugu': 'Fogo',
        'Goldfish': 'goldfish',
        'Medaka': 'Medaka',
        'Nile_tilapia': 'niletilapia',
        'Rainbow_trout': 'rainbowtrout',
        'Tetraodon': 'Tetraodon',
        'Zebrafish': 'zebfish'
    }
    data_dir = r"D:\zebfish\data"
    
    print("\nLoading sequences from species folders...")
    for folder in species_folders.values():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            continue
        fasta_files = [f for f in os.listdir(folder_path) if f.endswith(('.fasta', '.fa'))]
        if not fasta_files:
            continue
        fasta_path = os.path.join(folder_path, fasta_files[0])
        current_id = None
        seq_lines = []
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id and seq_lines:
                        all_sequences[current_id] = ''.join(seq_lines)
                    header = line[1:]
                    current_id = extract_uniprot_id(header)
                    seq_lines = []
                elif current_id:
                    seq_lines.append(line)
            if current_id and seq_lines:
                all_sequences[current_id] = ''.join(seq_lines)
    return all_sequences

sequences = load_sequences()

# ============================================
# CREATE FASTA FOR CD-HIT
# ============================================
def find_sequence(pid):
    if pid in sequences:
        return sequences[pid]
    if '.' in pid:
        base = pid.split('.')[0]
        if base in sequences:
            return sequences[base]
    return None

os.makedirs(OUTPUT_DIR, exist_ok=True)
fasta_path = os.path.join(OUTPUT_DIR, "sequences.fasta")

print("\nPreparing FASTA file for clustering...")
found = 0
with open(fasta_path, 'w') as f:
    for pid in df['protein_id']:
        seq = find_sequence(pid)
        if seq:
            f.write(f">{pid}\n")
            f.write(f"{seq}\n")
            found += 1

print(f"  Sequences prepared: {found}/{len(df)}")

# ============================================
# RUN CD-HIT
# ============================================
def win_to_wsl_path(path):
    path = path.replace('\\', '/')
    import re
    match = re.match(r'([A-Za-z]):/(.*)', path)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return path

def run_cdhit(fasta_path):
    print("\nRunning CD-HIT clustering...")
    print("  This may take a few minutes...")
    
    fasta_wsl = win_to_wsl_path(fasta_path)
    temp_wsl = "/tmp/cdhit_temp"
    out_prefix = f"{temp_wsl}/output"
    cluster_file = f"{out_prefix}.clstr"
    
    cmd = f"""
    mkdir -p {temp_wsl} && \
    cd-hit -i {fasta_wsl} -o {out_prefix} -c {CDHIT_IDENTITY} -n 4 -M 2000 -T 4 -d 0 && \
    cat {cluster_file}
    """
    
    try:
        result = subprocess.run(["wsl", "-e", "bash", "-c", cmd], capture_output=True, text=True, check=True)
        print("  CD-HIT completed successfully")
        
        temp_dir = tempfile.mkdtemp()
        cluster_win = os.path.join(temp_dir, "clusters.clstr")
        subprocess.run(["wsl", "-e", "bash", "-c", f"cp {cluster_file} {win_to_wsl_path(cluster_win)}"], capture_output=True)
        
        clusters = {}
        current = None
        with open(cluster_win, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>Cluster'):
                    current = int(line.split()[1])
                    clusters[current] = []
                elif line and current is not None:
                    m = re.search(r'>(\S+)\.\.\.', line)
                    if m:
                        pid = m.group(1).split('.')[0]
                        clusters[current].append(pid)
        
        subprocess.run(["wsl", "-e", "bash", "-c", f"rm -rf {temp_wsl}"], capture_output=True)
        shutil.rmtree(temp_dir)
        return clusters
    except Exception as e:
        print(f"  CD-HIT failed: {e}")
        return None

clusters = run_cdhit(fasta_path)

# ============================================
# CREATE SPLITS
# ============================================
if clusters is None:
    print("\nCreating random splits (fallback)...")
    train_df, temp_df = train_test_split(df, test_size=TEST_SIZE+VAL_SIZE, random_state=RANDOM_SEED, stratify=df['is_enzyme'])
    val_df, test_df = train_test_split(temp_df, test_size=TEST_SIZE/(TEST_SIZE+VAL_SIZE), random_state=RANDOM_SEED, stratify=temp_df['is_enzyme'])
    homology_aware = False
else:
    print(f"\nCreating homology-aware splits from {len(clusters)} clusters...")
    protein_to_cluster = {}
    for cid, proteins in clusters.items():
        for p in proteins:
            protein_to_cluster[p] = cid
    
    # Initialize cluster column as string to avoid dtype issues
    df['cluster'] = df['protein_id'].map(protein_to_cluster).astype(object)
    
    unassigned = df['cluster'].isna().sum()
    if unassigned > 0:
        df.loc[df['cluster'].isna(), 'cluster'] = 'singleton_' + df.loc[df['cluster'].isna()].index.astype(str)
    
    unique_clusters = df['cluster'].unique()
    np.random.seed(RANDOM_SEED)
    shuffled = np.random.permutation(unique_clusters)
    
    n = len(shuffled)
    n_test = int(n * TEST_SIZE)
    n_val = int(n * VAL_SIZE)
    
    train_clusters = shuffled[:n-n_test-n_val]
    val_clusters = shuffled[n-n_test-n_val:n-n_test]
    test_clusters = shuffled[n-n_test:]
    
    train_df = df[df['cluster'].isin(train_clusters)].drop('cluster', axis=1)
    val_df = df[df['cluster'].isin(val_clusters)].drop('cluster', axis=1)
    test_df = df[df['cluster'].isin(test_clusters)].drop('cluster', axis=1)
    homology_aware = True

# ============================================
# SAVE SPLITS
# ============================================
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("CD-HIT CLUSTERING COMPLETE")
print("=" * 80)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Split type: {'Homology-aware (CD-HIT)' if homology_aware else 'Random (fallback)'}")

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)} samples ({train_df['is_enzyme'].sum()} enzymes)")
print(f"  Val:   {len(val_df)} samples ({val_df['is_enzyme'].sum()} enzymes)")
print(f"  Test:  {len(test_df)} samples ({test_df['is_enzyme'].sum()} enzymes)")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)