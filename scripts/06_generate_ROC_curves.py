# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:43:55 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Generate REAL ROC Curves for PAPER 2: FIGNet (Feature Importance Gate Network)
Based on actual predictions and true labels from VARDON_GATE_Results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

base_dirs = [
    r"D:\zebfish1\VARDON_GATE_Results\cv_runs",
]

# Methods for Paper 2 (FIGNet variants)
methods = [
    'Logistic_Regression',
    'MLP_Baseline',
    'VARDON_FeatureGate',
    'VARDON_Gate_RealVD',
    'VARDON_Gate_AdaptiveVD',
    'VARDON_Gate_Sparsity',
    'VARDON_Gate_Full'
]

# Map original method names to FIGNet display names
method_display_names = {
    'Logistic_Regression': 'Logistic Regression',
    'MLP_Baseline': 'MLP Baseline',
    'VARDON_FeatureGate': 'FIGNet + Gate',
    'VARDON_Gate_RealVD': 'FIGNet + Gate + RealVD',
    'VARDON_Gate_AdaptiveVD': 'FIGNet + Gate + AdaptiveVD',
    'VARDON_Gate_Sparsity': 'FIGNet + Gate + Sparsity',
    'VARDON_Gate_Full': 'FIGNet + Gate + Full'
}

# Best configurations from your Paper 2 results (based on Test MCC)
best_configs = {
    'VARDON_Gate_RealVD': {'lr': 0.0001, 'bs': 32},
    'VARDON_FeatureGate': {'lr': 0.0001, 'bs': 64},
    'MLP_Baseline': {'lr': 0.0001, 'bs': 32},
    'VARDON_Gate_AdaptiveVD': {'lr': 0.0001, 'bs': 32},
    'VARDON_Gate_Sparsity': {'lr': 0.0001, 'bs': 32},
    'VARDON_Gate_Full': {'lr': 0.001, 'bs': 64},
    'Logistic_Regression': {'lr': 0.001, 'bs': 64}
}

# Colors for methods (based on FIGNet display names)
method_colors = {
    'Logistic Regression': '#808080',           # Gray
    'MLP Baseline': '#1f77b4',                  # Blue
    'FIGNet + Gate': '#2ca02c',                 # Green
    'FIGNet + Gate + RealVD': '#d62728',        # Red
    'FIGNet + Gate + AdaptiveVD': '#ff7f0e',    # Orange
    'FIGNet + Gate + Sparsity': '#9467bd',      # Purple
    'FIGNet + Gate + Full': '#8c564b'           # Brown
}

# Line styles: solid for FIGNet variants, dashed for baselines
def get_line_style(method):
    if method in ['Logistic_Regression', 'MLP_Baseline']:
        return '--'
    else:
        return '-'

def load_roc_data(method, lr, bs):
    """
    Load predictions and true labels from all CV folds to compute ROC
    """
    
    all_probs = []
    all_labels = []
    
    # Format folder name based on learning rate
    if lr == 0.0001:
        lr_folder = f"lr_0_00010_bs_{bs}"
    elif lr == 0.001:
        lr_folder = f"lr_0_00100_bs_{bs}"
    else:
        lr_folder = f"lr_0_01000_bs_{bs}"
    
    npy_dir = os.path.join(r"D:\zebfish1\VARDON_GATE_Results\cv_runs", lr_folder, method, "npy_files")
    
    if os.path.exists(npy_dir):
        for fold in range(1, 11):
            # Load predictions (probability for class 1 - Enzyme)
            pred_file = os.path.join(npy_dir, f"fold{fold}_predictions.npy")
            labels_file = os.path.join(npy_dir, f"fold{fold}_true_labels.npy")
            
            if os.path.exists(pred_file) and os.path.exists(labels_file):
                pred = np.load(pred_file)
                labels = np.load(labels_file)
                
                # Get probability for class 1 (Enzyme)
                if pred.shape[1] == 2:
                    probs = pred[:, 1]
                else:
                    probs = pred
                
                all_probs.extend(probs)
                all_labels.extend(labels)
                print(f"      Loaded fold {fold}: {len(probs)} samples")
    
    if len(all_probs) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    return None, None, None

# =============================================
# FIRST, CHECK WHAT FILES ARE AVAILABLE
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: Checking available prediction files")
print("="*60)

for base_dir in base_dirs:
    print(f"\n📁 Checking: {base_dir}")
    if os.path.exists(base_dir):
        configs = glob.glob(os.path.join(base_dir, "lr_*"))
        print(f"   Found {len(configs)} configuration folders")
        for config in configs[:5]:
            print(f"     - {os.path.basename(config)}")
            for method in methods[:3]:
                method_dir = os.path.join(config, method)
                if os.path.exists(method_dir):
                    npy_dir = os.path.join(method_dir, "npy_files")
                    if os.path.exists(npy_dir):
                        pred_files = glob.glob(os.path.join(npy_dir, "fold*_predictions.npy"))
                        if pred_files:
                            print(f"         ✅ {method}: {len(pred_files)} prediction files found")

# =============================================
# GENERATE REAL ROC CURVES
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: Generating REAL ROC Curves from Actual Data")
print("FIGNet - Feature Importance Gate Network")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

loaded_methods = []

for method in methods:
    config = best_configs.get(method, {'lr': 0.001, 'bs': 64})
    print(f"\n  Loading {method} (LR={config['lr']}, BS={config['bs']})...")
    
    fpr, tpr, roc_auc = load_roc_data(method, config['lr'], config['bs'])
    
    display_name = method_display_names.get(method, method)
    color = method_colors.get(display_name, '#1f77b4')
    linestyle = get_line_style(method)
    
    if fpr is not None:
        ax.plot(fpr, tpr, linewidth=2.5, 
                color=color, linestyle=linestyle,
                label=f'{display_name} (AUC = {roc_auc:.4f})')
        print(f"  ✅ {display_name}: AUC = {roc_auc:.4f}")
        loaded_methods.append(method)
    else:
        print(f"  ⚠️ {display_name}: No ROC data found")

# Diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.5)')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2. Receiver Operating Characteristic (ROC) Curves\nFIGNet (Feature Importance Gate Network)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Save figures
output_dir = r"D:\zebfish1\FIGNet_Paper_Figures_ROC"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

if loaded_methods:
    png_path = os.path.join(png_dir, "Figure2_ROC_Curves_FIGNET.png")
    tiff_path = os.path.join(tiff_dir, "Figure2_ROC_Curves_FIGNET.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"\n✅ Saved: {png_path}")
    print(f"✅ Saved: {tiff_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, "Figure2_ROC_Curves_FIGNET.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {pdf_path}")
else:
    print("\n❌ No ROC data found for any method!")
    
    # Show what folders exist
    print("\n📁 Available folders in your results:")
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            configs = glob.glob(os.path.join(base_dir, "lr_*"))
            print(f"\n  {os.path.basename(base_dir)}:")
            for config in configs[:5]:
                print(f"    - {os.path.basename(config)}")

plt.close(fig)

# =============================================
# OPTIONAL: Create a second figure with only FIGNet variants
# =============================================

if loaded_methods:
    print("\n" + "="*60)
    print("Generating ROC Curves for FIGNet Variants Only")
    print("="*60)
    
    fignet_methods = ['VARDON_FeatureGate', 'VARDON_Gate_RealVD', 
                      'VARDON_Gate_AdaptiveVD', 'VARDON_Gate_Sparsity', 
                      'VARDON_Gate_Full']
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    
    for method in fignet_methods:
        config = best_configs.get(method, {'lr': 0.001, 'bs': 64})
        fpr, tpr, roc_auc = load_roc_data(method, config['lr'], config['bs'])
        
        display_name = method_display_names.get(method, method)
        color = method_colors.get(display_name, '#1f77b4')
        
        if fpr is not None:
            ax2.plot(fpr, tpr, linewidth=2.5, 
                    color=color, linestyle='-',
                    label=f'{display_name} (AUC = {roc_auc:.4f})')
            print(f"  ✅ {display_name}: AUC = {roc_auc:.4f}")
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random (AUC = 0.5)')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 2. ROC Curves - FIGNet Variants Only\n(Feature Importance Gate Network)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    png_path2 = os.path.join(png_dir, "Figure2_ROC_Curves_FIGNET_Only.png")
    tiff_path2 = os.path.join(tiff_dir, "Figure2_ROC_Curves_FIGNET_Only.tiff")
    fig2.savefig(png_path2, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(tiff_path2, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"\n✅ Saved: {png_path2}")
    print(f"✅ Saved: {tiff_path2}")
    
    plt.close(fig2)

print("\n" + "="*60)
print("✅ PAPER 2 - FIGNet ROC CURVE GENERATION COMPLETE!")
print(f"📁 Output directory: {output_dir}")
print("="*60)