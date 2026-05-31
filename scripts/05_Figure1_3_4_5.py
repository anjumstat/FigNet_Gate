# -*- coding: utf-8 -*-
"""
Generate Publication-Ready Figures for PAPER 2: FIGNet (Feature Importance Gate Network)
Based on actual experimental results from VARDON_GATE_Results
Figures: 1, 2 (4 subplots), 3, 4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# =============================================
# CONFIGURATION
# =============================================

# Path to your combined CV results for Paper 2
cv_results_path = r"D:\zebfish1\VARDON_GATE_Results\Paper2_CV_Summary_Results.csv"

# Base directories for .npy files (training curves)
base_dirs = [
    r"D:\zebfish1\VARDON_GATE_Results\cv_runs",
]

# Output directories
output_dir = r"D:\zebfish1\FIGNet_Paper_Figures"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

# =============================================
# METHOD MAPPING (FIGNet instead of VARDON)
# =============================================

# Map original method names to display names
method_display_names = {
    'Logistic_Regression': 'Logistic Regression',
    'MLP_Baseline': 'MLP Baseline',
    'VARDON_FeatureGate': 'FIGNet + Gate',
    'VARDON_Gate_RealVD': 'FIGNet + Gate + RealVD',
    'VARDON_Gate_AdaptiveVD': 'FIGNet + Gate + AdaptiveVD',
    'VARDON_Gate_Sparsity': 'FIGNet + Gate + Sparsity',
    'VARDON_Gate_Full': 'FIGNet + Gate + Full'
}

# Colors for methods
method_colors = {
    'Logistic Regression': '#808080',           # Gray
    'MLP Baseline': '#1f77b4',                  # Blue
    'FIGNet + Gate': '#2ca02c',                 # Green
    'FIGNet + Gate + RealVD': '#d62728',        # Red
    'FIGNet + Gate + AdaptiveVD': '#ff7f0e',    # Orange
    'FIGNet + Gate + Sparsity': '#9467bd',      # Purple
    'FIGNet + Gate + Full': '#8c564b'           # Brown
}

# Method order for consistent display
methods_order = [
    'Logistic_Regression',
    'MLP_Baseline',
    'VARDON_FeatureGate',
    'VARDON_Gate_RealVD',
    'VARDON_Gate_AdaptiveVD',
    'VARDON_Gate_Sparsity',
    'VARDON_Gate_Full'
]

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def save_figure(fig, filename):
    """Save figure as both PNG and TIFF"""
    png_path = os.path.join(png_dir, f"{filename}.png")
    tiff_path = os.path.join(tiff_dir, f"{filename}.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"  ✅ Saved: {filename}")

# =============================================
# FIGURE 1: HYPERPARAMETER EFFECTS (LR × BS)
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: Generating Figure 1: Hyperparameter Effects (LR × BS)")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    # Calculate mean MCC for each (LR, BS) combination
    lr_values = [0.0001, 0.001, 0.01]
    bs_values = [32, 64, 128]
    
    heatmap_data = np.zeros((len(bs_values), len(lr_values)))
    
    for i, bs in enumerate(bs_values):
        for j, lr in enumerate(lr_values):
            subset = df[(df['learning_rate'] == lr) & (df['batch_size'] == bs)]
            if len(subset) > 0:
                heatmap_data[i, j] = subset['mean_mcc'].mean()
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap
    im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.75, vmax=0.86)
    ax1.set_xticks(np.arange(len(lr_values)))
    ax1.set_yticks(np.arange(len(bs_values)))
    ax1.set_xticklabels([f'{lr:.4f}' for lr in lr_values], fontweight='bold')
    ax1.set_yticklabels(bs_values, fontweight='bold')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 1A. Mean MCC Score (LR × BS)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(bs_values)):
        for j in range(len(lr_values)):
            ax1.text(j, i, f'{heatmap_data[i, j]:.4f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax1, label='Mean MCC Score')
    
    # Bar plot of best LR per BS
    best_by_lr = []
    for lr in lr_values:
        subset = df[df['learning_rate'] == lr]
        best_mcc = subset['mean_mcc'].mean() if len(subset) > 0 else 0
        best_by_lr.append(best_mcc)
    
    bars = ax2.bar([f'LR={lr}' for lr in lr_values], best_by_lr, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax2.set_ylabel('Mean MCC Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 1B. Mean MCC by Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.80, 0.86])
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, best_by_lr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig1, "Figure1_Hyperparameter_Effects")
    plt.close(fig1)

# =============================================
# FIGURE 2: TRAINING AND VALIDATION CURVES (2x2 Subplots)
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: Generating Figure 2: Training and Validation Curves")
print("="*60)

# Top methods to display (best 3)
top_methods = ['VARDON_Gate_RealVD', 'VARDON_FeatureGate', 'MLP_Baseline']

# Best configurations for these methods
best_configs = {
    'VARDON_Gate_RealVD': {'lr': 0.0001, 'bs': 32},
    'VARDON_FeatureGate': {'lr': 0.0001, 'bs': 64},
    'MLP_Baseline': {'lr': 0.0001, 'bs': 32}
}

def load_training_and_validation_data(method, lr, bs):
    """Load training and validation history from .npy files"""
    
    if lr == 0.0001:
        lr_folder = f"lr_0_00010_bs_{bs}"
    elif lr == 0.001:
        lr_folder = f"lr_0_00100_bs_{bs}"
    else:
        lr_folder = f"lr_0_01000_bs_{bs}"
    
    npy_dir = os.path.join(r"D:\zebfish1\VARDON_GATE_Results\cv_runs", lr_folder, method, "npy_files")
    
    if os.path.exists(npy_dir):
        all_train_acc = []
        all_val_acc = []
        all_train_loss = []
        all_val_loss = []
        
        for fold in range(1, 11):
            train_acc_file = os.path.join(npy_dir, f"fold{fold}_accuracy.npy")
            if os.path.exists(train_acc_file):
                all_train_acc.append(np.load(train_acc_file))
            
            val_acc_file = os.path.join(npy_dir, f"fold{fold}_val_accuracy.npy")
            if os.path.exists(val_acc_file):
                all_val_acc.append(np.load(val_acc_file))
            elif os.path.exists(train_acc_file):
                all_val_acc.append(np.load(train_acc_file))
            
            train_loss_file = os.path.join(npy_dir, f"fold{fold}_loss.npy")
            if os.path.exists(train_loss_file):
                all_train_loss.append(np.load(train_loss_file))
            
            val_loss_file = os.path.join(npy_dir, f"fold{fold}_val_loss.npy")
            if os.path.exists(val_loss_file):
                all_val_loss.append(np.load(val_loss_file))
            elif os.path.exists(train_loss_file):
                all_val_loss.append(np.load(train_loss_file))
        
        if all_train_acc:
            max_len = max(len(acc) for acc in all_train_acc)
            
            padded_train_acc = np.array([np.pad(acc, (0, max_len - len(acc)), constant_values=acc[-1]) for acc in all_train_acc])
            padded_val_acc = np.array([np.pad(acc, (0, max_len - len(acc)), constant_values=acc[-1]) for acc in all_val_acc]) if all_val_acc else padded_train_acc
            
            if all_train_loss:
                padded_train_loss = np.array([np.pad(loss, (0, max_len - len(loss)), constant_values=loss[-1]) for loss in all_train_loss])
                train_loss_mean = padded_train_loss.mean(axis=0)
            else:
                train_loss_mean = None
            
            if all_val_loss:
                padded_val_loss = np.array([np.pad(loss, (0, max_len - len(loss)), constant_values=loss[-1]) for loss in all_val_loss])
                val_loss_mean = padded_val_loss.mean(axis=0)
            else:
                val_loss_mean = None
            
            return {
                'train_acc_mean': padded_train_acc.mean(axis=0),
                'train_acc_std': padded_train_acc.std(axis=0),
                'val_acc_mean': padded_val_acc.mean(axis=0),
                'val_acc_std': padded_val_acc.std(axis=0),
                'train_loss_mean': train_loss_mean,
                'val_loss_mean': val_loss_mean,
                'epochs': max_len
            }
    return None

# Create 2x2 subplots
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Figure 2. Training and Validation Curves - FIGNet Methods', fontsize=14, fontweight='bold')

for method in top_methods:
    config = best_configs.get(method, {'lr': 0.001, 'bs': 64})
    data = load_training_and_validation_data(method, config['lr'], config['bs'])
    
    if data:
        epochs = np.arange(1, data['epochs'] + 1)
        display_name = method_display_names.get(method, method)
        color = method_colors.get(display_name, '#1f77b4')
        
        # Training Accuracy
        axes[0, 0].plot(epochs, data['train_acc_mean'], linewidth=2.5,
                        color=color, label=f'{display_name}')
        axes[0, 0].fill_between(epochs,
                                data['train_acc_mean'] - data['train_acc_std'],
                                data['train_acc_mean'] + data['train_acc_std'],
                                alpha=0.15, color=color)
        
        # Validation Accuracy
        axes[0, 1].plot(epochs, data['val_acc_mean'], linewidth=2.5,
                        color=color, label=f'{display_name}')
        axes[0, 1].fill_between(epochs,
                                data['val_acc_mean'] - data['val_acc_std'],
                                data['val_acc_mean'] + data['val_acc_std'],
                                alpha=0.15, color=color)
        
        # Training Loss
        if data['train_loss_mean'] is not None:
            axes[1, 0].plot(epochs, data['train_loss_mean'], linewidth=2.5,
                            color=color, label=f'{display_name}')
        
        # Validation Loss
        if data['val_loss_mean'] is not None:
            axes[1, 1].plot(epochs, data['val_loss_mean'], linewidth=2.5,
                            color=color, label=f'{display_name}')

# Configure axes
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Figure 2A. Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.7, 1.0])

axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Figure 2B. Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.7, 1.0])

axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Figure 2C. Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 0.6])

axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Figure 2D. Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 0.6])

plt.tight_layout()
save_figure(fig2, "Figure2_Training_Validation_Curves")
plt.close(fig2)

# =============================================
# FIGURE 3: PERFORMANCE RANKING (All Methods)
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: Generating Figure 3: Performance Ranking")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    # Get best configuration for each method
    best_configs_df = df.loc[df.groupby('method')['mean_mcc'].idxmax()]
    best_configs_df = best_configs_df.sort_values('mean_mcc', ascending=False)
    
    # Prepare data for plotting
    methods_display = [method_display_names.get(m, m) for m in best_configs_df['method'].tolist()]
    mcc_values = best_configs_df['mean_mcc'].tolist()
    f1_values = best_configs_df['mean_f1'].tolist()
    acc_values = best_configs_df['mean_accuracy'].tolist()
    
    fig3, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods_display))
    width = 0.25
    
    bars1 = ax.bar(x - width, mcc_values, width, label='MCC', 
                   color='#2ca02c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, f1_values, width, label='F1 Score', 
                   color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, acc_values, width, label='Accuracy', 
                   color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('FIGNet Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3. Performance Ranking of FIGNet Methods', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_display, rotation=45, ha='right', fontsize=10)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_ylim([0.6, 0.96])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.003,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Highlight best performer
    bars1[0].set_edgecolor('gold')
    bars1[0].set_linewidth(2.5)
    bars2[0].set_edgecolor('gold')
    bars2[0].set_linewidth(2.5)
    bars3[0].set_edgecolor('gold')
    bars3[0].set_linewidth(2.5)
    
    plt.tight_layout()
    save_figure(fig3, "Figure3_Performance_Ranking")
    plt.close(fig3)

# =============================================
# FIGURE 4: CROSS-VALIDATION BOXPLOTS
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: Generating Figure 4: Cross-Validation Boxplots")
print("="*60)

if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    
    fig4, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for boxplot
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []
    
    for method in methods_order:
        method_data = df[df['method'] == method]['mean_mcc'].values
        if len(method_data) > 0:
            boxplot_data.append(method_data)
            display_name = method_display_names.get(method, method)
            boxplot_labels.append(display_name)
            boxplot_colors.append(method_colors.get(display_name, '#1f77b4'))
    
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))
    
    for patch, color in zip(bp['boxes'], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('MCC Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('FIGNet Method', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4. Cross-Validation MCC Distribution\n(Best configuration per method)', 
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0.6, 0.88])
    
    plt.tight_layout()
    save_figure(fig4, "Figure4_CV_Boxplots")
    plt.close(fig4)

# =============================================
# SUMMARY
# =============================================

print("\n" + "="*60)
print("PAPER 2 - FIGNet: FIGURE GENERATION COMPLETE")
print("="*60)
print(f"\n📁 PNG files saved to: {png_dir}")
print(f"📁 TIFF files saved to: {tiff_dir}")
print("\nGenerated figures:")
print("  - Figure 1: Hyperparameter Effects (LR × BS Heatmap + Bar plot)")
print("  - Figure 2: Training and Validation Curves (4 subplots)")
print("  - Figure 3: Performance Ranking (MCC, F1, Accuracy)")
print("  - Figure 4: Cross-Validation Boxplots")
print("\n" + "="*60)