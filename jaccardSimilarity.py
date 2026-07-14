import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

import json

# Load data from bandSelect_Config.json
with open('bandSelect_Config.json', 'r') as f:
    data = json.load(f)

# Define methods mapping (json keys -> plot titles)
method_map = {
    "cmim": "CMIM",
    "jmim": "JMIM",
    "jmim_csnr": "JMIM + CSNR",
    "jmim_csnr_corr": "JMIM + CSNR + Corr",
    "mrmr": "mRMR",
    "sim_lp": "SIM-LP"
}

methods = list(method_map.keys())
seeds = list(data.keys())

# 1. Compute Pairwise Jaccard Similarity (Mean & Standard Deviation)
jaccard_stats = {}
for m in methods:
    j_sims = []
    for s1, s2 in combinations(seeds, 2):
        set1 = set(data[s1][m])
        set2 = set(data[s2][m])
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        j_sim = intersection / union if union != 0 else 0
        j_sims.append(j_sim)
        
    jaccard_stats[m] = {
        "mean": np.mean(j_sims),
        "std": np.std(j_sims)
    }

# Print Summary Table
print(f"{'Method':<20} | {'Mean Jaccard':<12} | {'Std Dev':<10}")
print("-" * 50)
for m in methods:
    print(f"{method_map[m]:<20} | {jaccard_stats[m]['mean']:<12.4f} | {jaccard_stats[m]['std']:<10.4f}")

# 2. Plot Frequencies and Jaccard Similarity Stats in a Single Plot
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()

max_band_idx = 128
band_indices = np.arange(max_band_idx)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, m in enumerate(methods):
    ax = axes[idx]
    
    # Calculate frequency of each band across all 10 seeds
    counts = np.zeros(max_band_idx, dtype=int)
    for s in seeds:
        for b in data[s][m]:
            counts[b] += 1
            
    # Bar plot for frequencies
    ax.bar(band_indices, counts, color=colors[idx], alpha=0.75, width=1.0, edgecolor='none')
    
    # Text box displaying Jaccard Similarity (Mean ± SD)
    mean_j = jaccard_stats[m]["mean"]
    std_j = jaccard_stats[m]["std"]
    text_str = f"Jaccard Sim: {mean_j:.3f} ± {std_j:.3f}"
    
    ax.text(0.03, 0.85, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_title(method_map[m], fontsize=14, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11, 2))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Axis labels and layout formatting
fig.text(0.5, 0.02, 'Band Index Number', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.5, 'Frequency across 10 Seeds', va='center', rotation='vertical', fontsize=14, fontweight='bold')

plt.suptitle('Band Selection Frequency & Stability Across 10 Seeds', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
plt.show()

plt.suptitle('Band Selection Frequency & Stability Across 10 Seeds', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
plt.show()
