import os
import matplotlib.pyplot as plt
import numpy as np

def plot_fidelity_scatter(df, output_dir):
    """
    Generate two scatter plots:
    1. shots vs MAE, coloured by entropy
    2. shots vs SSIM, coloured by entropy
    """
    os.makedirs(output_dir, exist_ok=True)
    entropy_min = df['entropy'].min()
    entropy_max = df['entropy'].max()
    cmap = 'plasma'
    fontsize = 18
    pointsize = 80

    # Plot 1: shots vs MAE
    plt.figure(figsize=(8,6))
    sc1 = plt.scatter(df['shots'], df['fidelity_mae'], c=df['entropy'], cmap=cmap, s=pointsize, vmin=entropy_min, vmax=entropy_max)
    cbar = plt.colorbar(sc1)
    cbar.set_label('Image Entropy', fontsize=fontsize)
    plt.xlabel('Shots', fontsize=fontsize)
    plt.ylabel('MAE Fidelity', fontsize=fontsize)
    plt.title('FRQI MNIST: MAE Fidelity vs Shots (coloured by Entropy)', fontsize=fontsize)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_mae_vs_shots_entropy.png'), dpi=300)
    plt.close()

    # Plot 2: shots vs SSIM
    plt.figure(figsize=(8,6))
    sc2 = plt.scatter(df['shots'], df['fidelity_ssim'], c=df['entropy'], cmap=cmap, s=pointsize, vmin=entropy_min, vmax=entropy_max)
    cbar = plt.colorbar(sc2)
    cbar.set_label('Image Entropy', fontsize=fontsize)
    plt.xlabel('Shots', fontsize=fontsize)
    plt.ylabel('SSIM Fidelity', fontsize=fontsize)
    plt.title('FRQI MNIST: SSIM Fidelity vs Shots (coloured by Entropy)', fontsize=fontsize)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_ssim_vs_shots_entropy.png'), dpi=300)
    plt.close()
