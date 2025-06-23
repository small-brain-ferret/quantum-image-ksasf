import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

def logistic(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def plot_metrics(shot_counts, avg_metric, metric_name, std_metric=None, prefix='debug'):
    """
    shot_counts: 1D array of shot counts (x-axis)
    avg_metric: 1D array of average metric values (y-axis)
    metric_name: string for labeling (e.g., 'SSIM' or 'MAE')
    std_metric: 1D array of standard deviations (optional, for error bars)
    prefix: 'debug' or 'batch_{start}'
    """
    # Print standard deviations
    if std_metric is not None:
        print("Standard deviations for each shot count:")
        for s, std in zip(shot_counts, std_metric):
            print(f"Shots: {s}, Std: {std}")

    # Sort by shot_counts for plotting
    sorted_indices = np.argsort(shot_counts)
    x = np.array(shot_counts)[sorted_indices]
    y = np.array(avg_metric)[sorted_indices]

    if std_metric is None:
        yerr = np.full_like(y, 0.01)
    else:
        yerr = np.array(std_metric)[sorted_indices]

    error_scale = 1.0
    yerr = yerr * error_scale

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=8, elinewidth=1, label='Average ± Std')

    # General logistic fit for a smooth trendline
    try:
        L_guess = max(y) - min(y)
        x0_guess = np.median(x)
        k_guess = 0.01
        b_guess = min(y)
        p0 = [L_guess, x0_guess, k_guess, b_guess]
        params, _ = curve_fit(logistic, x, y, p0, maxfev=20000)
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = logistic(x_fit, *params)
        ax.plot(x_fit, y_fit, color='black', linestyle='-', label='Logistic Trendline')
        print(f"Logistic fit parameters: L={params[0]}, x0={params[1]}, k={params[2]}, b={params[3]}")
    except Exception as e:
        print("Logistic fit failed:", e)

    ax.set_title(f'FRQI Debug: Shots vs Average {metric_name}')
    ax.set_xlabel('Shots')
    ax.set_ylabel(f'Average {metric_name}')
    ax.grid(True)
    ax.legend()
    plot_filename = f'{prefix}_plot_{metric_name.lower()}.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)