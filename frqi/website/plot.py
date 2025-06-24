import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

def inverse_power(x, a, b, c):
    return a * np.power(x, b) + c

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

    # Inverse power fit for a smooth trendline
    try:
        # Initial guess: a=1, b=-0.5, c=0.5
        p0 = [1, -0.5, 0.5]
        params, _ = curve_fit(inverse_power, x, y, p0, maxfev=20000)
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = inverse_power(x_fit, *params)
        ax.plot(x_fit, y_fit, color='black', linestyle='-', label='Inverse Power Trendline')
        print(f"Inverse power fit parameters: a={params[0]}, b={params[1]}, c={params[2]}")
    except Exception as e:
        print("Inverse power fit failed:", e)

    ax.set_title(f'FRQI Debug: Shots vs Average {metric_name}')
    ax.set_xlabel('Shots')
    ax.set_ylabel(f'Average {metric_name}')
    ax.grid(True)
    ax.legend()
    plot_filename = f'{prefix}_plot_{metric_name.lower()}.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)