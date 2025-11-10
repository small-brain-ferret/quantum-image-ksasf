import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
import random

def inverse_power(x, a, b, c):
    return a * np.power(x, b) + c

def inverse_power_plateau(x, a, b, plateau):
    return a * np.power(x, b) + plateau

def shot_fidelity_curve(shots, c, b):
    shots_safe = np.maximum(shots, 1e-8) # avoid division by 0
    return 1.0 - c / np.power(shots_safe, b)

def plot_trendline(ax, x, y, yerr=None):
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=float)

        mask = x > 0
        x, y = x[mask], y[mask]
        if yerr is not None:
            yerr = yerr[mask]

        if x.size == 0:
            print("No valid shot counts to fit.")
            return None

        # initial guess
        p0 = [1.0, 0.5] 

        bounds = (0, np.inf)

        if yerr is not None:
            params, _ = curve_fit(shot_fidelity_curve, x, y, p0=p0,
            sigma=yerr, absolute_sigma=True,
            bounds=bounds, maxfev=20000)
        else:
            params, _ = curve_fit(shot_fidelity_curve, x, y, p0=p0,
            bounds=bounds, maxfev=20000)

        c_fit, b_fit = params

        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = shot_fidelity_curve(x_fit, c_fit, b_fit)
        ax.plot(x_fit, y_fit, color='black', linestyle='-', label='trendline')

        print(f"Fitted shot-fidelity parameters: c={c_fit:.4g}, b={b_fit:.4g}")
        return params

    except Exception as e:
        print("Shot-fidelity fit failed:", e)
        return None

def plot_metrics(shot_counts, avg_metric, metric_name, std_metric=None, prefix='debug', title=None):
    if std_metric is not None:
        print("Standard deviations for each shot count:")
        for s, std in zip(shot_counts, std_metric):
            print(f"Shots: {s}, Std: {std}")

    sorted_indices = np.argsort(shot_counts)
    x = np.array(shot_counts)[sorted_indices]
    y = np.array(avg_metric)[sorted_indices]

    if std_metric is None:
        yerr = np.full_like(y, 0.01)
    else:
        yerr = np.array(std_metric)[sorted_indices]

    # ensure yerr is finite and non-zero to avoid division-by-zero in curve fitting
    # raised minimum sigma so fits are less sensitive to tiny/zero stds
    min_sigma = 1e-3
    yerr = np.array(yerr, dtype=float)
    # replace non-finite values with min_sigma
    yerr[~np.isfinite(yerr)] = min_sigma
    # replace zeros or negatives with min_sigma
    yerr = np.maximum(yerr, min_sigma)

    error_scale = 1.0
    yerr = yerr * error_scale

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=8, elinewidth=1, label='average ± sd')
    plot_trendline(ax, x, y, yerr=yerr)

    if title is not None:
        plot_title = title
    elif prefix.lower().startswith('neqr'):
        plot_title = 'NEQR: Shots vs Average Fidelity'
    elif prefix.lower().startswith('frqi') or prefix.lower().startswith('debug'):
        plot_title = 'FRQI: Shots vs Average Fidelity'
    else:
        plot_title = 'Shots vs Average Fidelity'
    ax.set_title(plot_title)
    ax.set_xlabel('Shots')
    ax.set_ylabel('Average Fidelity')
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    # Force correct legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plot_filename = f'{prefix}_plot_ssim.png' if metric_name.lower() == 'ssim' else f'{prefix}_plot_mae.png'
    plt.savefig(os.path.join(os.getcwd(), plot_filename), format='png')
    plt.close(fig)