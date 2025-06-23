import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def logistic(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def plot_metrics(shot_counts, avg_fidelity, metric_name, std_fidelity=None):
    """
    shot_counts: 1D array of shot counts (x-axis)
    avg_fidelity: 1D array of average fidelity values (y-axis)
    metric_name: string for labeling
    std_fidelity: 1D array of standard deviations (optional, for error bars)
    """
    # Print standard deviations
    if std_fidelity is not None:
        print("Standard deviations for each shot count:")
        for s, std in zip(shot_counts, std_fidelity):
            print(f"Shots: {s}, Std: {std}")

    # Sort by shot_counts for plotting
    sorted_indices = np.argsort(shot_counts)
    x = np.array(shot_counts)[sorted_indices]
    y = np.array(avg_fidelity)[sorted_indices]

    # TEST: add a test yerr if none provided
    if std_fidelity is None:
        yerr = np.full_like(y, 0.01)  # test error bars of 0.01
    else:
        yerr = np.array(std_fidelity)[sorted_indices]

    error_scale = 1.0  # Adjust this value as needed
    yerr = yerr * error_scale

    fig, ax = plt.subplots()

    # Plot with vertical error bars (yerr)
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=8, elinewidth=1, label='Average ± Std')

    # General logistic fit for a smooth trendline (plateau and shape fit to data)
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
    ax.set_xlim(left=min(x), right=max(x))
    plot_filename = f'debug_plotx_{metric_name.replace(' ', '_').lower()}.png'
    plt.savefig(plot_filename, format='png')
    plt.close(fig)