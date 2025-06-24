import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from entropy_utils import image_entropy
from plot_scatter import plot_fidelity_scatter

# Import your FRQI encode/decode functions here
# from your_frqi_module import frqi_encode, frqi_decode

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Progress bars will be disabled. To enable, install tqdm via pip.")
    def tqdm(x, **kwargs):
        return x

def mae_fidelity(original, reconstructed):
    return 1.0 - np.mean(np.abs(original.astype(float) - reconstructed.astype(float)))/255.0

# Load MNIST data (adjust path as needed)
import pandas as pd
dataset = pd.read_csv('../website/static/mnist_dataset.csv')

# Parameters
num_images = 20  # or 100 for full analysis
shot_counts = np.arange(100, 4001, 100)

results = []

for img_idx in tqdm(range(num_images), desc='Images'):
    image_data = dataset.to_numpy()[img_idx, 1:]
    image = image_data.reshape(8, 8).astype(np.uint8)  # or (28,28) if using full MNIST
    entropy_val = image_entropy(image)
    for shots in shot_counts:
        # --- FRQI encode and reconstruct ---
        # angles = frqi_encode(image)
        # reconstructed = frqi_decode(angles, shots)
        # For demonstration, use noisy image as placeholder:
        reconstructed = image + np.random.normal(0, 10, image.shape)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        mae = mae_fidelity(image, reconstructed)
        ssim_val = ssim(image, reconstructed, data_range=255)
        results.append({
            'image_id': img_idx,
            'shots': shots,
            'entropy': entropy_val,
            'fidelity_mae': mae,
            'fidelity_ssim': ssim_val
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('frqi_entropy_fidelity_results.csv', index=False)

# Plot
plot_fidelity_scatter(results_df, output_dir='figures')

print('Analysis complete. Results and plots saved.')
