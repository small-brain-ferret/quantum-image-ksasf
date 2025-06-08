
import pandas as pd
import numpy as np

def load_and_process_image(selected_index):
    dataset = pd.read_csv('mnist_dataset.csv')
    images = dataset.to_numpy()[:, 1:].reshape(42000, 8, 8)
    pixel_values = images.reshape(42000, 64)
    normalized_pixels = pixel_values / 255.0
    angles = np.arcsin(normalized_pixels[selected_index, :])
    return images, angles