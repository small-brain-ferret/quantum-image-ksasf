
import pandas as pd
import numpy as np
import os

def load_and_process_image(selected_index):
    dataset_path = os.path.join(os.path.dirname(__file__), 'static', 'mnist_dataset.csv')
    dataset = pd.read_csv(dataset_path)
    images = dataset.to_numpy()[:, 1:].reshape(42000, 8, 8)
    pixel_values = images.reshape(42000, 64)
    normalized_pixels = pixel_values / 255.0
    angles = np.arcsin(normalized_pixels[selected_index, :])
    return images, angles