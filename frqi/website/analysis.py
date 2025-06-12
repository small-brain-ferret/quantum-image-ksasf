import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def balanced_weighted_mae(original, retrieved):
    # Convert to torch tensors and normalize
    orig = torch.tensor(original, dtype=torch.float32) / 255.0
    retr = torch.tensor(retrieved, dtype=torch.float32) / 255.0

    # Create a mask to ignore all-black pixels in the original
    mask = (orig > 0)

    # Apply mask
    orig_masked = orig[mask]
    retr_masked = retr[mask]

    if orig_masked.numel() == 0:
        return 0.0  # or 1.0 depending on desired behavior for all-black

    # Compute MAE on masked values
    loss = F.l1_loss(orig_masked, retr_masked)
    return 1.0 - loss.item()  # Higher = better fidelity

def compute_fidelity(original, retrieved):
    return ssim(original, retrieved, data_range=255)
 