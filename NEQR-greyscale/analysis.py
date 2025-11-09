import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from qiskit.quantum_info import Statevector, state_fidelity


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

def mae(original, retrieved, normalize=True):
    orig = np.asarray(original, dtype=np.float32)
    retr = np.asarray(retrieved, dtype=np.float32)
    if orig.shape != retr.shape:
        raise ValueError(f"shape mismatch: original {orig.shape} vs retrieved {retr.shape}")

    diff = np.abs(orig - retr)
    mae_val = diff.mean()
    if normalize:
        mae_val = mae_val / 255.0
    return float(1.0 - mae_val)

def SSIM(original, retrieved):
    return ssim(original, retrieved, data_range=255)

def quantum_state_fidelity(original, retrieved):
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(retrieved, torch.Tensor):
        retrieved = retrieved.detach().cpu().numpy()

    vec1 = original.flatten().astype(np.complex128)
    vec2 = retrieved.flatten().astype(np.complex128)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    vec1 /= norm1
    vec2 /= norm2

    sv1 = Statevector(vec1)
    sv2 = Statevector(vec2)

    fid = state_fidelity(sv1, sv2)
    return float(fid)