import torch
import numpy as np
import scipy.linalg
import time
from scipy import linalg

def calc_fid_torch(kps_gen, kps_gt, device):
    start_time = time.time()  
    
    # Convert data to PyTorch tensors and move to GPU/CPU as per device setting
    kps_gen = torch.tensor(kps_gen, dtype=torch.float32, device=device)
    kps_gt = torch.tensor(kps_gt, dtype=torch.float32, device=device)

    # Calculate means
    mu_gen = torch.mean(kps_gen, axis=0)
    mu_gt = torch.mean(kps_gt, axis=0)

    # Center the data
    kps_gen_centered = kps_gen - mu_gen
    kps_gt_centered = kps_gt - mu_gt

    # Compute covariance matrices on the specified device
    sigma_gen = (kps_gen_centered.t() @ kps_gen_centered) / (kps_gen.size(0) - 1)
    sigma_gt = (kps_gt_centered.t() @ kps_gt_centered) / (kps_gt.size(0) - 1)

    if device == 'cuda':
        # Transfer matrices to CPU and convert to NumPy for using SciPy
        sigma_gen_np = sigma_gen.cpu().numpy()
        sigma_gt_np = sigma_gt.cpu().numpy()
    else:
        sigma_gen_np = sigma_gen.numpy()
        sigma_gt_np = sigma_gt.numpy()

    # Calculate sqrt of product of covariance matrices using SciPy
    eps = 1e-6
    product = sigma_gen_np.dot(sigma_gt_np)
    covmean = scipy.linalg.sqrtm(product + eps * np.eye(product.shape[0]))

    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    # Convert back to PyTorch tensor and move to specified device
    covmean = torch.tensor(covmean, dtype=torch.float32, device=device)

    diff = mu_gen - mu_gt
    tr_covmean = torch.trace(covmean)

    # Final FID calculation
    fid_score = (diff.dot(diff) + torch.trace(sigma_gen) + torch.trace(sigma_gt) - 2 * tr_covmean).item()

    elapsed_time = time.time() - start_time 
    return fid_score, elapsed_time


def calc_fid(kps_gen, kps_gt):

    start_time = time.time()  

    kps_gen, kps_gt = np.array(kps_gen), np.array(kps_gt)
    # mean, std = kps_gt.

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)
    elapsed_time = time.time() - start_time  

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean), elapsed_time

# Example data
n, c = 120, 3072  # Larger datasets
a = np.random.rand(n, c)
b = np.random.rand(n, c)

# Calculate FID score on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fid_score_gpu, time_gpu = calc_fid_torch(a, b, device)
print("FID score calculated with GPU:", fid_score_gpu, "Time:", time_gpu, "seconds")

# Calculate FID score on CPU
fid_score_cpu, time_cpu = calc_fid_torch(a, b, 'cpu')
print("FID score calculated with CPU:", fid_score_cpu, "Time:", time_cpu, "seconds")

# Calculate FID score on CPU
fid_score_cpu, time_cpu = calc_fid(a, b)
print("FID score calculated with CPU:", fid_score_cpu, "Time:", time_cpu, "seconds")
