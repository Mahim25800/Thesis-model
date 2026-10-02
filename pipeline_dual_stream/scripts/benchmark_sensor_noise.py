import cv2
import numpy as np
from pathlib import Path
import glob
from scipy import signal, ndimage

def compute_sensor_noise_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w, c = img.shape
    if min(h, w) < 64:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_f = img.astype(np.float32) / 255.0

    # 1. 5x5 SRM / KV high-pass filter residual
    kv_kernel = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0
    res = cv2.filter2D(gray, -1, kv_kernel)

    # 2. Local gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Flat regions (bokeh, skin, sky, flat surfaces)
    flat_mask = grad_mag < 0.025
    flat_ratio = float(np.mean(flat_mask))

    # Noise std in flat areas (scaled by 1000)
    if flat_ratio > 0.02:
        flat_noise_std = float(np.std(res[flat_mask])) * 1000.0
    else:
        flat_noise_std = float(np.std(res)) * 1000.0

    # 3. Noise Kurtosis & Skewness of residual (Gaussian physical sensor noise has kurtosis ~ 3 / excess kurtosis ~ 0)
    # AI generated images often have super-Gaussian or heavy-tailed residual distributions (kurtosis > 10)
    r_flat = res[flat_mask] if flat_ratio > 0.02 else res.flatten()
    r_mean = np.mean(r_flat)
    r_std = np.std(r_flat) + 1e-8
    r_norm = (r_flat - r_mean) / r_std
    kurtosis = float(np.mean(r_norm**4))

    # 4. Sensor Cross-Color Correlation (Bayer CFA trace)
    # In physical Bayer sensors, Green is sampled on a quincunx grid (2x Red and Blue).
    # The high-pass residual of G vs (R+B)/2 has specific variance and correlation
    cfa_res = img_f[:, :, 1] - 0.5 * (img_f[:, :, 0] + img_f[:, :, 2])
    cfa_hp = cv2.filter2D(cfa_res, -1, kv_kernel)
    cfa_std = float(np.std(cfa_hp)) * 1000.0

    # 5. Patch-level noise consistency / Spatial uniformity
    # A physical sensor has a consistent noise floor across all flat patches.
    # We partition the image into 32x32 blocks, compute noise std in flat blocks,
    # and compute the Coefficient of Variation (CV = std(patch_noise) / mean(patch_noise)).
    # Authentic cameras have uniform noise (low CV); AI models have patchy / non-uniform noise (high CV).
    bs = 32
    patch_stds = []
    for r in range(0, h - bs, bs):
        for c in range(0, w - bs, bs):
            p_flat = flat_mask[r:r+bs, c:c+bs]
            if np.mean(p_flat) > 0.6:  # mostly flat patch
                p_res = res[r:r+bs, c:c+bs]
                patch_stds.append(np.std(p_res) * 1000.0)
                
    if len(patch_stds) >= 4:
        patch_mean = np.mean(patch_stds)
        patch_std = np.std(patch_stds)
        noise_uniformity = float(patch_std / (patch_mean + 1e-4))
        flat_patch_count = len(patch_stds)
    else:
        noise_uniformity = 0.5
        flat_patch_count = len(patch_stds)

    # 6. Spectral Roll-off in 2D Fourier domain
    # Authentic camera noise has a flat, white noise floor at the highest octave.
    # Latent diffusion models (due to 8x downsampling and deconvolution) exhibit high-frequency roll-off.
    f = np.fft.fft2(res)
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift)**2
    cy, cx = h // 2, w // 2
    r_max = min(cy, cx)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # Band 1: Mid-high frequencies (0.4 to 0.7 r_max)
    # Band 2: Nyquist highest frequencies (0.7 to 1.0 r_max)
    mask_mid = (dist > 0.4 * r_max) & (dist <= 0.7 * r_max)
    mask_high = (dist > 0.7 * r_max) & (dist <= 1.0 * r_max)
    mid_p = np.mean(psd[mask_mid]) if np.any(mask_mid) else 1e-6
    high_p = np.mean(psd[mask_high]) if np.any(mask_high) else 1e-6
    hf_ratio = float(high_p / (mid_p + 1e-6))

    return {
        'flat_noise_std': flat_noise_std,
        'kurtosis': kurtosis,
        'cfa_std': cfa_std,
        'noise_uniformity': noise_uniformity,
        'hf_ratio': hf_ratio,
        'flat_ratio': flat_ratio,
        'flat_patch_count': flat_patch_count,
    }

def summarize_set(name, paths):
    feats = []
    for p in paths:
        f = compute_sensor_noise_features(p)
        if f is not None:
            feats.append(f)
    if not feats:
        print(f"No valid images for {name}")
        return
    print(f"\n=== Dataset: {name} (N={len(feats)}) ===")
    for k in ['flat_noise_std', 'kurtosis', 'cfa_std', 'noise_uniformity', 'hf_ratio']:
        vals = [x[k] for x in feats]
        print(f"  {k:<18}: mean={np.mean(vals):.3f} +/- {np.std(vals):.3f} (min={np.min(vals):.3f}, max={np.max(vals):.3f})")

def main():
    # Individual targets
    targets = [
        ('Smartphone Clean', 'data/test_smartphone_portrait_clean.png'),
        ('Blue Bedroom Anime', 'data/test_blue_bedroom.png'),
        ('Catwoman AI', 'data/catwoman_extracted.png'),
    ]
    for name, p in targets:
        f = compute_sensor_noise_features(p)
        print(f"\nTarget: {name}")
        for k, v in f.items():
            print(f"  {k}: {v:.4f}")

    # Datasets
    raise_paths = glob.glob('G:/Thesis/pipeline_40k/data/external_synthbuster/raise/real_RAISE_1k/*.png')[:25]
    summarize_set('RAISE Camera Photos (Real)', raise_paths)

    mj_paths = glob.glob('G:/Thesis/pipeline_40k/data/external_genimage_confirmatory_disjoint/selected/midjourney/fake/*.png')[:25]
    summarize_set('Midjourney (Fake)', mj_paths)

    adm_paths = glob.glob('G:/Thesis/pipeline_40k/data/external_genimage_dev/selected/adm/fake/*.png')[:25]
    summarize_set('ADM Diffusion (Fake)', adm_paths)

if __name__ == '__main__':
    main()
