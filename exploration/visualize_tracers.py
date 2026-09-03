import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import os
from skimage import exposure

def enhance_and_save(image_path, output_png_path, title_name):
    print(f"Loading {image_path}...")
    img = iio.imread(image_path).astype(np.float32)
    
    img_min, img_max = img.min(), img.max()
    
    # We will generate a comparison figure showing standard vs log vs Strong CLAHE + Gamma
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.patch.set_facecolor('#121214')
    
    for ax in axes:
        ax.set_facecolor('#121214')
        ax.tick_params(colors='white', labelsize=12)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333336')
            
    # Method 1: Standard Linear Scaling (0 to 1 normalized)
    ax1 = axes[0]
    img_linear = (img - img_min) / (img_max - img_min) if (img_max - img_min) > 0 else img
    ax1.imshow(img_linear, cmap='gray')
    ax1.set_title("1. Standard Linear Scaling (Clipped)", fontsize=16, pad=15, fontweight='bold')
    ax1.set_xlabel("X [pixels]", fontsize=14)
    ax1.set_ylabel("Y [pixels]", fontsize=14)
    
    # Method 2: Logarithmic Compression (magma colormap)
    ax2 = axes[1]
    img_log = np.log1p(img)
    img_log_norm = (img_log - img_log.min()) / (img_log.max() - img_log.min())
    ax2.imshow(img_log_norm, cmap='magma')
    ax2.set_title("2. Logarithmic Scaling (Magma Colormap)", fontsize=16, pad=15, fontweight='bold')
    ax2.set_xlabel("X [pixels]", fontsize=14)
    ax2.set_ylabel("Y [pixels]", fontsize=14)
    
    # Method 3: Strong CLAHE + Gamma Correction (highly boosted local contrast)
    ax3 = axes[2]
    # Normalize
    img_norm = (img - img_min) / (img_max - img_min) if (img_max - img_min) > 0 else img
    # Apply strong CLAHE (clip_limit raised to 0.08 for extremely crisp edges)
    img_clahe = exposure.equalize_adapthist(img_norm, kernel_size=64, clip_limit=0.08)
    # Apply power-law gamma correction (gamma = 0.5) to lift mid-tones and dark background
    img_boosted = img_clahe ** 0.5
    
    ax3.imshow(img_boosted, cmap='gray')
    ax3.set_title("3. Strong CLAHE + Gamma=0.5 (Maximum Visibility Boost)", fontsize=16, pad=15, fontweight='bold')
    ax3.set_xlabel("X [pixels]", fontsize=14)
    ax3.set_ylabel("Y [pixels]", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_png_path, facecolor='#121214', edgecolor='none', dpi=200)
    plt.close()
    print(f"Saved comparison: {output_png_path}")
    
    # Save full-resolution strongly-enhanced gray image
    full_res_path = f"enhanced_full_{image_path.lower().replace('.tif', '.png')}"
    plt.figure(figsize=(12, 20))
    plt.imshow(img_boosted, cmap='gray')
    plt.axis('off')
    plt.savefig(full_res_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Saved full-res strong gray image: {full_res_path}")

print("Processing tracer image visualizations with Strong CLAHE + Gamma...")
enhance_and_save("B0001.tif", "tracers_B0001.png", "Frame B0001")
enhance_and_save("B0002.tif", "tracers_B0002.png", "Frame B0002")
enhance_and_save("B0003.tif", "tracers_B0003.png", "Frame B0003")
print("\n=== Finished generating strong tracer visualizations! ===")
