import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
from skimage import exposure
import os

def create_displacement_visual(tif_filename, output_png_filename, prefix_name):
    # Construct correct path inside the 'tiff' folder
    tif_path = os.path.join("tiff", tif_filename)
    print(f"Loading {tif_path} for displacement visualization...")
    if not os.path.exists(tif_path):
        # Fallback to root just in case
        tif_path = tif_filename
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Could not find {tif_filename} in 'tiff/' or root directory!")
            
    img = iio.imread(tif_path).astype(np.float32)
    h_total, w_total = img.shape
    
    # Split into Frame A (top) and Frame B (bottom) at 2048
    frame_a = img[0:2048, :]
    frame_b = img[2048:4096, :]
    
    # 1. Enhance both frames using a strong CLAHE + Gamma so tracers are highly visible
    print("Enhancing frames with CLAHE + Gamma...")
    def enhance(f):
        f_norm = (f - f.min()) / (f.max() - f.min()) if (f.max() - f.min()) > 0 else f
        f_clahe = exposure.equalize_adapthist(f_norm, kernel_size=64, clip_limit=0.08)
        return f_clahe ** 0.5
        
    enh_a = enhance(frame_a)
    enh_b = enhance(frame_b)
    
    # 2. Build Red-Cyan Overlay composite
    h_split, w_split = enh_a.shape
    rgb_composite = np.zeros((h_split, w_split, 3), dtype=np.float32)
    rgb_composite[..., 0] = enh_a         # Red channel (Pulse 1)
    rgb_composite[..., 1] = enh_b         # Green channel (Pulse 2)
    rgb_composite[..., 2] = enh_b         # Blue channel (Pulse 2)
    
    rgb_composite = np.clip(rgb_composite, 0.0, 1.0)
    
    # 3. Choose a region of interest with tracers for zoom-in (center of the image)
    zoom_size = 250
    cy, cx = h_split // 2, w_split // 2
    crop_composite = rgb_composite[cy-zoom_size:cy+zoom_size, cx-zoom_size:cx+zoom_size, :]
    
    # 4. Generate the comparison plot
    print("Rendering composite plot...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.patch.set_facecolor('#121214')
    
    # Full Frame Overlay
    ax1 = axes[0]
    ax1.imshow(rgb_composite)
    ax1.set_title(f"1. Full-Frame Red-Cyan Overlay ({prefix_name})", fontsize=18, color='white', fontweight='bold', pad=15)
    ax1.set_xlabel("X [pixels]", fontsize=14, color='white')
    ax1.set_ylabel("Y [pixels]", fontsize=14, color='white')
    ax1.tick_params(colors='white', labelsize=12)
    for spine in ax1.spines.values():
        spine.set_color('#333336')
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF1744', label='Frame A (Red, Pulse 1)'),
        Patch(facecolor='#00E5FF', label='Frame B (Cyan, Pulse 2)'),
        Patch(facecolor='#FFFFFF', label='Perfect Overlap (No Shift)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', facecolor='#1A1A1E', edgecolor='#333336', labelcolor='white', fontsize=12)
    
    # Zoomed Close-Up showing tracer shift
    ax2 = axes[1]
    ax2.imshow(crop_composite)
    ax2.set_title(f"2. Tracer Shifts / Displacement (500x500 Zoom Close-Up)", fontsize=18, color='white', fontweight='bold', pad=15)
    ax2.set_xlabel("X [pixels]", fontsize=14, color='white')
    ax2.set_ylabel("Y [pixels]", fontsize=14, color='white')
    ax2.tick_params(colors='white', labelsize=12)
    for spine in ax2.spines.values():
        spine.set_color('#333336')
        
    # Draw border around the zoom region
    rect = plt.Rectangle((cx-zoom_size, cy-zoom_size), zoom_size*2, zoom_size*2, fill=False, color='#00E5FF', linewidth=2, linestyle='--')
    ax1.add_patch(rect)
    ax1.text(cx-zoom_size, cy-zoom_size-20, "Zoom Area", color='#00E5FF', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_png_filename, facecolor='#121214', edgecolor='none', dpi=200)
    plt.close()
    print(f"Saved displacement figure: {output_png_filename}")

print("=== Starting Tracer Displacement Visualizations ===")
create_displacement_visual("B0001.tif", "displacement_B0001.png", "Frame B0001")
create_displacement_visual("B0002.tif", "displacement_B0002.png", "Frame B0002")
create_displacement_visual("B0003.tif", "displacement_B0003.png", "Frame B0003")
print("=== All Displacement Visualizations Completed! ===")
