import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import lvpyio as lv
from openpiv.piv import simple_piv
from openpiv import tools
import time
import json
import os

def load_parameters():
    params_path = "piv_parameters.json"
    if os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
            print(f"Loaded parameters from {params_path}: {params}")
            return params
        except Exception as e:
            print(f"Error loading {params_path}, using defaults. Error: {e}")
    else:
        print("piv_parameters.json not found, using default parameters.")
    
    return {
        "window_size": 64,
        "overlap": 32,
        "search_area_size": 64,
        "dt": 1.0,
        "validation_method": "sig2noise",
        "s2n_thresh": 1.1,
        "scale_plot": 80,
        "arrow_skip": 2
    }

def process_tif_as_piv_pair(tif_filename, im7_filename, prefix_name, params):
    print(f"\n=======================================================")
    print(f"Processing {prefix_name}: {tif_filename}")
    print(f"=======================================================")
    
    t0 = time.time()
    
    # 1. Load the raw 16-bit TIFF image
    print(f"Loading {tif_filename}...")
    img = iio.imread(tif_filename).astype(np.float32)
    h_total, w_total = img.shape
    print(f"Raw image size: {w_total}x{h_total}")
    
    # 2. Split vertically into two frames of height 2048
    h_split = 2048
    print(f"Splitting vertically into two frames: 0:2048 and 2048:4096...")
    frame_a = img[0:h_split, :]
    frame_b = img[h_split:4096, :]
    
    # 3. Save as a LaVision multi-frame IM7 file using lvpyio
    print(f"Writing split frames as dual-frame IM7 file: {im7_filename}...")
    lv.write_buffer([frame_a, frame_b], im7_filename)
    
    # 4. Read back using lvpyio to verify and run PIV
    print(f"Reading back {im7_filename} using lvpyio.read_buffer...")
    buffer = lv.read_buffer(im7_filename)
    print(f"Loaded LaVision buffer. Frame count: {len(buffer)}")
    
    # Extract the frames as numpy arrays from the lvpyio masked arrays
    im_a_masked = buffer.as_masked_array(0)
    im_b_masked = buffer.as_masked_array(1)
    
    # Extract underlying data for PIV processing
    im_a = im_a_masked.data
    im_b = im_b_masked.data
    
    # 5. Perform simple_piv between Frame A and Frame B of the same file!
    window_size = params.get("window_size", 64)
    overlap = params.get("overlap", 32)
    search_area_size = params.get("search_area_size", 64)
    dt = params.get("dt", 1.0)
    validation_method = params.get("validation_method", "sig2noise")
    s2n_thresh = params.get("s2n_thresh", 1.1)
    
    print(f"Running simple_piv (window={window_size}, overlap={overlap}, search={search_area_size}, dt={dt}, validation={validation_method})...")
    x_phys, y_phys, u_final_phys, v_final_phys, s2n = simple_piv(
        im_a, im_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        dt=dt,
        plot=False,
        validation_method=validation_method,
        s2n_thresh=s2n_thresh
    )
    
    # Create invalid mask based on S/N threshold
    invalid_mask = s2n <= s2n_thresh
    
    t_piv = time.time()
    print(f"PIV analysis and validation completed in {t_piv - t0:.2f} seconds.")
    
    # 6. Save intermediate steps into a .npz file
    npz_filename = f"piv_intermediate_{prefix_name.lower().replace(' ', '_')}.npz"
    np.savez(
        npz_filename,
        x=x_phys,
        y=y_phys,
        u_final=u_final_phys,
        v_final=v_final_phys,
        s2n=s2n,
        invalid_mask=invalid_mask
    )
    print(f"Saved intermediate arrays to: {npz_filename}")
    
    # 7. Create a high-contrast display image for the background
    im_max = np.max(im_a)
    im_display = (im_a.astype(np.float32) / im_max * 255.0).astype(np.uint8) if im_max > 0 else im_a.astype(np.uint8)
    
    # Calculate velocity magnitude
    speed = np.sqrt(u_final_phys**2 + v_final_phys**2)
    
    print("Generating high-resolution 3-panel visualization...")
    
    # Create beautiful 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.patch.set_facecolor('#121214')
    
    # Shared styling
    for ax in axes:
        ax.set_facecolor('#121214')
        ax.tick_params(colors='white', labelsize=12)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333336')
            
    height, width = im_display.shape
    extent = [0, width, 0, height]
    
    # Arrow skip and scale parameters
    skip = params.get("arrow_skip", 2)
    scale = params.get("scale_plot", 80)
    
    # Panel 1: Raw Vector Field representation (quiver plot highlighting valid vs invalid)
    ax1 = axes[0]
    ax1.imshow(im_display[::-1, :], cmap='gray', extent=extent, alpha=0.85)
    
    valid_mask_skip = ~invalid_mask[::skip, ::skip]
    ax1.quiver(
        x_phys[::skip, ::skip][valid_mask_skip],
        y_phys[::skip, ::skip][valid_mask_skip],
        u_final_phys[::skip, ::skip][valid_mask_skip],
        v_final_phys[::skip, ::skip][valid_mask_skip],
        color='#00E5FF',  # Electric Cyan
        scale=scale,
        width=0.002,
        headwidth=3.5,
        headlength=4.5,
        label='Valid Vectors'
    )
    
    invalid_mask_skip = invalid_mask[::skip, ::skip]
    if np.any(invalid_mask_skip):
        ax1.quiver(
            x_phys[::skip, ::skip][invalid_mask_skip],
            y_phys[::skip, ::skip][invalid_mask_skip],
            u_final_phys[::skip, ::skip][invalid_mask_skip],
            v_final_phys[::skip, ::skip][invalid_mask_skip],
            color='#FF1744',  # Bright Red
            scale=scale,
            width=0.002,
            headwidth=3.5,
            headlength=4.5,
            label='Low S/N Vectors'
        )
    ax1.set_title(f"1. Vector Field & S/N Validation ({prefix_name})", fontsize=16, pad=15, fontweight='bold')
    ax1.set_xlabel("X [pixels]", fontsize=14)
    ax1.set_ylabel("Y [pixels]", fontsize=14)
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', facecolor='#1A1A1E', edgecolor='#333336', labelcolor='white', fontsize=12)
    
    # Panel 2: Validated / Cleaned Vector Field
    ax2 = axes[1]
    ax2.imshow(im_display[::-1, :], cmap='gray', extent=extent, alpha=0.85)
    ax2.quiver(
        x_phys[::skip, ::skip],
        y_phys[::skip, ::skip],
        u_final_phys[::skip, ::skip],
        v_final_phys[::skip, ::skip],
        color='#00E5FF',  # Electric Cyan
        scale=scale,
        width=0.002,
        headwidth=3.5,
        headlength=4.5
    )
    ax2.set_title(f"2. Validated Field (After Outlier Replacement)", fontsize=16, pad=15, fontweight='bold')
    ax2.set_xlabel("X [pixels]", fontsize=14)
    ax2.set_ylabel("Y [pixels]", fontsize=14)
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect('equal')
    
    # Panel 3: Velocity Magnitude Contour Map
    ax3 = axes[2]
    cf = ax3.contourf(x_phys, y_phys, speed, cmap='plasma', levels=100, extend='both')
    
    cbar = fig.colorbar(cf, ax=ax3, orientation='vertical', pad=0.03, shrink=0.7)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=12)
    cbar.set_label('Velocity Magnitude [px/dt]', color='white', fontsize=14, labelpad=15)
    cbar.outline.set_color('#333336')
    
    ax3.quiver(
        x_phys[::skip, ::skip],
        y_phys[::skip, ::skip],
        u_final_phys[::skip, ::skip],
        v_final_phys[::skip, ::skip],
        color='white',
        scale=scale,
        width=0.0018,
        headwidth=3.5,
        headlength=4.5
    )
    ax3.set_title(f"3. Velocity Magnitude Contour Plot", fontsize=16, pad=15, fontweight='bold')
    ax3.set_xlabel("X [pixels]", fontsize=14)
    ax3.set_ylabel("Y [pixels]", fontsize=14)
    ax3.set_xlim(0, width)
    ax3.set_ylim(0, height)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    png_filename = f"piv_result_{prefix_name.lower().replace(' ', '_')}.png"
    plt.savefig(png_filename, facecolor='#121214', edgecolor='none', dpi=200)
    plt.close()
    
    t_done = time.time()
    print(f"Saved figure to: {png_filename} (done in {t_done - t_piv:.2f}s)")
    
    stats = {
        "mean_u": np.nanmean(u_final_phys),
        "mean_v": np.nanmean(v_final_phys),
        "mean_speed": np.nanmean(speed),
        "max_speed": np.nanmax(speed),
        "outlier_count": np.sum(invalid_mask),
        "outlier_percentage": (np.sum(invalid_mask) / invalid_mask.size) * 100.0,
        "grid_shape": u_final_phys.shape
    }
    return stats

def main():
    params = load_parameters()
    
    stats_1 = process_tif_as_piv_pair("B0001.tif", "B0001.im7", "Pair 1", params)
    stats_2 = process_tif_as_piv_pair("B0002.tif", "B0002.im7", "Pair 2", params)
    stats_3 = process_tif_as_piv_pair("B0003.tif", "B0003.im7", "Pair 3", params)
    
    print("\n=======================================================")
    print("=== All Dual-Frame PIV Processing Completed! ===")
    print("=======================================================")
    print(f"Pair 1 (B0001): Grid={stats_1['grid_shape']}, Outliers={stats_1['outlier_count']} ({stats_1['outlier_percentage']:.2f}%), Mean Speed={stats_1['mean_speed']:.2f} px")
    print(f"Pair 2 (B0002): Grid={stats_2['grid_shape']}, Outliers={stats_2['outlier_count']} ({stats_2['outlier_percentage']:.2f}%), Mean Speed={stats_2['mean_speed']:.2f} px")
    print(f"Pair 3 (B0003): Grid={stats_3['grid_shape']}, Outliers={stats_3['outlier_count']} ({stats_3['outlier_percentage']:.2f}%), Mean Speed={stats_3['mean_speed']:.2f} px")

if __name__ == "__main__":
    main()
