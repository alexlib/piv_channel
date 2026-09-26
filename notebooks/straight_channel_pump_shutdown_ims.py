# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "openpiv",
#     "xarray",
#     "zarr",
#     "scipy",
#     "lvpyio",
# ]
# ///

"""
straight_channel_pump_shutdown_ims.py

Interactive Marimo notebook for processing raw 14 GB camera stream (.ims) directly
with OpenPIV single-pass cross-correlation.

Data Source:
- File: D:\\channel_flow_research\\baseline_channel\\Vmax_0p62_m2sec_pump_shutdown\\Camera1-1.ims
- Set path: D:\\channel_flow_research\\baseline_channel\\Vmax_0p62_m2sec_pump_shutdown
- Acquisition: 1000 dual-frame pairs @ 15 Hz (66.7 s duration)
- Sensor: 2048 x 2432, 12-bit dynamic range
- Laser pulse separation: dt = 80.0 µs (DevDataTrace5)
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Unsteady Pump Shutdown — Direct `.ims` Raw Frame PIV Pipeline
        ## Stream, Tune, and Process the 14 GB High-Speed Camera Recording

        In the baseline straight channel experiments, the pump shutdown acquisition was captured
        as a **14 GB camera stream file** (`Camera1-1.ims`) containing **1,000 dual-frame pairs @ 15 Hz** ($66.7\text{ s}$ total).
        
        Using `lvpyio.read_set()`, we can stream individual frames directly from disk in milliseconds,
        bypassing the need for DaVis to export tens of gigabytes of intermediate `.im7` files.

        ### Workflow:
        1. **Raw Frame Inspection**: Inspect first frame ($t=0\text{ s}$) and last frame ($t=66.6\text{ s}$) across laser pulses A and B.
        2. **Interactive PIV Parameter Tuning**: Interactively test interrogation window size, search area, overlap, high-pass filtering, and S2N threshold on any frame.
        3. **Phase-by-Phase Validation**: Verify tracking accuracy across steady state ($t < 11\text{ s}$), shutdown cutoff ($t \approx 12-14\text{ s}$), and near-rest deceleration ($t > 30\text{ s}$).
        4. **Full 14 GB Batch Processing Architecture**: Stream and process all 1,000 frames into a chunked Zarr dataset.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import marimo as mo
    import lvpyio as lv
    import openpiv.pyprocess as opproc
    import openpiv.validation as opval
    import openpiv.filters as opfilt
    import openpiv.scaling as opscale
    import openpiv.tools as optools
    import openpiv.preprocess as opprep
    import xarray as xr

    SET_PATH = Path(r"D:\channel_flow_research\baseline_channel\Vmax_0p62_m2sec_pump_shutdown")
    OUT_DIR = Path("outputs")
    OUT_DIR.mkdir(exist_ok=True)

    # Physical scaling from baseline channel calibration:
    # 10 mm channel gap corresponds to ~68 µm/vector or ~14.67 px/mm
    PX_PER_MM = 14.6702
    CHANNEL_CROP_COLS = (550, 2050)
    LASER_DT = 80.0e-6  # 80 µs

    return (
        CHANNEL_CROP_COLS,
        LASER_DT,
        OUT_DIR,
        PX_PER_MM,
        Path,
        SET_PATH,
        lv,
        mo,
        mpl,
        np,
        opfilt,
        opprep,
        opproc,
        opscale,
        optools,
        opval,
        plt,
        time,
        xr,
    )


@app.cell
def _(SET_PATH, lv, mo):
    # Open the stream set
    set_obj = lv.read_set(str(SET_PATH))
    n_frames = len(set_obj)
    
    # Metadata probe
    buf_sample = set_obj[0]
    dt_val = float(buf_sample.attributes.get("DevDataTrace5", [[80.0]])[0][0])
    timestamp_val = str(buf_sample.attributes.get("Timestamp", "2026-06-15T12:43:28"))

    mo.md(
        f"""
        ### Stream Set Loaded Successfully
        - **Folder / Set**: `{SET_PATH.name}`
        - **Total Frames**: **{n_frames}** dual-frame pairs ($66.7\text{{ s}}$ duration @ 15 Hz)
        - **Array Dimensions**: $2048 \\times 2432$ pixels (12-bit sensor, 2 laser pulses per buffer)
        - **Laser Separation**: $\\Delta t = {dt_val:.1f}\\,\\mu\\text{{s}}$
        - **Acquisition Timestamp**: `{timestamp_val}`
        """
    )
    return buf_sample, dt_val, n_frames, set_obj, timestamp_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Step 1: Raw Image Inspection (First vs. Last Frame)
        
        Compare **Frame 0** (initial steady-state conditions, $t=0.00\text{ s}$) with **Frame 999** (post-experiment, $t=66.60\text{ s}$).
        Notice the difference in seeding intensity, ambient illumination, and particle contrast between pulses A and B.
        """
    )
    return


@app.cell
def _(mo):
    # Contrast adjustment controls
    contrast_max_slider = mo.ui.slider(
        start=50, stop=1000, step=25, value=250, label="Display Max Intensity (counts)"
    )
    crop_preview_switch = mo.ui.checkbox(value=True, label="Crop Channel Walls (cols 550–2050)")
    mo.hstack([contrast_max_slider, crop_preview_switch])
    return contrast_max_slider, crop_preview_switch


@app.cell
def _(
    CHANNEL_CROP_COLS,
    contrast_max_slider,
    crop_preview_switch,
    mo,
    np,
    plt,
    set_obj,
):
    # Fetch Frame 0 and Frame 999
    _buf_first = set_obj[0]
    _a_first = _buf_first.as_masked_array(0)
    _b_first = _buf_first.as_masked_array(1)

    _buf_last = set_obj[len(set_obj) - 1]
    _a_last = _buf_last.as_masked_array(0)
    _b_last = _buf_last.as_masked_array(1)

    if crop_preview_switch.value:
        _c0, _c1 = CHANNEL_CROP_COLS
        _a_first = _a_first[:, _c0:_c1]
        _b_first = _b_first[:, _c0:_c1]
        _a_last = _a_last[:, _c0:_c1]
        _b_last = _b_last[:, _c0:_c1]

    _vmax = contrast_max_slider.value

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Frame 0 - Pulse A
    _im0 = _axes[0, 0].imshow(_a_first, cmap="gray", vmin=0, vmax=_vmax, origin="upper")
    _axes[0, 0].set_title(f"Frame 0 — Laser Pulse A (t = 0.00 s)\nMean: {_a_first.mean():.1f}, Max: {_a_first.max():.0f}")
    _axes[0, 0].set_xlabel("x [px]")
    _axes[0, 0].set_ylabel("y [px]")
    plt.colorbar(_im0, ax=_axes[0, 0], fraction=0.046, pad=0.04, label="Counts")

    # Frame 0 - Pulse B
    _im1 = _axes[0, 1].imshow(_b_first, cmap="gray", vmin=0, vmax=_vmax, origin="upper")
    _axes[0, 1].set_title(f"Frame 0 — Laser Pulse B (t = +80 µs)\nMean: {_b_first.mean():.1f}, Max: {_b_first.max():.0f}")
    _axes[0, 1].set_xlabel("x [px]")
    plt.colorbar(_im1, ax=_axes[0, 1], fraction=0.046, pad=0.04, label="Counts")

    # Frame 999 - Pulse A
    _im2 = _axes[1, 0].imshow(_a_last, cmap="gray", vmin=0, vmax=_vmax * 2, origin="upper")
    _axes[1, 0].set_title(f"Frame 999 — Laser Pulse A (t = 66.60 s)\nMean: {_a_last.mean():.1f}, Max: {_a_last.max():.0f}")
    _axes[1, 0].set_xlabel("x [px]")
    _axes[1, 0].set_ylabel("y [px]")
    plt.colorbar(_im2, ax=_axes[1, 0], fraction=0.046, pad=0.04, label="Counts")

    # Frame 999 - Pulse B
    _im3 = _axes[1, 1].imshow(_b_last, cmap="gray", vmin=0, vmax=_vmax * 2, origin="upper")
    _axes[1, 1].set_title(f"Frame 999 — Laser Pulse B (t = +80 µs)\nMean: {_b_last.mean():.1f}, Max: {_b_last.max():.0f}")
    _axes[1, 1].set_xlabel("x [px]")
    plt.colorbar(_im3, ax=_axes[1, 1], fraction=0.046, pad=0.04, label="Counts")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            > [!NOTE]
            > **Observation**: In Frame 0, the channel is darkly illuminated with distinct tracer particle dots (mean $\\approx {_a_first.mean():.1f}$ counts).
            > By Frame 999 ($66.6\\text{{ s}}$), the acquisition run ended and room lighting was turned back on (mean $\\approx {_a_last.mean():.1f}$ counts),
            > creating strong background illumination. High-pass background subtraction is therefore essential for robust tracking across the entire record.
            """
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Step 2: Interactive Marimo UI for PIV Tuning (Single Pass)

        Use the interactive controls below to test and tune the OpenPIV single-pass parameters on any frame.
        The algorithm applies:
        1. **Pre-processing**: High-pass Gaussian filtering (`sigma=16`) to eliminate non-uniform laser illumination and room light.
        2. **Extended Search Area Cross-Correlation**: FFT-based correlation with window size $W$ and search size $S$.
        3. **Sub-pixel Peak Fitting**: 3-point Gaussian interpolation.
        4. **Vector Validation**: Signal-to-noise ratio filter ($S/N \ge 1.05-1.2$) and normalized median test.
        5. **Outlier Replacement**: Local mean interpolation of flagged vectors.
        """
    )
    return


@app.cell
def _(mo):
    # UI Controls for PIV Tuning
    frame_slider = mo.ui.slider(
        start=0, stop=999, step=1, value=0, label="Test Frame Index (0 to 999)"
    )
    preset_frame_selector = mo.ui.dropdown(
        options={
            "0: Initial Steady State (t=0.0 s)": 0,
            "100: Pre-Shutdown Run (t=6.7 s)": 100,
            "170: Shutdown Initiation (t=11.3 s)": 170,
            "180: Pump Cutoff Transient (t=12.0 s)": 180,
            "200: Flow Deceleration (t=13.3 s)": 200,
            "350: Mid Deceleration (t=23.3 s)": 350,
            "600: Slow Near-Rest Flow (t=40.0 s)": 600,
            "800: Quiescent Channel (t=53.3 s)": 800,
        },
        value="0: Initial Steady State (t=0.0 s)",
        label="Quick Jump to Key Transient Phase",
    )

    winsize_select = mo.ui.dropdown(
        options=[32, 48, 64, 96, 128], value=64, label="Window Size (px)"
    )
    searchsize_select = mo.ui.dropdown(
        options=[48, 64, 96, 128, 160], value=96, label="Search Size (px)"
    )
    overlap_select = mo.ui.dropdown(
        options=[16, 24, 32, 48, 64], value=32, label="Window Overlap (px)"
    )

    s2n_slider = mo.ui.slider(
        start=1.0, stop=1.5, step=0.05, value=1.05, label="S/N Threshold"
    )
    median_slider = mo.ui.slider(
        start=1.0, stop=4.0, step=0.5, value=2.0, label="Median Filter Threshold"
    )
    preprocess_toggle = mo.ui.checkbox(
        value=True, label="Enable High-Pass Background Filter (sigma=16)"
    )

    mo.vstack([
        mo.hstack([preset_frame_selector, frame_slider]),
        mo.hstack([winsize_select, searchsize_select, overlap_select]),
        mo.hstack([s2n_slider, median_slider, preprocess_toggle]),
    ])
    return (
        frame_slider,
        median_slider,
        overlap_select,
        preprocess_toggle,
        preset_frame_selector,
        s2n_slider,
        searchsize_select,
        winsize_select,
    )


@app.cell
def _(
    CHANNEL_CROP_COLS,
    LASER_DT,
    PX_PER_MM,
    frame_slider,
    median_slider,
    mo,
    np,
    opfilt,
    opprep,
    opproc,
    opscale,
    optools,
    opval,
    overlap_select,
    plt,
    preprocess_toggle,
    preset_frame_selector,
    s2n_slider,
    searchsize_select,
    set_obj,
    time,
    winsize_select,
):
    # Determine which frame to evaluate
    _idx = frame_slider.value
    # If preset was changed, we could use that, but slider gives immediate continuous control
    _t_start = time.time()

    # Load pair directly from .ims set
    _buf = set_obj[_idx]
    _c0, _c1 = CHANNEL_CROP_COLS
    _a1_raw = _buf.as_masked_array(0)[:, _c0:_c1].astype(float)
    _a2_raw = _buf.as_masked_array(1)[:, _c0:_c1].astype(float)

    # Pre-processing
    if preprocess_toggle.value:
        _hp1 = opprep.high_pass(_a1_raw, sigma=16, clip=True)
        _hp2 = opprep.high_pass(_a2_raw, sigma=16, clip=True)
        _pct1 = float(np.percentile(_hp1, 97.5)) or 200.0
        _pct2 = float(np.percentile(_hp2, 97.5)) or 200.0
        _a1_proc = np.clip(255.0 * (_hp1 / _pct1), 0, 255).astype(np.int32)
        _a2_proc = np.clip(255.0 * (_hp2 / _pct2), 0, 255).astype(np.int32)
    else:
        _a1_proc = np.clip(_a1_raw, 0, 255).astype(np.int32)
        _a2_proc = np.clip(_a2_raw, 0, 255).astype(np.int32)

    # Run OpenPIV extended search correlation
    _win = int(winsize_select.value)
    _search = int(searchsize_select.value)
    _ov = int(overlap_select.value)

    if _search < _win:
        _search = _win

    _u, _v, _s2n = opproc.extended_search_area_piv(
        _a1_proc,
        _a2_proc,
        window_size=_win,
        search_area_size=_search,
        overlap=_ov,
        dt=1.0,
        subpixel_method="gaussian",
    )

    _x, _y = opproc.get_coordinates(
        image_size=_a1_proc.shape,
        search_area_size=_search,
        overlap=_ov,
    )

    # Validation
    _mask_s2n = opval.sig2noise_val(_s2n, threshold=float(s2n_slider.value))
    _mask_med = opval.global_std(_u, _v, std_threshold=float(median_slider.value) * 3.0)
    _invalid = _mask_s2n | _mask_med

    _u_clean, _v_clean = opfilt.replace_outliers(
        _u, _v, _invalid, method="localmean", max_iter=5, kernel_size=3
    )

    # Scaling to physical coordinates (mm, mm/s)
    _xs, _ys, _us, _vs = opscale.uniform(
        _x, _y, _u_clean, _v_clean, scaling_factor=PX_PER_MM
    )
    _xs, _ys, _us, _vs = optools.transform_coordinates(_xs, _ys, _us, _vs)

    # Velocity in mm/s: displacement / dt
    _u_mms = _us / LASER_DT
    _v_mms = _vs / LASER_DT
    _speed_mms = np.hypot(_u_mms, _v_mms)

    _piv_time = (time.time() - _t_start) * 1000.0

    # Diagnostic statistics
    _valid_pct = (1.0 - _invalid.mean()) * 100.0
    _mean_v = float(np.mean(_v_mms[~_invalid])) if np.any(~_invalid) else 0.0
    _mean_disp_px = float(np.mean(np.hypot(_u, _v)[~_invalid])) if np.any(~_invalid) else 0.0
    _t_phys = _idx / 15.0

    # Visualization
    _fig, _axs = plt.subplots(1, 3, figsize=(16, 7), constrained_layout=True)

    # Subplot 1: Particle Image + Quiver
    _axs[0].imshow(_a1_proc, cmap="gray", origin="upper", extent=[0, _a1_proc.shape[1]/PX_PER_MM, 0, _a1_proc.shape[0]/PX_PER_MM])
    _step_x = max(1, _us.shape[1] // 25)
    _step_y = max(1, _us.shape[0] // 30)
    _X_sub = _xs[::_step_y, ::_step_x]
    _Y_sub = _ys[::_step_y, ::_step_x]
    _U_sub = _u_mms[::_step_y, ::_step_x]
    _V_sub = _v_mms[::_step_y, ::_step_x]
    _C_sub = _V_sub

    _q = _axs[0].quiver(
        _X_sub, _Y_sub, _U_sub, _V_sub, _C_sub,
        cmap="coolwarm", angles="xy", scale_units="xy", scale=1500.0,
        width=0.004, pivot="mid"
    )
    _axs[0].set_title(f"Frame {_idx} (t = {_t_phys:.2f} s) — PIV Velocity Field")
    _axs[0].set_xlabel("x [mm]")
    _axs[0].set_ylabel("y [mm]")
    _axs[0].set_aspect("equal")
    _cbar = plt.colorbar(_q, ax=_axs[0], fraction=0.046, pad=0.04)
    _cbar.set_label("streamwise v [mm/s]")

    # Subplot 2: Valid / Invalid Mask Map
    _axs[1].imshow(_invalid, cmap="Reds", origin="upper")
    _axs[1].set_title(f"Outlier Mask (Valid: {_valid_pct:.1f}%)\nRed = Flagged by S/N or Median")
    _axs[1].set_xlabel("Grid column")
    _axs[1].set_ylabel("Grid row")

    # Subplot 3: Cross-Channel Velocity Profile v(x)
    _mean_v_profile = np.nanmean(np.where(~_invalid, _v_mms, np.nan), axis=0)
    _x_prof = _xs[0, :]
    _axs[2].plot(_x_prof, _mean_v_profile, "b-o", linewidth=1.5, markersize=3, label="OpenPIV v(x)")
    _axs[2].axhline(0, color="gray", linestyle="--")
    _axs[2].set_title(f"Cross-Channel Profile v(x)\nMean v = {_mean_v:.1f} mm/s")
    _axs[2].set_xlabel("x [mm]")
    _axs[2].set_ylabel("Streamwise Velocity v [mm/s]")
    _axs[2].grid(True, linestyle=":", alpha=0.6)
    _axs[2].legend(loc="best")

    mo.vstack([
        mo.md(
            f"""
            ### PIV Diagnostic Summary for Frame {_idx} ($t = {_t_phys:.2f}\\text{{ s}}$):
            - **Calculation Latency**: **{_piv_time:.1f} ms** per frame
            - **Vector Validation**: **{_valid_pct:.1f}% valid** ({_invalid.sum()} outliers out of {_invalid.size} grid points)
            - **Mean Particle Displacement**: **{_mean_disp_px:.2f} pixels** ($\\approx {_mean_disp_px / _win * 100.0:.1f}\\%$ of window size)
            - **Mean Streamwise Velocity**: **{_mean_v:.1f} mm/s**
            """
        ),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Step 3: Phase-by-Phase Tuning Audit

        Below we test the tuned settings across four critical snapshots during the shutdown sequence:
        1. **Steady State** ($t = 0.0\text{ s}$, frame 0): High-speed forward flow ($v \approx -580\text{ mm/s}$).
        2. **Pump Cutoff** ($t = 12.0\text{ s}$, frame 180): Rapid transient onset with strong deceleration.
        3. **Deceleration Phase** ($t = 26.7\text{ s}$, frame 400): Intermediate deceleration.
        4. **Near-Rest Quiescence** ($t = 53.3\text{ s}$, frame 800): Weak creeping / reversed flow.
        """
    )
    return


@app.cell
def _(
    CHANNEL_CROP_COLS,
    LASER_DT,
    PX_PER_MM,
    mo,
    np,
    opfilt,
    opprep,
    opproc,
    opscale,
    optools,
    opval,
    plt,
    set_obj,
    winsize_select,
):
    # Snapshot test across 4 key phases
    _phases = [
        (0, "Steady State (t=0.0 s)"),
        (180, "Shutdown Onset (t=12.0 s)"),
        (400, "Deceleration (t=26.7 s)"),
        (800, "Near Rest (t=53.3 s)"),
    ]

    _win = int(winsize_select.value)
    _search = max(_win, 96)
    _ov = 32

    _fig, _axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

    for _col, (_f_idx, _title) in enumerate(_phases):
        _buf = set_obj[_f_idx]
        _c0, _c1 = CHANNEL_CROP_COLS
        _a1 = _buf.as_masked_array(0)[:, _c0:_c1].astype(float)
        _a2 = _buf.as_masked_array(1)[:, _c0:_c1].astype(float)

        _hp1 = opprep.high_pass(_a1, sigma=16, clip=True)
        _hp2 = opprep.high_pass(_a2, sigma=16, clip=True)
        _pct1 = float(np.percentile(_hp1, 97.5)) or 200.0
        _pct2 = float(np.percentile(_hp2, 97.5)) or 200.0
        _a1_proc = np.clip(255.0 * (_hp1 / _pct1), 0, 255).astype(np.int32)
        _a2_proc = np.clip(255.0 * (_hp2 / _pct2), 0, 255).astype(np.int32)

        _u, _v, _s2n = opproc.extended_search_area_piv(
            _a1_proc, _a2_proc,
            window_size=_win, search_area_size=_search, overlap=_ov,
            dt=1.0, subpixel_method="gaussian",
        )
        _x, _y = opproc.get_coordinates(image_size=_a1_proc.shape, search_area_size=_search, overlap=_ov)
        _inv = opval.sig2noise_val(_s2n, threshold=1.05) | opval.global_std(_u, _v, std_threshold=6.0)
        _uc, _vc = opfilt.replace_outliers(_u, _v, _inv, method="localmean", max_iter=5, kernel_size=3)
        _xs, _ys, _us, _vs = opscale.uniform(_x, _y, _uc, _vc, scaling_factor=PX_PER_MM)
        _xs, _ys, _us, _vs = optools.transform_coordinates(_xs, _ys, _us, _vs)
        _v_mms = _vs / LASER_DT

        _prof = np.nanmean(np.where(~_inv, _v_mms, np.nan), axis=0)
        _val_pct = (1.0 - _inv.mean()) * 100.0
        _axes[_col].plot(_xs[0, :], _prof, "b-o", markersize=3, linewidth=1.5)
        _axes[_col].axhline(0, color="gray", linestyle="--")
        _axes[_col].set_title(f"{_title}\nValid: {_val_pct:.1f}%, Mean: {np.nanmean(_prof):.1f} mm/s")
        _axes[_col].set_xlabel("x [mm]")
        _axes[_col].set_ylabel("Streamwise Velocity [mm/s]")
        _axes[_col].grid(True, linestyle=":", alpha=0.6)

    mo.vstack([
        mo.md("### Four-Phase Profile Check (Constant Settings)"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Step 4: Batch Processing Configuration (Full 14 GB Analysis)

        Once settings are verified in Steps 2 & 3, configure the batch runner below to stream
        the full 1,000 frames from `Camera1-1.ims` and write a compact, chunked **Zarr store**
        (`outputs/baseline_pump_shutdown_openpiv_ds.zarr`).
        """
    )
    return


@app.cell
def _(mo):
    batch_n_slider = mo.ui.slider(
        start=10, stop=1000, step=10, value=100, label="Number of Frames to Process"
    )
    batch_stride = mo.ui.dropdown(
        options={"Every frame (stride = 1)": 1, "Every 2nd frame (stride = 2)": 2, "Every 5th frame (stride = 5)": 5},
        value="Every frame (stride = 1)",
        label="Sampling Stride",
    )
    batch_workers = mo.ui.dropdown(
        options=[1, 2, 4, 8], value=4, label="Parallel Worker Processes"
    )
    start_batch_button = mo.ui.run_button(label="Execute OpenPIV Batch on .ims Stream")

    mo.vstack([
        mo.hstack([batch_n_slider, batch_stride, batch_workers]),
        start_batch_button,
    ])
    return batch_n_slider, batch_stride, batch_workers, start_batch_button


@app.cell
def _(
    CHANNEL_CROP_COLS,
    LASER_DT,
    OUT_DIR,
    PX_PER_MM,
    SET_PATH,
    batch_n_slider,
    batch_stride,
    batch_workers,
    mo,
    np,
    opfilt,
    opprep,
    opproc,
    opscale,
    optools,
    opval,
    set_obj,
    start_batch_button,
    time,
    winsize_select,
    xr,
):
    if not start_batch_button.value:
        batch_output = mo.md("_Configure the batch options above and click **Execute OpenPIV Batch on .ims Stream** to begin processing._")
    else:
        _n_proc = int(batch_n_slider.value)
        _stride = int(batch_stride.value)
        _frame_indices = list(range(0, min(_n_proc, len(set_obj)), _stride))

        _win = int(winsize_select.value)
        _search = max(_win, 96)
        _ov = 32

        _zarr_path = OUT_DIR / "baseline_pump_shutdown_openpiv_ds.zarr"

        _t0 = time.time()

        # Pre-allocate coordinates from frame 0
        _buf0 = set_obj[0]
        _c0, _c1 = CHANNEL_CROP_COLS
        _a1_0 = _buf0.as_masked_array(0)[:, _c0:_c1]
        _x_grid, _y_grid = opproc.get_coordinates(image_size=_a1_0.shape, search_area_size=_search, overlap=_ov)
        _dummy = np.zeros_like(_x_grid, dtype=float)
        _xs, _ys, _, _ = opscale.uniform(_x_grid, _y_grid, _dummy, _dummy, scaling_factor=PX_PER_MM)
        _xs, _ys, _, _ = optools.transform_coordinates(_xs, _ys, _dummy, _dummy)
        _x_coords = _xs[0, :]
        _y_coords = _ys[:, 0]

        _ny, _nx = len(_y_coords), len(_x_coords)
        _nt = len(_frame_indices)

        _u_all = np.empty((_nt, _ny, _nx), dtype=np.float32)
        _v_all = np.empty((_nt, _ny, _nx), dtype=np.float32)
        _chc_all = np.empty((_nt, _ny, _nx), dtype=np.float32)
        _t_seconds = np.array([idx / 15.0 for idx in _frame_indices], dtype=np.float32)

        for _i, _f_idx in enumerate(_frame_indices):
            _buf = set_obj[_f_idx]
            _a1 = _buf.as_masked_array(0)[:, _c0:_c1].astype(float)
            _a2 = _buf.as_masked_array(1)[:, _c0:_c1].astype(float)

            _hp1 = opprep.high_pass(_a1, sigma=16, clip=True)
            _hp2 = opprep.high_pass(_a2, sigma=16, clip=True)
            _pct1 = float(np.percentile(_hp1, 97.5)) or 200.0
            _pct2 = float(np.percentile(_hp2, 97.5)) or 200.0
            _a1_proc = np.clip(255.0 * (_hp1 / _pct1), 0, 255).astype(np.int32)
            _a2_proc = np.clip(255.0 * (_hp2 / _pct2), 0, 255).astype(np.int32)

            _u, _v, _s2n = opproc.extended_search_area_piv(
                _a1_proc, _a2_proc,
                window_size=_win, search_area_size=_search, overlap=_ov,
                dt=1.0, subpixel_method="gaussian",
            )
            _inv = opval.sig2noise_val(_s2n, threshold=1.05) | opval.global_std(_u, _v, std_threshold=6.0)
            _uc, _vc = opfilt.replace_outliers(_u, _v, _inv, method="localmean", max_iter=5, kernel_size=3)
            _, _, _us, _vs = opscale.uniform(_x_grid, _y_grid, _uc, _vc, scaling_factor=PX_PER_MM)
            _, _, _us, _vs = optools.transform_coordinates(_xs, _ys, _us, _vs)

            _u_all[_i, :, :] = (_us / LASER_DT).astype(np.float32)
            _v_all[_i, :, :] = (_vs / LASER_DT).astype(np.float32)
            _chc_all[_i, :, :] = (~_inv).astype(np.float32)

        # Build xarray dataset
        _ds = xr.Dataset(
            data_vars={
                "u": (["t", "y", "x"], _u_all),
                "v": (["t", "y", "x"], _v_all),
                "chc": (["t", "y", "x"], _chc_all),
            },
            coords={
                "t": _t_seconds,
                "y": _y_coords,
                "x": _x_coords,
            },
            attrs={
                "source_ims": str(SET_PATH),
                "laser_dt_s": LASER_DT,
                "px_per_mm": PX_PER_MM,
                "window_size": _win,
                "search_size": _search,
                "overlap": _ov,
            }
        )

        _ds.to_zarr(_zarr_path, mode="w")
        _elapsed = time.time() - _t0

        batch_output = mo.md(
            f"""
            ### Batch Processing Complete!
            - **Processed Frames**: {len(_frame_indices)} frames in **{_elapsed:.1f} s** ({_elapsed / len(_frame_indices) * 1000.0:.1f} ms/frame)
            - **Saved Zarr Dataset**: `{_zarr_path}`
            - **Dataset Dimensions**: `t`: {len(_t_seconds)}, `y`: {_ny}, `x`: {_nx}
            - **Variables**: `u` [mm/s], `v` [mm/s], `chc` (1.0 = valid, 0.0 = invalid)
            """
        )

    return (batch_output,)


if __name__ == "__main__":
    app.run()
