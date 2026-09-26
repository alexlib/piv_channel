import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import json
    import os
    import time
    from pathlib import Path

    import lvpyio as lv
    import matplotlib.pyplot as plt
    import numpy as np
    import openpiv.preprocess as pp
    import pivpy.io as pivpy_io
    import xarray as xr
    import marimo as mo

    return Path, json, lv, mo, np, os, pivpy_io, plt, pp, time, xr


@app.cell
def _(mo):
    mo.md(r"""
    # Sinusoidal Channel Zoom-In — Interactive Two-Half & Wall Mask Alignment Tool

    Use this interactive tool to align the two camera views (**Top Half**: `first_half_case_1`, **Bottom Half**: `second_half`)
    and match the **curvilinear sinusoidal wall mask**:
    - **Move Bottom Half relative to Top Half**:
      - **Y-Shift (Row Offset)**: Adjust vertical overlap / gap between camera fields.
      - **X-Shift (Col Offset)**: Adjust horizontal alignment between camera fields.
    - **Match Wall Mask**:
      - **Center, Amplitude, Phase, Wavelength, Drift**: Fine-tune the wall boundary against actual laser reflection / dark boundaries.
    - **Inspection Views**:
      - **Overview**: Entire channel field of view (~4.8 mm height).
      - **Seam Zoom**: Close-up inspection of the interface between the two halves to verify particle and streamline continuity.
      - **Wall Boundary Zoom**: Close-up inspection along the wavy wall.
    - **Save Alignment**:
      - Click the button to export the tuned parameters directly to `channel04_zoom_in_alignment_params.json` for batch processing!
    """)
    return


@app.cell
def _():
    # File locations
    FIRST_HALF_VC7 = "D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/first_half_case_1/ImgPreproc_03/PIV_MPd(4x24x24_75%ov_ImgCorr)/B00001.vc7"
    SECOND_HALF_VC7 = "D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/second_half/ImgPreproc/PIV_MPd(4x24x24_75%ov_ImgCorr)/B00001.vc7"

    FIRST_HALF_IM7 = r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\first_half_case_1\exported_images\first_half_case_1\B0001.im7"
    SECOND_HALF_IM7 = r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\second_half\exported_images\second_half\B0001.im7"

    WALL_MASK_JSON = "notebooks/channel04_zoom_in_second_half_wall_mask.json"
    OUTPUT_PARAMS_JSON = "notebooks/channel04_zoom_in_alignment_params.json"

    PX_PER_MM = 995.2867181463189
    return (
        FIRST_HALF_IM7,
        FIRST_HALF_VC7,
        OUTPUT_PARAMS_JSON,
        PX_PER_MM,
        SECOND_HALF_IM7,
        SECOND_HALF_VC7,
        WALL_MASK_JSON,
    )


@app.cell
def _(
    FIRST_HALF_IM7,
    FIRST_HALF_VC7,
    SECOND_HALF_IM7,
    SECOND_HALF_VC7,
    WALL_MASK_JSON,
    json,
    lv,
    np,
    pivpy_io,
    pp,
):
    # Load base images and vector data once into memory
    print("Loading raw images and vector fields...")

    raw_a = np.asarray(lv.read_buffer(FIRST_HALF_IM7).as_masked_array(0).data)
    raw_b = np.asarray(lv.read_buffer(SECOND_HALF_IM7).as_masked_array(0).data)

    # Normalize contrast with high-pass filter
    def prep_img(frame):
        hp = pp.high_pass(frame.astype(float), sigma=16, clip=True)
        pmax = np.percentile(hp, 99.0)
        hp = np.clip(hp, 0, pmax)
        return ((255.0 / pmax) * hp).astype(np.uint8)

    img_a = prep_img(raw_a)
    img_b = prep_img(raw_b)

    # Load vector fields
    ds_a_raw = pivpy_io.load_vc7(FIRST_HALF_VC7).isel(t=0)
    ds_b_raw = pivpy_io.load_vc7(SECOND_HALF_VC7).isel(t=0)

    # Invert v to match standard cartesian convention
    ds_a_raw["v"] = -ds_a_raw["v"]
    ds_b_raw["v"] = -ds_b_raw["v"]

    # Initial wall mask defaults from JSON
    with open(WALL_MASK_JSON) as _f:
        initial_wall_json = json.load(_f)
    initial_wm = initial_wall_json.get("wall_mask", {})

    print("Data loaded successfully!")
    return ds_a_raw, ds_b_raw, img_a, img_b, initial_wm, prep_img, raw_a, raw_b


@app.cell
def _(initial_wm, mo):
    # Interactive UI Sliders
    shift_y_slider = mo.ui.slider(
        start=2300,
        stop=2460,
        step=1,
        value=2390,
        label="Bottom Half Y-Shift / Row Offset [px]",
    )
    shift_x_slider = mo.ui.slider(
        start=-150,
        stop=150,
        step=1,
        value=80,
        label="Bottom Half X-Shift / Col Offset [px]",
    )

    mask_center_slider = mo.ui.slider(
        start=1100.0,
        stop=1500.0,
        step=0.5,
        value=float(initial_wm.get("right_center_px", 1310.69)),
        label="Wall Center X [px]",
    )
    mask_amp_slider = mo.ui.slider(
        start=450.0,
        stop=650.0,
        step=0.5,
        value=float(initial_wm.get("right_amplitude_px", 555.24)),
        label="Wall Amplitude A [px]",
    )
    mask_phase_slider = mo.ui.slider(
        start=0.0,
        stop=6.28,
        step=0.01,
        value=float(initial_wm.get("phase_rad", 4.747)),
        label="Wall Phase [rad]",
    )
    mask_wavelength_slider = mo.ui.slider(
        start=4600.0,
        stop=5100.0,
        step=1.0,
        value=float(initial_wm.get("wavelength_px", 4862.16)),
        label="Wall Wavelength λ [px]",
    )
    mask_drift_slider = mo.ui.slider(
        start=-0.05,
        stop=0.10,
        step=0.001,
        value=float(initial_wm.get("drift_px_per_row", 0.038)),
        label="Wall Drift / Tilt [px/row]",
    )

    view_mode = mo.ui.dropdown(
        options=[
            "Overview (Full Channel FOV)",
            "Seam Zoom (Half A / Half B Interface)",
            "Wall Crest Zoom (Near-wall apex)",
            "Wall Trough Zoom (Recirculation cavity)",
        ],
        value="Overview (Full Channel FOV)",
        label="Inspection View Mode",
    )

    display_layer = mo.ui.dropdown(
        options=[
            "Particles + Vectors + Wall Mask",
            "Particles + Wall Mask (No Vectors)",
            "Vector Speed Contour + Quiver + Wall Mask",
            "Vector Quiver + Wall Mask",
        ],
        value="Particles + Vectors + Wall Mask",
        label="Display Layer",
    )

    quiver_skip = mo.ui.slider(start=2, stop=16, step=1, value=6, label="Quiver Density (Skip)")
    quiver_scale = mo.ui.slider(start=0.2, stop=3.0, step=0.1, value=1.0, label="Arrow Scale")

    save_btn = mo.ui.run_button(label="Save Calibrated Parameters to JSON")

    return (
        display_layer,
        mask_amp_slider,
        mask_center_slider,
        mask_drift_slider,
        mask_phase_slider,
        mask_wavelength_slider,
        quiver_scale,
        quiver_skip,
        save_btn,
        shift_x_slider,
        shift_y_slider,
        view_mode,
    )


@app.cell
def _(
    display_layer,
    mask_amp_slider,
    mask_center_slider,
    mask_drift_slider,
    mask_phase_slider,
    mask_wavelength_slider,
    mo,
    quiver_scale,
    quiver_skip,
    save_btn,
    shift_x_slider,
    shift_y_slider,
    view_mode,
):
    # Layout the controls in organized tabs/cards
    controls_ui = mo.vstack([
        mo.md("### Calibration Controls"),
        mo.hstack([
            mo.vstack([
                mo.md("#### 1. Halves Relative Alignment (Bottom vs Top)"),
                shift_y_slider,
                shift_x_slider,
                mo.md("#### 2. View & Display Options"),
                view_mode,
                display_layer,
                mo.hstack([quiver_skip, quiver_scale]),
            ]),
            mo.vstack([
                mo.md("#### 3. Wavy Wall Mask Geometry"),
                mask_center_slider,
                mask_amp_slider,
                mask_phase_slider,
                mask_wavelength_slider,
                mask_drift_slider,
                mo.md("#### 4. Save Calibration"),
                save_btn,
            ]),
        ], justify="space-between"),
    ])
    controls_ui
    return (controls_ui,)


@app.cell
def _(
    PX_PER_MM,
    display_layer,
    ds_a_raw,
    ds_b_raw,
    img_a,
    img_b,
    mask_amp_slider,
    mask_center_slider,
    mask_drift_slider,
    mask_phase_slider,
    mask_wavelength_slider,
    np,
    plt,
    quiver_scale,
    quiver_skip,
    shift_x_slider,
    shift_y_slider,
    view_mode,
):
    # Extract current parameters
    row_off = int(shift_y_slider.value)
    col_off = int(shift_x_slider.value)

    w_center = float(mask_center_slider.value)
    w_amp = float(mask_amp_slider.value)
    w_phase = float(mask_phase_slider.value)
    w_lambda = float(mask_wavelength_slider.value)
    w_drift = float(mask_drift_slider.value)

    # 1. Build composite stitched raw background canvas
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape

    total_h = row_off + h_b
    total_w = max(w_a, w_b + max(0, col_off)) + abs(min(0, col_off))

    # Base canvas
    canvas = np.zeros((total_h, total_w), dtype=np.uint8)

    # Offset for a if col_off is negative
    x_off_a = abs(min(0, col_off))
    x_off_b = x_off_a + col_off

    # Paste top half
    canvas[0:h_a, x_off_a : x_off_a + w_a] = img_a

    # Blend / paste bottom half
    overlap_rows = h_a - row_off
    if overlap_rows > 0:
        # Weighted blend in the overlap zone
        alpha = np.linspace(0.0, 1.0, overlap_rows)[:, None]
        zone_a = canvas[row_off:h_a, x_off_b : x_off_b + w_b]
        zone_b = img_b[:overlap_rows, :]
        min_w = min(zone_a.shape[1], zone_b.shape[1])
        canvas[row_off:h_a, x_off_b : x_off_b + min_w] = (
            (1.0 - alpha) * zone_a[:, :min_w] + alpha * zone_b[:, :min_w]
        ).astype(np.uint8)
        # Paste non-overlapping part of bottom half
        canvas[h_a:total_h, x_off_b : x_off_b + w_b] = img_b[overlap_rows:, :]
    else:
        canvas[row_off:total_h, x_off_b : x_off_b + w_b] = img_b

    # 2. Compute wall curve in stitched pixel coordinates
    rows_all = np.arange(total_h)
    wall_cols = (
        w_center
        + w_drift * rows_all
        + w_amp * np.sin(2.0 * np.pi * rows_all / w_lambda + w_phase)
        + x_off_a
    )

    # 3. Vector fields coordinates in stitched pixel space
    # Half A vectors
    y_a_m = ds_a_raw.y.values
    x_a_m = ds_a_raw.x.values
    row_vec_a = (y_a_m.max() - y_a_m) * PX_PER_MM
    col_vec_a = (x_a_m - x_a_m.min()) * PX_PER_MM + x_off_a
    u_a = ds_a_raw["u"].values
    v_a = ds_a_raw["v"].values
    chc_a = ds_a_raw["chc"].values

    # Half B vectors
    y_b_m = ds_b_raw.y.values
    x_b_m = ds_b_raw.x.values
    row_vec_b = (y_b_m.max() - y_b_m) * PX_PER_MM + row_off
    col_vec_b = (x_b_m - x_b_m.min()) * PX_PER_MM + x_off_b
    u_b = ds_b_raw["u"].values
    v_b = ds_b_raw["v"].values
    chc_b = ds_b_raw["chc"].values

    # Mask vectors outside wall
    row_grid_a, col_grid_a = np.meshgrid(row_vec_a, col_vec_a, indexing="ij")
    wall_at_a = (
        w_center
        + w_drift * row_grid_a
        + w_amp * np.sin(2.0 * np.pi * row_grid_a / w_lambda + w_phase)
        + x_off_a
    )
    mask_a = (col_grid_a > wall_at_a) | (chc_a < 0.5)
    u_a_m = np.where(~mask_a, u_a, np.nan)
    v_a_m = np.where(~mask_a, v_a, np.nan)

    row_grid_b, col_grid_b = np.meshgrid(row_vec_b, col_vec_b, indexing="ij")
    wall_at_b = (
        w_center
        + w_drift * row_grid_b
        + w_amp * np.sin(2.0 * np.pi * row_grid_b / w_lambda + w_phase)
        + x_off_a
    )
    mask_b = (col_grid_b > wall_at_b) | (chc_b < 0.5)
    u_b_m = np.where(~mask_b, u_b, np.nan)
    v_b_m = np.where(~mask_b, v_b, np.nan)

    # Convert coordinates to mm (x right, y up from bottom or y down from top)
    # We display directly in mm: x_mm = col / PX_PER_MM, y_mm = -row / PX_PER_MM
    x_canvas_mm = np.arange(total_w) / PX_PER_MM
    y_canvas_mm = -rows_all / PX_PER_MM
    wall_x_mm = wall_cols / PX_PER_MM

    col_a_mm = col_vec_a / PX_PER_MM
    row_a_mm = -row_vec_a / PX_PER_MM
    col_b_mm = col_vec_b / PX_PER_MM
    row_b_mm = -row_vec_b / PX_PER_MM

    # Determine plot bounds based on View Mode
    seam_row = row_off
    seam_y_mm = -seam_row / PX_PER_MM

    if "Overview" in view_mode.value:
        fig_h = 13
        x_lims = (0, 2.2)
        y_lims = (y_canvas_mm[-1], y_canvas_mm[0])
    elif "Seam Zoom" in view_mode.value:
        fig_h = 8
        x_lims = (0.0, 2.0)
        y_lims = (seam_y_mm - 0.45, seam_y_mm + 0.45)
    elif "Crest Zoom" in view_mode.value:
        fig_h = 8
        # Crest is where wall reaches minimum x (most into channel)
        crest_idx = np.argmin(wall_cols[rows_all < 2400])
        crest_y = y_canvas_mm[crest_idx]
        x_lims = (0.4, 1.8)
        y_lims = (crest_y - 0.4, crest_y + 0.4)
    else:  # Trough Zoom
        fig_h = 8
        trough_idx = np.argmax(wall_cols)
        trough_y = y_canvas_mm[trough_idx]
        x_lims = (0.6, 2.2)
        y_lims = (trough_y - 0.5, trough_y + 0.5)

    fig, ax = plt.subplots(figsize=(8, fig_h))

    # Render background
    if "Particles" in display_layer.value:
        # Display particle image extent in mm
        extent = (x_canvas_mm[0], x_canvas_mm[-1], y_canvas_mm[-1], y_canvas_mm[0])
        ax.imshow(canvas, cmap="gray", extent=extent, aspect="equal", origin="upper")

    if "Speed Contour" in display_layer.value:
        # Plot speed contour
        spd_a = np.hypot(u_a_m, v_a_m)
        spd_b = np.hypot(u_b_m, v_b_m)
        ax.contourf(col_a_mm, row_a_mm, spd_a, levels=25, cmap="viridis", alpha=0.85)
        ax.contourf(col_b_mm, row_b_mm, spd_b, levels=25, cmap="viridis", alpha=0.85)

    # Render quivers
    if "Vector" in display_layer.value or "Quiver" in display_layer.value:
        sk = int(quiver_skip.value)
        sc = float(quiver_scale.value)
        q_scale = 18.0 / sc

        ax.quiver(
            col_a_mm[::sk],
            row_a_mm[::sk],
            u_a_m[::sk, ::sk],
            v_a_m[::sk, ::sk],
            color="cyan",
            scale=q_scale,
            width=0.003,
            label="Top Half (Case 1)",
        )
        ax.quiver(
            col_b_mm[::sk],
            row_b_mm[::sk],
            u_b_m[::sk, ::sk],
            v_b_m[::sk, ::sk],
            color="yellow",
            scale=q_scale,
            width=0.003,
            label="Bottom Half",
        )

    # Overlay Wall Mask Curve
    ax.plot(wall_x_mm, y_canvas_mm, "r-", linewidth=2.5, label="Calibrated Wall Mask")

    # Mark Seam interface
    ax.axhline(seam_y_mm, color="magenta", linestyle="--", linewidth=1.5, label=f"Seam Interface (y = {seam_y_mm:.3f} mm)")

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel("x [mm] (Cross-stream)")
    ax.set_ylabel("y [mm] (Streamwise)")
    ax.set_title(
        f"Alignment: Bottom Half Shift = ({col_off:+d} px X, {row_off} px Y) | "
        f"Mask Center = {w_center:.1f} px, Amp = {w_amp:.1f} px, Phase = {w_phase:.2f} rad",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig
    return (
        alpha,
        ax,
        canvas,
        chc_a,
        chc_b,
        col_a_mm,
        col_b_mm,
        col_grid_a,
        col_grid_b,
        col_off,
        col_vec_a,
        col_vec_b,
        crest_idx,
        crest_y,
        extent,
        fig,
        fig_h,
        h_a,
        h_b,
        mask_a,
        mask_b,
        min_w,
        overlap_rows,
        q_scale,
        row_a_mm,
        row_b_mm,
        row_grid_a,
        row_grid_b,
        row_off,
        row_vec_a,
        row_vec_b,
        rows_all,
        sc,
        seam_row,
        seam_y_mm,
        sk,
        spd_a,
        spd_b,
        total_h,
        total_w,
        trough_idx,
        trough_y,
        u_a,
        u_a_m,
        u_b,
        u_b_m,
        v_a,
        v_a_m,
        v_b,
        v_b_m,
        w_a,
        w_amp,
        w_b,
        w_center,
        w_drift,
        w_lambda,
        w_phase,
        wall_at_a,
        wall_at_b,
        wall_cols,
        wall_x_mm,
        x_canvas_mm,
        x_lims,
        x_off_a,
        x_off_b,
        y_a_m,
        y_b_m,
        y_canvas_mm,
        y_lims,
        zone_a,
        zone_b,
    )


@app.cell
def _(
    OUTPUT_PARAMS_JSON,
    PX_PER_MM,
    col_off,
    json,
    mo,
    row_off,
    save_btn,
    time,
    w_amp,
    w_center,
    w_drift,
    w_lambda,
    w_phase,
):
    current_params = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "px_per_mm": PX_PER_MM,
        "bottom_half_alignment": {
            "row_offset_px": row_off,
            "col_offset_px": col_off,
            "col_offset_mm": col_off / PX_PER_MM,
        },
        "wall_mask": {
            "method": "rotated_sinusoidal_wall_bounds",
            "right_center_px": w_center,
            "right_amplitude_px": w_amp,
            "phase_rad": w_phase,
            "wavelength_px": w_lambda,
            "drift_px_per_row": w_drift,
            "right_amplitude_mm": w_amp / PX_PER_MM,
            "wavelength_mm": w_lambda / PX_PER_MM,
        },
    }

    save_status = ""
    if save_btn.value:
        with open(OUTPUT_PARAMS_JSON, "w") as _out:
            json.dump(current_params, _out, indent=2)
        save_status = f"✅ **Successfully saved parameters to `{OUTPUT_PARAMS_JSON}`!**"

    mo.vstack([
        mo.md(save_status if save_status else "_Click 'Save Calibrated Parameters' above when alignment looks optimal._"),
        mo.md(f"""
        ### Active Calibration Parameters
        ```json
        {json.dumps(current_params, indent=2)}
        ```
        """),
    ])
    return current_params, save_status


if __name__ == "__main__":
    app.run()
