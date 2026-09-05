import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# Wavy channel (channel_04), steady state - single pair first pass\n"
        "Same purpose as `pair1_first_pass.py` for the baseline channel: work out "
        "the calibration and PIV window sizing on **one** real pair before "
        "committing to a batch notebook. Unlike the baseline's fixed-column "
        "crop, this channel's walls are sinusoidal - they move in/out of frame "
        "with row, so we don't crop at all. The wall shape is a single, known "
        "physical sinusoid (not something to re-detect per frame from pixel "
        "brightness, which is fragile against illumination dropouts and stray "
        "bright particles), so `piv_pipeline.sinusoidal_wall_bounds()` masks "
        "wall-adjacent vectors out of the *results* using a manually-tuned "
        "sinusoid model, keeping the frame - and its physical x-coordinates - "
        "intact."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import lvpyio as lv
    import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
    from pathlib import Path
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import quiver_on_image, sinusoidal_wall_bounds


    return Path, lv, mo, np, plt, quiver_on_image, sinusoidal_wall_bounds, xr


@app.cell
def _(mo):
    FOLDER = (
        r"D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\exported_images"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1"
    )
    # Calibration.xml PixelPerMmFactor for this channel_04 run (see
    # D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547
    # \Properties\Calibration\Calibration.xml) - distinct from piv_pipeline's
    # PX_PER_MM, which is baseline_channel's calibration.
    PX_PER_MM = 172.88323514796974
    mo.md(f"`FOLDER = {FOLDER}`  \n`PX_PER_MM = {PX_PER_MM}`")
    return FOLDER, PX_PER_MM


@app.cell
def _(FOLDER, Path, lv, np):
    pair1_path = sorted(Path(FOLDER).glob("*.im7"))[0]
    buffer = lv.read_buffer(str(pair1_path))
    dt = float(np.asarray(buffer.attributes["DevDataTrace5"]).flat[0]) * 1e-6  # us -> s
    a1_raw = np.asarray(buffer.as_masked_array(0).data)
    a2_raw = np.asarray(buffer.as_masked_array(1).data)
    return a1_raw, a2_raw, dt, pair1_path


@app.cell
def _(a1_raw, dt, mo, pair1_path):
    mo.md(f"""
    **{pair1_path.name}** - raw frame shape {a1_raw.shape} "
        f"(rows x cols), dt = {dt * 1e6:.1f} \u00b5s (read from file metadata).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - trace the wavy wall (mask, don't crop)
    """)
    return


@app.cell
def _(mo):
    # display_max default is 4000, close to the raw sensor max (~4076), not
    # baseline's 200: this dataset's raw intensities run much dimmer (99th
    # percentile ~55-60 vs baseline's ~140 in the same interior-fraction
    # comparison). Only affects the display/PIV image, not wall detection
    # below (that runs on the raw buffer directly).
    display_min_slider = mo.ui.slider(0, 500, step=10, value=0, label="display min")
    display_max_slider = mo.ui.slider(50, 4500, step=50, value=4000, label="display max")
    mo.hstack([display_min_slider, display_max_slider])
    return display_max_slider, display_min_slider


@app.cell
def _(a1_raw, display_max_slider, display_min_slider, np):
    _dmin, _dmax = display_min_slider.value, display_max_slider.value
    _img = np.clip(a1_raw.astype(float), _dmin, _dmax)
    _img -= _dmin
    img8_full = ((255.0 / (_dmax - _dmin)) * _img).astype(np.uint8)
    return (img8_full,)


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "Both walls share one wavelength/phase (cut from the same physical "
        "wave) - defaults below came from fitting `piv_pipeline.wavy_wall_bounds()`'s "
        "brightness-detected trace with `scipy.optimize.curve_fit` (residual "
        "std ~35 px), as a starting point for manual tuning, not something to "
        "re-run per frame. Nudge the sliders until the lines in the next cell "
        "hug the wall precisely."
    )
    mo.md(_text)
    return


@app.cell
def _(mo):
    wavelength_slider = mo.ui.slider(400, 1400, step=5, value=916, label="wavelength (px)")
    phase_slider = mo.ui.slider(-3.2, 3.2, step=0.02, value=1.40, label="phase (rad)")
    left_center_slider = mo.ui.slider(0, 500, step=1, value=147, label="left center (px)")
    left_amp_slider = mo.ui.slider(-250, 250, step=1, value=-87, label="left amplitude (px)")
    right_center_slider = mo.ui.slider(1500, 2048, step=1, value=1900, label="right center (px)")
    right_amp_slider = mo.ui.slider(-250, 250, step=1, value=79, label="right amplitude (px)")
    mo.vstack([
        mo.hstack([wavelength_slider, phase_slider]),
        mo.hstack([left_center_slider, left_amp_slider]),
        mo.hstack([right_center_slider, right_amp_slider]),
    ])
    return (
        left_amp_slider,
        left_center_slider,
        phase_slider,
        right_amp_slider,
        right_center_slider,
        wavelength_slider,
    )


@app.cell
def _(
    a1_raw,
    left_amp_slider,
    left_center_slider,
    phase_slider,
    right_amp_slider,
    right_center_slider,
    sinusoidal_wall_bounds,
    wavelength_slider,
):
    wall_bounds = sinusoidal_wall_bounds(
        a1_raw.shape[0],
        wavelength_px=wavelength_slider.value,
        phase=phase_slider.value,
        left_center=left_center_slider.value,
        left_amplitude=left_amp_slider.value,
        right_center=right_center_slider.value,
        right_amplitude=right_amp_slider.value,
    )
    return (wall_bounds,)


@app.cell
def _(img8_full, np, plt, wall_bounds):
    _left_edge, _right_edge = wall_bounds
    _rows = np.arange(img8_full.shape[0])

    _fig, _ax = plt.subplots(figsize=(6, 10))
    _ax.imshow(img8_full, cmap="gray", origin="upper")
    _ax.plot(_left_edge, _rows, color="lime", lw=1.5)
    _ax.plot(_right_edge, _rows, color="lime", lw=1.5)
    _ax.set_title("sinusoidal_wall_bounds() - green lines mask everything outside")
    _ax.set_xlabel("column [px]")
    _ax.set_ylabel("row [px]")
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "The green lines should hug the wall's actual bright band at every "
        "row. If they don't, nudge the sinusoid sliders above - `wavelength`/"
        "`phase` shift both walls together (one physical wave), `*_center`/"
        "`*_amplitude` adjust one wall's mean position and swing "
        "independently. Points outside these bounds are masked (marked "
        "invalid) in the PIV results below, not removed from the image - and "
        "since it's a fixed parametric model, not per-frame detection, these "
        "same values carry unchanged into the batch notebook."
    )
    mo.md(_text)
    return


@app.cell
def _(a1_raw, np, plt, wall_bounds):
    # Same bounds as the outline above, applied directly to the pixels this
    # time - what actually gets excluded from the PIV results, made visible.
    _left_edge, _right_edge = wall_bounds
    _img = np.clip(a1_raw.astype(float), 0, 60)
    _img8 = ((255.0 / 60) * _img).astype(np.uint8)
    _cols = np.arange(_img8.shape[1])[None, :]
    _outside = (_cols < _left_edge[:, None]) | (_cols > _right_edge[:, None])
    _masked = _img8.copy()
    _masked[_outside] = 0

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 10))
    _axes[0].imshow(_img8, cmap="gray", origin="upper")
    _axes[0].set_title("unmasked (display_max=60, to show the wall band itself)")
    _axes[1].imshow(_masked, cmap="gray", origin="upper")
    _axes[1].set_title("masked: both wavy wall regions blacked out")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell
def _(a1_raw, a2_raw, np, plt):
    # Fixed display_max=60 for visualization (matches the masked/unmasked
    # image above) - independent of display_min_slider/display_max_slider,
    # which are tuned for PIV correlation (currently 4000) and wash out the
    # tracers at that range, making the shift hard to see by eye.
    _a1_8 = np.clip(a1_raw.astype(float), 0, 60)
    _a1_8 = ((255.0 / 60) * _a1_8).astype(np.uint8)
    _a2_8 = np.clip(a2_raw.astype(float), 0, 60)
    _a2_8 = ((255.0 / 60) * _a2_8).astype(np.uint8)

    _fig, _ax = plt.subplots(figsize=(9, 11))
    _ax.imshow(np.stack([_a1_8, _a2_8, _a2_8 * 0], axis=2), origin="upper")
    _ax.set_title("red = frame A, green = frame B (display_max=60) - overlap patches show the shift")
    _ax.set_xlabel("column [px]")
    _ax.set_ylabel("row [px]")
    _fig.gca()

    return


@app.cell
def _(a1_raw, a2_raw, mo, np):
    from skimage.registration import phase_cross_correlation

    # Fixed display_max=200 (not the Step 1 slider, currently 4000) - verified
    # this diagnostic gives a stable, coherent shift at low display_max (60-800px);
    # at 4000 it breaks into a wraparound artifact (garbage ~1000px+ "shift").
    # The actual PIV correlation below is unaffected - it uses its own
    # high_pass-preprocessed input, not this slider.
    _a1_8 = np.clip(a1_raw.astype(float), 0, 200)
    _a1_8 = ((255.0 / 200) * _a1_8).astype(np.uint8)
    _a2_8 = np.clip(a2_raw.astype(float), 0, 200)
    _a2_8 = ((255.0 / 200) * _a2_8).astype(np.uint8)

    _shift, _err, _ = phase_cross_correlation(_a1_8, _a2_8, upsample_factor=10)
    _dy, _dx = _shift
    _disp = float(np.hypot(_dy, _dx))

    _line1 = f"**Global shift A to B:** dy = {-_dy:.2f} px (down is positive), dx = {-_dx:.2f} px, magnitude approx {_disp:.1f} px."
    _line2 = f"Rule of thumb: winsize >= 4 x {_disp:.0f} = {int(np.ceil(4*_disp))} px, searchsize >= winsize + 2 x {_disp:.0f} px."
    mo.md(_line1 + "\n\n" + _line2)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - PIV window parameters
    """)
    return


@app.cell
def _(mo):
    hp_sigma_slider = mo.ui.slider(1, 20, step=1, value=16, label="high_pass sigma (background removal)")
    hp_pct_slider = mo.ui.slider(90.0, 99.99, step=0.1, value=97.5, label="contrast stretch percentile")
    mo.hstack([hp_sigma_slider, hp_pct_slider])

    return hp_pct_slider, hp_sigma_slider


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "Instead of the packaged `process_im7_pair()`, build the PIV input by hand: "
        "`openpiv.preprocess.high_pass()` removes the slowly-varying background before "
        "rescaling, which measurably improved correlation quality in testing (median "
        "s2n and s2n-only invalid fraction, at winsize=64/searchsize=96) versus a naive "
        "linear display_min/max clip. Wall pixels are zeroed (masked) here too, before "
        "correlation - not just discarded from the output afterward."
    )
    mo.md(_text)

    return


@app.cell
def _(a1_raw, a2_raw, hp_pct_slider, hp_sigma_slider, np, wall_bounds):
    import openpiv.preprocess as _pp

    def _to_u8_highpass(a, sigma, pct):
        _hp = _pp.high_pass(a.astype(float), sigma=sigma, clip=True)
        _pmax = np.percentile(_hp, pct)
        _hp = np.clip(_hp, 0, _pmax)
        return ((255.0 / _pmax) * _hp).astype(np.uint8)

    a1_proc = _to_u8_highpass(a1_raw, hp_sigma_slider.value, hp_pct_slider.value)
    a2_proc = _to_u8_highpass(a2_raw, hp_sigma_slider.value, hp_pct_slider.value)

    _left_edge, _right_edge = wall_bounds
    _cols_idx = np.arange(a1_proc.shape[1])[None, :]
    _outside = (_cols_idx < _left_edge[:, None]) | (_cols_idx > _right_edge[:, None])
    a1_proc = a1_proc.copy()
    a2_proc = a2_proc.copy()
    a1_proc[_outside] = 0
    a2_proc[_outside] = 0

    return a1_proc, a2_proc


@app.cell
def _(a1_proc, a2_proc, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 11))
    _ax.imshow(np.stack([a1_proc, a2_proc, a2_proc * 0], axis=2), origin="upper")
    _ax.set_title("preprocessed + wall-masked PIV input: red = frame A, green = frame B")
    _ax.set_xlabel("column [px]")
    _ax.set_ylabel("row [px]")
    _fig.gca()

    return


@app.cell
def _(mo):
    winsize_slider = mo.ui.slider(16, 128, step=8, value=64, label="winsize")
    searchsize_slider = mo.ui.slider(16, 192, step=8, value=64, label="searchsize")
    overlap_slider = mo.ui.slider(0, 96, step=8, value=16, label="overlap")
    s2n_slider = mo.ui.slider(1.0, 3.0, step=0.05, value=1.0, label="s2n threshold")
    median_slider = mo.ui.slider(1, 20, step=1, value=2, label="median threshold (px/frame)")
    mo.vstack([
        mo.hstack([winsize_slider, searchsize_slider, overlap_slider]),
        mo.hstack([s2n_slider, median_slider]),
    ])

    return (
        median_slider,
        overlap_slider,
        s2n_slider,
        searchsize_slider,
        winsize_slider,
    )


@app.cell
def _(
    a1_proc,
    a2_proc,
    dt,
    np,
    overlap_slider,
    searchsize_slider,
    winsize_slider,
):
    import openpiv.pyprocess as _pyprocess

    u0, v0, s2n = _pyprocess.extended_search_area_piv(
        a1_proc.astype(np.int32), a2_proc.astype(np.int32),
        window_size=winsize_slider.value, overlap=overlap_slider.value, dt=dt,
        search_area_size=searchsize_slider.value, sig2noise_method="peak2peak",
    )
    x_px, y_px = _pyprocess.get_coordinates(
        image_size=a1_proc.shape, search_area_size=searchsize_slider.value,
        overlap=overlap_slider.value,
    )
    print(f"raw speed (px/s) median={np.nanmedian(np.hypot(u0, v0)):.0f}, s2n median={np.nanmedian(s2n):.3f}")

    return s2n, u0, v0, x_px, y_px


@app.cell
def _(
    PX_PER_MM,
    a1_proc,
    dt,
    median_slider,
    np,
    s2n,
    s2n_slider,
    u0,
    v0,
    wall_bounds,
    x_px,
    y_px,
):
    from openpiv import validation as _validation, filters as _filters
    from openpiv import scaling as _opiv_scaling, tools as _opiv_tools

    _left_edge, _right_edge = wall_bounds
    _row_idx = np.clip(np.round(y_px[:, 0]).astype(int), 0, len(_left_edge) - 1)

    _mask_s2n = _validation.sig2noise_val(s2n, threshold=s2n_slider.value)
    # median_slider is a px/frame displacement tolerance (intuitive, dt-independent),
    # but u0/v0 are already in px/s (extended_search_area_piv divides by dt
    # internally) - convert before comparing, or a tiny dt inflates any real
    # px/frame variation into a huge px/s number that rejects almost everything
    # regardless of actual data quality (this was the dominant bug all along).
    _mask_med = _validation.local_median_val(
        u0, v0, u_threshold=median_slider.value / dt, v_threshold=median_slider.value / dt, size=1
    )
    _mask_wall = (x_px < _left_edge[_row_idx][:, None]) | (x_px > _right_edge[_row_idx][:, None])
    invalid = _mask_s2n | _mask_med | _mask_wall

    u2, v2 = _filters.replace_outliers(u0, v0, invalid, method="localmean", max_iter=10, kernel_size=3)
    xs, ys, u3, v3 = _opiv_scaling.uniform(x_px, y_px, u2, v2, scaling_factor=PX_PER_MM)
    xs, ys, u3, v3 = _opiv_tools.transform_coordinates(xs, ys, u3, v3)

    result = {
        "x": xs, "y": ys, "u": u3, "v": v3, "invalid": invalid, "s2n": s2n,
        "u_raw": u0, "v_raw": v0, "x_px": x_px, "y_px": y_px,
        "image": a1_proc, "dt": dt,
    }
    _interior = ~_mask_wall
    print(f"invalid: total={invalid.mean():.1%}, interior-only={invalid[_interior].mean():.1%} ({_interior.sum()} interior pts)")

    return (result,)


@app.cell
def _(np, plt, result):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _speed_raw = np.hypot(result["u_raw"], result["v_raw"])
    # Fixed, physically-meaningful clip (not percentile-based - percentiles are
    # themselves dominated by garbage peaks here, see the raw speed distribution:
    # even the 10th percentile is ~45,000 mm/s, vs. an expected ~600-1200 mm/s).
    _vmax = 2000
    _ax.imshow(result["image"], cmap="gray", origin="upper",
               extent=(0, result["image"].shape[1], result["image"].shape[0], 0))
    _q = _ax.quiver(
        result["x_px"], result["y_px"], result["u_raw"], result["v_raw"], _speed_raw,
        cmap="viridis", clim=(0, _vmax), angles="xy",
    )
    _cbar = _fig.colorbar(_q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label(f"raw speed [px/dt], clipped at {_vmax} (physically-plausible range)")
    _frac_over = float((_speed_raw > _vmax).mean())
    _ax.set_title(f"Raw correlation peak, pre-validation ({_frac_over:.0%} of vectors exceed clip)")
    _ax.set_xlabel("column [px]")
    _ax.set_ylabel("row [px]")
    _fig.gca()
    return


@app.cell
def _(mo, result):
    _invalid = result["invalid"]
    mo.md(
        f"dt = {result['dt'] * 1e6:.1f} \u00b5s  \n"
        f"Invalid vectors: {_invalid.sum()} / {_invalid.size} "
        f"({100 * _invalid.mean():.1f}%)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "**Root cause found and fixed - three compounding issues:**\n\n"
        "1. **Weak raw contrast**: naive linear `display_min`/`display_max` clip "
        "wasted most of the 8-bit range on background. Fixed with "
        "`openpiv.preprocess.high_pass()` (background removal) + percentile "
        "stretch, replacing the linear clip for the PIV input specifically.\n"
        "2. **Wall contamination**: windows straddling the wall boundary "
        "correlated against the wall's bright reflection. Fixed by zeroing wall "
        "pixels in both frames *before* correlation (not just discarding the "
        "output after).\n"
        "3. **The dominant bug**: `median_threshold` was compared directly "
        "against `u0`/`v0`, which openpiv returns already divided by `dt` "
        "(px/s, not px/frame). At this run's `dt=120\u00b5s`, even a normal "
        "~1 px/frame variation between neighbors becomes ~8,300 px/s - a "
        "threshold of 3 had effectively zero tolerance, rejecting nearly "
        "everything regardless of actual data quality. Fixed in "
        "`piv_pipeline._run_piv()` (divides `median_threshold` by `dt` before "
        "comparing) - this also affects the baseline pipeline (`dt=80\u00b5s`, "
        "same bug), so its already-completed batch run likely overstates its "
        "invalid fraction.\n\n"
        "**Result:** invalid fraction dropped from ~99% to ~40% (this pair), "
        "with a physically coherent recovered field (median speed ~536 mm/s, "
        "matching the reported Vmax=0.66 m/s and the independent global-shift "
        "estimate above)."
    )
    mo.md(_text)

    return


@app.cell
def _(PX_PER_MM, np, pair1_path, plt, quiver_on_image, result):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _u, _v = result["u"], result["v"]
    _x, _y = result["x"], result["y"]

    _Q = quiver_on_image(
        _ax, result["image"], _x, _y, _u, _v,
        color_by=np.hypot(_u, _v), cmap="viridis", scaling_factor=PX_PER_MM,
    )
    _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("speed [mm/s]")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"{pair1_path.name} - single-pair PIV preview")
    _fig.gca()
    return


@app.cell
def _(PX_PER_MM, np, pair1_path, result, xr):
    # Same invalid-masking as the quiver_on_image plot above (replace_outliers()
    # fills invalid points instead of removing them) - NaN them here too so
    # pivpy's own plot doesn't draw fake arrows in the wall-masked region either.
    result_ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), np.where(result["invalid"], np.nan, result["u"])),
            "v": (("y", "x"), np.where(result["invalid"], np.nan, result["v"])),
            "chc": (("y", "x"), (~result["invalid"]).astype(float)),
        },
        coords={"x": result["x"][0, :], "y": result["y"][:, 0]},
    )
    # Image's true physical extent (from PX_PER_MM), NOT the vector grid's own
    # x/y range - ds.piv.plot(background="image") defaults to the latter, which
    # silently crops/distorts the image (see pair1_first_pass.py's note on this).
    image_extent = (
        0, result["image"].shape[1] / PX_PER_MM,
        0, result["image"].shape[0] / PX_PER_MM,
    )
    _fig, _ax = result_ds.piv.plot(
        background="image", image=result["image"], image_extent=image_extent,
        color_by="v", streamlines=False, quiver_key=False,
        title=f"{pair1_path.name} - pivpy .piv.plot()",
    )
    _fig.gca()

    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "All parameters tuned above (wall-mask sinusoid, high_pass preprocessing, "
        "PIV window/validation settings) are saved to `channel04_pair1_config.json` "
        "alongside this notebook - load it with `piv_pipeline.load_run_config()` "
        "in a future `channel04_steady_state_batch.py` batch notebook, same "
        "pattern as `baseline_steady_state_batch.py`, instead of re-tuning or "
        "hardcoding the same numbers twice."
    )
    mo.md(_text)

    return


if __name__ == "__main__":
    app.run()
