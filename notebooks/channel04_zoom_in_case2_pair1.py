import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# channel_04_zoom_in / first_half_case_2 - single pair first pass\n"
        "A boundary-layer **zoom-in** case, not the full two-wall sinusoidal "
        "view - confirmed visually: a single *diagonal* wall band crosses "
        "the frame (upper-left to lower-right as row increases), tracer "
        "particles sit on its left, the right side is empty (outside the "
        "channel/wall material). Calibration and timing differ from the "
        "steady-state cases too: `995.29 px/mm` (much higher magnification), "
        "`dt=50\\u00b5s`.\n\n"
        "Since the wall here is a straight line, not a sinusoid, "
        "`piv_pipeline.sinusoidal_wall_bounds()` doesn't apply - this "
        "notebook builds a simple two-point linear wall model instead, kept "
        "local to this notebook until we know whether other zoom-in cases "
        "need the same treatment (not worth generalizing into "
        "`piv_pipeline.py` from a single case)."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import lvpyio as lv
    import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
    from pathlib import Path
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import quiver_on_image

    return Path, lv, mo, np, plt, quiver_on_image


@app.cell
def _(mo):
    FOLDER = (
        r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445"
        r"\first_half_case_2\exported_images\first_half_case_2"
    )
    # Properties/Calibration/Calibration.xml, this project - distinct from
    # both piv_pipeline.PX_PER_MM (baseline) and channel04_pair1_config.json
    # (channel_04 steady-state).
    PX_PER_MM = 995.28671814631889
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
    mo.md(
        f"**{pair1_path.name}** - raw frame shape {a1_raw.shape} "
        f"(rows x cols), dt = {dt * 1e6:.1f} µs (read from file metadata)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - pick the diagonal wall line (mask, don't crop)
    """)
    return


@app.cell
def _(mo):
    display_min_slider = mo.ui.slider(0, 200, step=5, value=0, label="display min")
    display_max_slider = mo.ui.slider(10, 500, step=5, value=40, label="display max")
    mo.hstack([display_min_slider, display_max_slider])
    return display_max_slider, display_min_slider


@app.cell
def _(a1_raw, display_max_slider, display_min_slider, np):
    _dmin, _dmax = display_min_slider.value, display_max_slider.value
    _img = np.clip(a1_raw.astype(float), _dmin, _dmax)
    _img -= _dmin
    img8_full = ((255.0 / (_dmax - _dmin)) * _img).astype(np.uint8)
    return (img8_full,)


@app.cell
def _(mo):
    col_top_slider = mo.ui.slider(0, 2048, step=5, value=950, label="wall col @ row 0")
    col_bottom_slider = mo.ui.slider(0, 2048, step=5, value=1750, label="wall col @ last row")
    mo.hstack([col_top_slider, col_bottom_slider])
    return col_bottom_slider, col_top_slider


@app.cell
def _(a1_raw, col_bottom_slider, col_top_slider, np):
    def wall_line_bounds(n_rows, col_top, col_bottom):
        """Linear wall model: the flow-side column boundary at each row,
        interpolated between (row=0, col_top) and (row=n_rows-1, col_bottom).
        Tracers sit to the *left* (smaller columns) of this line for this
        case - points with column >= boundary are outside the flow.
        """
        rows = np.arange(n_rows)
        return col_top + (col_bottom - col_top) * (rows / (n_rows - 1))

    wall_col = wall_line_bounds(a1_raw.shape[0], col_top_slider.value, col_bottom_slider.value)
    return (wall_col,)


@app.cell
def _(img8_full, np, plt, wall_col):
    _rows = np.arange(img8_full.shape[0])

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 10))
    _axes[0].imshow(img8_full, cmap="gray", origin="upper")
    _axes[0].plot(wall_col, _rows, color="lime", lw=1.5)
    _axes[0].set_title("wall line - green marks the flow/wall boundary")
    _axes[0].set_xlabel("column [px]")
    _axes[0].set_ylabel("row [px]")

    _cols = np.arange(img8_full.shape[1])[None, :]
    _outside = _cols >= wall_col[:, None]
    _masked = img8_full.copy()
    _masked[_outside] = 0
    _axes[1].imshow(_masked, cmap="gray", origin="upper")
    _axes[1].set_title("masked: wall side blacked out")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "Adjust the two sliders until the green line hugs the wall band's "
        "*inner* edge (where tracers stop) at both the top and bottom of the "
        "frame - the masked panel should show no wall band, only the "
        "seeded flow region."
    )
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - preprocessing and PIV parameters
    """)
    return


@app.cell
def _(mo):
    hp_sigma_slider = mo.ui.slider(1, 20, step=1, value=3, label="high_pass sigma")
    hp_pct_slider = mo.ui.slider(90.0, 99.99, step=0.1, value=99.5, label="contrast stretch percentile")
    mo.hstack([hp_sigma_slider, hp_pct_slider])
    return hp_pct_slider, hp_sigma_slider


@app.cell
def _(a1_raw, a2_raw, hp_pct_slider, hp_sigma_slider, np, wall_col):
    import openpiv.preprocess as _pp

    def _to_u8_highpass(a, sigma, pct):
        _hp = _pp.high_pass(a.astype(float), sigma=sigma, clip=True)
        _pmax = np.percentile(_hp, pct)
        _hp = np.clip(_hp, 0, _pmax)
        return ((255.0 / _pmax) * _hp).astype(np.uint8)

    a1_proc = _to_u8_highpass(a1_raw, hp_sigma_slider.value, hp_pct_slider.value)
    a2_proc = _to_u8_highpass(a2_raw, hp_sigma_slider.value, hp_pct_slider.value)

    _cols_idx = np.arange(a1_proc.shape[1])[None, :]
    _outside = _cols_idx >= wall_col[:, None]
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
    _fig.gca()
    return


@app.cell
def _(mo):
    winsize_slider = mo.ui.slider(16, 128, step=8, value=32, label="winsize")
    searchsize_slider = mo.ui.slider(16, 192, step=8, value=48, label="searchsize")
    overlap_slider = mo.ui.slider(0, 96, step=8, value=16, label="overlap")
    s2n_slider = mo.ui.slider(1.0, 3.0, step=0.05, value=1.1, label="s2n threshold")
    median_slider = mo.ui.slider(1, 20, step=1, value=3, label="median threshold (px/frame)")
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
    dt,
    median_slider,
    np,
    s2n,
    s2n_slider,
    u0,
    v0,
    wall_col,
    x_px,
    y_px,
):
    from openpiv import validation as _validation, filters as _filters
    from openpiv import scaling as _opiv_scaling, tools as _opiv_tools

    _row_idx = np.clip(np.round(y_px[:, 0]).astype(int), 0, len(wall_col) - 1)

    _mask_s2n = _validation.sig2noise_val(s2n, threshold=s2n_slider.value)
    # median_slider is a px/frame displacement tolerance (dt-independent) -
    # u0/v0 are already in px/s (extended_search_area_piv divides by dt
    # internally), so convert before comparing (same fix as
    # channel04_steady_state_pair1.py - without it a tiny dt gives the
    # threshold effectively zero tolerance regardless of data quality).
    _mask_med = _validation.local_median_val(
        u0, v0, u_threshold=median_slider.value / dt, v_threshold=median_slider.value / dt, size=1
    )
    _mask_wall = x_px >= wall_col[_row_idx][:, None]
    invalid = _mask_s2n | _mask_med | _mask_wall

    u2, v2 = _filters.replace_outliers(u0, v0, invalid, method="localmean", max_iter=10, kernel_size=3)
    xs, ys, u3, v3 = _opiv_scaling.uniform(x_px, y_px, u2, v2, scaling_factor=PX_PER_MM)
    xs, ys, u3, v3 = _opiv_tools.transform_coordinates(xs, ys, u3, v3)

    result = {
        "x": xs, "y": ys, "u": u3, "v": v3, "invalid": invalid, "s2n": s2n,
        "u_raw": u0, "v_raw": v0, "x_px": x_px, "y_px": y_px, "dt": dt,
    }
    _interior = ~_mask_wall
    print(f"invalid: total={invalid.mean():.1%}, interior-only={invalid[_interior].mean():.1%} ({_interior.sum()} interior pts)")
    return (result,)


@app.cell
def _(mo, result):
    _invalid = result["invalid"]
    mo.md(
        f"dt = {result['dt'] * 1e6:.1f} \u00b5s  \n"
        f"Invalid vectors: {_invalid.sum()} / {_invalid.size} "
        f"({100 * _invalid.mean():.1f}%)"
    )
    return


@app.cell
def _(PX_PER_MM, a1_proc, np, pair1_path, plt, quiver_on_image, result):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _u = np.where(result["invalid"], np.nan, result["u"])
    _v = np.where(result["invalid"], np.nan, result["v"])
    _x, _y = result["x"], result["y"]

    _Q = quiver_on_image(
        _ax, a1_proc, _x, _y, _u, _v,
        color_by=np.hypot(_u, _v), cmap="viridis", scaling_factor=PX_PER_MM,
    )
    _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("speed [mm/s]")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"{pair1_path.name} - single-pair PIV preview ({result['invalid'].mean():.0%} invalid, hidden)")
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "Once the wall line and PIV parameters above look right (low "
        "invalid fraction, sane speeds, no wall pixels in the flow region), "
        "we'll save the tuned values to a config JSON (same pattern as "
        "`channel04_pair1_config.json`) and build a batch notebook over all "
        "3000 pairs, same as `channel04_steady_state_batch.py`."
    )
    mo.md(_text)
    return


if __name__ == "__main__":
    app.run()
