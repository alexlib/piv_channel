import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# DaVis .vc7 vs. our own openpiv analysis - one pair\n"
        "Compares DaVis's own pre-analyzed vector field "
        "(`ImgPreproc/PIV_MPd(4x16x16_25%ov_ImgCorr)/B00001.vc7`, multi-pass "
        "deformation down to a 16x16 final window, 25% overlap) against our "
        "own single-pass pipeline (`process_im7_pair()`, tuned in "
        "`channel04_steady_state_pair1.py` / `channel04_pair1_config.json`) "
        "run on the exact same raw pair (`B0001.im7`).\n\n"
        "Loaded via `pivpy.io.load_vc7()` - straight into the same "
        "pivpy-shaped `u`/`v`/`chc` convention this project already uses, "
        "so both sides are directly comparable as xarray Datasets.\n\n"
        "**Two things to resolve together, surfaced rather than silently "
        "assumed:** DaVis's `v` came out positive (~0.24 m/s mean) where "
        "our own convention has downward flow as *negative*; and DaVis's "
        "x-coordinates are centered near 0 (its own calibration origin) "
        "while ours run 0 to ~11.5 mm (image-column origin) - these need "
        "reconciling before any pixel-for-pixel overlay/diff means anything."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)
    from pivpy import io as pivpy_io

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import process_im7_pair, sinusoidal_wall_bounds, load_run_config, quiver_on_image

    return (
        ROOT,
        load_run_config,
        mo,
        np,
        pivpy_io,
        plt,
        process_im7_pair,
        sinusoidal_wall_bounds,
        xr,
    )


@app.cell
def _(mo):
    IM7_PATH = (
        r"D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\exported_images"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\B0001.im7"
    )
    VC7_PATH = (
        r"D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\ImgPreproc"
        r"\PIV_MPd(4x16x16_25%ov_ImgCorr)\B00001.vc7"
    )
    mo.md(f"`IM7_PATH = {IM7_PATH}`  \n`VC7_PATH = {VC7_PATH}`")
    return IM7_PATH, VC7_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - load DaVis's own result
    """)
    return


@app.cell
def _(VC7_PATH, mo, pivpy_io):
    davis_ds = pivpy_io.load_vc7(VC7_PATH).isel(t=0)
    mo.md(
        f"grid: {davis_ds.sizes['y']} x {davis_ds.sizes['x']}  \n"
        f"x range: {float(davis_ds.x.min()):.2f} to {float(davis_ds.x.max()):.2f} mm  \n"
        f"y range: {float(davis_ds.y.min()):.2f} to {float(davis_ds.y.max()):.2f} mm  \n"
        f"u mean/min/max: {float(davis_ds.u.mean()):.4f} / {float(davis_ds.u.min()):.4f} / {float(davis_ds.u.max()):.4f} m/s  \n"
        f"v mean/min/max: {float(davis_ds.v.mean()):.4f} / {float(davis_ds.v.min()):.4f} / {float(davis_ds.v.max()):.4f} m/s  \n"
        f"chc valid fraction: {float(davis_ds.chc.mean()):.1%}"
    )
    return (davis_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - run our own pipeline on the same raw pair
    """)
    return


@app.cell
def _(IM7_PATH, ROOT, load_run_config, sinusoidal_wall_bounds):
    import lvpyio as lv
    import numpy as _np

    config = load_run_config(ROOT / "notebooks" / "channel04_pair1_config.json")
    _a1_raw = _np.asarray(lv.read_buffer(IM7_PATH).as_masked_array(0).data)
    _wm = config["wall_mask"]
    wall_bounds = sinusoidal_wall_bounds(
        _a1_raw.shape[0],
        wavelength_px=_wm["wavelength_px"], phase=_wm["phase_rad"],
        left_center=_wm["left_center_px"], left_amplitude=_wm["left_amplitude_px"],
        right_center=_wm["right_center_px"], right_amplitude=_wm["right_amplitude_px"],
    )
    return config, wall_bounds


@app.cell
def _(IM7_PATH, config, process_im7_pair, wall_bounds):
    our_result = process_im7_pair(
        IM7_PATH, crop_cols=None, wall_bounds=wall_bounds,
        mask_input=config["piv"]["mask_input"], preprocess="high_pass",
        hp_sigma=config["preprocessing"]["high_pass_sigma"],
        hp_pct=config["preprocessing"]["contrast_stretch_percentile"],
        winsize=config["piv"]["winsize"], searchsize=config["piv"]["searchsize"],
        overlap=config["piv"]["overlap"], s2n_threshold=config["piv"]["s2n_threshold"],
        median_threshold=config["piv"]["median_threshold_px_per_frame"],
        scaling_factor=config["px_per_mm"],
    )
    return (our_result,)


@app.cell
def _(mo, np, our_result, xr):
    our_ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), our_result["u"] / 1000.0),  # mm/s -> m/s, matching davis_ds's units
            "v": (("y", "x"), our_result["v"] / 1000.0),
            "chc": (("y", "x"), (~our_result["invalid"]).astype(float)),
        },
        coords={"x": our_result["x"][0, :], "y": our_result["y"][:, 0]},
    )
    mo.md(
        f"grid: {our_ds.sizes['y']} x {our_ds.sizes['x']}  \n"
        f"x range: {float(our_ds.x.min()):.2f} to {float(our_ds.x.max()):.2f} mm  \n"
        f"y range: {float(our_ds.y.min()):.2f} to {float(our_ds.y.max()):.2f} mm  \n"
        f"u mean/min/max: {float(our_ds.u.mean()):.4f} / {float(np.nanmin(our_ds.u)):.4f} / {float(np.nanmax(our_ds.u)):.4f} m/s  \n"
        f"v mean/min/max: {float(our_ds.v.mean()):.4f} / {float(np.nanmin(our_ds.v)):.4f} / {float(np.nanmax(our_ds.v)):.4f} m/s  \n"
        f"chc valid fraction: {float(our_ds.chc.mean()):.1%}"
    )
    return (our_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3 - side by side, each in its own native coordinates
    """)
    return


@app.cell
def _(davis_ds, np, our_ds, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 8))

    _dspeed = np.hypot(davis_ds.u.values, davis_ds.v.values)
    _q0 = _axes[0].quiver(
        davis_ds.x.values, davis_ds.y.values, davis_ds.u.values, davis_ds.v.values,
        _dspeed, cmap="viridis", angles="xy",
    )
    _axes[0].set_title(f"DaVis MPd(16x16, 25%ov) - {davis_ds.sizes['y']}x{davis_ds.sizes['x']}")
    _axes[0].set_xlabel("x [mm] (DaVis calibration origin)")
    _axes[0].set_ylabel("y [mm]")
    _fig.colorbar(_q0, ax=_axes[0], shrink=0.8, label="speed [m/s]")

    _ospeed = np.hypot(our_ds.u.values, our_ds.v.values)
    _q1 = _axes[1].quiver(
        our_ds.x.values, our_ds.y.values, our_ds.u.values, our_ds.v.values,
        _ospeed, cmap="viridis", angles="xy",
    )
    _axes[1].set_title(f"our single-pass (64px) - {our_ds.sizes['y']}x{our_ds.sizes['x']}")
    _axes[1].set_xlabel("x [mm] (image-column origin)")
    _axes[1].set_ylabel("y [mm]")
    _fig.colorbar(_q1, ax=_axes[1], shrink=0.8, label="speed [m/s]")

    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "**Open questions for us to resolve together, not yet answered "
        "here:**\n\n"
        "1. **Sign convention** - is DaVis's positive-`v`-downward the "
        "opposite of our `tools.transform_coordinates()`-derived "
        "negative-`v`-downward, or is something else going on (e.g. a flow "
        "direction difference)? Check by comparing the *sign pattern* "
        "against the known flow direction (top to bottom), not just the "
        "mean.\n"
        "2. **Coordinate origin** - DaVis's `x` is centered on its own "
        "calibration target origin; ours starts at the raw image's column "
        "0. Need the actual pixel offset between the two (from "
        "`Calibration.xml` or by matching a recognizable feature - e.g. "
        "the wavy wall position - in both) before any per-point diff means "
        "anything.\n"
        "3. Once 1-2 are resolved, interpolate our coarser grid onto "
        "DaVis's finer one (or vice versa) with `xr.interp()` for an actual "
        "residual/difference map."
    )
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 4 - align by corner-shift, sign-correct, and interpolate onto a common grid
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "Neither dataset's coordinate origin is a real physical reference (both "
        "are just wherever each pipeline's own crop/transform happened to start), "
        "so an exact calibration-based alignment isn't available here. As a "
        "working hypothesis: shift DaVis's (already-cropped) lower-left corner "
        "to (0, 0), the same convention `our_ds` already uses.\n\n"
        "**Sign check, not assumed**: at the channel center (the easiest, "
        "highest-magnitude point to compare), DaVis's `v` = +0.67 m/s and ours "
        "= -0.65 m/s - nearly equal magnitude, opposite sign. DaVis's own y-axis "
        "scale has a *negative* slope (row increases downward, y_mm decreases "
        "downward), which suggests `V0` was reported in the raw row-increasing-"
        "downward sense without being re-signed for that flipped axis - exactly "
        "what `openpiv.tools.transform_coordinates()` does for us automatically. "
        "So: flip DaVis's `v` sign to match, don't touch `u`.\n\n"
        "Extents are close but not identical (DaVis 11.2x12.5 mm vs ours "
        "11.4x13.6 mm) - expected, since the two pipelines crop differently, not "
        "evidence the corner-shift is wrong on its own."
    )
    mo.md(_text)

    return


@app.cell
def _(davis_ds, mo, our_ds):
    # Anchor DaVis's cropped lower-left corner at (0, 0), matching our_ds's
    # own origin convention, and flip v to match our sign convention (see note
    # above - not touching u, which agreed in sign already).
    davis_aligned = davis_ds.assign_coords(
        x=davis_ds.x - davis_ds.x.min(),
        y=davis_ds.y - davis_ds.y.min(),
    )
    davis_aligned["v"] = -davis_aligned["v"]

    # Interpolate DaVis's finer grid onto our coarser one (downsampling, not up-)
    # so the comparison happens on real openpiv output points, not invented ones.
    davis_on_our_grid = davis_aligned.interp(x=our_ds.x, y=our_ds.y)

    _both_valid = (davis_on_our_grid["chc"] > 0.5) & (our_ds["chc"] > 0.5)
    _n_both_valid = int(_both_valid.sum())
    mo.md(
        f"Interpolated DaVis onto our {our_ds.sizes['y']}x{our_ds.sizes['x']} "
        f"grid. {_n_both_valid} / {_both_valid.size} points are valid in *both* "
        f"datasets at once - only those are used for the residuals below."
    )

    return (davis_on_our_grid,)


@app.cell
def _(davis_on_our_grid, mo, our_ds):
    u_diff = (davis_on_our_grid["u"] - our_ds["u"]).where(_both_valid)
    v_diff = (davis_on_our_grid["v"] - our_ds["v"]).where(_both_valid)

    _line1 = f"u residual (DaVis - ours): mean={float(u_diff.mean()):.4f}, std={float(u_diff.std()):.4f}, RMS={float((u_diff**2).mean()**0.5):.4f} m/s"
    _line2 = f"v residual (DaVis - ours): mean={float(v_diff.mean()):.4f}, std={float(v_diff.std()):.4f}, RMS={float((v_diff**2).mean()**0.5):.4f} m/s"
    mo.md(_line1 + "  \n" + _line2)

    return u_diff, v_diff


@app.cell
def _(np, our_ds, plt, u_diff, v_diff):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 7))
    _vmax_u = float(np.nanpercentile(np.abs(u_diff), 95))
    _vmax_v = float(np.nanpercentile(np.abs(v_diff), 95))

    _im0 = _axes[0].pcolormesh(our_ds.x, our_ds.y, u_diff, cmap="RdBu_r", vmin=-_vmax_u, vmax=_vmax_u, shading="auto")
    _axes[0].set_title("u residual: DaVis - ours [m/s]")
    _axes[0].set_xlabel("x [mm] (corner-shift aligned)")
    _axes[0].set_ylabel("y [mm]")
    _fig.colorbar(_im0, ax=_axes[0], shrink=0.8)

    _im1 = _axes[1].pcolormesh(our_ds.x, our_ds.y, v_diff, cmap="RdBu_r", vmin=-_vmax_v, vmax=_vmax_v, shading="auto")
    _axes[1].set_title("v residual: DaVis - ours [m/s]")
    _axes[1].set_xlabel("x [mm] (corner-shift aligned)")
    _axes[1].set_ylabel("y [mm]")
    _fig.colorbar(_im1, ax=_axes[1], shrink=0.8)

    _fig.tight_layout()
    _fig.gca()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 5 - overlay quiver: DaVis (red) vs ours (blue), same grid
    """)
    return


@app.cell
def _(both_valid, davis_on_our_grid, np, our_ds, plt):
    _X, _Y = np.meshgrid(our_ds.x.values, our_ds.y.values)

    _fig, _ax = plt.subplots(figsize=(9, 11))
    _q_davis = _ax.quiver(
        _X, _Y, davis_on_our_grid["u"].where(both_valid), davis_on_our_grid["v"].where(both_valid),
        color="red", angles="xy", scale_units="xy", scale=0.15, width=0.003, label="DaVis (MPd, 16x16)",
    )
    _q_ours = _ax.quiver(
        _X, _Y, our_ds["u"].where(both_valid), our_ds["v"].where(both_valid),
        color="blue", angles="xy", scale_units="xy", scale=0.15, width=0.003, label="ours (single-pass, 64px)",
    )
    _ax.set_xlabel("x [mm] (corner-shift aligned)")
    _ax.set_ylabel("y [mm]")
    _ax.set_title("DaVis (red) vs our openpiv (blue) - same grid, both-valid points only")
    _ax.legend(loc="upper right")
    _ax.set_aspect("equal")
    _fig.tight_layout()
    _fig.gca()

    return


if __name__ == "__main__":
    app.run()
