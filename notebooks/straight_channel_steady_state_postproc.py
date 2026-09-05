# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "xarray",
#     "zarr",
#     "matplotlib",
#     "pivpy",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    from piv_pipeline import load_vc7_directory, quiver_on_image

    OUT_DIR = Path("outputs")
    return Path, OUT_DIR, load_vc7_directory, mo, np, plt, quiver_on_image, xr


@app.cell
def _(mo):
    mo.md(
        """
        # Straight channel, steady state - postprocessing

        Scope: `baseline_channel`, both the full-width run and the
        right-boundary-layer zoom-in. Step 1 loads everything into xarray
        Datasets and stores them locally; Step 2 is deeper analysis
        (profiles, mean/turbulent fields).
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Step 1a - zoom-out (our own openpiv reprocessing, already in outputs/)")
    return


@app.cell
def _(OUT_DIR, xr):
    # Own reprocessed run (re-run 2026-09 with the median-threshold/dt fix -
    # the batch predating that fix overstated invalid fraction, see README).
    zoomout_ds = xr.open_zarr(OUT_DIR / "baseline_steady_state_ds.zarr")
    zoomout_stats = xr.open_zarr(OUT_DIR / "baseline_steady_state_stats.zarr")
    zoomout_ds, zoomout_stats
    return zoomout_ds, zoomout_stats


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 1b - zoom-in, right boundary layer (DaVis-only, 100% valid coverage - not reprocessed)

        Loaded via `load_vc7_directory()` (new shared helper): reads every
        `.vc7` in the folder, stacks into one time-indexed Dataset, flips
        `v` to match our sign convention (validated on channel_04's
        DaVis-vs-openpiv comparison), and crops to the outer bounding box of
        any valid data. `dt` is read from this run's own raw `.im7` metadata
        (`DevDataTrace5`), the same method `process_im7_pair` uses - DaVis's
        vc7 itself doesn't carry it (`delta_t` comes back 0).
        """
    )
    return


@app.cell
def _(OUT_DIR, load_vc7_directory):
    RIGHT_BL_VC7_FOLDER = (
        r"D:\channel_flow_research\baseline_channel"
        r"\Vmax_0p62_m2sec_steady_state_right_boundary_layer"
        r"\PIV_MPd(4x16x16_25%ov_ImgCorr)"
    )
    RIGHT_BL_DT_S = 5.0e-5  # from this run's own B0001.im7 DevDataTrace5

    zoomin_ds_path = OUT_DIR / "baseline_right_boundary_layer_ds.zarr"
    return RIGHT_BL_DT_S, RIGHT_BL_VC7_FOLDER, zoomin_ds_path


@app.cell
def _(RIGHT_BL_DT_S, RIGHT_BL_VC7_FOLDER, load_vc7_directory, xr, zoomin_ds_path):
    if zoomin_ds_path.exists():
        zoomin_ds = xr.open_zarr(zoomin_ds_path)
    else:
        zoomin_ds = load_vc7_directory(RIGHT_BL_VC7_FOLDER)
        # DaVis reports u,v in m/s; convert to mm/s to match our own
        # pipeline's units_u/units_v convention (see piv_run_metadata).
        zoomin_ds["u"] = zoomin_ds["u"] * 1000.0
        zoomin_ds["v"] = zoomin_ds["v"] * 1000.0
        zoomin_ds.attrs.update(
            units_x="mm", units_y="mm", units_u="mm/s", units_v="mm/s",
            dt=RIGHT_BL_DT_S,
            history="load_vc7_directory(right_boundary_layer)",
        )
        zoomin_ds.to_zarr(zoomin_ds_path, mode="w")
    zoomin_ds
    return (zoomin_ds,)


@app.cell
def _(zoomin_ds):
    zoomin_stats = zoomin_ds.piv.reynolds_decomposition()
    zoomin_stats
    return (zoomin_stats,)


@app.cell
def _(mo):
    mo.md("## Step 2 - mean flow fields")
    return


@app.cell
def _(plt, zoomout_stats):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _Q = _ax.quiver(
        zoomout_stats.x, zoomout_stats.y,
        zoomout_stats.u_mean, zoomout_stats.v_mean,
        zoomout_stats.u_mean**2 + zoomout_stats.v_mean**2,
        cmap="viridis", angles="xy",
    )
    _fig.colorbar(_Q, ax=_ax, shrink=0.85, label="speed$^2$ [mm$^2$/s$^2$]")
    _ax.set_title("Zoom-out - mean flow field")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(plt, zoomin_stats):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _Q = _ax.quiver(
        zoomin_stats.x, zoomin_stats.y,
        zoomin_stats.u_mean, zoomin_stats.v_mean,
        zoomin_stats.u_mean**2 + zoomin_stats.v_mean**2,
        cmap="viridis", angles="xy",
    )
    _fig.colorbar(_Q, ax=_ax, shrink=0.85, label="speed$^2$ [mm$^2$/s$^2$]")
    _ax.set_title("Zoom-in (right boundary layer) - mean flow field")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(mo):
    mo.md("## Step 3 - Reynolds shear stress")
    return


@app.cell
def _(plt, zoomout_stats):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _cf = _ax.contourf(
        zoomout_stats.x, zoomout_stats.y, zoomout_stats.uv_prime,
        levels=60, cmap="RdBu_r",
    )
    _fig.colorbar(_cf, ax=_ax, shrink=0.85, label="$-\\overline{u'v'}$ [mm$^2$/s$^2$]")
    _ax.set_title("Zoom-out - Reynolds shear stress")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(plt, zoomin_stats):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _cf = _ax.contourf(
        zoomin_stats.x, zoomin_stats.y, zoomin_stats.uv_prime,
        levels=60, cmap="RdBu_r",
    )
    _fig.colorbar(_cf, ax=_ax, shrink=0.85, label="$-\\overline{u'v'}$ [mm$^2$/s$^2$]")
    _ax.set_title("Zoom-in (right boundary layer) - Reynolds shear stress")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 4 - streamwise velocity profiles

        Zoom-out: full-width `v(x)` profile at a few heights `y`. Zoom-in:
        `v(x)` across the right-boundary-layer sub-region at the same
        heights, on its own (DaVis-calibrated) local coordinate frame - the
        two crops are not corner-registered to each other (unlike the
        channel_04 vc7-vs-openpiv comparison), since this notebook doesn't
        need them overlaid on one plot.
        """
    )
    return


@app.cell
def _(mo, zoomout_stats):
    y_choices = [float(v) for v in zoomout_stats.y.values[::max(1, zoomout_stats.sizes["y"] // 6)]]
    y_slider = mo.ui.multiselect(
        options=[f"{v:.2f}" for v in y_choices], value=[f"{y_choices[0]:.2f}"],
        label="zoom-out profile heights y [mm]",
    )
    y_slider
    return (y_slider,)


@app.cell
def _(plt, y_slider, zoomout_stats):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _y in y_slider.value:
        _profile = zoomout_stats.v_mean.sel(y=float(_y), method="nearest")
        _ax.plot(zoomout_stats.x, _profile, label=f"y={_y} mm")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("v (streamwise) [mm/s]")
    _ax.set_title("Zoom-out - streamwise velocity profiles")
    _ax.legend()
    _fig.gca()
    return


@app.cell
def _(mo, zoomin_stats):
    y_choices_zi = [float(v) for v in zoomin_stats.y.values[::max(1, zoomin_stats.sizes["y"] // 6)]]
    y_slider_zi = mo.ui.multiselect(
        options=[f"{v:.2f}" for v in y_choices_zi], value=[f"{y_choices_zi[0]:.2f}"],
        label="zoom-in profile heights y [mm] (local frame)",
    )
    y_slider_zi
    return (y_slider_zi,)


@app.cell
def _(plt, y_slider_zi, zoomin_stats):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _y in y_slider_zi.value:
        _profile = zoomin_stats.v_mean.sel(y=float(_y), method="nearest")
        _ax.plot(zoomin_stats.x, _profile, label=f"y={_y} mm")
    _ax.set_xlabel("x [mm] (local frame)"); _ax.set_ylabel("v (streamwise) [mm/s]")
    _ax.set_title("Zoom-in (right boundary layer) - streamwise velocity profiles")
    _ax.legend()
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
