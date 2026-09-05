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
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    from piv_pipeline import load_vc7_directory

    OUT_DIR = Path("outputs")
    return OUT_DIR, Path, json, load_vc7_directory, mo, np, plt, xr


@app.cell
def _(mo):
    mo.md(
        """
        # Sinusoidal channel, steady state - postprocessing

        Scope: `channel_04` (zoom-out, parts 1-4) and `channel_04_zoom_in
        /second_half` (zoom-in). Step 1 loads everything into xarray
        Datasets and stores/pools them locally; Step 2 is deeper analysis
        (profiles, mean/turbulent fields).
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Step 1a - zoom-out (our own openpiv reprocessing, ensemble of 4 parts)")
    return


@app.cell
def _(OUT_DIR, json, xr):
    with open(OUT_DIR / "channel04_all_parts" / "channel04_all_parts_run_log.json") as _f:
        _run_log = json.load(_f)

    _part_dss = []
    _t_offset = 0
    for _run in _run_log:
        _ds = xr.open_zarr(_run["ds_path"])
        _part_dss.append(_ds.assign_coords(t=_ds.t + _t_offset))
        _t_offset += _ds.sizes["t"]

    zoomout_ds = xr.concat(_part_dss, dim="t")
    zoomout_ds
    return (zoomout_ds,)


@app.cell
def _(zoomout_ds):
    zoomout_stats = zoomout_ds.piv.reynolds_decomposition()
    zoomout_stats
    return (zoomout_stats,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 1b - zoom-in, second_half (DaVis-only, 100% valid coverage - not reprocessed)

        Same `load_vc7_directory()` helper as the straight-channel notebook:
        loads all 3000 `.vc7` files, flips `v` to our sign convention,
        crops to the valid bounding box, converts DaVis's m/s to mm/s.
        `dt` read from this run's own raw `.im7` metadata (`DevDataTrace5`).
        """
    )
    return


@app.cell
def _(OUT_DIR, load_vc7_directory, xr):
    SECOND_HALF_VC7_FOLDER = (
        r"D:\channel_flow_research\channel_04_zoom_in"
        r"\Project_FlowMaster_260705_110445\second_half\ImgPreproc"
        r"\PIV_MPd(4x24x24_75%ov_ImgCorr)"
    )
    SECOND_HALF_DT_S = 5.0e-5  # from this run's own B0001.im7 DevDataTrace5

    zoomin_ds_path = OUT_DIR / "channel04_second_half_ds.zarr"
    if zoomin_ds_path.exists():
        zoomin_ds = xr.open_zarr(zoomin_ds_path)
    else:
        zoomin_ds = load_vc7_directory(SECOND_HALF_VC7_FOLDER)
        zoomin_ds["u"] = zoomin_ds["u"] * 1000.0
        zoomin_ds["v"] = zoomin_ds["v"] * 1000.0
        zoomin_ds.attrs.update(
            units_x="mm", units_y="mm", units_u="mm/s", units_v="mm/s",
            dt=SECOND_HALF_DT_S,
            history="load_vc7_directory(second_half)",
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
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _Q = _ax.quiver(
        zoomout_stats.x, zoomout_stats.y,
        zoomout_stats.u_mean, zoomout_stats.v_mean,
        zoomout_stats.u_mean**2 + zoomout_stats.v_mean**2,
        cmap="viridis", angles="xy",
    )
    _fig.colorbar(_Q, ax=_ax, shrink=0.85, label="speed$^2$ [mm$^2$/s$^2$]")
    _ax.set_title("Zoom-out (4 parts pooled) - mean flow field")
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
    _ax.set_title("Zoom-in (second_half) - mean flow field")
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
    _fig, _ax = plt.subplots(figsize=(10, 6))
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
    _ax.set_title("Zoom-in (second_half) - Reynolds shear stress")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 4 - phase-based streamwise velocity profiles (zoom-out)

        Reuses `channel04_ensemble_average.py`'s phase convention (the
        wall's own sinusoid model, not re-detected) - see that notebook for
        the interactive figure/export tooling this reuses the same
        `zoomout_stats` for.
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
    _fig, _ax = plt.subplots(figsize=(10, 5))
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
    _ax.set_title("Zoom-in (second_half) - streamwise velocity profiles")
    _ax.legend()
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
