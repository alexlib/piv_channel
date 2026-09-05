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

    from piv_pipeline import load_vc7_directory

    OUT_DIR = Path("outputs")
    return OUT_DIR, Path, load_vc7_directory, mo, np, plt, xr


@app.cell
def _(mo):
    mo.md(
        """
        # Unsteady - after pump shutdown - postprocessing

        Scope: `baseline_channel/Vmax_after_pump_shutdown`, DaVis-only (100
        `.vc7` frames). This run's raw acquisition is `.ims` format, which
        `lvpyio` cannot read (`RuntimeError: .ims isn't an allowed set
        extension`) - independent openpiv reprocessing is blocked, so
        DaVis's own vc7 vectors are the only source. `dt` comes from this
        run's own `Settings_Acquisition_Timing_*.xml` (no raw `.im7` pair
        frames are exported here to read `DevDataTrace5` from directly,
        unlike the other DaVis-only runs).

        This is a transient (not steady-state) run - Step 2 below looks at
        the time evolution frame-by-frame rather than time-averaged
        statistics.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Step 1 - load and store")
    return


@app.cell
def _(OUT_DIR, load_vc7_directory, xr):
    PUMP_SHUTDOWN_VC7_FOLDER = (
        r"D:\channel_flow_research\baseline_channel\Vmax_after_pump_shutdown"
        r"\PIV_MPd(4x16x16_25%ov_ImgCorr)"
    )
    PUMP_SHUTDOWN_DT_S = 8.0e-5  # Settings_Acquisition_Timing_*.xml, this run's own folder

    ds_path = OUT_DIR / "baseline_pump_shutdown_ds.zarr"
    if ds_path.exists():
        ds = xr.open_zarr(ds_path)
    else:
        ds = load_vc7_directory(PUMP_SHUTDOWN_VC7_FOLDER)
        ds["u"] = ds["u"] * 1000.0
        ds["v"] = ds["v"] * 1000.0
        ds.attrs.update(
            units_x="mm", units_y="mm", units_u="mm/s", units_v="mm/s",
            dt=PUMP_SHUTDOWN_DT_S,
            history="load_vc7_directory(Vmax_after_pump_shutdown)",
        )
        ds.to_zarr(ds_path, mode="w")
    ds
    return (ds,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 2 - transient evolution

        Mean speed over the field vs. frame index (time), and a
        frame-by-frame flow-field viewer.
        """
    )
    return


@app.cell
def _(ds, np, plt):
    _speed = np.hypot(ds.u, ds.v).where(ds.chc > 0.5)
    _mean_speed_t = _speed.mean(dim=("y", "x"), skipna=True)
    _time_s = ds.t.values * ds.attrs["dt"]

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(_time_s, _mean_speed_t)
    _ax.set_xlabel("time [s] (frame index * dt)")
    _ax.set_ylabel("mean speed [mm/s]")
    _ax.set_title("Decay of mean flow speed after pump shutdown")
    _fig.gca()
    return


@app.cell
def _(ds, mo):
    frame_slider = mo.ui.slider(0, ds.sizes["t"] - 1, value=0, label="frame")
    frame_slider
    return (frame_slider,)


@app.cell
def _(ds, frame_slider, np, plt):
    _frame = ds.isel(t=frame_slider.value)
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _Q = _ax.quiver(
        _frame.x, _frame.y, _frame.u.where(_frame.chc > 0.5), _frame.v.where(_frame.chc > 0.5),
        np.hypot(_frame.u, _frame.v).where(_frame.chc > 0.5),
        cmap="viridis", angles="xy",
    )
    _fig.colorbar(_Q, ax=_ax, shrink=0.85, label="speed [mm/s]")
    _ax.set_title(f"frame {frame_slider.value} (t={frame_slider.value * ds.attrs['dt']:.4f} s)")
    _ax.set_xlabel("x [mm]"); _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
