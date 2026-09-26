# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "xarray",
#     "zarr",
#     "matplotlib",
#     "pivpy",
#     "lvpyio",
# ]
# ///

"""Straight channel, transient (pump shutdown) - time-resolved analysis.

Scope: `baseline_channel` zoom-out (`Vmax_after_pump_shutdown`, 100 DaVis
`.vc7` frames) + zoom-in (no transient acquisition exists yet - probed and
reported in the last section).

Data honesty notes (verified 2026-09-26, see cells below):
- Inter-frame time is 1/15 s from the run's own `.set` recording-rate
  statistics ("Average: 15 Hz"). The 80 us laser-pulse `dt` only scales
  velocity inside each pair - it is NOT the time between vector maps. The
  earlier `unsteady_pump_shutdown_postproc.py` multiplied frame index by
  the pulse `dt`, understating elapsed time by ~800x.
- The 100-frame `Vmax_after_pump_shutdown` series is quasi-steady
  (~500 mm/s, <1% reversed vectors, flat across ~6.6 s) - no slowdown is
  visible in it. The true shutdown transient is almost certainly the
  unprocessed 14 GB `Vmax_0p62_m2sec_pump_shutdown/Camera1-1.ims`
  (1000 images @ 15 Hz, no `.vc7`, `.ims` unreadable by `lvpyio`) - needs
  a DaVis export to `exported_images/*.im7` before this notebook's
  machinery can run on it. This notebook is written so that export only
  requires changing `ZOOMOUT_VC7_FOLDER` / the loader cell.
- No raw background image exists for this run (no `.im7` frames, only
  dark frames), so the frame viewer uses a computed speed background
  with one fixed colormap + one colorbar instead of a raw-image overlay.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Straight channel, transient - time-resolved flow analysis

    Zoom-out first (`Vmax_after_pump_shutdown`, 100 frames @ 15 Hz),
    then zoom-in (probed below - no transient acquisition found yet).
    Each frame gets the same fixed-scale quiver + fixed colorbar so
    frames are comparable through the slowdown.
    """)
    return


@app.cell
def _(mo):
    md_provenance = mo.md(
        """
        ## Data provenance - where this comes from

        - **Run**: `baseline_channel/Vmax_after_pump_shutdown` on
          `D:\\channel_flow_research` - straight (baseline) channel, zoom-out
          field of view, recorded after the pump was shut down.
        - **Format**: 100 DaVis `.vc7` vector maps from
          `PIV_MPd(4x16x16_25%ov_ImgCorr)` (multi-pass deformation down to a
          16x16 final window, 25% overlap, image correction + median
          postprocessing).
        - **Time base**: 15 Hz recording rate (the run's own `.set` file:
          *"Recording rate statistics of 100 images: Average: 15 Hz"*) - 0.0667 s
          between maps, 6.6 s total. The 80 us laser-pulse `dt` only scales
          velocity *inside* each frame pair; it is not the inter-frame time.
        - **Calibration**: `PX_PER_MM = 173.419` from this project's
          `Properties/Calibration/Calibration.xml`.
        - **Raw**: `Camera1-*.ims` (proprietary stream, unreadable by `lvpyio`);
          no exported `.im7` frames exist for this run, so there is no raw-image
          background - viewer backgrounds are computed speed fields.
        - **In-notebook format**: xarray Dataset `(t: 100, y: 119, x: 147)` with
          `u, v` in mm/s, `chc` validity flag and a `time_s` coordinate; stored
          chunked as `outputs/baseline_pump_shutdown_ds.zarr` for fast reload.
        - **Caveats**: only ~50% of vectors are valid (invalids cluster at the
          FOV borders - see the Step 8 map); the 100 frames are quasi-steady
          (~499 mm/s mean, <1% reversal), i.e. no slowdown is visible here. The
          true shutdown transient is the unprocessed 14 GB
          `Vmax_0p62_m2sec_pump_shutdown/Camera1-1.ims`.
        """
    )
    md_provenance
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import pivpy  # noqa: F401 (registers the .piv xarray accessor)

    from piv_pipeline import load_vc7_directory

    ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = ROOT / "outputs"

    ZOOMOUT_VC7_FOLDER = (
        r"D:\channel_flow_research\baseline_channel\Vmax_after_pump_shutdown"
        r"\PIV_MPd(4x16x16_25%ov_ImgCorr)"
    )
    # Pulse separation inside each pair (velocity scaling, already baked
    # into the vc7 m/s values) vs. recording rate between vector maps.
    PULSE_DT_S = 8.0e-5
    FRAME_DT_S = 1.0 / 15.0  # from Vmax_after_pump_shutdown.set
    PX_PER_MM = 173.41900170673784  # baseline_channel Calibration.xml
    return (
        FRAME_DT_S,
        OUT_DIR,
        PULSE_DT_S,
        PX_PER_MM,
        ZOOMOUT_VC7_FOLDER,
        load_vc7_directory,
        mo,
        np,
        plt,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - load zoom-out series into pivpy, store as zarr

    `load_vc7_directory()` stacks every `.vc7` into one `(t, y, x)`
    Dataset (`u, v, chc`), flips `v` to our sign convention, crops to
    the valid bounding box, converts DaVis m/s to mm/s. Stored
    chunked (`t=1`) so later reloads take seconds.
    """)
    return


@app.cell
def _(
    FRAME_DT_S,
    OUT_DIR,
    PULSE_DT_S,
    PX_PER_MM,
    ZOOMOUT_VC7_FOLDER,
    load_vc7_directory,
    mo,
    xr,
):
    ds_path = OUT_DIR / "baseline_pump_shutdown_ds.zarr"
    if ds_path.exists():
        ds = xr.open_zarr(ds_path)
        _src = f"reloaded {ds_path.name}"
    else:
        ds = load_vc7_directory(ZOOMOUT_VC7_FOLDER)
        ds["u"] = ds["u"] * 1000.0
        ds["v"] = ds["v"] * 1000.0
        ds.attrs.update(
            units_x="mm", units_y="mm", units_u="mm/s", units_v="mm/s",
            dt=PULSE_DT_S, frame_dt_s=FRAME_DT_S,
            px_per_mm=PX_PER_MM,
            history="straight_channel_transient.py: load_vc7_directory(Vmax_after_pump_shutdown)",
        )
        ds.to_zarr(ds_path, mode="w", encoding={"u": {"chunks": (1, -1, -1)}})
        _src = f"built + saved {ds_path.name}"
    # Attach frame time in seconds; keep frame index as the coordinate.
    ds = ds.assign_coords(time_s=("t", ds.t.values * FRAME_DT_S))
    mo.md(
        f"Zoom-out series: **{ds.sizes['t']}** frames, "
        f"grid **{ds.sizes['y']} x {ds.sizes['x']}**, "
        f"elapsed **{float(ds.time_s.max()):.2f} s** @ 15 Hz ({_src})."
    )
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - data audit: is there a slowdown in these 100 frames?

    Mean speed, valid fraction and reversed-flow fraction vs. time.
    `reynolds_decomposition()` is deliberately NOT used here - the
    flow is non-stationary, so a full-record mean is not a valid
    base flow (per-frame spatial TKE comes later in Step 6).
    """)
    return


@app.cell
def _(ds, mo, np, plt):
    _speed = np.hypot(ds.u, ds.v).where(ds.chc > 0.5)
    mean_speed_t = _speed.mean(dim=("y", "x"), skipna=True).load()
    valid_frac_t = (ds.chc > 0.5).mean(dim=("y", "x")).load()
    reversed_frac_t = (ds.v.where(ds.chc > 0.5) > 0).mean(
        dim=("y", "x"), skipna=True
    ).load()

    _fig, _ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    _ax[0].plot(ds.time_s, mean_speed_t)
    _ax[0].set_ylabel("mean speed [mm/s]")
    _ax[0].set_title("Field-mean speed vs. time (15 Hz frame rate)")
    _ax[1].plot(ds.time_s, valid_frac_t)
    _ax[1].set_ylabel("valid fraction")
    _ax[2].plot(ds.time_s, reversed_frac_t)
    _ax[2].set_ylabel("reversed (v>0) fraction")
    _ax[2].set_xlabel("time [s]")
    _fig.tight_layout()

    mo.md(
        f"Mean speed **{float(mean_speed_t.mean()):.0f} mm/s**, "
        f"range {float(mean_speed_t.min()):.0f}-{float(mean_speed_t.max()):.0f} mm/s; "
        f"valid **{float(valid_frac_t.mean()):.1%}**; "
        f"reversed **{float(reversed_frac_t.mean()):.2%}** - "
        f"quasi-steady, no decay visible. True transient likely lives in "
        f"`Vmax_0p62_m2sec_pump_shutdown/Camera1-1.ims` (needs DaVis export)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3 - fixed-scale frame viewer (slider through time)

    One fixed speed colormap + one colorbar for all frames, quiver
    scale fixed from the global median speed so arrows are neither
    too long nor too short. Invalid (`chc<0.5`) vectors are masked.
    """)
    return


@app.cell
def _(ds, np):
    _all_speed = np.hypot(ds.u.values, ds.v.values)
    _valid = ds.chc.values > 0.5
    SPEED_VMIN, SPEED_VMAX = float(np.nanpercentile(_all_speed[_valid], 2)), float(
        np.nanpercentile(_all_speed[_valid], 98)
    )
    _vv_all = ds.v.values[_valid]
    VVMIN, VVMAX = float(np.nanpercentile(_vv_all, 1)), float(np.nanpercentile(_vv_all, 99))
    VCENTER = float(np.nanmean(_vv_all))
    _step = max(1, ds.sizes["x"] // 40)
    _dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    _med = float(np.nanmedian(_all_speed[_valid]))
    QUIVER_SCALE = _med / (0.9 * _step * _dx)
    return QUIVER_SCALE, SPEED_VMAX, SPEED_VMIN, VCENTER, VVMAX, VVMIN


@app.cell
def _(ds, mo):
    frame_slider = mo.ui.slider(
        0, ds.sizes["t"] - 1, value=0, label="frame (time-ordered)"
    )
    frame_slider
    return (frame_slider,)


@app.cell
def _(
    QUIVER_SCALE,
    SPEED_VMAX,
    SPEED_VMIN,
    VCENTER,
    VVMAX,
    VVMIN,
    ds,
    frame_slider,
    np,
    plt,
):
    import matplotlib as _mpl

    _k = frame_slider.value
    _fr = ds.isel(t=_k)
    _u = _fr.u.where(_fr.chc > 0.5)
    _v = _fr.v.where(_fr.chc > 0.5)

    _fig, _ax = plt.subplots(figsize=(8, 10))
    _ax.contourf(
        _fr.x, _fr.y, np.hypot(_u, _v), levels=40, cmap="Greys",
        vmin=SPEED_VMIN, vmax=SPEED_VMAX, extend="both",
    )
    _vnorm = _mpl.colors.TwoSlopeNorm(vmin=VVMIN, vcenter=VCENTER, vmax=VVMAX)
    _step = max(1, ds.sizes["x"] // 40)
    _X, _Y = np.meshgrid(_fr.x.values, _fr.y.values)
    _uv = _u.values[::2, ::_step]
    _vv = _v.values[::2, ::_step]
    _ax.quiver(
        _X[::2, ::_step], _Y[::2, ::_step], _uv, _vv, _vv,
        cmap="coolwarm", norm=_vnorm,
        angles="xy", scale_units="xy", scale=QUIVER_SCALE,
        width=0.004, pivot="mid",
    )
    _sm = plt.cm.ScalarMappable(norm=_vnorm, cmap="coolwarm")
    _cbar = _fig.colorbar(_sm, ax=_ax, pad=0.02, shrink=0.85)
    _cbar.set_label("streamwise v [mm/s] (arrow color, fixed scale)")
    _ax.set_title(
        f"frame {_k} (t={float(_fr.time_s):.2f} s) - straight zoom-out transient"
    )
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 4 - slowdown animation (phases of the transient)

    Fixed norm/scale identical to the viewer above, so the animation
    is directly comparable frame to frame. Saved under `outputs/`.
    """)
    return


@app.cell
def _(OUT_DIR, QUIVER_SCALE, SPEED_VMAX, SPEED_VMIN, ds, mo, np, plt):
    import matplotlib.animation as animation
    import matplotlib as _mpl
    from scipy.ndimage import gaussian_filter

    _STRIDE = 10
    _SIGMA = 1.0

    def _smooth_nan(a, sigma=_SIGMA):
        m = np.isfinite(a)
        af = np.where(m, a, 0.0)
        w = gaussian_filter(m.astype(float), sigma, mode="nearest")
        return gaussian_filter(af, sigma, mode="nearest") / np.maximum(w, 1e-6)

    # Temporal (5-frame, centered) + spatial (NaN-aware gaussian) smoothing.
    _ua = ds.u.where(ds.chc > 0.5).rolling(t=5, center=True, min_periods=1).mean().values
    _va = ds.v.where(ds.chc > 0.5).rolling(t=5, center=True, min_periods=1).mean().values
    _plo, _phi = float(np.nanpercentile(_va, 1)), float(np.nanpercentile(_va, 99))
    _vc = float(np.nanmean(_va))
    _vnorm = _mpl.colors.TwoSlopeNorm(vmin=_plo, vcenter=_vc, vmax=_phi)
    _vcmap = plt.get_cmap("coolwarm")

    # Static backdrop: smoothed first-frame speed (no raw image exists).
    _bg = _smooth_nan(np.hypot(_ua[0], _va[0]))

    _idx = list(range(0, ds.sizes["t"], _STRIDE))
    _fig, _ax = plt.subplots(figsize=(7, 9))
    _X, _Y = np.meshgrid(ds.x.values, ds.y.values)

    def _draw(k):
        _ax.clear()
        _ax.contourf(
            ds.x, ds.y, _bg, levels=24, cmap="Greys",
            vmin=SPEED_VMIN, vmax=SPEED_VMAX, extend="both",
        )
        _uf = _smooth_nan(_ua[k])[::4, ::6]
        _vf = _smooth_nan(_va[k])[::4, ::6]
        _ax.quiver(
            _X[::4, ::6], _Y[::4, ::6], _uf, _vf, _vf,
            cmap=_vcmap, norm=_vnorm,
            angles="xy", scale_units="xy", scale=QUIVER_SCALE,
            width=0.005, pivot="mid",
        )
        _ax.set_title(f"t={float(ds.time_s.isel(t=k)):.2f} s (frame {k})")
        _ax.set_xlabel("x [mm]")
        _ax.set_ylabel("y [mm]")
        _ax.set_aspect("equal")

    _draw(_idx[0])
    _sm = plt.cm.ScalarMappable(norm=_vnorm, cmap=_vcmap)
    _cbar = _fig.colorbar(_sm, ax=_ax, pad=0.02, shrink=0.85)
    _cbar.set_label("streamwise velocity v [mm/s] (smoothed arrow color)")
    _anim = animation.FuncAnimation(_fig, _draw, frames=_idx, interval=350, repeat=True)
    _gif_path = OUT_DIR / "straight_transient_zoomout.gif"
    _anim.save(_gif_path, writer="pillow", fps=3)
    plt.close(_fig)
    mo.md(f"Saved `{_gif_path.name}` ({len(_idx)} frames, every 10th; smoothed, coolwarm v scale {_plo:.0f}..{_phi:.0f} centered {_vc:.0f} mm/s).")
    return


@app.cell(hide_code=True)
def _(mo):
    md_step5 = mo.md(
        """
        ## Step 5 - velocity profiles in time + wall shear

        `profile_evolution()` (shared helper in `piv_pipeline.py`): row-averaged
        (`y`), 5-frame smoothed streamwise profiles `v(x)` drawn every ~60 s
        (up to 10 time-colored curves, refined spacing on short runs), centered
        at x = 0, shown once dimensional (mm/s vs. mm) and once normalized
        (`V/U_bulk` vs. `x/b`, single reference bulk velocity) - plus a
        bulk-velocity-vs-time trace with matching markers. Below that:
        centreline `v(t)` and a near-wall shear proxy `dv/dx` from the two grid
        columns closest to each wall (resolution-limited - the zoom-in, when a
        transient exists, is the proper shear measurement).
        """
    )
    md_step5
    return


@app.cell
def _(ds, mo):
    import importlib as _il
    import piv_pipeline as _pp
    _il.reload(_pp)
    _fig1, _axes1, _prof1 = _pp.profile_evolution(
        ds, component="v", window=5, every_s=60.0, max_curves=10,
    )
    _fig2, _axes2, _prof2 = _pp.profile_evolution(
        ds, component="v", window=5, every_s=60.0, max_curves=10, normalize=True,
    )
    mo.vstack([_fig1, _fig2])
    return


@app.cell
def _(ds, plt):
    _cx = ds.sizes["x"] // 2
    _vc = ds.v.isel(x=_cx).where(ds.chc.isel(x=_cx) > 0.5).mean(
        dim="y", skipna=True
    )
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(ds.time_s, _vc)
    _ax.set_xlabel("time [s]")
    _ax.set_ylabel("centreline v [mm/s]")
    _ax.set_title("Centreline streamwise velocity vs. time")
    _fig.gca()

    # Wall-shear proxy: dv/dx at the first valid columns near each wall.
    _dudx = ds.v.differentiate("x")
    _left = _dudx.isel(x=2).where(ds.chc.isel(x=2) > 0.5).mean(dim="y", skipna=True)
    _right = _dudx.isel(x=-3).where(ds.chc.isel(x=-3) > 0.5).mean(dim="y", skipna=True)
    _fig2, _ax2 = plt.subplots(figsize=(9, 3.5))
    _ax2.plot(ds.time_s, _left, label="left wall proxy")
    _ax2.plot(ds.time_s, _right, label="right wall proxy")
    _ax2.set_xlabel("time [s]")
    _ax2.set_ylabel("dv/dx [1/s] (2nd valid column)")
    _ax2.set_title("Near-wall shear proxy vs. time (zoom-out resolution)")
    _ax2.legend()
    _fig2.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    md_tke = mo.md(
        r"""
        ## Step 6 - turbulence = spatial fluctuations about the row-mean profile

        Per frame, on the same 5-frame smoothed fields as Step 5:

        mean profile: \(\langle V\rangle(x,t) =\) mean over valid rows \(y\) of \(V(x,y,t)\)

        fluctuations: \(u'(x,y,t) = U - \langle U\rangle(x,t)\), \(v'(x,y,t) = V - \langle V\rangle(x,t)\)

        frame TKE: \(\mathrm{TKE}(t) = \langle \tfrac{1}{2}(u'^2+v'^2)\rangle\) averaged over valid \((x,y)\)

        Deliberately **not** temporal fluctuations about a time mean (those would
        leak the slowdown trend itself into "turbulence"), and not full-record
        `reynolds_decomposition()` (invalid for non-stationary flow). Components
        \(\langle u'^2\rangle/2\) and \(\langle v'^2\rangle/2\) are plotted
        alongside the total.
        """
    )
    md_tke
    return


@app.cell
def _(ds, plt):
    import importlib as _il
    import piv_pipeline as _pp
    _il.reload(_pp)
    _tke_ds = _pp.spatial_tke(ds, window=5)
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(_tke_ds.time_s, _tke_ds.tke, color="k", linewidth=1.5,
             label="TKE = <1/2 (u'^2+v'^2)>")
    _ax.plot(_tke_ds.time_s, _tke_ds.u_var, label="<u'^2>/2", alpha=0.7)
    _ax.plot(_tke_ds.time_s, _tke_ds.v_var, label="<v'^2>/2", alpha=0.7)
    _ax.set_xlabel("time [s]")
    _ax.set_ylabel("spatial TKE [mm^2/s^2]")
    _ax.set_title("Per-frame spatial TKE about the row-mean profile (5-frame smoothed)")
    _ax.legend(fontsize=8)
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 7 - zoom-in transient (probe)

    Looks for any pump-shutdown acquisition under the
    right-boundary-layer folder and the `Vmax_0p62_m2sec_pump_shutdown`
    raw set. Expected outcome today: none processed - the steady
    right-BL zarr is listed as the reference to reuse once a
    transient zoom-in is acquired/exported.
    """)
    return


@app.cell
def _(OUT_DIR, mo):
    from pathlib import Path as _P

    _candidates = [
        _P(r"D:\channel_flow_research\baseline_channel\Vmax_0p62_m2sec_pump_shutdown"),
        _P(r"D:\channel_flow_research\baseline_channel\Vmax_0p62_m2sec_steady_state_right_boundary_layer"),
    ]
    _rows = []
    for _c in _candidates:
        _vc7 = list(_c.glob("**/*.vc7"))[:3]
        _im7 = list(_c.glob("**/exported_images/**/*.im7"))[:3]
        _rows.append(
            f"| `{_c.name}` | exists={_c.exists()} | "
            f"vc7 sample={len(_vc7)} | exported .im7 sample={len(_im7)} |"
        )
    _steady_bl = OUT_DIR / "baseline_right_boundary_layer_ds.zarr"
    mo.md(
        "| candidate | status |\n|---|---|\n" + "\n".join(_rows) + "\n\n"
        f"Steady right-BL reference zarr present: **{_steady_bl.exists()}** "
        f"(`{_steady_bl.name}`). No transient zoom-in vectors found - "
        f"acquire/export before running Steps 2-6 on the zoom-in."
    )
    return


@app.cell
def _(mo):
    md_insight = mo.md(
        """
        ## Step 8 - where is the flow trustworthy, and where does it reverse?

        Three follow-up views from the live session (2026-09-26):

        - **Valid-fraction map**: `chc>0.5` averaged over time shows *where*
          the vectors are reliable. Expect bright borders / dark interior edge
          effects from the DaVis multi-pass grid.
        - **Reversal-hotspot map**: fraction of frames with `v>0` (up-flow)
          shows whether reversal is scattered noise or a persistent zone.
        - **Steady overlay**: the same cross-channel `v(x)` profile from the
          3000-frame steady run, on the same axes - if the pump shutdown left
          a trace, the curves should differ.
        """
    )
    md_insight
    return


@app.cell
def _(ds, np, plt):
    _valid_map = (ds.chc > 0.5).mean(dim="t").load()
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _cf = _ax.contourf(
        ds.x, ds.y, _valid_map, levels=np.linspace(0, 1, 21),
        cmap="Greys", vmin=0, vmax=1, extend="both",
    )
    _cbar = _fig.colorbar(_cf, ax=_ax, pad=0.02, shrink=0.85)
    _cbar.set_label("valid fraction over 100 frames")
    _ax.set_title("Valid-fraction map - dark borders = DaVis edge effect")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(ds, np, plt):
    _rev_map = ((ds.v > 0) & (ds.chc > 0.5)).mean(dim="t", skipna=True).load()
    _u_mean = ds.u.where(ds.chc > 0.5).mean(dim="t", skipna=True)
    _v_mean = ds.v.where(ds.chc > 0.5).mean(dim="t", skipna=True)
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _cf = _ax.contourf(ds.x, ds.y, _rev_map, levels=60, cmap="Reds", extend="max")
    _cbar = _fig.colorbar(_cf, ax=_ax, pad=0.02, shrink=0.85)
    _cbar.set_label("reversed-flow (v>0) fraction over 100 frames")
    _step = max(1, ds.sizes["x"] // 30)
    _X, _Y = np.meshgrid(ds.x.values, ds.y.values)
    _ax.quiver(
        _X[::3, ::_step], _Y[::3, ::_step],
        _u_mean.values[::3, ::_step], _v_mean.values[::3, ::_step],
        angles="xy", color="k", width=0.003, alpha=0.5,
    )
    _ax.set_title("Reversal-hotspot map with time-mean flow vectors")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _fig.gca()
    return


@app.cell
def _(OUT_DIR, ds, plt, xr):
    import importlib as _il
    import piv_pipeline as _pp
    _il.reload(_pp)
    _steady_stats = xr.open_zarr(OUT_DIR / "baseline_steady_state_stats.zarr")
    _s_prof = _steady_stats.v_mean.mean(dim="y", skipna=True).load()
    _t_prof = ds.v.where(ds.chc > 0.5).mean(dim=("t", "y"), skipna=True).load()
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for _ax, _norm in ((_ax1, False), (_ax2, True)):
        for _prof, _xx, _label in (
            (_s_prof, _steady_stats.x.values, "steady (3000 frames)"),
            (_t_prof, ds.x.values, "after-shutdown (100 frames)"),
        ):
            _xc, _b, _xcc = _pp.channel_center(_xx)
            _ub = _pp.bulk_velocity(_prof.values, _xx, _b)
            if _norm:
                _ax.plot(_xcc / _b, _prof.values / _ub,
                         label=f"{_label} (b={_b:.2f}, Ub={_ub:.0f})")
            else:
                _ax.plot(_xcc, _prof.values, label=f"{_label} (b={_b:.2f} mm)")
        _ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        _ax.legend(fontsize=8)
    _ax1.set_xlabel("x - center [mm] (0 = centerline)")
    _ax1.set_ylabel("v [mm/s]")
    _ax1.set_title("dimensional, centered")
    _ax2.set_xlabel("x/b [-] (0 = centerline)")
    _ax2.set_ylabel("V/U_bulk [-]")
    _ax2.set_title("normalized by own bulk velocity")
    _fig.suptitle("Cross-channel profile: steady vs. after-shutdown")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
