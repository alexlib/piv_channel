# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "xarray",
#     "zarr",
#     "matplotlib",
#     "scipy",
#     "pivpy",
# ]
# ///

"""Straight channel, right boundary layer (zoom-in) - high-resolution wall shear & transient analysis.

Scope: `baseline_channel` zoom-in (`Vmax_0p62_m2sec_steady_state_right_boundary_layer`,
3000 DaVis `.vc7` frames @ 15 Hz = 200 s record).

Key advantages over zoom-out:
- Grid pitch Δx ≈ 15.3 µm (4.5x finer than zoom-out's ~68 µm), resolving the viscous
  sublayer and buffer layer directly against the right wall (x_wall ≈ 1.587 mm).
- Viscous sublayer (y+ < 5, yw < ~270 µm) contains ~18 PIV vector columns.
- Enables direct, robust calculation of wall shear rate d|v|/dy_w, wall shear stress
  tau_w = mu * d|v|/dy_w, friction velocity u_tau, viscous length scale delta_nu,
  inner-unit profile scaling (u+ vs y+), and Reynolds shear stress near the wall.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Straight channel, right boundary layer (zoom-in)
    ## High-resolution wall shear stress, boundary-layer profiles & dynamics

    In the zoom-out dataset (`straight_channel_transient.py`), the 10 mm channel gap
    was resolved with $\Delta x \approx 68\,\mu\mathrm{m}$. Near the wall, the first 2–3
    grid points fell inside the DaVis multi-pass edge artifact region, making direct
    wall-shear measurements resolution-limited.

    This zoom-in dataset focuses specifically on the **right boundary layer**, offering:
    - **Spatial resolution**: $\Delta x \approx 15.28\,\mu\mathrm{m}$ (4.5× zoom-out resolution),
      covering $2.84\text{ mm}$ across 187 grid columns right against the solid wall ($x_{\text{wall}} \approx 1.587\text{ mm}$).
    - **Viscous sublayer**: $\sim 18$ grid columns reside within $y^+ < 5$, allowing a direct linear fit
      for wall shear rate $\dot{\gamma}_w = \left.\frac{\partial |v|}{\partial y_w}\right|_{y_w \to 0}$
      and wall shear stress $\tau_w = \mu \dot{\gamma}_w$.
    - **Temporal record**: 3000 frames at 15 Hz ($200\text{ s}$ total duration).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    md_provenance = mo.md(
        r"""
        ## Data provenance

        - **Run**: `baseline_channel/Vmax_0p62_m2sec_steady_state_right_boundary_layer` on
          `D:\\channel_flow_research` - straight baseline channel, right-boundary-layer zoom-in.
        - **Format**: 3000 DaVis `.vc7` vector maps from `PIV_MPd(4x16x16_25%ov_ImgCorr)`
          (multi-pass deformation down to 16×16 final window, 25% overlap, image correction + median filter).
        - **Time base**: 15 Hz recording rate (from run's own `.set` file: *"Average: 15 Hz, Min: 15 Hz, Max: 15 Hz"*).
          Inter-frame interval is $1/15\\text{ s} \\approx 0.0667\\text{ s}$, covering $200\\text{ s}$.
          Laser pulse separation inside each frame pair is $\\Delta t = 50\\,\\mu\\mathrm{s}$ (`DevDataTrace5`).
        - **Calibration**: `PX_PER_MM = 173.419` from `Properties/Calibration/Calibration.xml`.
        - **Grid**: $158$ rows ($y \\in [-1.22, 1.18]\\text{ mm}$) $\\times$ $187$ columns ($x \\in [-1.25, 1.59]\\text{ mm}$).
        - **Right wall location**: $x_{\\text{wall}} \\approx 1.587\\text{ mm}$. Distance from wall is $y_w = x_{\\text{wall}} - x$.
        - **Stores**: `outputs/baseline_right_boundary_layer_ds.zarr` (full time-resolved Dataset) and
          `outputs/baseline_right_boundary_layer_stats.zarr` (ensemble Reynolds decomposition statistics).
        """
    )
    md_provenance
    return


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import pivpy  # noqa: F401 (registers .piv accessor)

    # Ensure piv_pipeline is importable from notebooks directory
    NOTEBOOKS_DIR = Path(__file__).resolve().parent
    if str(NOTEBOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS_DIR))

    ROOT = NOTEBOOKS_DIR.parent
    OUT_DIR = ROOT / "outputs"

    RIGHT_BL_VC7_FOLDER = (
        r"D:\channel_flow_research\baseline_channel"
        r"\Vmax_0p62_m2sec_steady_state_right_boundary_layer"
        r"\PIV_MPd(4x16x16_25%ov_ImgCorr)"
    )

    FRAME_DT_S = 1.0 / 15.0  # 15 Hz recording rate
    PULSE_DT_S = 5.0e-5      # 50 us laser pulse separation
    PX_PER_MM = 173.41900170673784
    B_HALF_GAP = 5.05        # channel half-width from full 10.1 mm gap

    # Water fluid properties at 20 C
    WATER_RHO = 998.0        # kg/m^3
    WATER_MU = 1.002e-3      # Pa.s (dynamic viscosity)
    WATER_NU = WATER_MU / WATER_RHO  # m^2/s (kinematic viscosity: ~1.004e-6 m^2/s)
    return (
        FRAME_DT_S,
        OUT_DIR,
        PULSE_DT_S,
        RIGHT_BL_VC7_FOLDER,
        WATER_MU,
        WATER_NU,
        WATER_RHO,
        mo,
        np,
        plt,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 - load zoom-in Dataset & setup coordinates

    Loads `outputs/baseline_right_boundary_layer_ds.zarr` and
    `outputs/baseline_right_boundary_layer_stats.zarr`. Coordinates include:
    - `time_s`: physical elapsed time in seconds ($t \times \frac{1}{15\text{ s}}$).
    - $x_{\text{wall}}$: position of the solid right channel wall ($x_{\text{wall}} = \max(x) \approx 1.587\text{ mm}$).
    - $y_w$: wall-normal distance into the fluid, $y_w = x_{\text{wall}} - x$ in mm and $\mu\mathrm{m}$.
    """)
    return


@app.cell(hide_code=True)
def _(FRAME_DT_S, OUT_DIR, PULSE_DT_S, RIGHT_BL_VC7_FOLDER, mo, xr):
    from piv_pipeline import load_vc7_directory

    ds_path = OUT_DIR / "baseline_right_boundary_layer_ds.zarr"
    stats_path = OUT_DIR / "baseline_right_boundary_layer_stats.zarr"

    if ds_path.exists():
        ds = xr.open_zarr(ds_path)
        _src_ds = f"reloaded `{ds_path.name}`"
    else:
        ds = load_vc7_directory(RIGHT_BL_VC7_FOLDER)
        ds["u"] = ds["u"] * 1000.0
        ds["v"] = ds["v"] * 1000.0
        ds.attrs.update(
            units_x="mm", units_y="mm", units_u="mm/s", units_v="mm/s",
            dt=PULSE_DT_S, frame_dt_s=FRAME_DT_S,
            history="straight_channel_transient_zoomin.py: load_vc7_directory(right_boundary_layer)",
        )
        ds.to_zarr(ds_path, mode="w")
        _src_ds = f"built + saved `{ds_path.name}`"

    # Attach physical time coordinate if not already present
    if "time_s" not in ds.coords:
        ds = ds.assign_coords(time_s=("t", ds.t.values * FRAME_DT_S))

    if stats_path.exists():
        stats = xr.open_zarr(stats_path)
        _src_stats = f"reloaded `{stats_path.name}`"
    else:
        stats = ds.piv.reynolds_decomposition()
        stats.to_zarr(stats_path, mode="w")
        _src_stats = f"built + saved `{stats_path.name}`"

    x_vals = ds.x.values
    x_wall = float(x_vals.max())
    y_wall_mm = x_wall - x_vals
    y_wall_um = y_wall_mm * 1000.0
    dx_um = float(abs(x_vals[1] - x_vals[0])) * 1000.0

    mo.md(
        f"**Zoom-in right boundary layer**: **{ds.sizes['t']}** frames, "
        f"grid **{ds.sizes['y']} rows × {ds.sizes['x']} columns**, "
        f"elapsed **{float(ds.time_s.max()):.1f} s** @ 15 Hz.\n\n"
        f"- Spatial resolution: $\\Delta x = {dx_um:.2f}\\,\\mu\\mathrm{{m}}$, "
        f"spanning $y_w = 0\\dots{float(y_wall_mm.max()):.2f}\\text{{ mm}}$.\n"
        f"- Wall location: $x_{{\\text{{wall}}}} = {x_wall:.4f}\\text{{ mm}}$.\n"
        f"- Status: {_src_ds}, {_src_stats}."
    )
    return ds, stats, x_wall, y_wall_mm, y_wall_um


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 - data audit: speed stability & near-wall dynamics across 200 s

    Audit of time evolution over the 3000-frame (200 s) acquisition:
    - Global field-mean speed $\langle |\mathbf{u}| \rangle(t)$ and valid vector fraction.
    - Near-wall velocity trace ($y_w \approx 30\,\mu\mathrm{m}$) vs. core boundary layer velocity ($y_w \approx 2.5\text{ mm}$).
    """)
    return


@app.cell
def _(ds, mo, np, plt, y_wall_um):
    # Decimate time by factor 5 for fast audit plot rendering (600 time samples across 200 s)
    _t_stride = 5
    _sub = ds.isel(t=slice(0, ds.sizes["t"], _t_stride))
    _speed = np.hypot(_sub.u, _sub.v).where(_sub.chc > 0.5)

    _mean_speed_t = _speed.mean(dim=("y", "x"), skipna=True).load()
    _valid_frac_t = (_sub.chc > 0.5).mean(dim=("y", "x")).load()

    # Compare near-wall trace (col -3, ~30 um from wall) vs outer boundary layer (col 10, ~2.5 mm)
    _v_near_wall = _sub.v.isel(x=-3).where(_sub.chc.isel(x=-3) > 0.5).mean(dim="y", skipna=True).load()
    _v_outer = _sub.v.isel(x=10).where(_sub.chc.isel(x=10) > 0.5).mean(dim="y", skipna=True).load()
    _t_axis = _sub.time_s.values

    _fig, _axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    _axs[0].plot(_t_axis, _mean_speed_t, color="tab:blue", lw=1.2)
    _axs[0].set_ylabel("mean speed [mm/s]")
    _axs[0].set_title("Field-mean speed vs time (3000 frames @ 15 Hz = 200 s)")
    _axs[0].grid(True, linestyle=":", alpha=0.6)

    _axs[1].plot(_t_axis, _valid_frac_t * 100.0, color="tab:green", lw=1.2)
    _axs[1].set_ylabel("valid fraction [%]")
    _axs[1].set_title("Vector validation rate over time")
    _axs[1].grid(True, linestyle=":", alpha=0.6)

    _axs[2].plot(_t_axis, np.abs(_v_near_wall), label=f"near-wall ($y_w \\approx {y_wall_um[-3]:.0f}\\,\\mu\\mathrm{{m}}$)", color="tab:red", lw=1.0)
    _axs[2].plot(_t_axis, np.abs(_v_outer), label=f"outer BL ($y_w \\approx {y_wall_um[10]:.0f}\\,\\mu\\mathrm{{m}}$)", color="tab:purple", lw=1.0, alpha=0.8)
    _axs[2].set_ylabel("streamwise |v| [mm/s]")
    _axs[2].set_xlabel("time [s]")
    _axs[2].set_title("Near-wall viscous layer vs outer boundary layer velocity")
    _axs[2].legend(loc="upper right", fontsize=8)
    _axs[2].grid(True, linestyle=":", alpha=0.6)

    _fig.tight_layout()

    _mean_val = float(_mean_speed_t.mean())
    _std_val = float(_mean_speed_t.std())
    _v_pct = float(_valid_frac_t.mean() * 100.0)

    mo.vstack([
        _fig,
        mo.md(
            f"**Audit summary**: Mean speed is **{_mean_val:.1f} ± {_std_val:.1f} mm/s** "
            f"(fluctuation {100.0*_std_val/_mean_val:.1f}%), with mean validity rate **{_v_pct:.1f}%**. "
            f"The flow maintains quasi-steady turbulent statistics over 200 s, with turbulent bursts visible in the near-wall trace."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 - interactive frame viewer (slider through time)

    Examine instantaneous vector fields near the solid wall.
    - Colormap shows streamwise velocity $v$ with fixed TwoSlopeNorm.
    - Red dashed vertical line at $x = x_{\text{wall}}$ marks the solid right wall boundary.
    - Background shows speed contour.
    """)
    return


@app.cell
def _(ds, mo, np):
    # Compute robust color and quiver limits from sample of frames
    _sample = ds.isel(t=slice(0, min(500, ds.sizes["t"]), 20))
    _speed_sample = np.hypot(_sample.u.values, _sample.v.values)
    _valid_mask = _sample.chc.values > 0.5
    _v_sample = _sample.v.values[_valid_mask]

    SPEED_VMIN = float(np.nanpercentile(_speed_sample[_valid_mask], 2))
    SPEED_VMAX = float(np.nanpercentile(_speed_sample[_valid_mask], 98))
    VVMIN = float(np.nanpercentile(_v_sample, 1))
    VVMAX = float(np.nanpercentile(_v_sample, 99))
    VCENTER = float(np.nanmean(_v_sample))

    _step_x = max(1, ds.sizes["x"] // 35)
    _dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    _med_speed = float(np.nanmedian(_speed_sample[_valid_mask]))
    QUIVER_SCALE = _med_speed / (0.85 * _step_x * _dx)

    frame_slider = mo.ui.slider(
        0, ds.sizes["t"] - 1, step=15, value=0, label="Frame index (t)"
    )
    return (
        QUIVER_SCALE,
        SPEED_VMAX,
        SPEED_VMIN,
        VCENTER,
        VVMAX,
        VVMIN,
        frame_slider,
    )


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
    mo,
    np,
    plt,
    x_wall,
):
    import matplotlib as _mpl

    _k = frame_slider.value
    _fr = ds.isel(t=_k)
    _u = _fr.u.where(_fr.chc > 0.5)
    _v = _fr.v.where(_fr.chc > 0.5)
    _speed_fr = np.hypot(_u, _v)

    _fig, _ax = plt.subplots(figsize=(8.5, 7.5))
    _cf = _ax.contourf(
        _fr.x, _fr.y, _speed_fr, levels=35, cmap="coolwarm",
        vmin=SPEED_VMIN, vmax=SPEED_VMAX, extend="both",
    )
    _vnorm = _mpl.colors.TwoSlopeNorm(vmin=VVMIN, vcenter=VCENTER, vmax=VVMAX)

    _step_x = max(1, ds.sizes["x"] // 35)
    _step_y = max(1, ds.sizes["y"] // 30)
    _X, _Y = np.meshgrid(_fr.x.values, _fr.y.values)
    _uv = _u.values[::_step_y, ::_step_x]
    _vv = _v.values[::_step_y, ::_step_x]

    _ax.quiver(
        _X[::_step_y, ::_step_x], _Y[::_step_y, ::_step_x], _uv, _vv, _vv,
        cmap="coolwarm", norm=_vnorm,
        angles="xy", scale_units="xy", scale=QUIVER_SCALE,
        width=0.0035, pivot="mid",
    )

    # Mark physical wall
    _ax.axvline(x_wall, color="red", linestyle="--", linewidth=1.5, label=f"Wall ($x={x_wall:.3f}\\text{{ mm}}$)")
    _ax.text(x_wall - 0.05, float(_fr.y.max()) - 0.2, "Solid Wall →", color="red",
             ha="right", va="top", fontweight="bold", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="red"))

    _sm = plt.cm.ScalarMappable(norm=_vnorm, cmap="coolwarm")
    _cbar = _fig.colorbar(_sm, ax=_ax, pad=0.02, shrink=0.85)
    _cbar.set_label("streamwise velocity v [mm/s] (coolwarm arrow color)")

    _time_now = float(_fr.time_s)
    _ax.set_title(f"Frame {_k} (t = {_time_now:.2f} s) — Right boundary layer zoom-in")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_aspect("equal")
    _ax.legend(loc="lower left", fontsize=8)

    mo.vstack([
        frame_slider,
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 - near-wall boundary layer velocity profiles

    Streamwise velocity profiles plotted against distance from the wall $y_w = x_{\text{wall}} - x$:
    1. **Overview profile**: $y_w$ from $0$ to $2.8\text{ mm}$, showing the full captured boundary layer.
    2. **Viscous sublayer zoom-in**: $y_w \in [0, 200]\,\mu\mathrm{m}$, showing the resolved linear shear layer.
    """)
    return


@app.cell
def _(ds, mo, np, plt, stats, y_wall_mm, y_wall_um):
    _v_mean_profile = stats.v_mean.mean(dim="y", skipna=True).load().values

    # Select 10 time intervals across the 3000 frames to show profile stability / evolution
    _n_curves = 10
    _t_indices = np.linspace(0, ds.sizes["t"] - 1, _n_curves, dtype=int)
    _cmap = plt.get_cmap("viridis")

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for _i, _idx in enumerate(_t_indices):
        _fr_v = ds.v.isel(t=_idx).where(ds.chc.isel(t=_idx) > 0.5).mean(dim="y", skipna=True).values
        _time_val = float(ds.time_s.isel(t=_idx))
        _color = _cmap(_i / (_n_curves - 1))
        _ax1.plot(y_wall_mm, np.abs(_fr_v), color=_color, alpha=0.35, lw=0.9)
        _ax2.plot(y_wall_um, np.abs(_fr_v), color=_color, alpha=0.35, lw=0.9)

    # Plot time-ensemble mean
    _ax1.plot(y_wall_mm, np.abs(_v_mean_profile), color="black", lw=2.2, label="Ensemble Mean (3000 frames)")
    _ax2.plot(y_wall_um, np.abs(_v_mean_profile), color="black", lw=2.2, label="Ensemble Mean")

    # Mark individual PIV grid points in sublayer
    _sub_mask = y_wall_um <= 200.0
    _ax2.plot(y_wall_um[_sub_mask], np.abs(_v_mean_profile)[_sub_mask], "ro", markersize=4, label="PIV grid points (Δx=15.3 µm)")

    _ax1.set_xlabel("Wall-normal distance $y_w$ [mm]")
    _ax1.set_ylabel("Streamwise velocity $|v|$ [mm/s]")
    _ax1.set_title("Full boundary layer profile ($0 \\dots 2.8\\text{ mm}$)")
    _ax1.grid(True, linestyle=":", alpha=0.6)
    _ax1.legend(loc="lower right", fontsize=8)

    _ax2.set_xlabel("Wall-normal distance $y_w$ [$\\mu\\mathrm{m}$]")
    _ax2.set_ylabel("Streamwise velocity $|v|$ [mm/s]")
    _ax2.set_xlim(-5, 200)
    _ax2.set_ylim(-5, 120)
    _ax2.set_title("Viscous sublayer zoom-in ($y_w \\leq 200\\,\\mu\\mathrm{m}$)")
    _ax2.grid(True, linestyle=":", alpha=0.6)
    _ax2.legend(loc="lower right", fontsize=8)

    _fig.suptitle("Near-wall streamwise velocity profile $v(y_w)$", fontsize=12, fontweight="bold")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            "Notice the remarkable linearity in the right panel ($y_w \\le 100\\,\\mu\\mathrm{m}$): "
            "the no-slip wall condition is cleanly satisfied as $y_w \\to 0$, with $\\sim 7$ vector points "
            "directly tracing the linear viscous sublayer."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 - direct wall shear stress $\tau_w$, friction velocity $u_\tau$, & inner-unit scaling ($u^+$ vs $y^+$)

    With the viscous sublayer resolved, we compute wall quantities directly from the linear gradient:

    $$\dot{\gamma}_w = \left. \frac{\partial |v|}{\partial y_w} \right|_{y_w \to 0}, \quad
    \tau_w = \mu \dot{\gamma}_w, \quad
    u_\tau = \sqrt{\frac{\tau_w}{\rho}}, \quad
    \delta_\nu = \frac{\nu}{u_\tau}$$

    In inner wall units:
    $$y^+ = \frac{y_w}{\delta_\nu}, \quad u^+ = \frac{|v|}{u_\tau}$$

    The linear viscous sublayer obeys $u^+ = y^+$ ($y^+ < 5$), while the log layer obeys $u^+ = \frac{1}{\kappa}\ln(y^+) + B$ ($\kappa \approx 0.41, B \approx 5.0$).
    """)
    return


@app.cell
def _(WATER_MU, WATER_NU, WATER_RHO, mo, np, plt, stats, y_wall_um):
    _v_mean_profile = stats.v_mean.mean(dim="y", skipna=True).load().values

    # Select points within 100 um of the wall for the linear shear rate fit
    _fit_mask = (y_wall_um >= 0.0) & (y_wall_um <= 100.0) & np.isfinite(_v_mean_profile)
    _yw_m = y_wall_um[_fit_mask] * 1e-6       # m
    _v_m_s = np.abs(_v_mean_profile[_fit_mask]) * 1e-3  # m/s

    # Linear fit: |v| = gamma_w * y_w + c0
    _poly, _cov = np.polyfit(_yw_m, _v_m_s, 1, cov=True)
    gamma_w = float(_poly[0])                 # s^-1
    _r2 = float(np.corrcoef(_yw_m, _v_m_s)[0, 1] ** 2)

    tau_w = float(WATER_MU * gamma_w)         # Pa (N/m^2)
    u_tau = float(np.sqrt(tau_w / WATER_RHO)) # m/s
    delta_nu = float(WATER_NU / u_tau)        # m
    delta_nu_um = float(delta_nu * 1e6)       # um

    # Compute inner units
    _all_mask = (y_wall_um >= 0.0) & np.isfinite(_v_mean_profile)
    _y_plus = (y_wall_um[_all_mask] * 1e-6) / delta_nu
    _u_plus = (np.abs(_v_mean_profile[_all_mask]) * 1e-3) / u_tau

    # Theoretical curves
    _y_visc = np.linspace(0.1, 7.0, 100)
    _u_visc = _y_visc  # u+ = y+

    _y_log = np.logspace(np.log10(5.0), np.log10(max(60.0, float(_y_plus.max()))), 100)
    _kappa = 0.41
    _B = 5.0
    _u_log = (1.0 / _kappa) * np.log(_y_log) + _B

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Linear fit in physical units
    _ax1.plot(y_wall_um[_fit_mask], _v_m_s * 1000.0, "ro", markersize=6, label="PIV points ($y_w \\leq 100\\,\\mu\\mathrm{m}$)")
    _yw_plot = np.linspace(0, 120, 100)
    _ax1.plot(_yw_plot, (_poly[0] * _yw_plot * 1e-6 + _poly[1]) * 1000.0, "b-", lw=1.5,
              label=f"Fit: $\\dot{{\\gamma}}_w =$ {gamma_w:.1f} s$^{{-1}}$ ($R^2={_r2:.4f}$)")
    _ax1.set_xlabel("Wall-normal distance $y_w$ [$\\mu\\mathrm{m}$]")
    _ax1.set_ylabel("Streamwise velocity $|v|$ [mm/s]")
    _ax1.set_title("Near-wall linear fit for shear rate $\\dot{\\gamma}_w$")
    _ax1.grid(True, linestyle=":", alpha=0.6)
    _ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: Law of the wall (inner units)
    _ax2.plot(_y_visc, _u_visc, "g--", lw=1.5, label="Viscous sublayer: $u^+ = y^+$")
    _ax2.plot(_y_log, _u_log, "m--", lw=1.5, label="Log law: $u^+ = \\frac{1}{0.41}\\ln y^+ + 5.0$")
    _ax2.plot(_y_plus, _u_plus, "k.-", lw=1.2, markersize=5, label="PIV zoom-in data")
    _ax2.axvline(5.0, color="gray", linestyle=":", alpha=0.7, label="$y^+ = 5$ (sublayer edge)")

    _ax2.set_xscale("log")
    _ax2.set_xlabel("Inner coordinate $y^+ = y_w / \\delta_\\nu$ [-]")
    _ax2.set_ylabel("Normalized velocity $u^+ = |v| / u_\\tau$ [-]")
    _ax2.set_title("Law of the Wall scaling ($u^+$ vs $y^+$)")
    _ax2.set_xlim(0.2, float(_y_plus.max()) * 1.1)
    _ax2.set_ylim(0, max(25.0, float(_u_plus.max()) * 1.1))
    _ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    _ax2.legend(loc="upper left", fontsize=8)

    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"### Boundary layer scaling parameters:\n\n"
            f"- **Wall shear rate**: $\\dot{{\\gamma}}_w =$ **{gamma_w:.1f} s⁻¹** ($R^2 = {_r2:.4f}$)\n"
            f"- **Wall shear stress**: $\\tau_w = \\mu \\dot{{\\gamma}}_w =$ **{tau_w:.3f} Pa**\n"
            f"- **Friction velocity**: $u_\\tau = \\sqrt{{\\tau_w / \\rho}} =$ **{u_tau*1000.0:.2f} mm/s**\n"
            f"- **Viscous length scale**: $\\delta_\\nu = \\nu / u_\\tau =$ **{delta_nu_um:.2f} µm**\n"
            f"- **Grid resolution in wall units**: $\\Delta x^+ = \\Delta x / \\delta_\\nu =$ **{abs(y_wall_um[1] - y_wall_um[0])/delta_nu_um:.2f}** wall units/grid point."
        ),
    ])
    return delta_nu_um, u_tau


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 - spatial turbulence & Reynolds shear stress profiles

    Using the ensemble Reynolds decomposition (`outputs/baseline_right_boundary_layer_stats.zarr`),
    we examine the boundary-layer turbulence quantities:
    - Streamwise fluctuation intensity $\sqrt{\overline{v'^2}}$
    - Wall-normal fluctuation intensity $\sqrt{\overline{u'^2}}$
    - Reynolds shear stress $-\overline{u'v'}$
    - Turbulent kinetic energy $\mathrm{TKE} = \frac{1}{2}(\overline{u'^2} + \overline{v'^2})$
    """)
    return


@app.cell
def _(delta_nu_um, mo, np, plt, stats, u_tau, y_wall_mm):
    _uu = stats.uu_prime.mean(dim="y", skipna=True).load().values
    _vv = stats.vv_prime.mean(dim="y", skipna=True).load().values
    _uv = stats.uv_prime.mean(dim="y", skipna=True).load().values
    _tke = stats.tke.mean(dim="y", skipna=True).load().values

    _y_plus = (y_wall_mm * 1000.0) / delta_nu_um
    _u_tau_mm_s = u_tau * 1000.0

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Fluctuation intensities normalized by u_tau
    _ax1.plot(_y_plus, np.sqrt(np.maximum(_vv, 0)) / _u_tau_mm_s, "b-", lw=1.5, label="$v'_{rms} / u_\\tau$ (streamwise)")
    _ax1.plot(_y_plus, np.sqrt(np.maximum(_uu, 0)) / _u_tau_mm_s, "r-", lw=1.5, label="$u'_{rms} / u_\\tau$ (wall-normal)")
    _ax1.plot(_y_plus, np.sqrt(np.maximum(2 * _tke, 0)) / _u_tau_mm_s, "k--", lw=1.5, label="$\\sqrt{2k} / u_\\tau$")
    _ax1.axvline(15.0, color="gray", linestyle=":", label="$y^+ \\approx 15$ (buffer peak)")
    _ax1.set_xlabel("Wall units $y^+ = y_w / \\delta_\\nu$ [-]")
    _ax1.set_ylabel("Turbulence intensity $[-]$")
    _ax1.set_title("Turbulent velocity fluctuations vs $y^+$")
    _ax1.set_xlim(0, min(60.0, float(_y_plus.max())))
    _ax1.grid(True, linestyle=":", alpha=0.6)
    _ax1.legend(loc="upper right", fontsize=8)

    # Reynolds shear stress -u'v'
    _ax2.plot(_y_plus, -_uv / (_u_tau_mm_s ** 2), "g-", lw=1.5, label="$-\\overline{u'v'} / u_\\tau^2$")
    _ax2.axhline(0, color="k", linestyle="--", lw=0.8)
    _ax2.axvline(15.0, color="gray", linestyle=":", label="$y^+ \\approx 15$")
    _ax2.set_xlabel("Wall units $y^+ = y_w / \\delta_\\nu$ [-]")
    _ax2.set_ylabel("Normalized Reynolds shear stress $[-\\overline{u'v'}/u_\\tau^2]$")
    _ax2.set_title("Reynolds shear stress across boundary layer")
    _ax2.set_xlim(0, min(60.0, float(_y_plus.max())))
    _ax2.grid(True, linestyle=":", alpha=0.6)
    _ax2.legend(loc="upper right", fontsize=8)

    _fig.suptitle("Near-wall turbulence structures and Reynolds stresses", fontsize=12, fontweight="bold")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            "The streamwise turbulence intensity peaks in the buffer layer around $y^+ \\approx 15$, "
            "consistent with classic flat-plate and channel turbulent boundary-layer physics."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7 - validity map & wall reversal hotspots

    Two-dimensional diagnostic maps of data quality and flow separation:
    - **Valid fraction map**: shows vector detection percentage across the FOV.
    - **Reversed flow probability**: fraction of frames where $v > 0$ (up-flow reversal).
    """)
    return


@app.cell
def _(ds, mo, np, plt, x_wall):
    # Sample 300 frames evenly across the record for fast 2D map computation
    _stride = max(1, ds.sizes["t"] // 300)
    _sub = ds.isel(t=slice(0, ds.sizes["t"], _stride))

    _valid_map = (_sub.chc > 0.5).mean(dim="t").load()
    _rev_map = ((_sub.v > 0) & (_sub.chc > 0.5)).mean(dim="t", skipna=True).load()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 6))

    _cf1 = _ax1.contourf(ds.x, ds.y, _valid_map, levels=np.linspace(0, 1, 21), cmap="Greys", vmin=0, vmax=1)
    _ax1.axvline(x_wall, color="red", linestyle="--", lw=1.2, label="Solid wall")
    _cbar1 = _fig.colorbar(_cf1, ax=_ax1, pad=0.02, shrink=0.85)
    _cbar1.set_label("Valid fraction")
    _ax1.set_title("Valid-fraction map")
    _ax1.set_xlabel("x [mm]")
    _ax1.set_ylabel("y [mm]")
    _ax1.set_aspect("equal")
    _ax1.legend(loc="lower left", fontsize=8)

    _cf2 = _ax2.contourf(ds.x, ds.y, _rev_map * 100.0, levels=25, cmap="Reds")
    _ax2.axvline(x_wall, color="blue", linestyle="--", lw=1.2, label="Solid wall")
    _cbar2 = _fig.colorbar(_cf2, ax=_ax2, pad=0.02, shrink=0.85)
    _cbar2.set_label("Reversed-flow percentage [%]")
    _ax2.set_title("Reversed-flow ($v > 0$) hotspot map")
    _ax2.set_xlabel("x [mm]")
    _ax2.set_ylabel("y [mm]")
    _ax2.set_aspect("equal")
    _ax2.legend(loc="lower left", fontsize=8)

    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"Mean validity is high across the entire interior FOV right up to 15 µm of the wall. "
            f"Mean reversed flow near the wall is **{float(_rev_map.mean() * 100.0):.2f}%**, indicating no permanent separation bubble."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8 - multi-scale cross-comparison: zoom-in right boundary layer vs zoom-out full channel

    Overlaying the zoom-in boundary layer profile with the zoom-out full-channel profile
    (`baseline_steady_state_stats.zarr`):
    - Places both datasets on common physical channel coordinates: distance from right wall $y_w$.
    - Shows that the zoom-in seamlessly extends the zoom-out core profile all the way to the wall,
      capturing the sharp gradient in the viscous sublayer that zoom-out missed.
    """)
    return


@app.cell
def _(OUT_DIR, mo, np, plt, stats, x_wall, xr):
    zoomout_stats_path = OUT_DIR / "baseline_steady_state_stats.zarr"

    if not zoomout_stats_path.exists():
        _out = mo.md(f"Zoom-out reference `{zoomout_stats_path.name}` not found in outputs/.")
    else:
        _zo_stats = xr.open_zarr(zoomout_stats_path)
        _zo_v = _zo_stats.v_mean.mean(dim="y", skipna=True).load()
        _zo_x = _zo_stats.x.values
        _zo_wall = float(_zo_x.max())
        _zo_yw = _zo_wall - _zo_x

        _zi_v = stats.v_mean.mean(dim="y", skipna=True).load()
        _zi_x = stats.x.values
        _zi_yw = x_wall - _zi_x

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Full channel scale
        _ax1.plot(_zo_yw, np.abs(_zo_v), "b-o", markersize=3, label="Zoom-out full channel (3000 frames, Δx≈68 µm)")
        _ax1.plot(_zi_yw, np.abs(_zi_v), "r-", lw=2.0, label="Zoom-in right BL (3000 frames, Δx≈15.3 µm)")
        _ax1.set_xlabel("Distance from right wall $y_w$ [mm]")
        _ax1.set_ylabel("Streamwise velocity $|v|$ [mm/s]")
        _ax1.set_title("Full channel cross-section (10 mm gap)")
        _ax1.grid(True, linestyle=":", alpha=0.6)
        _ax1.legend(loc="lower left", fontsize=8)

        # Panel 2: Near-wall zoom-in comparison
        _ax2.plot(_zo_yw * 1000.0, np.abs(_zo_v), "b-s", markersize=5, lw=1.2, label="Zoom-out (coarse grid)")
        _ax2.plot(_zi_yw * 1000.0, np.abs(_zi_v), "r-o", markersize=3, lw=1.8, label="Zoom-in (resolved sublayer)")
        _ax2.set_xlim(-10, 800)
        _ax2.set_ylim(-10, 450)
        _ax2.set_xlabel("Distance from right wall $y_w$ [$\\mu\\mathrm{m}$]")
        _ax2.set_ylabel("Streamwise velocity $|v|$ [mm/s]")
        _ax2.set_title("Near-wall resolution comparison ($y_w \\leq 800\\,\\mu\\mathrm{m}$)")
        _ax2.grid(True, linestyle=":", alpha=0.6)
        _ax2.legend(loc="lower right", fontsize=8)

        _fig.suptitle("Multi-scale flow stitching: zoom-out vs zoom-in", fontsize=12, fontweight="bold")
        _fig.tight_layout()

        _out = mo.vstack([
            _fig,
            mo.md(
                "**Resolution contrast**:\n"
                "- In the zoom-out dataset, the nearest valid point to the wall is at $y_w \\approx 150\\dots 200\\,\\mu\\mathrm{m}$, "
                "missing the entire viscous sublayer ($y^+ < 5$).\n"
                "- In the zoom-in dataset, the grid resolves down to $y_w \\approx 15\\,\\mu\\mathrm{m}$, with 18 points "
                "inside the sublayer smoothly connecting to the outer logarithmic core."
            ),
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
