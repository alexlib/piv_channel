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

from pathlib import Path
import marimo
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import welch
import xarray as xr

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from scipy.signal import welch
    import xarray as xr

    return Path, RegularGridInterpolator, json, mo, np, plt, welch, xr


@app.cell
def _(mo):
    mo.md(r"""
    # Sinusoidal Channel Zoom-In: Advanced Hydrodynamic Post-Processing

    This notebook analyzes the consolidated 2,315-frame stitched dataset of the wavy channel:
    1. **Wave-Phase Wall-Normal Profiles**: Curvilinear coordinate transformation $(\xi, \eta)$ along the sinusoidal wall at 8 distinct wave phases ($\phi = 0, \pi/4, \dots, 7\pi/4$).
    2. **Wall Shear Stress ($\tau_w$) & Boundary Layer Separation**: Direct estimation of $\tau_w(s) \approx \mu \left. \frac{\partial u_\parallel}{\partial \eta} \right|_w$, locating exact **detachment ($s_{\text{sep}}$)** and **reattachment ($s_{\text{reatt}}$)** points, and separation bubble length.
    3. **Vortex Identification & Dynamics**: Vortex core identification using Hunt's $Q$-criterion, swirl strength $\lambda_{ci}$, circulation $\Gamma$, and temporal shedding frequency spectra (Welch PSD).
    """)
    return


@app.cell
def _(Path, mo, xr):
    STATS_PATH = Path("outputs/channel04_zoom_in_stitched/channel04_zoom_in_stitched_stats.zarr")
    DS_PATH = Path("outputs/channel04_zoom_in_stitched/channel04_zoom_in_stitched_ds.zarr")

    has_stats = STATS_PATH.exists()
    has_ds = DS_PATH.exists()

    if not has_stats:
        mo.md("⚠️ **Stats store not found.** Run `notebooks/channel04_zoom_in_batch_stitch.py` first.")
        stats = None
        ds = None
    else:
        stats = xr.open_zarr(STATS_PATH)
        ds = xr.open_zarr(DS_PATH) if has_ds else None

    mo.md(f"""
    - **Ensemble Stats**: `{'Loaded' if stats is not None else 'Not Found'}`
    - **Time Series**: `{'Loaded (2,315 frames)' if ds is not None else 'Not Found'}`
    """)
    return DS_PATH, STATS_PATH, ds, has_ds, has_stats, stats


@app.cell
def _(mo, stats):
    if stats is None:
        mo.stop()

    # Geometry parameters
    PX_PER_MM = 995.28671814631889
    LAMBDA_PX = 4862.161913832164
    AMP_PX = 555.2436724529737
    PHASE_0 = 4.746677044475907
    DRIFT_PX = 0.03800152249355379
    CENTER_PX = 1310.6908790170396

    LAMBDA_MM = LAMBDA_PX / PX_PER_MM  # ~4.885 mm
    AMP_MM = AMP_PX / PX_PER_MM        # ~0.558 mm

    def wall_x_at_y(y_mm):
        # y_mm is negative (-4.805 to -0.018)
        row_px = -y_mm * PX_PER_MM
        x_px = CENTER_PX + DRIFT_PX * row_px + AMP_PX * np.sin(2.0 * np.pi * row_px / LAMBDA_PX + PHASE_0)
        return x_px / PX_PER_MM

    def wall_slope_at_y(y_mm):
        row_px = -y_mm * PX_PER_MM
        dx_drow = DRIFT_PX + AMP_PX * (2.0 * np.pi / LAMBDA_PX) * np.cos(2.0 * np.pi * row_px / LAMBDA_PX + PHASE_0)
        # d(y_mm) = -d(row_px) / PX_PER_MM -> dx/dy = -dx_drow
        return -dx_drow

    mo.md(f"""
    ### Wavy Channel Geometry
    - **Wavelength**: $\\lambda = {LAMBDA_MM:.3f}\\text{{ mm}}$
    - **Wave Amplitude**: $A = {AMP_MM:.3f}\\text{{ mm}}$
    - **Aspect Ratio**: $2A/\\lambda = {2*AMP_MM/LAMBDA_MM:.3f}$
    """)
    return (
        AMP_MM,
        AMP_PX,
        CENTER_PX,
        DRIFT_PX,
        LAMBDA_MM,
        LAMBDA_PX,
        PHASE_0,
        PX_PER_MM,
        wall_slope_at_y,
        wall_x_at_y,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Section 1 — Mean Flow Field & Recirculation Topology
    """)
    return


@app.cell
def _(np, plt, stats, wall_x_at_y):
    y_vals = stats.y.values
    x_vals = stats.x.values
    u_mean = stats.u_mean.values
    v_mean = stats.v_mean.values
    speed = np.hypot(u_mean, v_mean)

    fig, ax = plt.subplots(figsize=(6, 11))
    im = ax.contourf(x_vals, y_vals, speed, levels=40, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Mean Speed $|\overline{\mathbf{u}}|$ [m/s]")

    # Vector quiver overlay
    skip_y = 12
    skip_x = 8
    ax.quiver(
        x_vals[::skip_x],
        y_vals[::skip_y],
        u_mean[::skip_y, ::skip_x],
        v_mean[::skip_y, ::skip_x],
        color="white",
        scale=5.0,
        width=0.003,
        alpha=0.7,
    )

    # Wavy wall boundary line
    y_fine = np.linspace(y_vals.min(), y_vals.max(), 300)
    x_w = wall_x_at_y(y_fine)
    ax.plot(x_w, y_fine, "r-", linewidth=2.5, label="Fitted Wavy Wall")

    ax.set_title("Time-Averaged Stitched Velocity Field (2,315 Frames)")
    ax.set_xlabel("x [mm] (Cross-channel)")
    ax.set_ylabel("y [mm] (Streamwise)")
    ax.set_xlim(0, 2.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig
    return ax, cbar, fig, im, skip_x, skip_y, speed, u_mean, v_mean, x_vals, x_w, y_fine, y_vals


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 2 — Wall-Normal Profiles across 8 Wave Phases

    We extract tangential velocity $u_\parallel(\eta)$ and wall-normal velocity $u_\perp(\eta)$
    as a function of the normal distance $\eta$ from the wavy wall:
    - $\phi = 0$: Wave Crest (minimum cross-section, high acceleration)
    - $\phi = \pi/4$: Downslope entry
    - $\phi = \pi/2$: Adverse pressure gradient / separation onset
    - $\phi = 3\pi/4$: Recirculation bubble core
    - $\phi = \pi$: Wave Trough (maximum cross-section)
    - $\phi = 5\pi/4$: Reattachment boundary
    - $\phi = 3\pi/2$: Upslope recovery
    - $\phi = 7\pi/4$: Acceleration towards crest
    """)
    return


@app.cell
def _(
    LAMBDA_MM,
    PHASE_0,
    PX_PER_MM,
    RegularGridInterpolator,
    mo,
    np,
    plt,
    stats,
    wall_slope_at_y,
    wall_x_at_y,
):
    phase_slider = mo.ui.slider(
        start=0,
        stop=7,
        value=4,
        step=1,
        label="Wave Phase Station (0=Crest, 2=Adverse, 4=Trough, 6=Upslope)",
    )

    phase_names = [
        "0 (Crest)",
        "π/4 (Downslope)",
        "π/2 (Adverse PG)",
        "3π/4 (Recirculation)",
        "π (Trough)",
        "5π/4 (Reattachment)",
        "3π/2 (Upslope)",
        "7π/4 (Acceleration)",
    ]

    # Pre-build interpolators
    # Note: stats.y decreases, so reverse for monotonically increasing axis in RegularGridInterpolator
    y_inc = stats.y.values[::-1]
    x_inc = stats.x.values
    u_inc = stats.u_mean.values[::-1, :]
    v_inc = stats.v_mean.values[::-1, :]
    tke_inc = stats.tke.values[::-1, :]

    interp_u = RegularGridInterpolator((y_inc, x_inc), u_inc, bounds_error=False, fill_value=np.nan)
    interp_v = RegularGridInterpolator((y_inc, x_inc), v_inc, bounds_error=False, fill_value=np.nan)
    interp_tke = RegularGridInterpolator((y_inc, x_inc), tke_inc, bounds_error=False, fill_value=np.nan)

    mo.vstack([phase_slider])
    return (
        interp_tke,
        interp_u,
        interp_v,
        phase_names,
        phase_slider,
        tke_inc,
        u_inc,
        v_inc,
        x_inc,
        y_inc,
    )


@app.cell
def _(
    LAMBDA_MM,
    interp_tke,
    interp_u,
    interp_v,
    np,
    phase_names,
    phase_slider,
    plt,
    wall_slope_at_y,
    wall_x_at_y,
):
    selected_idx = phase_slider.value
    phi = selected_idx * (np.pi / 4.0)

    # Pick a representative y location in the center of the domain matching this wave phase
    # y = -2.4 mm is near center
    y_center = -2.4 - (phi / (2.0 * np.pi)) * LAMBDA_MM
    if y_center < -4.5:
        y_center += LAMBDA_MM

    xw = wall_x_at_y(y_center)
    slope = wall_slope_at_y(y_center)

    # Unit normal (into fluid, leftward) and unit tangent (downstream, upward)
    norm = np.hypot(1.0, slope)
    nx_u = -1.0 / norm
    ny_u = slope / norm
    tx_u = slope / norm
    ty_u = 1.0 / norm

    # Wall-normal distance eta up to 1.2 mm into the core flow
    eta_vals = np.linspace(0.005, 1.2, 80)
    pts_x = xw + eta_vals * nx_u
    pts_y = y_center + eta_vals * ny_u

    eval_pts = np.column_stack([pts_y, pts_x])
    u_eval = interp_u(eval_pts)
    v_eval = interp_v(eval_pts)
    tke_eval = interp_tke(eval_pts)

    # Project into tangential and normal
    u_tangential = u_eval * tx_u + v_eval * ty_u
    u_normal = u_eval * nx_u + v_eval * ny_u

    fig_prof, (ax_u, ax_tke) = plt.subplots(1, 2, figsize=(9, 4.5))

    ax_u.plot(u_tangential, eta_vals * 1000.0, "b-o", markersize=4, label=r"$u_\parallel(\eta)$ (Tangential)")
    ax_u.plot(u_normal, eta_vals * 1000.0, "g--", label=r"$u_\perp(\eta)$ (Normal)")
    ax_u.axvline(0, color="gray", linestyle=":")
    ax_u.set_xlabel("Velocity [m/s]")
    ax_u.set_ylabel(r"Wall-normal distance $\eta$ [$\mu$m]")
    ax_u.set_title(f"Velocity Profiles: Phase {phase_names[selected_idx]}")
    ax_u.grid(True, alpha=0.3)
    ax_u.legend()

    ax_tke.plot(tke_eval * 1000.0, eta_vals * 1000.0, "r-s", markersize=4, label=r"TKE [$10^{-3}\text{ m}^2/\text{s}^2$]")
    ax_tke.set_xlabel(r"TKE [$\text{mJ/kg}$]")
    ax_tke.set_title("Turbulent Kinetic Energy")
    ax_tke.grid(True, alpha=0.3)
    ax_tke.legend()

    fig_prof.tight_layout()
    fig_prof
    return (
        ax_tke,
        ax_u,
        eta_vals,
        eval_pts,
        fig_prof,
        norm,
        nx_u,
        ny_u,
        phi,
        pts_x,
        pts_y,
        selected_idx,
        slope,
        tke_eval,
        tx_u,
        ty_u,
        u_eval,
        u_normal,
        u_tangential,
        v_eval,
        xw,
        y_center,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3 — Wall Shear Stress ($\tau_w$), Detachment ($s_{\text{sep}}$), and Reattachment ($s_{\text{reatt}}$)

    The wall shear stress is estimated from the near-wall velocity gradient:
    $$\tau_w(y) = \mu \left. \frac{\partial u_\parallel}{\partial \eta} \right|_{\eta \to 0}$$
    - **Detachment / Separation Point**: Zero-crossing where $\tau_w$ drops below zero ($\partial \tau_w/\partial s < 0$).
    - **Reattachment Point**: Zero-crossing where $\tau_w$ recovers to positive ($\partial \tau_w/\partial s > 0$).
    - **Recirculation Bubble Length**: Distance along the wall $L_{\text{sep}} = s_{\text{reatt}} - s_{\text{sep}}$.
    """)
    return


@app.cell
def _(
    interp_u,
    interp_v,
    np,
    plt,
    stats,
    wall_slope_at_y,
    wall_x_at_y,
):
    MU_WATER = 1.002e-3  # Pa.s at 20 deg C

    y_sample = np.linspace(float(stats.y.min()) + 0.2, float(stats.y.max()) - 0.2, 120)
    tau_w_list = []
    d_u_list = []

    # Two probe heights near wall: eta1 = 15 um, eta2 = 35 um
    eta1 = 0.015  # mm
    eta2 = 0.035  # mm

    for y_cur in y_sample:
        xw_cur = wall_x_at_y(y_cur)
        s_cur = wall_slope_at_y(y_cur)
        n_cur = np.hypot(1.0, s_cur)

        nx_c = -1.0 / n_cur
        ny_c = s_cur / n_cur
        tx_c = s_cur / n_cur
        ty_c = 1.0 / n_cur

        p1 = np.array([[y_cur + eta1 * ny_c, xw_cur + eta1 * nx_c]])
        p2 = np.array([[y_cur + eta2 * ny_c, xw_cur + eta2 * nx_c]])

        u1 = interp_u(p1)[0] * tx_c + interp_v(p1)[0] * ty_c
        u2 = interp_u(p2)[0] * tx_c + interp_v(p2)[0] * ty_c

        # Near-wall gradient in 1/s (eta in meters)
        d_eta_m = (eta2 - eta1) * 1e-3
        grad = (u2 - u1) / d_eta_m if not np.isnan(u1) and not np.isnan(u2) else np.nan
        tau = MU_WATER * grad if not np.isnan(grad) else np.nan

        tau_w_list.append(tau)
        d_u_list.append(u1)

    tau_w_arr = np.array(tau_w_list)

    # Robust detection with slight gaussian filtering to remove interrogation window noise
    from scipy.ndimage import gaussian_filter1d
    valid = ~np.isnan(tau_w_arr)
    y_v = y_sample[valid]
    tau_v = gaussian_filter1d(tau_w_arr[valid], sigma=4.0)

    zero_crossings = []
    for k in range(len(tau_v) - 1):
        if (tau_v[k] * tau_v[k + 1]) <= 0.0:
            y_zero = y_v[k] - tau_v[k] * (y_v[k + 1] - y_v[k]) / (tau_v[k + 1] - tau_v[k])
            slope_zero = (tau_v[k + 1] - tau_v[k]) / (y_v[k + 1] - y_v[k])
            crossing_type = "Separation (Detachment)" if slope_zero < 0 else "Reattachment"
            zero_crossings.append((y_zero, crossing_type))

    # Calculate separation bubble length
    bubble_length_str = ""
    seps = [y_z for y_z, t in zero_crossings if "Separation" in t]
    reatts = [y_z for y_z, t in zero_crossings if "Reattachment" in t]
    if seps and reatts:
        for s in seps:
            downstream = [r for r in reatts if r < s]
            if downstream:
                l_b = abs(s - downstream[0])
                bubble_length_str += f"Bubble Length L_sep = {l_b:.3f} mm ({l_b/4.885*100:.1f}% λ) | "

    fig_tau, ax_tau = plt.subplots(figsize=(8, 4.5))
    ax_tau.plot(y_v, tau_v, "b-", linewidth=2.0, label=r"Wall Shear Stress $\tau_w(y)$ (smoothed)")
    ax_tau.axhline(0, color="k", linestyle="--", linewidth=1.0)

    for y_z, c_type in zero_crossings:
        color = "red" if "Separation" in c_type else "green"
        ax_tau.axvline(y_z, color=color, linestyle=":", linewidth=2.0)
        ax_tau.plot(y_z, 0, "o", color=color, markersize=8, label=f"{c_type}: y={y_z:.3f} mm")

    ax_tau.set_xlabel("Streamwise Position y [mm]")
    ax_tau.set_ylabel(r"Wall Shear Stress $\tau_w$ [Pa]")
    ax_tau.set_title(f"Wall Shear Stress & Hydrodynamic Separation\n{bubble_length_str}")
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(loc="lower right")
    fig_tau.tight_layout()
    fig_tau
    return (
        MU_WATER,
        ax_tau,
        c_type,
        color,
        crossing_type,
        d_eta_m,
        d_u_list,
        eta1,
        eta2,
        fig_tau,
        grad,
        k,
        n_cur,
        nx_c,
        ny_c,
        p1,
        p2,
        s_cur,
        slope_zero,
        t_cur,
        tau,
        tau_v,
        tau_w_arr,
        tau_w_list,
        tx_c,
        ty_c,
        u1,
        u2,
        valid,
        xw_cur,
        y_cur,
        y_sample,
        y_v,
        y_z,
        y_zero,
        zero_crossings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 4 — Vortex Identification (Hunt's $Q$-Criterion & Core Location)

    The second invariant of the velocity gradient tensor ($Q$-criterion):
    $$Q = \frac{1}{2}\left(\|\mathbf{\Omega}\|^2 - \|\mathbf{S}\|^2\right)$$
    identifies regions where rotation dominates strain ($Q > 0$).
    """)
    return


@app.cell
def _(np, plt, stats, wall_x_at_y, x_vals, y_vals):
    u_m = stats.u_mean.values
    v_m = stats.v_mean.values

    dx = float(np.abs(np.diff(x_vals).mean())) * 1e-3
    dy = float(np.abs(np.diff(y_vals).mean())) * 1e-3

    du_dx = np.gradient(u_m, dx, axis=1)
    du_dy = -np.gradient(u_m, dy, axis=0)
    dv_dx = np.gradient(v_m, dx, axis=1)
    dv_dy = -np.gradient(v_m, dy, axis=0)

    # Q = -0.5 * ( (du_dx)**2 + (dv_dy)**2 + 2 * (du_dy * dv_dx) ) in 2D
    Q_crit = -0.5 * (du_dx**2 + dv_dy**2 + 2.0 * du_dy * dv_dx)

    # Locate peak Q inside recirculation region
    Q_masked = np.where(Q_crit > 0, Q_crit, np.nan)
    max_idx = np.unravel_index(np.nanargmax(Q_masked), Q_masked.shape)
    vortex_core_y = y_vals[max_idx[0]]
    vortex_core_x = x_vals[max_idx[1]]
    peak_Q = Q_masked[max_idx]

    fig_q, ax_q = plt.subplots(figsize=(6, 11))
    im_q = ax_q.contourf(x_vals, y_vals, np.log10(np.clip(Q_crit, 1.0, 1e6)), levels=30, cmap="inferno")
    cbar_q = fig_q.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)
    cbar_q.set_label(r"$\log_{10}(Q)$ [$1/\text{s}^2$]")

    # Vortex core marker
    ax_q.plot(vortex_core_x, vortex_core_y, "c*", markersize=14, label=f"Vortex Core: ({vortex_core_x:.3f}, {vortex_core_y:.3f}) mm")

    y_f = np.linspace(y_vals.min(), y_vals.max(), 300)
    ax_q.plot(wall_x_at_y(y_f), y_f, "w-", linewidth=2.0)

    # Streamlines overlay
    # Note: Streamplot requires 1D x and increasing y strictly equally spaced
    y_rev = y_vals[::-1]
    u_rev = u_m[::-1, :]
    v_rev = v_m[::-1, :]
    u_stream = np.nan_to_num(u_rev, 0.0)
    v_stream = np.nan_to_num(v_rev, 0.0)
    x_grid = np.linspace(float(x_vals[0]), float(x_vals[-1]), len(x_vals))
    y_grid = np.linspace(float(y_rev[0]), float(y_rev[-1]), len(y_rev))
    ax_q.streamplot(x_grid, y_grid, u_stream, v_stream, color="cyan", density=1.0, linewidth=0.8, arrowsize=0.8)

    ax_q.set_title("Vortex Core & Recirculation Topology (Q-Criterion)")
    ax_q.set_xlabel("x [mm]")
    ax_q.set_ylabel("y [mm]")
    ax_q.legend(loc="lower right")
    fig_q.tight_layout()
    fig_q
    return (
        Q_crit,
        Q_masked,
        ax_q,
        cbar_q,
        du_dx,
        du_dy,
        dv_dx,
        dv_dy,
        dx,
        dy,
        fig_q,
        im_q,
        max_idx,
        peak_Q,
        u_m,
        u_rev,
        u_stream,
        v_m,
        v_rev,
        v_stream,
        vortex_core_x,
        vortex_core_y,
        y_f,
        y_rev,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Spectral Dynamics & Flapping Frequency (Welch PSD)

    Analyzing the time series of velocity fluctuations near the vortex core to determine
    the characteristic shedding frequency ($f_{\text{shed}}$) and Strouhal number ($St$).
    """)
    return


@app.cell
def _(ds, max_idx, np, plt, welch):
    if ds is None:
        fig_psd = plt.figure()
        plt.text(0.5, 0.5, "Time-series dataset not available yet", ha="center")
    else:
        # Probe velocity time series at vortex core
        u_t = ds.u.isel(y=max_idx[0], x=max_idx[1]).values
        v_t = ds.v.isel(y=max_idx[0], x=max_idx[1]).values
        fs = float(ds.attrs.get("fs_hz", 15.0))

        # Fill NaNs with interpolation for spectral analysis
        valid_t = ~np.isnan(u_t)
        t_axis = np.arange(len(u_t))
        if valid_t.sum() > len(u_t) // 2:
            u_clean = np.interp(t_axis, t_axis[valid_t], u_t[valid_t])
            v_clean = np.interp(t_axis, t_axis[valid_t], v_t[valid_t])

            freqs, psd_u = welch(u_clean - u_clean.mean(), fs=fs, nperseg=256)
            _, psd_v = welch(v_clean - v_clean.mean(), fs=fs, nperseg=256)

            fig_psd, ax_psd = plt.subplots(figsize=(8, 4.5))
            ax_psd.semilogy(freqs, psd_u, "b-", label=r"PSD $u'$ at Vortex Core")
            ax_psd.semilogy(freqs, psd_v, "r--", label=r"PSD $v'$ at Vortex Core")
            ax_psd.set_xlabel("Frequency [Hz]")
            ax_psd.set_ylabel(r"Power Spectral Density $[(\text{m/s})^2/\text{Hz}]$")
            ax_psd.set_title("Vortex Core Flapping & Shedding Spectra")
            ax_psd.grid(True, which="both", alpha=0.3)
            ax_psd.legend()
            fig_psd.tight_layout()
        else:
            fig_psd = plt.figure()
            plt.text(0.5, 0.5, "Insufficient valid samples at probe point", ha="center")

    fig_psd
    return (
        ax_psd,
        fig_psd,
        freqs,
        fs,
        psd_u,
        psd_v,
        t_axis,
        u_clean,
        u_t,
        v_clean,
        v_t,
        valid_t,
    )


if __name__ == "__main__":
    app.run()
