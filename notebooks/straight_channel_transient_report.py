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

"""Report figures for the straight-channel transient - LaTeX-ready.

Same analysis as `straight_channel_transient.py`, but rendering to
`outputs/report_figs/*.pdf` + `*.png` for inclusion in a manuscript, with
values also emitted to `outputs/report_values.json` so the LaTeX template
can `\input` real numbers instead of hand-copied ones.

Draft figures: caption text lives in the LaTeX file, not here.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Straight-channel transient - report figures

        Regenerates every figure/table for the LaTeX report straight from the
        stored zarr (fast, no PIV re-run), and writes the numbers to
        `outputs/report_values.json` consumed by
        `outputs/straight_channel_transient_report.tex`.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import json

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import pivpy  # noqa: F401 (registers the .piv xarray accessor)

    import piv_pipeline as pp

    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "legend.fontsize": 7,
        "axes.titlesize": 9, "figure.constrained_layout.use": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = ROOT / "outputs"
    FIG_DIR = OUT_DIR / "report_figs"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    return FIG_DIR, FIG_DIR, json, mpl, mo, np, OUT_DIR, plt, pp, ROOT, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Data
        """
    )
    return


@app.cell
def _(OUT_DIR, xr):
    ds = xr.open_zarr(OUT_DIR / "baseline_pump_shutdown_ds.zarr")
    ds = ds.assign_coords(time_s=("t", ds.t.values / 15.0))
    steady_stats = xr.open_zarr(OUT_DIR / "baseline_steady_state_stats.zarr")
    ds
    return ds, steady_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Derived quantities (shared `piv_pipeline` helpers)")
    return


@app.cell
def _(ds, np, pp):
    x = np.asarray(ds.x.values, dtype=float)
    xc, b, x_cc = pp.channel_center(x)
    bulk_t = np.array([
        pp.bulk_velocity(v, x, b)
        for v in ds.v.where(ds.chc > 0.5).mean(dim="y", skipna=True).values
    ])
    tke_ds = pp.spatial_tke(ds, window=5)
    return b, bulk_t, tke_ds, x, x_cc, xc


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Fig. 1 - time series: bulk velocity, speed, validity, reversal
        """
    )
    return


@app.cell
def _(FIG_DIR, bulk_t, ds, np, plt):
    _speed = np.hypot(ds.u, ds.v).where(ds.chc > 0.5)
    mean_speed_t = _speed.mean(dim=("y", "x"), skipna=True).load()
    valid_t = (ds.chc > 0.5).mean(dim=("y", "x")).load()
    rev_t = (ds.v.where(ds.chc > 0.5) > 0).mean(dim=("y", "x"), skipna=True).load()

    _fig, _axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    _axs[0, 0].plot(ds.time_s, mean_speed_t, color="k")
    _axs[0, 0].set_ylabel(r"$\langle|\mathbf{u}|\rangle$ [mm s$^{-1}$]")
    _axs[0, 1].plot(ds.time_s, bulk_t, color="C0")
    _axs[0, 1].set_ylabel(r"$U_{\mathrm{bulk}}$ [mm s$^{-1}$]")
    _axs[1, 0].plot(ds.time_s, valid_t, color="C2")
    _axs[1, 0].set_ylabel("valid fraction")
    _axs[1, 1].plot(ds.time_s, rev_t, color="C3")
    _axs[1, 1].set_ylabel(r"reversed fraction ($v>0$)")
    for _a in _axs[1]:
        _a.set_xlabel("time [s]")
    for _a in (_axs[0, 0], _axs[0, 1]):
        _a.set_xticklabels([])
    for _a in _axs.flat:
        _a.set_xlim(ds.time_s.min(), ds.time_s.max())
    _fig.suptitle("Straight channel, after pump shutdown: time series "
                 "(100 maps at 15 Hz)")
    _fig.savefig(FIG_DIR / "fig01_timeseries.png")
    _fig.savefig(FIG_DIR / "fig01_timeseries.pdf")
    plt.close(_fig)
    return mean_speed_t, rev_t, valid_t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Fig. 2 - cross-channel profiles: dimensional + normalized
        """
    )
    return


@app.cell
def _(FIG_DIR, b, bulk_t, ds, mpl, np, plt, x_cc):
    prof = ds.v.where(ds.chc > 0.5).rolling(
        t=5, center=True, min_periods=1).mean().mean(dim="y", skipna=True).load()
    idx = np.unique(np.linspace(0, ds.sizes["t"] - 1, 10).round().astype(int))
    ub = float(np.mean(bulk_t))
    norm = mpl.colors.Normalize(ds.time_s.min().item(), ds.time_s.max().item())
    cmap = plt.get_cmap("plasma")

    _fig, _axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for _k in idx:
        _axs[0].plot(x_cc, prof.isel(t=_k).values, color=cmap(norm(ds.time_s[_k])),
                     lw=1.2)
        _axs[1].plot(x_cc / b, prof.isel(t=_k).values / ub,
                     color=cmap(norm(ds.time_s[_k])), lw=1.2)
    _axs[0].set_xlabel("x - centre [mm]")
    _axs[0].set_ylabel(r"$V$ [mm s$^{-1}$]")
    _axs[0].set_title("dimensional")
    _axs[1].set_xlabel(r"$x/b$")
    _axs[1].set_ylabel(r"$V/U_{\mathrm{bulk}}$ [-]")
    _axs[1].set_title("normalized")
    _sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    _cb = _fig.colorbar(_sm, ax=_axs, pad=0.02, fraction=0.03)
    _cb.set_label("time [s]")
    _fig.suptitle(f"Row-averaged streamwise profiles (5-frame smoothed); "
                 f"$b$={b:.2f} mm, $U_{{bulk}}$={ub:.0f} mm s$^{{-1}}$")
    _fig.savefig(FIG_DIR / "fig02_profiles.png")
    _fig.savefig(FIG_DIR / "fig02_profiles.pdf")
    plt.close(_fig)
    return prof, ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Fig. 3 - spatial TKE about the row-mean profile
        """
    )
    return


@app.cell
def _(FIG_DIR, ds, plt, tke_ds):
    _fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(tke_ds.time_s, tke_ds.tke, color="k", lw=1.4,
            label=r"TKE $=\langle\frac{1}{2}(u'^2+v'^2)\rangle$")
    ax.plot(tke_ds.time_s, tke_ds.u_var, lw=1.0, alpha=0.8,
            label=r"$\langle u'^2\rangle/2$")
    ax.plot(tke_ds.time_s, tke_ds.v_var, lw=1.0, alpha=0.8,
            label=r"$\langle v'^2\rangle/2$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"TKE [mm$^2$ s$^{-2}$]")
    ax.legend()
    ax.set_title("Spatial fluctuations about the row-mean profile "
                 "(5-frame smoothed)")
    _fig.savefig(FIG_DIR / "fig03_tke.png")
    _fig.savefig(FIG_DIR / "fig03_tke.pdf")
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Fig. 4 - steady vs. after-shutdown comparison
        """
    )
    return


@app.cell
def _(FIG_DIR, b, ds, np, plt, pp, steady_stats):
    s_prof = steady_stats.v_mean.mean(dim="y", skipna=True).load()
    t_prof = ds.v.where(ds.chc > 0.5).mean(dim=("t", "y"), skipna=True).load()
    _fig, _axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for _prof, _xx, _label in ((s_prof, steady_stats.x.values, "steady"),
                               (t_prof, ds.x.values, "after shutdown")):
        xc_, b_, xcc = pp.channel_center(np.asarray(_xx, dtype=float))
        ub_ = pp.bulk_velocity(_prof.values, np.asarray(_xx, dtype=float), b_)
        _axs[0].plot(xcc, _prof.values, label=f"{_label} "
                  rf"($U_{{\rm bulk}}$={ub_:.0f} mm s$^{{-1}}$)")
        _axs[1].plot(xcc / b_, _prof.values / ub_, label=_label)
    for _a in _axs:
        _a.axhline(0, color="k", lw=0.7, ls="--")
        _a.legend()
    _axs[0].set_xlabel("x - centre [mm]")
    _axs[0].set_ylabel(r"$V$ [mm s$^{-1}$]")
    _axs[0].set_title("dimensional")
    _axs[1].set_xlabel(r"$x/b$")
    _axs[1].set_ylabel(r"$V/U_{\mathrm{bulk}}$ [-]")
    _axs[1].set_title("normalized")
    _fig.suptitle("Cross-channel profile: steady state vs. after pump shutdown")
    _fig.savefig(FIG_DIR / "fig04_steady_vs_shutdown.png")
    _fig.savefig(FIG_DIR / "fig04_steady_vs_shutdown.pdf")
    plt.close(_fig)
    return s_prof, t_prof


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Fig. 5 - validity map, reversal map, quiver example
        """
    )
    return


@app.cell
def _(FIG_DIR, ds, np, plt):
    valid_map = (ds.chc > 0.5).mean(dim="t").load()
    rev_map = ((ds.v > 0) & (ds.chc > 0.5)).mean(dim="t", skipna=True).load()
    u_mean = ds.u.where(ds.chc > 0.5).mean(dim="t", skipna=True).load()
    v_mean = ds.v.where(ds.chc > 0.5).mean(dim="t", skipna=True).load()
    _vv = ds.v.where(ds.chc > 0.5).values
    vnorm = mpl.colors.TwoSlopeNorm(
        vmin=float(np.nanpercentile(_vv, 1)),
        vcenter=float(np.nanmean(_vv)),
        vmax=float(np.nanpercentile(_vv, 99)),
    )
    _sp = np.hypot(ds.u, ds.v).where(ds.chc > 0.5).values
    _snorm = mpl.colors.Normalize(
        vmin=float(np.nanpercentile(_sp, 2)), vmax=float(np.nanpercentile(_sp, 98)))

    _fig, _axs = plt.subplots(1, 3, figsize=(7.4, 5.2))
    _cf = _axs[0].contourf(ds.x, ds.y, valid_map, levels=np.linspace(0, 1, 11),
                         cmap="Greys", vmin=0, vmax=1, extend="both")
    _fig.colorbar(_cf, ax=_axs[0], fraction=0.046, pad=0.03,
                 label="valid fraction")
    _axs[0].set_title("(a) vector validity")
    _cf = _axs[1].contourf(ds.x, ds.y, rev_map, levels=40, cmap="Reds",
                         extend="max")
    _fig.colorbar(_cf, ax=_axs[1], fraction=0.046, pad=0.03,
                 label=r"reversed fraction")
    step = max(1, ds.sizes["x"] // 22)
    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    _axs[1].quiver(X[::4, ::step], Y[::4, ::step],
                  u_mean.values[::4, ::step], v_mean.values[::4, ::step],
                  angles="xy", color="k", width=0.003, alpha=0.5)
    _axs[1].set_title("(b) reversal map + mean flow")
    fr = ds.isel(t=0)
    _axs[2].contourf(ds.x, ds.y, np.hypot(fr.u, fr.v).where(fr.chc > 0.5),
                    levels=40, cmap="Greys", norm=snorm)
    _Q = _axs[2].quiver(
        X[::2, ::step], Y[::2, ::step],
        fr.u.where(fr.chc > 0.5).values[::2, ::step],
        fr.v.where(fr.chc > 0.5).values[::2, ::step],
        fr.v.where(fr.chc > 0.5).values[::2, ::step],
        cmap="coolwarm", norm=vnorm, angles="xy", width=0.004)
    _fig.colorbar(_Q, ax=_axs[2], fraction=0.046, pad=0.03,
                 label=r"$v$ [mm s$^{-1}$]")
    _axs[2].set_title(r"(c) frame 0: $v$-coloured arrows")
    for _a in _axs:
        _a.set_xlabel("x [mm]")
        _a.set_aspect("equal")
    _axs[0].set_ylabel("y [mm]")
    _fig.savefig(FIG_DIR / "fig05_maps.png")
    _fig.savefig(FIG_DIR / "fig05_maps.pdf")
    plt.close(_fig)
    return rev_map, u_mean, valid_map, v_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Numbers for the LaTeX template
        """
    )
    return


@app.cell
def _(FIG_DIR, OUT_DIR, b, bulk_t, ds, json, mean_speed_t, np,
      rev_map, rev_t, tke_ds, u_mean, valid_map, valid_t, x, xc):
    sp_ref = float(np.nanmean(np.hypot(ds.u, ds.v).where(ds.chc > 0.5)))
    values = {
        "n_frames": int(ds.sizes["t"]),
        "frame_rate_hz": 15.0,
        "duration_s": float(ds.time_s.max().item()),
        "pulse_dt_s": 8.0e-5,
        "px_per_mm": 173.41900170673784,
        "grid_nx": int(ds.sizes["x"]),
        "grid_ny": int(ds.sizes["y"]),
        "center_mm": float(xc),
        "half_width_mm": float(b),
        "channel_gap_mm": float(2 * b),
        "mean_speed_mm_s": float(mean_speed_t.mean()),
        "mean_speed_std_mm_s": float(mean_speed_t.std()),
        "mean_speed_rel_std": float(mean_speed_t.std() / mean_speed_t.mean()),
        "bulk_velocity_mm_s": float(np.mean(bulk_t)),
        "bulk_velocity_rel_std": float(np.std(bulk_t) / np.mean(bulk_t)),
        "valid_fraction_mean": float(valid_t.mean()),
        "reversed_fraction_mean": float(rev_t.mean()),
        "reversed_fraction_max": float(rev_t.max()),
        "tke_mean_mm2_s2": float(tke_ds.tke.mean()),
        "tke_min_mm2_s2": float(tke_ds.tke.min()),
        "tke_max_mm2_s2": float(tke_ds.tke.max()),
        "u_var_mean_mm2_s2": float(tke_ds.u_var.mean()),
        "v_var_mean_mm2_s2": float(tke_ds.v_var.mean()),
        "profile_centerline_ratio": float(
            (ds.v.where(ds.chc > 0.5).mean(dim=("t", "y"), skipna=True).max()
             / np.mean(bulk_t))),
        "reversal_hotspot_max": float(rev_map.max()),
        "reversal_hotspot_x_mm": float(ds.x.values[
            int(np.unravel_index(int(np.argmax(rev_map)), rev_map.shape)[1])]),
        "reversal_hotspot_y_mm": float(ds.y.values[
            int(np.unravel_index(int(np.argmax(rev_map)), rev_map.shape)[0])]),
        "valid_map_min": float(valid_map.min()),
        "valid_map_median": float(np.median(valid_map)),
        "mean_v_centre_mm_s": float(
            ds.v.where(ds.chc > 0.5).mean(dim=("t", "y"), skipna=True)
            .sel(x=float(xc), method="nearest")),
    }
    out_json = OUT_DIR / "report_values.json"
    out_json.write_text(json.dumps(values, indent=2), encoding="utf-8")
    mo.md(f"Wrote `{out_json.name}` ({len(values)} entries) and "
          f"{len(list(FIG_DIR.glob('*.pdf')))} PDF figures.")
    return out_json, values


if __name__ == "__main__":
    app.run()
