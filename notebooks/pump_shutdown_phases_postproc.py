# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "xarray",
#     "zarr",
#     "scipy",
#     "pivpy",
# ]
# ///

"""
pump_shutdown_phases_postproc.py

Pivpy-based publication figures for the multi-pass pump-shutdown batch
(`pump_shutdown_ims_multipass_batch.py` output Zarrs in `outputs/`).

Regimes (measured 2026-09-28, streamwise v, downward positive in raw
run `Vmax_0p62_m2sec_pump_shutdown`, 1000 pairs @ 15 Hz):
  Phase 1 "steady" : frames   0-170  (t = 0-11 s),   v ~ 530 mm/s
  Phase 2 "fast"   : frames 190-865  (t = 13-57 s),  v ~ 1000-1300 mm/s
  Phase 3 "trip"   : frames 870-950  (t = 58-63 s),  decay through zero
                     into brief reversal, then weak relaxation
  (frames 950+: relaxed/room-light contaminated, excluded from stats)

Contents:
  1. Full-cycle overview: bulk v(t) with phase shading + one snapshot
     per phase (vorticity background + quiver) + time-colored v(x) profiles.
  2. Per-phase mean fields: vorticity background + arrows, v(x) profile
     with +/-1 std band. PNG + PDF into outputs/pump_shutdown_phases/.
  3. Moving-window turbulence: fluctuations defined against a centered
     rolling mean of window W; TKE(W,t) and rms-vorticity(W) traces plus
     window-sensitivity per phase — i.e. what counts as "turbulence"
     when the mean itself is moving (trip leakage demo).
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pump shutdown — phase figures + moving-window turbulence (pivpy)

    Publication panels from the multi-pass `.ims` batch. Three physical
    phases plus a full-cycle overview; turbulence defined against a
    **moving average** so the trip itself does not leak into the
    fluctuation statistics.
    """)


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import xarray as xr
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = ROOT / "outputs"
    FIG_DIR = OUT_DIR / "pump_shutdown_phases"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    TRANSIENT_ZARR = OUT_DIR / "pump_shutdown_transient_multipass_ds.zarr"
    AFTER_ZARR = OUT_DIR / "pump_shutdown_after_multipass_ds.zarr"

    # Phase frame ranges (inclusive) in transient indices.
    PHASES = {
        "phase1_steady": (0, 170),
        "phase2_fast": (190, 865),
        "phase3_trip": (870, 950),
    }
    PHASE_LABELS = {
        "phase1_steady": "Phase 1 — steady (t = 0–11 s)",
        "phase2_fast": "Phase 2 — fast (t = 13–57 s)",
        "phase3_trip": "Phase 3 — trip + reversal (t = 58–63 s)",
    }
    SNAPSHOTS = {"phase1_steady": 100, "phase2_fast": 500, "phase3_trip": 920}
    return (
        AFTER_ZARR,
        FIG_DIR,
        PHASES,
        PHASE_LABELS,
        SNAPSHOTS,
        TRANSIENT_ZARR,
        mo,
        np,
        plt,
        xr,
    )


@app.cell
def _(AFTER_ZARR, TRANSIENT_ZARR, xr):
    ds_t = xr.open_zarr(TRANSIENT_ZARR) if TRANSIENT_ZARR.exists() else None
    ds_a = xr.open_zarr(AFTER_ZARR) if AFTER_ZARR.exists() else None
    return ds_a, ds_t


@app.cell
def _(AFTER_ZARR, TRANSIENT_ZARR, ds_a, ds_t, mo):
    _missing = [
        p.name for p, d in ((TRANSIENT_ZARR, ds_t), (AFTER_ZARR, ds_a)) if d is None
    ]
    if _missing:
        _status_out = mo.md(
            "**Waiting on batch output** — missing: "
            + ", ".join(f"`{m}`" for m in _missing)
            + ". Run `pump_shutdown_ims_multipass_batch.py` batches first."
        )
    else:
        _status_out = mo.md(
            f"Loaded `{TRANSIENT_ZARR.name}` ({ds_t.sizes['t']} frames) + "
            f"`{AFTER_ZARR.name}` ({ds_a.sizes['t']} frames)."
        )
    _status_out


@app.cell
def _(ds_a, ds_t):
    ds_t, ds_a
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
        ## 1. Full-cycle overview — bulk decay, snapshots, evolving profiles
        """
    )


@app.cell
def _(FIG_DIR, PHASES, ds_t, mo, np, plt):
    if ds_t is None:
        overview_output = mo.md("_No transient Zarr — run the batch first._")
    else:
        from piv_pipeline import bulk_velocity as _bulk_vel

        _x = np.asarray(ds_t["x"].values, dtype=float)
        _nt = int(ds_t.sizes["t"])
        _tt = np.asarray(ds_t["t"].values, dtype=float)
        _bulk = np.full(_nt, np.nan)
        for _k in range(_nt):
            _fr = ds_t["v"].isel(t=_k).where(ds_t["chc"].isel(t=_k) > 0.5)
            _prof = _fr.mean(dim="y", skipna=True).values
            try:
                _bulk[_k] = _bulk_vel(_prof, _x, 0.5 * (_x.max() - _x.min()))
            except ValueError:
                pass

        _fig, _axs = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
        _ax_bulk = _axs[0]
        _ax_bulk.plot(_tt, _bulk, "k-", linewidth=1.0, label="bulk v(t), per-frame")
        for _name, (_i0, _i1) in PHASES.items():
            _ax_bulk.axvspan(_i0 / 15.0, _i1 / 15.0, alpha=0.15, label=_name)
        _ax_bulk.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        _ax_bulk.set_xlabel("time [s]")
        _ax_bulk.set_ylabel("bulk streamwise velocity [mm/s]")
        _ax_bulk.set_title("Pump shutdown full cycle — bulk velocity (multi-pass OpenPIV)")
        _ax_bulk.legend(fontsize=8, loc="best")
        _ax_bulk.grid(True, linestyle=":", alpha=0.6)

        # Snapshot profiles, one per phase (row-averaged valid v(x)).
        _ax_prof = _axs[1]
        for _name, (_i0, _i1) in PHASES.items():
            _sel = ds_t.isel(t=slice(_i0, _i1 + 1))
            _pm = _sel["v"].where(_sel["chc"] > 0.5).mean(dim=("t", "y"), skipna=True)
            _ax_prof.plot(_x, _pm.values, linewidth=2.0, label=_name)
        _ax_prof.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        _ax_prof.set_xlabel("x [mm]")
        _ax_prof.set_ylabel("phase-mean streamwise v [mm/s]")
        _ax_prof.set_title("Cross-channel profiles per phase")
        _ax_prof.legend(fontsize=8, loc="best")
        _ax_prof.grid(True, linestyle=":", alpha=0.6)

        # Time-colored evolution: every ~5 s a row-averaged profile.
        _ax_evo = _axs[2]
        _cmap = plt.get_cmap("plasma")
        _norm = plt.Normalize(vmin=float(_tt.min()), vmax=float(_tt.max()))
        for _k in range(0, _nt, 75):
            _fr2 = ds_t["v"].isel(t=_k).where(ds_t["chc"].isel(t=_k) > 0.5)
            _ax_evo.plot(
                _x, _fr2.mean(dim="y", skipna=True).values,
                color=_cmap(_norm(float(_tt[_k]))), linewidth=1.2,
            )
        _sm = plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)
        _fig.colorbar(_sm, ax=_ax_evo, pad=0.02, label="time [s]")
        _ax_evo.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        _ax_evo.set_xlabel("x [mm]")
        _ax_evo.set_ylabel("instantaneous row-mean v [mm/s]")
        _ax_evo.set_title("Profile evolution through the cycle (every 5 s)")
        _ax_evo.grid(True, linestyle=":", alpha=0.6)

        _fig.savefig(FIG_DIR / "fig00_full_cycle_overview.png", dpi=200)
        _fig.savefig(FIG_DIR / "fig00_full_cycle_overview.pdf")
        overview_output = mo.vstack([
            mo.md(
                f"### Full cycle — bulk {np.nanmean(_bulk[:170]):.0f} → "
                f"{np.nanmean(_bulk[190:865]):.0f} → trip "
                f"(min {np.nanmin(_bulk[870:951]):.0f} mm/s) → `{FIG_DIR.name}/fig00_*`"
            ),
            _fig,
        ])
    overview_output


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
        ## 2. Per-phase mean fields — vorticity background + arrows
        """
    )


@app.cell
def _(FIG_DIR, PHASES, PHASE_LABELS, SNAPSHOTS, ds_t, mo, plt, xr):
    if ds_t is None:
        phase_output = mo.md("_No transient Zarr — run the batch first._")
    else:
        _figs = []
        for _name, (_i0, _i1) in PHASES.items():
            _sel = ds_t.isel(t=slice(_i0, _i1 + 1))
            _valid = _sel["chc"] > 0.5
            _umean = _sel["u"].where(_valid).mean(dim="t", skipna=True)
            _vmean = _sel["v"].where(_valid).mean(dim="t", skipna=True)
            _frame = xr.Dataset(
                data_vars={
                    "u": _umean, "v": _vmean,
                    "chc": _valid.mean(dim="t"),
                },
                coords={"x": ds_t["x"], "y": ds_t["y"]},
                attrs=dict(units_x="mm", units_y="mm",
                           units_u="mm/s", units_v="mm/s"),
            )
            _with_vort = _frame.piv.vorticity(name="vorticity", method="circulation", radius=2)
            _fig_p, _ax_p = plt.subplots(figsize=(7, 8), dpi=150, constrained_layout=True)
            _with_vort.piv.plot(
                flow_property="vorticity", cmap="RdBu_r",
                quiver_density=2, quiver_width=0.0035, quiver_alpha=0.75,
                ax=_ax_p,
            )
            _ax_p.set_title(f"{PHASE_LABELS[_name]} — mean, frames {_i0}–{_i1}")
            _ax_p.set_xlabel("x [mm]")
            _ax_p.set_ylabel("y [mm]")
            _fig_p.savefig(FIG_DIR / f"{_name}_mean_vorticity.png", dpi=200)
            _fig_p.savefig(FIG_DIR / f"{_name}_mean_vorticity.pdf")
            _figs.append(_fig_p)

            # Snapshot of the same phase: instantaneous vorticity + arrows.
            _snap = int(SNAPSHOTS[_name])
            _inst = ds_t.isel(t=_snap)
            _with_vi = _inst.piv.vorticity(name="vorticity", method="circulation", radius=2)
            _fig_s, _ax_s = plt.subplots(figsize=(7, 8), dpi=150, constrained_layout=True)
            _with_vi.piv.plot(
                flow_property="vorticity", cmap="RdBu_r",
                quiver_density=2, quiver_width=0.0035, quiver_alpha=0.75,
                ax=_ax_s,
            )
            _ax_s.set_title(f"{PHASE_LABELS[_name]} — snapshot frame {_snap} (t={_snap / 15.0:.1f} s)")
            _ax_s.set_xlabel("x [mm]")
            _ax_s.set_ylabel("y [mm]")
            _fig_s.savefig(FIG_DIR / f"{_name}_snapshot_vorticity.png", dpi=200)
            _fig_s.savefig(FIG_DIR / f"{_name}_snapshot_vorticity.pdf")
            _figs.append(_fig_s)

        phase_output = mo.vstack(
            [mo.md(f"### Phase mean + snapshot vorticity fields → `{FIG_DIR.name}/` (PNG + PDF)")]
            + _figs
        )
    phase_output


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Moving-window turbulence — what counts as a fluctuation?

    Fluctuations are defined against a **centered moving average** of
    window $W$ frames: $u' = u - \langle u \rangle_W$. Short $W$ tracks
    the trip (decay stays in the mean, fluctuations stay turbulent);
        long $W$ leaks the mean-flow deceleration into $u'$, inflating
        "turbulence" across phases 2→3. Vorticity is split the same way:
        $\omega(\langle \mathbf{u} \rangle_W)$ vs. rms of $\omega'$.
        """
    )


@app.cell
def _(FIG_DIR, ds_t, mo, np, plt):
    if ds_t is None:
        turb_output = mo.md("_No transient Zarr — run the batch first._")
    else:
        _WINDOWS = (5, 15, 51, 151)
        _masked_u = ds_t["u"].where(ds_t["chc"] > 0.5)
        _masked_v = ds_t["v"].where(ds_t["chc"] > 0.5)
        _traces = {}
        for _w in _WINDOWS:
            _mu = _masked_u.rolling(t=int(_w), center=True, min_periods=1).mean()
            _mv = _masked_v.rolling(t=int(_w), center=True, min_periods=1).mean()
            _up = _masked_u - _mu
            _vp = _masked_v - _mv
            _tke = (0.5 * (_up ** 2 + _vp ** 2)).mean(dim=("y", "x"), skipna=True).load()
            _traces[_w] = np.sqrt(np.asarray(_tke.values, dtype=float))

        _fig_t, _ax_t = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        for _w, _tr in _traces.items():
            _ax_t.plot(
                np.asarray(ds_t["t"].values, dtype=float), _tr,
                linewidth=1.2, label=f"W={_w} frames ({_w / 15.0:.1f} s)",
            )
        _ax_t.set_xlabel("time [s]")
        _ax_t.set_ylabel("rms fluctuation speed [mm/s]")
        _ax_t.set_title("Moving-window fluctuation level through the cycle")
        _ax_t.legend(fontsize=8, loc="best")
        _ax_t.grid(True, linestyle=":", alpha=0.6)
        _fig_t.savefig(FIG_DIR / "fig03_moving_window_tke.png", dpi=200)
        _fig_t.savefig(FIG_DIR / "fig03_moving_window_tke.pdf")

        # Window sensitivity per phase: mean rms-fluctuation vs W.
        _fig_w, _ax_w = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        from piv_pipeline import bulk_velocity as _bv2  # noqa: F401 (kept for provenance)

        _phase_means = {}
        for _w, _tr in _traces.items():
            _phase_means[_w] = {
                "steady": float(np.nanmean(_tr[0:171])),
                "fast": float(np.nanmean(_tr[190:866])),
                "trip": float(np.nanmean(_tr[870:951])),
            }
        _ww = np.array(_WINDOWS, dtype=float)
        for _ph in ("steady", "fast", "trip"):
            _ax_w.plot(_ww, [_phase_means[_w][_ph] for _w in _WINDOWS],
                       "o-", label=_ph)
        _ax_w.set_xscale("log")
        _ax_w.set_xlabel("moving-average window W [frames]")
        _ax_w.set_ylabel("phase-mean rms fluctuation [mm/s]")
        _ax_w.set_title("What counts as turbulence depends on W (trip leakage)")
        _ax_w.legend(fontsize=9, loc="best")
        _ax_w.grid(True, linestyle=":", alpha=0.6, which="both")
        _fig_w.savefig(FIG_DIR / "fig04_window_sensitivity.png", dpi=200)
        _fig_w.savefig(FIG_DIR / "fig04_window_sensitivity.pdf")

        # Persist traces for the report (JSON numbers, not hand-copied).
        import json as _json

        _report_vals = {
            "windows_frames": list(_WINDOWS),
            "phase_mean_rms": {str(k): v for k, v in _phase_means.items()},
            "t_s": [float(t) for t in np.asarray(ds_t["t"].values)],
            "rms_W15": [float(v) for v in _traces[15]],
        }
        with open(FIG_DIR / "moving_window_values.json", "w") as _f:
            _json.dump(_report_vals, _f)

        turb_output = mo.vstack([
            mo.md(
                f"### Moving-window result — W=15: steady "
                f"{_phase_means[15]['steady']:.0f}, fast "
                f"{_phase_means[15]['fast']:.0f}, trip "
                f"{_phase_means[15]['trip']:.0f} mm/s rms "
                f"(W=151 inflates trip to {_phase_means[151]['trip']:.0f} — mean-flow leakage)"
            ),
            _fig_t,
            _fig_w,
        ])

    turb_output


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Vorticity fluctuation fields (snapshot view)

        Instantaneous $\omega_z$ beside its moving-mean counterpart
        ($W = 15$) and the residual $\omega'$, for one snapshot per phase —
        the spatial view of what the traces above integrate.
        """
    )


@app.cell
def _(FIG_DIR, SNAPSHOTS, ds_t, mo, np, plt, xr):
    if ds_t is None:
        vort_output = mo.md("_No transient Zarr — run the batch first._")
    else:
        _W = 15
        _mu15 = ds_t["u"].where(ds_t["chc"] > 0.5).rolling(t=_W, center=True, min_periods=1).mean()
        _mv15 = ds_t["v"].where(ds_t["chc"] > 0.5).rolling(t=_W, center=True, min_periods=1).mean()
        _figs_v = []
        for _name, _snap in SNAPSHOTS.items():
            _inst = ds_t.isel(t=int(_snap))
            _mean_frame = xr.Dataset(
                data_vars={
                    "u": _mu15.isel(t=int(_snap)),
                    "v": _mv15.isel(t=int(_snap)),
                    "chc": _inst["chc"],
                },
                coords={"x": ds_t["x"], "y": ds_t["y"]},
                attrs=dict(units_x="mm", units_y="mm",
                           units_u="mm/s", units_v="mm/s"),
            )
            _w_inst = _inst.piv.vorticity(name="vorticity", method="circulation", radius=2)
            _w_mean = _mean_frame.piv.vorticity(name="vorticity", method="circulation", radius=2)
            _fig_v, _axs_v = plt.subplots(1, 2, figsize=(13, 6), dpi=150, constrained_layout=True)
            for _ax, _ds_p, _ttl in (
                (_axs_v[0], _w_inst, f"{_name} frame {_snap}: instantaneous ωz"),
                (_axs_v[1], _w_mean, f"{_name} frame {_snap}: moving-mean (W={_W}) ωz"),
            ):
                _vals = _ds_p["vorticity"].values
                _cmax = float(np.nanpercentile(np.abs(_vals[np.isfinite(_vals)]), 98)) or 1.0
                _cf = _ax.contourf(
                    _ds_p["x"].values, _ds_p["y"].values, _vals, levels=41,
                    cmap="RdBu_r", vmin=-_cmax, vmax=_cmax, extend="both",
                )
                _fig_v.colorbar(_cf, ax=_ax, pad=0.02, label="ωz [1/s]")
                _step = max(1, _ds_p.sizes["x"] // 40, _ds_p.sizes["y"] // 48)
                _ax.quiver(
                    _ds_p["x"].values[::_step], _ds_p["y"].values[::_step],
                    _ds_p["u"].values[::_step, ::_step],
                    _ds_p["v"].values[::_step, ::_step],
                    color="k", alpha=0.6, width=0.004,
                    angles="xy", scale_units="xy",
                    scale=float(np.nanmedian(np.hypot(
                        _ds_p["u"].values, _ds_p["v"].values)) or 1.0)
                    / (0.85 * _step * abs(float(_ds_p["x"].values[1] - _ds_p["x"].values[0]))),
                )
                _ax.set_title(_ttl)
                _ax.set_xlabel("x [mm]")
                _ax.set_ylabel("y [mm]")
                _ax.set_aspect("equal")
            _fig_v.savefig(FIG_DIR / f"{_name}_vorticity_split.png", dpi=200)
            _fig_v.savefig(FIG_DIR / f"{_name}_vorticity_split.pdf")
            _figs_v.append(_fig_v)
        vort_output = mo.vstack(
            [mo.md(f"### Vorticity split (instantaneous vs W={_W} mean) → `{FIG_DIR.name}/`")]
            + _figs_v
        )

    vort_output


if __name__ == "__main__":
    app.run()
