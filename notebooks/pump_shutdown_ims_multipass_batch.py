# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "openpiv",
#     "xarray",
#     "zarr",
#     "scipy",
#     "pivpy",
#     "readim7",
# ]
# ///

"""
pump_shutdown_ims_multipass_batch.py

Multi-pass OpenPIV batch for the two straight-channel pump-shutdown `.ims`
streams, read directly from disk (no DaVis `.im7` export):

- Set A (transient): `Vmax_0p62_m2sec_pump_shutdown/Camera1-1.ims`,
  1000 dual-frame pairs @ 15 Hz, 2048x2432, dt = 80 us.
  Flow starts at full speed (~0.6 m/s) and decays to near rest, so one
  fixed single-pass window cannot cover the whole record: early frames
  need a large search range, late frames need small windows for the
  creeping flow. A 3-pass window-deformation schedule (128 -> 64 -> 32)
  handles both ends.
- Set B (post-shutdown reference): `Vmax_after_pump_shutdown/Camera1-1.ims`,
  100 dual-frame pairs @ 15 Hz, same camera/dt.

Results are stored as pivpy-canonical `xarray` Datasets
(dims `t, y, x`; vars `u, v, chc` in mm/s) in chunked Zarr stores under
`outputs/`, ready for post-analysis (`reynolds_decomposition`,
`profile_evolution`, TKE) and the LaTeX report.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Pump shutdown — multi-pass `.ims` batch (transient + post-shutdown)

        Direct-from-stream OpenPIV with **window deformation multi-pass**
        (`openpiv.windef.simple_multipass`). The transient record spans a
        large dynamic range, but NOT as a simple fast→slow decay — measured
        single-pass survey (2026-09-28, `dt = 80 us`, `173.42 px/mm`):

        - `t = 0–11 s` (frames 0–170): steady `~530 mm/s` downward;
        - `t ≈ 11.5–12.5 s` (frames ~172–185): abrupt jump to `~1100–1300 mm/s`
          (pump speed-up / valve event — *not* the shutdown);
        - `t = 12.5–57 s` (frames ~190–865): sustained fast `~1000 mm/s`;
        - `t ≈ 58–60 s` (frames ~870–900): true pump trip — rapid decay
          through zero into brief reversal (`~-360 mm/s`), then weak
          relaxation; room lights come on by frame 999.

        One fixed single-pass window cannot cover `~7 px → ~18 px → 0 px`
        displacements plus reversal, hence the 3-pass deformation schedule
        (128 -> 64 -> 32). Earlier notebook comments placing the shutdown at
        `t ≈ 12 s` are wrong — that is the speed-up; the trip is at `t ≈ 58.5 s`.

        ### Runs
        1. **Preview** — multi-pass on one fast + one slow frame, check
           valid-fraction and cross-channel profiles before committing.
        2. **Batch A** — full 1000-pair transient → `outputs/pump_shutdown_transient_multipass_ds.zarr`
        3. **Batch B** — 100-pair post-shutdown reference → `outputs/pump_shutdown_after_multipass_ds.zarr`
        4. **Post-analysis** — pivpy stats + bulk-velocity decay trace for the report.
        """
    )


@app.cell
def _():
    from pathlib import Path

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import xarray as xr
    from readim7 import ims as ims_mod
    import openpiv.preprocess as opprep
    import openpiv.windef as windef
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    # Either the Windows acquisition path or the macOS-mounted drive.
    _CANDIDATES = [
        Path("/Volumes/Yossi's_Hard_Drive/channel_flow_research/baseline_channel"),
        Path(r"D:\channel_flow_research\baseline_channel"),
    ]
    BASE = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
    SHUTDOWN_SET = BASE / "Vmax_0p62_m2sec_pump_shutdown"
    AFTER_SET = BASE / "Vmax_after_pump_shutdown"

    # Physical scaling: DaVis Calibration.xml PixelPerMmFactor for this
    # camera/lens (LinearScale 0.00576638 mm/px). Laser pulse separation
    # from the buffer metadata channel "Reference time dt" (FrameDt0 = 80 us
    # on the post-shutdown .vc7 files). Acquisition rate of the stream.
    PX_PER_MM = 173.41900170673784
    LASER_DT = 80.0e-6
    ACQ_HZ = 15.0
    CROP_COLS = (550, 2050)

    return (
        ACQ_HZ,
        AFTER_SET,
        BASE,
        CROP_COLS,
        LASER_DT,
        OUT_DIR,
        PX_PER_MM,
        ROOT,
        SHUTDOWN_SET,
        ims_mod,
        mo,
        np,
        opprep,
        plt,
        time,
        windef,
        xr,
    )


@app.cell
def _(AFTER_SET, SHUTDOWN_SET, ims_mod):
    shutdown_info = ims_mod.ims_info(str(SHUTDOWN_SET))
    after_info = ims_mod.ims_info(str(AFTER_SET))
    return after_info, shutdown_info


@app.cell
def _(AFTER_SET, SHUTDOWN_SET, after_info, mo, shutdown_info):
    mo.md(
        f"### Stream sets found\n"
        f"- **Transient**: `{SHUTDOWN_SET.name}` — **{shutdown_info['n_pairs']}** pairs, "
        f"{shutdown_info['ny']}x{shutdown_info['nx']} px\n"
        f"- **Post-shutdown**: `{AFTER_SET.name}` — **{after_info['n_pairs']}** pairs, "
        f"{after_info['ny']}x{after_info['nx']} px\n"
        f"- Timing: `dt = 80 us` (pulse separation), frame rate `15 Hz` "
        f"(`t = idx / 15`, transient spans `66.7 s`)"
    )


@app.cell
def _(mo):
    pass_preset = mo.ui.dropdown(
        options={
            "wide-dynamic 128->64->32 (transient, Recommended)": "wide",
            "fine 64->32->16 (post-shutdown / slow tail)": "fine",
        },
        value="wide-dynamic 128->64->32 (transient, Recommended)",
        label="Multi-pass schedule",
    )
    s2n_slider_mp = mo.ui.slider(
        start=1.0, stop=1.5, step=0.05, value=1.1, label="S/N validation threshold"
    )
    preview_frames = mo.ui.multiselect(
        options={
            "100: steady 0.5 m/s (t=6.7 s)": 100,
            "500: fast 1.1 m/s (t=33.3 s)": 500,
            "885: trip decay (t=59.0 s)": 885,
            "920: reversal (t=61.3 s)": 920,
        },
        value=["100: steady 0.5 m/s (t=6.7 s)", "500: fast 1.1 m/s (t=33.3 s)",
               "885: trip decay (t=59.0 s)", "920: reversal (t=61.3 s)"],
        label="Preview phases (one multi-pass run each)",
    )
    mo.vstack([
        mo.hstack([pass_preset, s2n_slider_mp]),
        preview_frames,
    ])
    return pass_preset, preview_frames, s2n_slider_mp


@app.cell
def _(
    CROP_COLS,
    LASER_DT,
    PX_PER_MM,
    SHUTDOWN_SET,
    ims_mod,
    np,
    opprep,
    pass_preset,
    preview_frames,
    s2n_slider_mp,
    time,
    windef,
):
    _schedules = {
        "wide": ((128, 64, 32), (64, 32, 16)),
        "fine": ((64, 32, 16), (32, 16, 8)),
    }
    _key = "wide" if str(pass_preset.value).startswith("wide") else "fine"
    _windows, _overlaps = _schedules["fine" if _key == "fine" else "wide"]

    def _run_pair(a_raw, b_raw, s2n_thr):
        _c0, _c1 = CROP_COLS
        _a = a_raw[:, _c0:_c1].astype(float)
        _b = b_raw[:, _c0:_c1].astype(float)
        _hp1 = opprep.high_pass(_a, sigma=16, clip=True)
        _hp2 = opprep.high_pass(_b, sigma=16, clip=True)
        _p1 = float(np.percentile(_hp1, 97.5)) or 200.0
        _p2 = float(np.percentile(_hp2, 97.5)) or 200.0
        _f1 = np.clip(255.0 * (_hp1 / _p1), 0, 255).astype(np.int32)
        _f2 = np.clip(255.0 * (_hp2 / _p2), 0, 255).astype(np.int32)
        _settings = windef.PIVSettings(
            windowsizes=_windows,
            overlap=_overlaps,
            subpixel_method="gaussian",
            sig2noise_method="peak2mean",
            sig2noise_threshold=float(s2n_thr),
            min_max_u_disp=(-60, 60),
            min_max_v_disp=(-60, 60),
        )
        _x, _y, _u_px, _v_px, _flags = windef.simple_multipass(_f1, _f2, _settings)
        _x_mm = np.asarray(_x, dtype=float) / PX_PER_MM
        _y_mm = np.asarray(_y, dtype=float) / PX_PER_MM
        _u = np.asarray(_u_px, dtype=float) / PX_PER_MM / LASER_DT
        _v = np.asarray(_v_px, dtype=float) / PX_PER_MM / LASER_DT
        _invalid = np.asarray(_flags, dtype=bool)
        return _x_mm, _y_mm, _u, _v, _invalid

    preview_payload = []
    for _idx in [int(v) for v in preview_frames.value] or [100]:
        _a_raw, _b_raw = ims_mod.read_ims_pair(str(SHUTDOWN_SET), _idx)
        _t0 = time.time()
        _px, _py, _pu, _pv, _inv = _run_pair(_a_raw, _b_raw, s2n_slider_mp.value)
        _ms = (time.time() - _t0) * 1000.0
        _prof = np.nanmean(np.where(~_inv, _pv, np.nan), axis=0)
        preview_payload.append({
            "idx": _idx,
            "x": np.asarray(_px[0, :], dtype=float),
            "prof": np.asarray(_prof, dtype=float),
            "valid_pct": (1.0 - np.asarray(_inv).mean()) * 100.0,
            "mean": float(np.nanmean(_prof)),
            "ms": _ms,
        })

    sched_windows, sched_overlaps = _windows, _overlaps
    return preview_payload, sched_overlaps, sched_windows


@app.cell
def _(mo, np, plt, preview_payload, s2n_slider_mp, sched_overlaps, sched_windows):
    _nprev = len(preview_payload)
    _fig, _axs = plt.subplots(1, max(1, _nprev), figsize=(5 * max(1, _nprev), 5), constrained_layout=True)
    if _nprev == 1:
        _axs = [_axs]
    for _ax, _row in zip(_axs, preview_payload):
        _ax.plot(_row["x"], _row["prof"], "b-o", markersize=3, linewidth=1.5)
        _ax.axhline(0, color="gray", linestyle="--")
        _ax.set_title(
            f"frame {_row['idx']} (t={_row['idx'] / 15.0:.1f} s) — valid {_row['valid_pct']:.1f}%, "
            f"mean {_row['mean']:.0f} mm/s, {_row['ms']:.0f} ms"
        )
        _ax.set_xlabel("x [mm]")
        _ax.set_ylabel("streamwise v [mm/s]")
        _ax.grid(True, linestyle=":", alpha=0.6)

    mo.vstack([
        mo.md(
            f"### Preview ({sched_windows[0]}→{sched_windows[1]}→{sched_windows[2]}, "
            f"overlap {sched_overlaps[0]}/{sched_overlaps[1]}/{sched_overlaps[2]}, S/N ≥ {s2n_slider_mp.value})"
        ),
        _fig,
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Batch runs (gated — headless smoke-test skips these)

        Set a small `N` first (e.g. 20) to validate throughput, then scale
        up. Each run writes a pivpy-canonical Zarr (`t, y, x` / `u, v, chc`
        in mm/s) that reloads in seconds for post-analysis.
        """
    )


@app.cell
def _(after_info, mo, shutdown_info):
    batch_n_shutdown = mo.ui.slider(
        start=5, stop=int(shutdown_info["n_pairs"]), step=5, value=20,
        label="Transient frames (N of 1000)",
    )
    batch_n_after = mo.ui.slider(
        start=5, stop=int(after_info["n_pairs"]), step=5, value=20,
        label="Post-shutdown frames (N of 100)",
    )
    run_shutdown_btn = mo.ui.run_button(label="Run transient multi-pass batch")
    run_after_btn = mo.ui.run_button(label="Run post-shutdown multi-pass batch")
    mo.vstack([
        mo.hstack([batch_n_shutdown, run_shutdown_btn]),
        mo.hstack([batch_n_after, run_after_btn]),
    ])
    return batch_n_after, batch_n_shutdown, run_after_btn, run_shutdown_btn


@app.cell
def _(
    ACQ_HZ,
    CROP_COLS,
    LASER_DT,
    OUT_DIR,
    PX_PER_MM,
    SHUTDOWN_SET,
    batch_n_shutdown,
    ims_mod,
    mo,
    np,
    opprep,
    run_shutdown_btn,
    sched_overlaps,
    sched_windows,
    time,
    windef,
    xr,
):
    if not run_shutdown_btn.value:
        shutdown_batch_output = mo.md(
            "_Set N and click **Run transient multi-pass batch**._"
        )
        shutdown_batch_result = {"ran": False}
    else:
        _n = int(batch_n_shutdown.value)
        _t0 = time.time()
        _us, _vs, _chcs, _ts = [], [], [], []
        _xg, _yg = None, None
        for _i in range(_n):
            _a_raw, _b_raw = ims_mod.read_ims_pair(str(SHUTDOWN_SET), _i)
            _c0, _c1 = CROP_COLS
            _f1 = opprep.high_pass(_a_raw[:, _c0:_c1].astype(float), sigma=16, clip=True)
            _f2 = opprep.high_pass(_b_raw[:, _c0:_c1].astype(float), sigma=16, clip=True)
            _p1 = float(np.percentile(_f1, 97.5)) or 200.0
            _p2 = float(np.percentile(_f2, 97.5)) or 200.0
            _settings = windef.PIVSettings(
                windowsizes=sched_windows, overlap=sched_overlaps,
                subpixel_method="gaussian",
                sig2noise_method="peak2mean", sig2noise_threshold=1.1,
                min_max_u_disp=(-60, 60), min_max_v_disp=(-60, 60),
            )
            _x, _y, _u_px, _v_px, _flags = windef.simple_multipass(
                np.clip(255.0 * (_f1 / _p1), 0, 255).astype(np.int32),
                np.clip(255.0 * (_f2 / _p2), 0, 255).astype(np.int32),
                _settings,
            )
            if _xg is None:
                _xg = np.asarray(_x, dtype=float) / PX_PER_MM
                _yg = np.asarray(_y, dtype=float) / PX_PER_MM
            _us.append(np.asarray(_u_px, dtype=float) / PX_PER_MM / LASER_DT)
            _vs.append(np.asarray(_v_px, dtype=float) / PX_PER_MM / LASER_DT)
            _chcs.append((np.asarray(_flags) == 0).astype(np.float32))
            _ts.append(_i / ACQ_HZ)
        from piv_pipeline import piv_run_metadata as _piv_meta

        _ds_t = xr.Dataset(
            data_vars={
                "u": (("t", "y", "x"), np.stack(_us).astype(np.float32)),
                "v": (("t", "y", "x"), np.stack(_vs).astype(np.float32)),
                "chc": (("t", "y", "x"), np.stack(_chcs).astype(np.float32)),
            },
            coords={"t": np.asarray(_ts, dtype=np.float32),
                    "x": _xg[0, :], "y": _yg[:, 0]},
        )
        _ds_t.attrs.update(_piv_meta(
            px_per_mm=PX_PER_MM,
            calibration_source="baseline_channel/Properties/Calibration/Calibration.xml",
            dt=LASER_DT, winsize=sched_windows[-1],
            searchsize=sched_windows[-1], overlap=sched_overlaps[-1],
            s2n_threshold=1.1, median_threshold="windef.typical_validation",
            source_folder=str(SHUTDOWN_SET), crop_cols=CROP_COLS,
            history=f"pump_shutdown_ims_multipass_batch.py: transient {sched_windows}",
        ))
        _zarr_t = OUT_DIR / "pump_shutdown_transient_multipass_ds.zarr"
        _ds_t.to_zarr(_zarr_t, mode="w")
        _el = time.time() - _t0
        shutdown_batch_result = {
            "ran": True, "n": _n, "zarr": str(_zarr_t),
            "valid_frac": float(np.mean(_chcs)),
            "elapsed_s": _el, "ms_per_frame": _el / _n * 1000.0,
        }
        shutdown_batch_output = mo.md(
            f"### Transient batch done — {_n} frames in {_el:.0f} s "
            f"({_el / _n * 1000.0:.0f} ms/frame), "
            f"mean valid {np.mean(_chcs):.1%} → `{_zarr_t.name}`"
        )
    shutdown_batch_output


@app.cell
def _(
    ACQ_HZ,
    AFTER_SET,
    CROP_COLS,
    LASER_DT,
    OUT_DIR,
    PX_PER_MM,
    batch_n_after,
    ims_mod,
    mo,
    np,
    opprep,
    run_after_btn,
    sched_overlaps,
    sched_windows,
    time,
    windef,
    xr,
):
    if not run_after_btn.value:
        after_batch_output = mo.md(
            "_Set N and click **Run post-shutdown multi-pass batch**._"
        )
        after_batch_result = {"ran": False}
    else:
        _n2 = int(batch_n_after.value)
        _t02 = time.time()
        _us2, _vs2, _chcs2, _ts2 = [], [], [], []
        _xg2, _yg2 = None, None
        for _i2 in range(_n2):
            _a_raw2, _b_raw2 = ims_mod.read_ims_pair(str(AFTER_SET), _i2)
            _c02, _c12 = CROP_COLS
            _f12 = opprep.high_pass(_a_raw2[:, _c02:_c12].astype(float), sigma=16, clip=True)
            _f22 = opprep.high_pass(_b_raw2[:, _c02:_c12].astype(float), sigma=16, clip=True)
            _p12 = float(np.percentile(_f12, 97.5)) or 200.0
            _p22 = float(np.percentile(_f22, 97.5)) or 200.0
            _settings2 = windef.PIVSettings(
                windowsizes=sched_windows, overlap=sched_overlaps,
                subpixel_method="gaussian",
                sig2noise_method="peak2mean", sig2noise_threshold=1.1,
                min_max_u_disp=(-60, 60), min_max_v_disp=(-60, 60),
            )
            _x2, _y2, _u_px2, _v_px2, _flags2 = windef.simple_multipass(
                np.clip(255.0 * (_f12 / _p12), 0, 255).astype(np.int32),
                np.clip(255.0 * (_f22 / _p22), 0, 255).astype(np.int32),
                _settings2,
            )
            if _xg2 is None:
                _xg2 = np.asarray(_x2, dtype=float) / PX_PER_MM
                _yg2 = np.asarray(_y2, dtype=float) / PX_PER_MM
            _us2.append(np.asarray(_u_px2, dtype=float) / PX_PER_MM / LASER_DT)
            _vs2.append(np.asarray(_v_px2, dtype=float) / PX_PER_MM / LASER_DT)
            _chcs2.append((np.asarray(_flags2) == 0).astype(np.float32))
            _ts2.append(_i2 / ACQ_HZ)
        from piv_pipeline import piv_run_metadata as _piv_meta2

        _ds_a = xr.Dataset(
            data_vars={
                "u": (("t", "y", "x"), np.stack(_us2).astype(np.float32)),
                "v": (("t", "y", "x"), np.stack(_vs2).astype(np.float32)),
                "chc": (("t", "y", "x"), np.stack(_chcs2).astype(np.float32)),
            },
            coords={"t": np.asarray(_ts2, dtype=np.float32),
                    "x": _xg2[0, :], "y": _yg2[:, 0]},
        )
        _ds_a.attrs.update(_piv_meta2(
            px_per_mm=PX_PER_MM,
            calibration_source="baseline_channel/Properties/Calibration/Calibration.xml",
            dt=LASER_DT, winsize=sched_windows[-1],
            searchsize=sched_windows[-1], overlap=sched_overlaps[-1],
            s2n_threshold=1.1, median_threshold="windef.typical_validation",
            source_folder=str(AFTER_SET), crop_cols=CROP_COLS,
            history=f"pump_shutdown_ims_multipass_batch.py: after {sched_windows}",
        ))
        _zarr_a = OUT_DIR / "pump_shutdown_after_multipass_ds.zarr"
        _ds_a.to_zarr(_zarr_a, mode="w")
        _stats_a = _ds_a.piv.reynolds_decomposition()
        _stats_path = OUT_DIR / "pump_shutdown_after_multipass_stats.zarr"
        _stats_a.to_zarr(_stats_path, mode="w")
        _el2 = time.time() - _t02
        after_batch_result = {
            "ran": True, "n": _n2, "zarr": str(_zarr_a),
            "stats": str(_stats_path),
            "valid_frac": float(np.mean(_chcs2)),
            "elapsed_s": _el2, "ms_per_frame": _el2 / _n2 * 1000.0,
        }
        after_batch_output = mo.md(
            f"### Post-shutdown batch done — {_n2} frames in {_el2:.0f} s "
            f"({_el2 / _n2 * 1000.0:.0f} ms/frame), "
            f"mean valid {np.mean(_chcs2):.1%} → `{_zarr_a.name}` + stats"
        )
    after_batch_output


@app.cell
def _(OUT_DIR, mo, np, plt, xr):
    _t_path = OUT_DIR / "pump_shutdown_transient_multipass_ds.zarr"
    _a_path = OUT_DIR / "pump_shutdown_after_multipass_ds.zarr"
    if not (_t_path.exists() and _a_path.exists()):
        postproc_output = mo.md(
            "_Post-analysis unlocks once **both** batch Zarrs exist "
            f"(`{_t_path.name}`, `{_a_path.name}`). Run both batches above first._"
        )
    else:
        _dst = xr.open_zarr(_t_path)
        _dsa = xr.open_zarr(_a_path)
        _bulk = []
        for _k in range(_dst.sizes["t"]):
            _fr = _dst["v"].isel(t=_k).where(_dst["chc"].isel(t=_k) > 0.5)
            _prof = _fr.mean(dim="y", skipna=True).values
            _xx = _dst["x"].values
            _ok = np.isfinite(_prof) & np.isfinite(_xx)
            _o = np.argsort(_xx[_ok])
            _bulk.append(
                float(np.trapezoid(_prof[_ok][_o], _xx[_ok][_o])
                      / (_xx.max() - _xx.min()))
            )
        _fig2, _ax2 = plt.subplots(figsize=(9, 4), constrained_layout=True)
        _ax2.plot(_dst["t"].values, _bulk, "k.-", markersize=4, label="transient bulk v(t)")
        _ax2.axhline(0, color="gray", linestyle="--")
        _ax2.set_xlabel("time [s]")
        _ax2.set_ylabel("bulk streamwise velocity [mm/s]")
        _ax2.set_title("Pump-shutdown decay: bulk velocity vs time (multi-pass, pivpy Zarr)")
        _ax2.legend()
        _ax2.grid(True, linestyle=":", alpha=0.6)
        _fig2.savefig(OUT_DIR / "pump_shutdown_multipass_decay.png", dpi=150)
        postproc_output = mo.vstack([
            mo.md(
                f"### Decay trace — {len(_bulk)} transient frames, "
                f"start {float(np.nanmean(_bulk[:3])):.0f} mm/s → "
                f"end {float(np.nanmean(_bulk[-3:])):.0f} mm/s"
            ),
            _fig2,
        ])
    postproc_output


if __name__ == "__main__":
    app.run()
