import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# Baseline channel, steady state - N-frame batch\n"
        "Reads the first `N` real `.im7` pairs from a DaVis export folder, "
        "runs the same single-pass PIV pipeline as `pair1_first_pass.py` "
        "(`process_im7_pair()` in `piv_pipeline.py`), computes the time-average "
        "and turbulence statistics via PIVPy's `.piv.reynolds_decomposition()`, "
        "and plots the mean flow field and the shear Reynolds stress "
        "(`-<u'v'>`) over the first raw image."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import (
        process_im7_pair, quiver_on_image, scalar_on_image, piv_run_metadata,
        PX_PER_MM, CHANNEL_CROP_COLS,
    )

    return (
        CHANNEL_CROP_COLS,
        PX_PER_MM,
        ROOT,
        mo,
        np,
        piv_run_metadata,
        plt,
        process_im7_pair,
        quiver_on_image,
        scalar_on_image,
        xr,
    )


@app.cell
def _(mo):
    N = 3000
    FOLDER = (
        r"D:\channel_flow_research\baseline_channel\Vmax_0p62_m2sec_steady_state"
        r"\exported_images\Vmax_0p62_m2sec_steady_state"
    )
    mo.md(f"`N = {N}`, folder: `{FOLDER}`")
    return FOLDER, N


@app.cell
def _(FOLDER, N, mo):
    from pathlib import Path as _Path

    im7_files = sorted(_Path(FOLDER).glob("*.im7"))[:N]
    _names = ', '.join(f.name for f in im7_files[:5])
    _more = f", ... +{len(im7_files) - 5} more" if len(im7_files) > 5 else ""
    mo.md(f"Found **{len(im7_files)}** pairs (of `N={N}` requested): {_names}{_more}")
    return (im7_files,)


@app.cell
def _(im7_files, mo, process_im7_pair):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    # Pairs are independent - farm them out across cores instead of one
    # process crunching FFT correlations serially (~3x faster on this box;
    # scaling tops out around the physical core count).
    # high_pass + these window/threshold settings cut invalid fraction from
    # ~66% (naive linear clip, original params) to ~2%, verified across the
    # sequence - same background-removal technique that fixed channel_04's
    # ~99% invalid, tuned here for this camera/dt (see README).
    _piv_kw = dict(
        preprocess="high_pass", hp_sigma=16, hp_pct=97.5,
        winsize=96, searchsize=96, overlap=32,
        s2n_threshold=1.0, median_threshold=2,
    )
    with ProcessPoolExecutor() as _ex:
        results = list(_ex.map(partial(process_im7_pair, return_image=False, **_piv_kw), im7_files))
    results[0] = process_im7_pair(im7_files[0], return_image=True, **_piv_kw)

    invalid_fracs = [r["invalid"].mean() for r in results]
    _preview = ', '.join(f'{f:.0%}' for f in invalid_fracs[:10])
    _more = f", ... (+{len(invalid_fracs) - 10} more)" if len(invalid_fracs) > 10 else ""
    mo.md(
        f"dt (from file metadata): **{results[0]['dt'] * 1e6:.1f} \u00b5s** "
        f"(consistent across all {len(results)} frames: {len({r['dt'] for r in results}) == 1})  \n"
        f"Invalid-vector fraction, first frames: {_preview}{_more}  \n"
        f"Mean over all {len(results)} frames: {sum(invalid_fracs) / len(invalid_fracs):.0%} "
        f"- high_pass preprocessing + tuned window/threshold (see README) brought "
        f"this down from ~66% (naive linear clip, dt-fixed median threshold)."
    )
    return (results,)


@app.cell
def _(CHANNEL_CROP_COLS, FOLDER, PX_PER_MM, piv_run_metadata, results, xr):
    # PIVPy-shaped Dataset (dims t, y, x; vars u, v, chc), same convention as
    # batch_process.py, in mm/s now that the real dt is known.
    def _to_frame(r, t):
        return xr.Dataset(
            data_vars={
                "u": (("y", "x"), r["u"]),
                "v": (("y", "x"), r["v"]),
                "chc": (("y", "x"), (~r["invalid"]).astype(float)),
            },
            coords={"x": r["x"][0, :], "y": r["y"][:, 0], "t": t},
        )

    ds = xr.concat([_to_frame(r, t) for t, r in enumerate(results)], dim="t")
    # See README.md's "Calibration" and "Timing" notes for how px_per_mm/dt
    # were found; ds.piv.reynolds_decomposition() carries these attrs
    # through to `stats` automatically, so no need to set them twice.
    ds.attrs.update(piv_run_metadata(
        px_per_mm=PX_PER_MM,
        calibration_source=r"D:\channel_flow_research\baseline_channel\Properties\Calibration\Calibration.xml",
        dt=results[0]["dt"],
        winsize=96, searchsize=96, overlap=32,
        s2n_threshold=1.0, median_threshold=2,
        source_folder=FOLDER,
        crop_cols=CHANNEL_CROP_COLS,
        history="baseline_steady_state_batch.py",
    ))
    return (ds,)


@app.cell
def _(ds, mo):
    stats = ds.piv.reynolds_decomposition()
    mo.md(
        f"Reynolds decomposition over {ds.sizes['t']} frames: "
        f"mean streamwise speed {float((stats['u_mean']**2 + stats['v_mean']**2).mean() ** 0.5):.1f} mm/s, "
        f"peak TKE {float(stats['tke'].max()):.1f} mm\u00b2/s\u00b2."
    )
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Average flow field over the first frame
    """)
    return


@app.cell
def _(PX_PER_MM, ROOT, np, plt, quiver_on_image, results, stats):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _u, _v = stats["u_mean"].values, stats["v_mean"].values
    _x, _y = stats["x"].values, stats["y"].values
    _X, _Y = np.meshgrid(_x, _y)

    _Q = quiver_on_image(
        _ax, results[0]["image"], _X, _Y, _u, _v,
        color_by=np.hypot(_u, _v), cmap="viridis", scaling_factor=PX_PER_MM,
    )
    _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("mean speed [mm/s]")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"Baseline, steady state - mean flow field (N={len(results)})")
    (ROOT / "outputs").mkdir(exist_ok=True)
    _fig.savefig(ROOT / "outputs" / "baseline_steady_state_mean_field.png", dpi=150)
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Turbulent (shear) Reynolds stress over the first frame\n"
        "`uv_prime` = $-\\overline{u'v'}$, computed by `reynolds_decomposition()`. "
        "`ds.piv.plot()` can't put a scalar contour *and* a raw image in its single "
        "`background=` slot at once, so this reuses `scalar_on_image()` (the scalar "
        "counterpart to `quiver_on_image()`) with the mean-flow quiver drawn on top "
        "for context."
    )
    mo.md(_text)
    return


@app.cell
def _(
    PX_PER_MM,
    ROOT,
    np,
    plt,
    quiver_on_image,
    results,
    scalar_on_image,
    stats,
):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _x, _y = stats["x"].values, stats["y"].values
    _X, _Y = np.meshgrid(_x, _y)

    _cf = scalar_on_image(
        _ax, results[0]["image"], _X, _Y, stats["uv_prime"].values,
        cmap="RdBu_r", scaling_factor=PX_PER_MM,
    )
    _cbar = _fig.colorbar(_cf, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("Reynolds shear stress $-\\overline{u'v'}$ [mm\u00b2/s\u00b2]")

    quiver_on_image(
        _ax, results[0]["image"], _X, _Y,
        stats["u_mean"].values, stats["v_mean"].values,
        scaling_factor=PX_PER_MM, image_alpha=0.0, arrow_width=0.003,
    )
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"Baseline, steady state - Reynolds shear stress (N={len(results)})")
    _fig.savefig(ROOT / "outputs" / "baseline_steady_state_reynolds_stress.png", dpi=150)
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Save results for fast reload\n"
        "Zarr instead of re-running PIV on the raw `.im7` pairs every time - "
        "`ds` (per-frame `u`, `v`, `chc`) and `stats` (mean/turbulence fields) "
        "each go to their own store; reload with `xr.open_zarr(path)`."
    )
    mo.md(_text)
    return


@app.cell
def _(ROOT, ds, mo, stats):
    _out_dir = ROOT / "outputs"
    _out_dir.mkdir(exist_ok=True)
    _ds_path = _out_dir / "baseline_steady_state_ds.zarr"
    _stats_path = _out_dir / "baseline_steady_state_stats.zarr"
    ds.to_zarr(_ds_path, mode="w")
    stats.to_zarr(_stats_path, mode="w")
    mo.md(f"Saved `{_ds_path.name}` ({ds.sizes['t']} frames) and `{_stats_path.name}`.")
    return


if __name__ == "__main__":
    app.run()
