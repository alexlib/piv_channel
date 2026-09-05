import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# Wavy channel (channel_04), steady state - N-frame batch\n"
        "Reads the first `N` real `.im7` pairs from a DaVis export folder, runs "
        "the same single-pass PIV pipeline tuned interactively in "
        "`channel04_steady_state_pair1.py` (`process_im7_pair(preprocess=\"high_pass\", "
        "mask_input=True, wall_bounds=...)` in `piv_pipeline.py`), computes the "
        "time-average and turbulence statistics via PIVPy's "
        "`.piv.reynolds_decomposition()`, and plots the mean flow field and the "
        "shear Reynolds stress (`-<u'v'>`) over the first raw image.\n\n"
        "All tunable parameters (calibration, wall-mask sinusoid, preprocessing, "
        "PIV window/validation settings) come from `channel04_pair1_config.json` "
        "- hand-tuned once on a single pair, not re-tuned here."
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
    import lvpyio as lv
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import (
        process_im7_pair, quiver_on_image, scalar_on_image, piv_run_metadata,
        sinusoidal_wall_bounds, load_run_config,
    )

    return (
        ROOT,
        lv,
        load_run_config,
        mo,
        np,
        piv_run_metadata,
        plt,
        process_im7_pair,
        quiver_on_image,
        scalar_on_image,
        sinusoidal_wall_bounds,
        xr,
    )


@app.cell
def _(ROOT, load_run_config, mo):
    config = load_run_config(ROOT / "notebooks" / "channel04_pair1_config.json")
    N = 3000
    FOLDER = config["source_folder"]
    mo.md(f"`N = {N}`, folder: `{FOLDER}`  \nconfig: `channel04_pair1_config.json` ({config['run_name']})")
    return FOLDER, N, config


@app.cell
def _(FOLDER, N, mo):
    from pathlib import Path as _Path

    im7_files = sorted(_Path(FOLDER).glob("*.im7"))[:N]
    _names = ', '.join(f.name for f in im7_files[:5])
    _more = f", ... +{len(im7_files) - 5} more" if len(im7_files) > 5 else ""
    mo.md(f"Found **{len(im7_files)}** pairs (of `N={N}` requested): {_names}{_more}")
    return (im7_files,)


@app.cell
def _(config, im7_files, lv, np, sinusoidal_wall_bounds):
    # Wall geometry is fixed for the whole run - compute once from the first
    # frame and reuse unchanged across every pair, instead of re-detecting
    # (or re-fitting) it per frame.
    _buffer = lv.read_buffer(str(im7_files[0]))
    _a1_raw = np.asarray(_buffer.as_masked_array(0).data)
    _wm = config["wall_mask"]
    wall_bounds = sinusoidal_wall_bounds(
        _a1_raw.shape[0],
        wavelength_px=_wm["wavelength_px"], phase=_wm["phase_rad"],
        left_center=_wm["left_center_px"], left_amplitude=_wm["left_amplitude_px"],
        right_center=_wm["right_center_px"], right_amplitude=_wm["right_amplitude_px"],
    )
    return (wall_bounds,)


@app.cell
def _(config, im7_files, mo, process_im7_pair, wall_bounds):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    _piv_kw = dict(
        crop_cols=None, wall_bounds=wall_bounds, mask_input=config["piv"]["mask_input"],
        preprocess="high_pass",
        hp_sigma=config["preprocessing"]["high_pass_sigma"],
        hp_pct=config["preprocessing"]["contrast_stretch_percentile"],
        winsize=config["piv"]["winsize"], searchsize=config["piv"]["searchsize"],
        overlap=config["piv"]["overlap"], s2n_threshold=config["piv"]["s2n_threshold"],
        median_threshold=config["piv"]["median_threshold_px_per_frame"],
        scaling_factor=config["px_per_mm"], return_image=False,
    )

    # Pairs are independent - farm them out across cores instead of one
    # process crunching FFT correlations + high_pass filtering serially.
    with ProcessPoolExecutor() as _ex:
        results = list(_ex.map(partial(process_im7_pair, **_piv_kw), im7_files))
    results[0] = process_im7_pair(im7_files[0], **{**_piv_kw, "return_image": True})

    invalid_fracs = [r["invalid"].mean() for r in results]
    _preview = ', '.join(f'{f:.0%}' for f in invalid_fracs[:10])
    _more = f", ... (+{len(invalid_fracs) - 10} more)" if len(invalid_fracs) > 10 else ""
    mo.md(
        f"dt (from file metadata): **{results[0]['dt'] * 1e6:.1f} µs** "
        f"(consistent across all {len(results)} frames: {len({r['dt'] for r in results}) == 1})  \n"
        f"Invalid-vector fraction, first frames: {_preview}{_more}  \n"
        f"Mean over all {len(results)} frames: {sum(invalid_fracs) / len(invalid_fracs):.0%}"
    )
    return (results,)


@app.cell
def _(config, piv_run_metadata, results, xr):
    # PIVPy-shaped Dataset (dims t, y, x; vars u, v, chc), same convention as
    # baseline_steady_state_batch.py.
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
    # ds.piv.reynolds_decomposition() carries these attrs through to `stats`
    # automatically, so no need to set them twice.
    ds.attrs.update(piv_run_metadata(
        px_per_mm=config["px_per_mm"],
        calibration_source=config["calibration_source"],
        dt=results[0]["dt"],
        winsize=config["piv"]["winsize"], searchsize=config["piv"]["searchsize"],
        overlap=config["piv"]["overlap"],
        s2n_threshold=config["piv"]["s2n_threshold"],
        median_threshold=config["piv"]["median_threshold_px_per_frame"],
        source_folder=config["source_folder"],
        crop_cols=None,
        wall_mask=True,
        history="channel04_steady_state_batch.py",
    ))
    return (ds,)


@app.cell
def _(ds, mo):
    stats = ds.piv.reynolds_decomposition()
    mo.md(
        f"Reynolds decomposition over {ds.sizes['t']} frames: "
        f"mean streamwise speed {float((stats['u_mean']**2 + stats['v_mean']**2).mean() ** 0.5):.1f} mm/s, "
        f"peak TKE {float(stats['tke'].max()):.1f} mm²/s²."
    )
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Average flow field over the first frame
    """)
    return


@app.cell
def _(ROOT, config, np, plt, quiver_on_image, results, stats):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _u, _v = stats["u_mean"].values, stats["v_mean"].values
    _x, _y = stats["x"].values, stats["y"].values
    _X, _Y = np.meshgrid(_x, _y)

    _Q = quiver_on_image(
        _ax, results[0]["image"], _X, _Y, _u, _v,
        color_by=np.hypot(_u, _v), cmap="viridis", scaling_factor=config["px_per_mm"],
    )
    _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("mean speed [mm/s]")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"channel_04, steady state - mean flow field (N={len(results)})")
    (ROOT / "outputs").mkdir(exist_ok=True)
    _fig.savefig(ROOT / "outputs" / "channel04_steady_state_mean_field.png", dpi=150)
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
    ROOT,
    config,
    np,
    plt,
    quiver_on_image,
    results,
    scalar_on_image,
    stats,
):
    _fig, _ax = plt.subplots(figsize=(8, 10))
    _x, _y = stats["x"].values, stats["y"].values
    _X, _Y = np.meshgrid(_x, _y)

    _cf = scalar_on_image(
        _ax, results[0]["image"], _X, _Y, stats["uv_prime"].values,
        cmap="RdBu_r", scaling_factor=config["px_per_mm"],
    )
    _cbar = _fig.colorbar(_cf, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("Reynolds shear stress $-\\overline{u'v'}$ [mm²/s²]")

    quiver_on_image(
        _ax, results[0]["image"], _X, _Y,
        stats["u_mean"].values, stats["v_mean"].values,
        scaling_factor=config["px_per_mm"], image_alpha=0.0, arrow_width=0.003,
    )
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"channel_04, steady state - Reynolds shear stress (N={len(results)})")
    _fig.savefig(ROOT / "outputs" / "channel04_steady_state_reynolds_stress.png", dpi=150)
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
    _ds_path = _out_dir / "channel04_steady_state_ds.zarr"
    _stats_path = _out_dir / "channel04_steady_state_stats.zarr"
    ds.to_zarr(_ds_path, mode="w")
    stats.to_zarr(_stats_path, mode="w")
    mo.md(f"Saved `{_ds_path.name}` ({ds.sizes['t']} frames) and `{_stats_path.name}`.")
    return


if __name__ == "__main__":
    app.run()
