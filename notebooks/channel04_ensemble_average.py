import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# channel_04 - ensemble average across all 4 parts\n"
        "Loads the 4 per-part `ds.zarr` stores (raw per-frame `u`, `v`, `chc` "
        "from `channel04_all_parts_batch.py`) and computes a true ensemble "
        "average via PIVPy's `.piv.reynolds_decomposition()` on all 12000 "
        "frames pooled together - not an average-of-per-part-averages, which "
        "would miss part-to-part variability in the turbulence quantities. "
        "**No merged dataset is written to disk** - the 4 zarr stores stay "
        "the source of truth, and the pooled dataset is rebuilt in memory "
        "each time this notebook runs (a few seconds for ~600 MB of vector "
        "data, per the timing check before writing this notebook).\n\n"
        "This is a live pairing scratchpad for publication figures - keep it "
        "open and we'll tune the plots together."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import lvpyio as lv
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import quiver_on_image, scalar_on_image, load_run_config

    return (
        ROOT,
        json,
        load_run_config,
        lv,
        mo,
        np,
        plt,
        quiver_on_image,
        scalar_on_image,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - the 4 runs (from the orchestration run log)
    """)
    return


@app.cell
def _(ROOT, json, mo):
    _log_path = ROOT / "outputs" / "channel04_all_parts" / "channel04_all_parts_run_log.json"
    with open(_log_path) as _f:
        run_log = json.load(_f)

    _rows = "\n".join(
        f"| {r['run_name']} | {r['n_pairs']} | {r['invalid_fraction']:.1%} | {r['mean_speed_mm_s']:.1f} |"
        for r in run_log
    )
    mo.md(
        f"Loaded `{_log_path.name}` - {len(run_log)} runs:\n\n"
        f"| part | pairs | invalid | mean speed [mm/s] |\n|---|---|---|---|\n{_rows}"
    )
    return (run_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - pool all 4 parts into one in-memory dataset (not saved to disk)
    """)
    return


@app.cell
def _(mo, np, run_log, xr):
    _parts = []
    _offset = 0
    for _r in run_log:
        _ds = xr.open_zarr(_r["ds_path"])
        _ds = _ds.assign_coords(t=_ds["t"] + _offset)
        _parts.append(_ds.load())
        _offset += _ds.sizes["t"]

    combined_ds = xr.concat(_parts, dim="t")
    mo.md(
        f"Pooled **{combined_ds.sizes['t']}** frames from {len(run_log)} parts "
        f"(in memory only, `{np.round(combined_ds.nbytes / 1e6)}` MB) into "
        f"`combined_ds` - same `u`/`v`/`chc` shape convention as any single "
        f"part, just a longer `t` axis."
    )
    return (combined_ds,)


@app.cell
def _(combined_ds, mo):
    ensemble_stats = combined_ds.piv.reynolds_decomposition()
    mo.md(
        f"Ensemble Reynolds decomposition over {combined_ds.sizes['t']} pooled frames: "
        f"mean streamwise speed {float((ensemble_stats['u_mean']**2 + ensemble_stats['v_mean']**2).mean() ** 0.5):.1f} mm/s, "
        f"peak TKE {float(ensemble_stats['tke'].max()):.1f} mm²/s²."
    )
    return (ensemble_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3 - a representative background image
    """)
    return


@app.cell
def _(ROOT, load_run_config, lv, np, run_log):
    from pathlib import Path as _Path2

    config = load_run_config(ROOT / "notebooks" / "channel04_pair1_config.json")
    _first_im7 = sorted(_Path2(run_log[0]["source_folder"]).glob("*.im7"))[0]
    _buffer = lv.read_buffer(str(_first_im7))
    _a1_raw = np.asarray(_buffer.as_masked_array(0).data)
    background_image = np.clip(_a1_raw.astype(float), 0, config["preprocessing"]["display_max"])
    background_image = ((255.0 / config["preprocessing"]["display_max"]) * background_image).astype(np.uint8)
    return background_image, config


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 4 - publication figure controls
    """)
    return


@app.cell
def _(mo):
    field_dd = mo.ui.dropdown(
        options=["mean speed", "uv_prime (shear stress)", "tke", "uu_prime", "vv_prime"],
        value="mean speed", label="field",
    )
    cmap_dd = mo.ui.dropdown(
        options=["viridis", "RdBu_r", "coolwarm", "magma", "cividis", "jet"],
        value="viridis", label="colormap",
    )
    show_bg_cb = mo.ui.checkbox(value=True, label="show background image")
    bg_alpha_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.6, label="background alpha")
    arrow_width_slider = mo.ui.slider(0.001, 0.02, step=0.0005, value=0.004, label="arrow width")
    fig_w_slider = mo.ui.slider(4, 16, step=0.5, value=8, label="figure width [in]")
    fig_h_slider = mo.ui.slider(4, 16, step=0.5, value=10, label="figure height [in]")
    dpi_slider = mo.ui.slider(100, 600, step=50, value=300, label="dpi")
    fontsize_slider = mo.ui.slider(8, 24, step=1, value=12, label="font size")

    mo.vstack([
        mo.hstack([field_dd, cmap_dd, show_bg_cb]),
        mo.hstack([bg_alpha_slider, arrow_width_slider]),
        mo.hstack([fig_w_slider, fig_h_slider, dpi_slider, fontsize_slider]),
    ])
    return (
        arrow_width_slider,
        bg_alpha_slider,
        cmap_dd,
        dpi_slider,
        field_dd,
        fig_h_slider,
        fig_w_slider,
        fontsize_slider,
        show_bg_cb,
    )


@app.cell
def _(
    arrow_width_slider,
    background_image,
    bg_alpha_slider,
    cmap_dd,
    config,
    dpi_slider,
    ensemble_stats,
    field_dd,
    fig_h_slider,
    fig_w_slider,
    fontsize_slider,
    np,
    plt,
    quiver_on_image,
    scalar_on_image,
    show_bg_cb,
):
    plt.rcParams.update({"font.size": fontsize_slider.value})

    _u = ensemble_stats["u_mean"].values
    _v = ensemble_stats["v_mean"].values
    _x = ensemble_stats["x"].values
    _y = ensemble_stats["y"].values
    _X, _Y = np.meshgrid(_x, _y)

    _fig, _ax = plt.subplots(figsize=(fig_w_slider.value, fig_h_slider.value))
    _bg = background_image if show_bg_cb.value else np.zeros_like(background_image)
    _bg_alpha = bg_alpha_slider.value if show_bg_cb.value else 0.0

    if field_dd.value == "mean speed":
        _Q = quiver_on_image(
            _ax, _bg, _X, _Y, _u, _v, color_by=np.hypot(_u, _v), cmap=cmap_dd.value,
            scaling_factor=config["px_per_mm"], image_alpha=_bg_alpha,
            arrow_width=arrow_width_slider.value,
        )
        _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
        _cbar.set_label("mean speed [mm/s]")
    else:
        _field_map = {
            "uv_prime (shear stress)": ("uv_prime", "Reynolds shear stress $-\\overline{u'v'}$ [mm²/s²]"),
            "tke": ("tke", "TKE [mm²/s²]"),
            "uu_prime": ("uu_prime", "$\\overline{u'u'}$ [mm²/s²]"),
            "vv_prime": ("vv_prime", "$\\overline{v'v'}$ [mm²/s²]"),
        }
        _var, _label = _field_map[field_dd.value]
        _cf = scalar_on_image(
            _ax, _bg, _X, _Y, ensemble_stats[_var].values, cmap=cmap_dd.value,
            scaling_factor=config["px_per_mm"], image_alpha=_bg_alpha,
            symmetric=(_var in ("uv_prime",)),
        )
        _cbar = _fig.colorbar(_cf, ax=_ax, pad=0.03, shrink=0.85)
        _cbar.set_label(_label)
        quiver_on_image(_ax, _bg, _X, _Y, _u, _v, scaling_factor=config["px_per_mm"],
                         image_alpha=0.0, arrow_width=arrow_width_slider.value * 0.75)

    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title(f"channel_04, steady state - ensemble average (N=12000, {field_dd.value})")
    _fig.set_dpi(dpi_slider.value)
    current_fig = _fig
    _fig.gca()
    return (current_fig,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "Uses the *same* sinusoid as the wall mask (`config[\"wall_mask\"]`) to "
        "define phase, not a re-detected wall - one full peak-to-peak period, "
        "centered on the domain (there are ~2.6 periods across the full 14 mm "
        "height, so we pick the single period closest to mid-channel), sampled "
        "at N evenly-spaced phases from 0 to 1 (phase 0 and 1 are the same wall "
        "position, one period apart - included so the profile set visibly closes "
        "the cycle). At each phase, `v_mean(x)` (streamwise, downward) is pulled "
        "at the nearest computed y."
    )
    mo.md(_text)

    return


@app.cell
def _(mo):
    n_phases_slider = mo.ui.slider(4, 20, step=1, value=10, label="number of phases")
    profile_cmap_dd = mo.ui.dropdown(
        options=["twilight", "hsv", "viridis", "plasma", "cividis"],
        value="twilight", label="phase colormap",
    )
    flip_sign_cb = mo.ui.checkbox(value=False, label="show downward speed as positive")
    mo.hstack([n_phases_slider, profile_cmap_dd, flip_sign_cb])

    return flip_sign_cb, n_phases_slider, profile_cmap_dd


@app.cell
def _(background_image, config, ensemble_stats, mo, n_phases_slider, np):
    _image_height_px = background_image.shape[0]
    _wm = config["wall_mask"]
    _wavelength_px = _wm["wavelength_px"]
    _phase0 = _wm["phase_rad"]

    # Peak of the wall sinusoid occurs where 2*pi*row/wavelength + phase0 = pi/2
    # (mod 2*pi); pick the period instance [row_start, row_start+wavelength)
    # whose peak sits nearest the vertical domain center.
    _row_peak_base = _wavelength_px * (np.pi / 2 - _phase0) / (2 * np.pi)
    _row_center = _image_height_px / 2
    _k = round((_row_center - _row_peak_base) / _wavelength_px)
    _row_start = _row_peak_base + _k * _wavelength_px

    phase_fractions = np.linspace(0, 1, n_phases_slider.value)
    _rows = _row_start + phase_fractions * _wavelength_px
    # Row 0 is the top of the image; ensemble_stats["y"] increases upward from
    # the bottom (row = image_height_px), matching quiver_on_image's convention.
    phase_y_mm = (_image_height_px - _rows) / config["px_per_mm"]

    phase_profiles = [
        ensemble_stats["v_mean"].sel(y=_y, method="nearest") for _y in phase_y_mm
    ]
    mo.md(
        f"Central period: rows {_row_start:.0f}-{_row_start + _wavelength_px:.0f} px "
        f"({phase_y_mm.max():.2f}-{phase_y_mm.min():.2f} mm), {n_phases_slider.value} phases."
    )

    return phase_fractions, phase_profiles


@app.cell
def _(
    ensemble_stats,
    flip_sign_cb,
    phase_fractions,
    phase_profiles,
    plt,
    profile_cmap_dd,
):
    _sign = -1.0 if flip_sign_cb.value else 1.0
    _cmap = plt.get_cmap(profile_cmap_dd.value)

    _fig, _ax = plt.subplots(figsize=(8, 6))
    for _frac, _prof in zip(phase_fractions, phase_profiles):
        _ax.plot(
            ensemble_stats["x"].values, _sign * _prof.values,
            color=_cmap(_frac), label=f"{_frac:.2f}",
        )
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel(f"{'downward speed' if flip_sign_cb.value else 'v_mean'} [mm/s]")
    _ax.set_title("Streamwise velocity profiles across one wave period")
    _ax.legend(title="phase (y/?)", fontsize=8, ncol=2, loc="best")
    _fig.tight_layout()
    phase_profile_fig = _fig
    _fig.gca()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 5 - export
    """)
    return


@app.cell
def _(mo):
    export_path_text = mo.ui.text(
        value="outputs/channel04_all_parts/ensemble_figure.pdf",
        label="export path (.png/.pdf/.svg)", full_width=True,
    )
    export_button = mo.ui.run_button(label="save figure")
    mo.hstack([export_path_text, export_button])
    return export_button, export_path_text


@app.cell
def _(current_fig, export_button, export_path_text, mo):
    if export_button.value:
        current_fig.savefig(export_path_text.value, bbox_inches="tight")
        _status = mo.md(f"Saved `{export_path_text.value}`")
    else:
        _status = mo.md("*(click \"save figure\" above to export the current figure)*")
    _status
    return


if __name__ == "__main__":
    app.run()
