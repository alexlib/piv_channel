import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import imageio.v3 as iio
    import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    # Repo root, independent of the runner's cwd (VS Code's marimo extension
    # uses the notebook's own folder; `uv run python notebooks/x.py` uses repo root).
    ROOT = Path(__file__).resolve().parent.parent
    print(ROOT)
    return ROOT, iio, mo, np, plt, xr


@app.cell
def _(ROOT, iio, np):
    # First test pair: a single dual-frame TIFF (frame A stacked on top of frame B).
    a = iio.imread(ROOT / "sample_data" / "tiff" / "B0001.tif")
    display_min = 0
    display_max = 150
    a = a.astype(float)
    np.clip(a, display_min, display_max, out=a)
    a -= display_min
    a = ((255.0 / (display_max - display_min)) * a).astype(np.uint8)
    return (a,)


@app.cell
def _(ROOT, a, np, plt):
    a1 = a[:2048, 200:1945]
    a2 = a[2048:, 200:1945]
    plt.figure(figsize=(10, 10))
    plt.imshow(np.stack([a1, a2, a2 * 0], axis=2))
    plt.show()
    (ROOT / "outputs").mkdir(exist_ok=True)
    plt.imsave(ROOT / "outputs" / "tmp.png", np.stack([a1, a2, a2 * 0], axis=2))
    return a1, a2


@app.cell
def _(a1, a2, mo, np):
    from skimage.registration import phase_cross_correlation

    _shift, _err, _ = phase_cross_correlation(a1, a2, upsample_factor=10)
    _dy, _dx = _shift  # (row, col): positive dy = frame B moved up relative to A
    _disp = float(np.hypot(_dy, _dx))

    mo.md(
        f"**Global shift A→B:** dy = {-_dy:.2f} px (down is positive), "
        f"dx = {-_dx:.2f} px, magnitude ≈ {_disp:.1f} px.  \n"
        f"Rule of thumb: `winsize ≥ 4 × {_disp:.0f} = {int(np.ceil(4*_disp))}` px, "
        f"`searchsize ≥ winsize + 2 × {_disp:.0f}` px."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Fast, reusable pipeline\n"
        "The step-by-step exploration above is now packaged as `process_pair()` "
        "in `piv_pipeline.py` (same parameters: winsize=64, searchsize=96, "
        "overlap=32, dt=1.0, s2n threshold=1.3). Tested here on the same image; "
        "reused as-is by `batch_process.py` for the rest of the pairs."
    )
    mo.md(_text)
    return


@app.cell
def _():
    from piv_pipeline import process_pair, quiver_on_image, PX_PER_MM

    return PX_PER_MM, process_pair, quiver_on_image


@app.cell
def _(ROOT, process_pair):
    result = process_pair(ROOT / "sample_data" / "tiff" / "B0001.tif")
    invalid = result["invalid"]
    print(f"Invalid vectors: {invalid.sum()} / {invalid.size} ({100 * invalid.mean():.1f}%)")
    return (result,)


@app.cell
def _(result, xr):
    # PIVPy-shaped Dataset (dims y, x; vars u, v, chc) for this single frame,
    # same convention batch_process.py uses for the full folder.
    result_ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), result["u"]),
            "v": (("y", "x"), result["v"]),
            "chc": (("y", "x"), (~result["invalid"]).astype(float)),
        },
        coords={"x": result["x"][0, :], "y": result["y"][:, 0]},
    )
    return (result_ds,)


@app.cell
def _(PX_PER_MM, a1):
    # The image's true physical extent (from its pixel size and the real
    # calibration) - NOT the same as the vector grid's own x/y range, which
    # is smaller (PIV leaves a margin the width of half an interrogation
    # window with no vectors near each edge). Always pass this explicitly
    # to ds.piv.plot(background="image", ...); its image_extent default is
    # the vector grid's bounding box, which crops/distorts the image.
    image_extent = (0, a1.shape[1] / PX_PER_MM, 0, a1.shape[0] / PX_PER_MM)
    return (image_extent,)


@app.cell
def _(plt, result_ds):
    # Overlay the PIVPy quiver+streamlines plot on the raw frame-A image.
    _fig, _ax = plt.subplots(figsize=(6, 6))
    # _ax.imshow(a1, cmap="gray", extent=image_extent, origin="upper", alpha=0.85)
    result_ds.piv.plot(ax=_ax, background='magnitude', title="B0001 — PIV overlay on frame A")
    # _fig.savefig(ROOT / "outputs" / "pair1_pivpy_overlay.png", dpi=150)
    # mo.mpl.interactive(_fig)
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Colored quiver on the image\n"
        "PIVPy's `.piv.plot()` only overlays vectors on a *computed* scalar "
        "contour (vorticity, magnitude, ...), never a raw image, and its "
        "`.piv.quiver(colorbar=True)` only colors arrows by speed magnitude — "
        "not by an arbitrary component like streamwise velocity. `quiver_on_image()` "
        "in `piv_pipeline.py` fills that gap: our own minimal, purpose-built "
        "function for exactly this overlay (PIVlab-style, arrows colored "
        "continuously by any field, auto-scaled). It's also the prototype for a "
        "`background=\"image\"` / `color_by=` addition now made upstream in pivpy "
        "itself (this project's `pivpy` dependency points at the local `../pivpy` "
        "clone) — see below.\n\n"
        "**The two aren't identical.** `ds.piv.plot()` is PIVPy's general-purpose "
        "entry point (also does streamlines, a quiver key, vorticity/divergence "
        "backgrounds, etc.), so it carries more machinery and defaults that don't "
        "always match a hand-tuned call. In particular: `plot(background=\"image\")` "
        "defaults `image_extent` to the *vector grid's own* x/y range, not the "
        "image's true physical size - since PIV leaves an unvectored margin near "
        "each edge, that silently crops/distorts the image unless you pass "
        "`image_extent=` explicitly (done below, via the `image_extent` cell above)."
    )
    mo.md(_text)
    return


@app.cell
def _(ROOT, a1, mo, np, plt, quiver_on_image, result_ds):
    _fig, _ax = plt.subplots(figsize=(9, 9))
    _u, _v = result_ds["u"].values, result_ds["v"].values
    _x, _y = np.meshgrid(result_ds["x"].values, result_ds["y"].values)

    _Q = quiver_on_image(_ax, a1, _x, _y, _u, _v, color_by=_v)
    _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("streamwise velocity v [mm/frame]")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title("B0001 — vectors colored by streamwise velocity")
    _fig.savefig(ROOT / "outputs" / "pair1_colored_quiver.png", dpi=150)
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Same thing, as a pivpy one-liner\n"
        "`quiver_on_image()` above is now `ds.piv.plot(background=\"image\", "
        "image=..., color_by=...)` in pivpy — the simple shortcut."
    )
    mo.md(_text)
    return


@app.cell
def _(ROOT, a1, image_extent, result_ds):
    _fig, _ax = result_ds.piv.plot(
        background="image", image=a1, image_extent=image_extent, color_by="v",
        streamlines=False, quiver_key=False,
        title="B0001 — ds.piv.plot(background='image', color_by='v')",
    )
    _fig.savefig(ROOT / "outputs" / "pair1_pivpy_shortcut.png", dpi=150)
    # mo.mpl.interactive(_fig)
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Interactive exploration\n"
        "Tune the plot live before deciding what to run in `batch_process.py`. "
        "This notebook is a scratchpad for picking parameters, not a place to "
        "commit to them — the batch pipeline still uses fixed, documented "
        "defaults from `piv_pipeline.py`."
    )
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    color_by_dd = mo.ui.dropdown(
        options=["v", "u", "mag"], value="v", label="color arrows by"
    )
    arrow_cmap_dd = mo.ui.dropdown(
        options=["RdBu_r", "coolwarm", "seismic", "viridis", "PuOr", "jet"],
        value="viridis", label="arrow colormap",
    )
    background_cb = mo.ui.checkbox(value=True, label="show background image")
    image_cmap_dd = mo.ui.dropdown(
        options=["gray", "bone", "magma", "viridis"], value="gray", label="image colormap"
    )
    alpha_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.6, label="image alpha")
    streamlines_cb = mo.ui.checkbox(value=False, label="streamlines")
    row1 = mo.hstack([color_by_dd, arrow_cmap_dd, background_cb, image_cmap_dd, alpha_slider, streamlines_cb])

    arrow_scale_auto_cb = mo.ui.checkbox(value=True, label="auto arrow scale")
    arrow_scale_slider = mo.ui.slider(0.001, 0.2, step=0.001, value=0.02, label="arrow scale (if not auto)")
    arrow_width_slider = mo.ui.slider(0.001, 0.02, step=0.0005, value=0.0065, label="arrow width")
    arrow_alpha_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.75, label="arrow alpha")
    row2 = mo.hstack([arrow_scale_auto_cb, arrow_scale_slider, arrow_width_slider, arrow_alpha_slider])

    skip_rows_slider = mo.ui.slider(1, 10, step=1, value=1, label="skip rows")
    skip_cols_slider = mo.ui.slider(1, 10, step=1, value=1, label="skip cols")
    row3 = mo.hstack([skip_rows_slider, skip_cols_slider])

    mo.vstack([row1, row2, row3])
    return (
        alpha_slider,
        arrow_alpha_slider,
        arrow_cmap_dd,
        arrow_scale_auto_cb,
        arrow_scale_slider,
        arrow_width_slider,
        background_cb,
        color_by_dd,
        image_cmap_dd,
        skip_cols_slider,
        skip_rows_slider,
        streamlines_cb,
    )


@app.cell(hide_code=True)
def _(
    a1,
    alpha_slider,
    arrow_alpha_slider,
    arrow_cmap_dd,
    arrow_scale_auto_cb,
    arrow_scale_slider,
    arrow_width_slider,
    background_cb,
    color_by_dd,
    image_cmap_dd,
    image_extent,
    result_ds,
    skip_cols_slider,
    skip_rows_slider,
    streamlines_cb,
):
    _fig, _ax = result_ds.piv.plot(
        background="image" if background_cb.value else None,
        image=a1 if background_cb.value else None,
        image_extent=image_extent,
        image_cmap=image_cmap_dd.value, image_alpha=alpha_slider.value,
        color_by=color_by_dd.value, cmap=arrow_cmap_dd.value,
        arrow_scale=None if arrow_scale_auto_cb.value else arrow_scale_slider.value,
        arrow_width=arrow_width_slider.value, arrow_alpha=arrow_alpha_slider.value,
        skip=(skip_rows_slider.value, skip_cols_slider.value),
        streamlines=streamlines_cb.value, quiver_key=False,
        title=f"B0001 - color_by={color_by_dd.value!r}, cmap={arrow_cmap_dd.value!r}",
    )
    # mo.mpl.interactive(_fig)
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
