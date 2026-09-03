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
    from piv_pipeline import process_pair

    return (process_pair,)


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
def _(ROOT, a1, plt, result_ds):
    # Overlay the PIVPy quiver+streamlines plot on the raw frame-A image.
    scaling_factor = 100  # px/mm, same as process_pair()
    extent = (0, a1.shape[1] / scaling_factor, 0, a1.shape[0] / scaling_factor)

    _fig, _ax = plt.subplots(figsize=(9, 9))
    _ax.imshow(a1, cmap="gray", extent=extent, origin="upper", alpha=0.85)
    result_ds.piv.plot(ax=_ax, background=None, title="B0001 — PIV overlay on frame A")
    _fig.savefig(ROOT / "outputs" / "pair1_pivpy_overlay.png", dpi=150)
    return


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "## Colored quiver on the image\n"
        "PIVPy's `.piv.plot()` only overlays vectors on a *computed* scalar "
        "contour (vorticity, magnitude, ...), never a raw image, and its "
        "`.piv.quiver(colorbar=True)` only colors arrows by speed magnitude — "
        "not by an arbitrary component like streamwise velocity. So this is "
        "composed by hand: the raw image via `imshow`, arrows colored "
        "continuously by `v` (streamwise) via `matplotlib`'s `quiver`, reusing "
        "PIVPy's own auto-scaling heuristic (from `pivpy.graphics.plot`) so "
        "arrow spacing/length stay readable instead of a solid wall of ink."
    )
    mo.md(_text)
    return


@app.cell
def _(ROOT, a1, np, plt, result_ds):
    _x, _y = result_ds["x"].values, result_ds["y"].values
    _X, _Y = np.meshgrid(_x, _y)
    _u, _v = result_ds["u"].values, result_ds["v"].values

    # Same auto-scale heuristic as pivpy.graphics.plot(): ~16 arrows across the
    # longer grid axis, arrow length set from the median speed.
    _ny, _nx = _u.shape
    _step = max(1, round(max(_nx, _ny) / 16))
    _dx = abs(_x[1] - _x[0])
    _med_speed = np.nanmedian(np.hypot(_u, _v)) or 1.0
    _auto_scale = _med_speed / (0.85 * _step * _dx)

    _fig, _ax = plt.subplots(figsize=(9, 9))
    _extent = (0, a1.shape[1] / 100, 0, a1.shape[0] / 100)  # 100 px/mm, same as process_pair()
    _ax.imshow(a1, cmap="gray", extent=_extent, origin="upper", alpha=0.6)

    _Q = _ax.quiver(
        _X[::_step, ::_step], _Y[::_step, ::_step],
        _u[::_step, ::_step], _v[::_step, ::_step],
        _v[::_step, ::_step],  # color by streamwise (vertical) velocity
        cmap="viridis", angles="xy", scale_units="xy", scale=_auto_scale,
        width=0.004, pivot="mid",
    )
    _cbar = _fig.colorbar(_Q, ax=_ax, pad=0.03, shrink=0.85)
    _cbar.set_label("streamwise velocity v [mm/frame]")
    _ax.set_aspect("equal")
    _ax.set_xlabel("x [mm]")
    _ax.set_ylabel("y [mm]")
    _ax.set_title("B0001 — vectors colored by streamwise velocity")
    _fig.savefig(ROOT / "outputs" / "pair1_colored_quiver.png", dpi=150)
    return


if __name__ == "__main__":
    app.run()
