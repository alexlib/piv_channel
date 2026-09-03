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
    from piv_pipeline import process_pair, quiver_on_image

    return process_pair, quiver_on_image


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
        "not by an arbitrary component like streamwise velocity. `quiver_on_image()` "
        "in `piv_pipeline.py` fills that gap (PIVlab-style: raw image + arrows "
        "colored continuously by any field, auto-scaled like PIVPy's own plot()). "
        "It's also the prototype for a `background=\"image\"` / `color_by=` shortcut "
        "now added upstream in pivpy itself (this project's `pivpy` dependency "
        "points at the local `../pivpy` clone) — see the last cell below."
    )
    mo.md(_text)
    return


@app.cell
def _(ROOT, a1, np, plt, quiver_on_image, result_ds):
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
def _(ROOT, a1, result_ds):
    _fig, _ax = result_ds.piv.plot(
        background="image", image=a1, color_by="v",
        streamlines=False, quiver_key=False,
        title="B0001 — ds.piv.plot(background='image', color_by='v')",
    )
    _fig.savefig(ROOT / "outputs" / "pair1_pivpy_shortcut.png", dpi=150)
    return


if __name__ == "__main__":
    app.run()
