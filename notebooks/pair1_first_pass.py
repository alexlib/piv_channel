import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import imageio.v3 as iio
    import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, iio, mo, np, plt


@app.cell
def _():
    from openpiv import tools, scaling, validation, filters
    import openpiv.pyprocess as pyprocess

    return filters, pyprocess, scaling, tools, validation


@app.cell
def _(iio, np):
    # First test pair: a single dual-frame TIFF (frame A stacked on top of frame B).
    a = iio.imread(r"sample_data\tiff\B0001.tif")
    display_min = 0
    display_max = 150
    a = a.astype(float)
    np.clip(a, display_min, display_max, out=a)
    a -= display_min
    a = ((255.0 / (display_max - display_min)) * a).astype(np.uint8)
    return (a,)


@app.cell
def _(a, np, plt):
    a1 = a[:2048, 200:1945]
    a2 = a[2048:, 200:1945]
    plt.figure(figsize=(10, 10))
    plt.imshow(np.stack([a1, a2, a2 * 0], axis=2))
    plt.show()
    plt.imsave(r"outputs\tmp.png", np.stack([a1, a2, a2 * 0], axis=2))
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


@app.cell
def _(a1, a2, filters, np, pyprocess, scaling, tools, validation):
    # dt is unknown for this camera (not a high-speed camera; see README "Timing").
    # Vectors below are in px/frame until dt is recovered from the DaVis metadata.
    winsize = 64  # >= 4x the measured displacement
    searchsize = 96  # winsize + 2x displacement, with margin
    overlap = 32  # 50% overlap
    dt = 1.0

    u0, v0, s2n = pyprocess.extended_search_area_piv(
        a1.astype(np.int32),
        a2.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method="peak2peak",
    )
    x, y = pyprocess.get_coordinates(
        image_size=a1.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )

    mask_s2n = validation.sig2noise_val(s2n, threshold=1.3)
    mask_med = validation.local_median_val(u0, v0, u_threshold=3, v_threshold=3, size=1)
    invalid = mask_s2n | mask_med
    print(f"Invalid vectors: {invalid.sum()} / {invalid.size} ({100 * invalid.mean():.1f}%)")

    u2, v2 = filters.replace_outliers(
        u0, v0, invalid,
        method="localmean", max_iter=10, kernel_size=3,
    )

    xs, ys, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=100)
    xs, ys, u3, v3 = tools.transform_coordinates(xs, ys, u3, v3)
    tools.save(r"outputs\pair1.txt", xs, ys, u3, v3, invalid)
    return


@app.cell
def _(Path, plt, tools):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    tools.display_vector_field(
        Path(r"outputs\pair1.txt"), ax=_ax, scaling_factor=100,
        scale=1, width=0.0035,
        on_img=True, image_name=r"outputs\tmp.png",
    )
    _fig.savefig(r"outputs\pair1_quiver.png", dpi=150)
    return


if __name__ == "__main__":
    app.run()
