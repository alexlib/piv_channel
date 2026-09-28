import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lvpyio as lv
    import imageio.v3 as iio
    import imagecodecs
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import exposure


    return Path, iio, mo, np, plt


@app.cell
def _():
    from openpiv import windef  # <---- see windef.py for details
    from openpiv import tools, scaling, validation, filters, preprocess
    import openpiv.pyprocess as process
    from openpiv import pyprocess
    from time import time
    import warnings

    return filters, pyprocess, scaling, tools, validation


@app.cell
def _(iio, np):
    a = iio.imread(r'tiff\B0001.tif')
    display_min = 0
    display_max = 150
    a = a.astype(float)
    np.clip(a, display_min, display_max, out=a)
    a -= display_min
    a = ((255. / (display_max - display_min)) * a).astype(np.uint8)


    # a = exposure.equalize_adapthist(a)
    # a = exposure.rescale_intensity(a)
    return (a,)


@app.cell
def _():
    return


@app.cell
def _(a, np, plt):
    a1 = a[:2048, 200:1945]
    a2 = a[2048:, 200:1945]
    # fig, ax = plt.subplots(1,3,figsize=(6,18))
    # ax[0].imshow(a1)
    # ax[1].imshow(a2)
    # ax[2].imshow(np.abs(a2-a1))
    plt.figure(figsize=(10,10))
    plt.imshow(np.stack([a1,a2,a2*0],axis=2))
    plt.show()
    plt.imsave('tmp.png', np.stack([a1,a2,a2*0],axis=2))
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
    _winsize = 64          # ≥ 4× the measured displacement
    _searchsize = 96       # winsize + 2× displacement, with margin
    _overlap = 32          # keep 50% overlap
    dt = 1.0

    _u0, _v0, _s2n = pyprocess.extended_search_area_piv(
        a1.astype(np.int32),
        a2.astype(np.int32),
        window_size=_winsize,
        overlap=_overlap,
        dt=dt,
        search_area_size=_searchsize,
        sig2noise_method='peak2peak',
    )
    _x, _y = pyprocess.get_coordinates(
        image_size=a1.shape,
        search_area_size=_searchsize,
        overlap=_overlap,
    )

    # 1) stricter signal-to-noise validation
    _mask_s2n = validation.sig2noise_val(_s2n, threshold=1.3)

    # 2) local median validation catches remaining outliers
    _mask_med = validation.local_median_val(_u0, _v0, u_threshold=3, v_threshold=3, size=1)

    _invalid = _mask_s2n | _mask_med
    print(f"Invalid vectors: {_invalid.sum()} / {_invalid.size} "
          f"({100*_invalid.mean():.1f}%)")

    _u2, _v2 = filters.replace_outliers(
        _u0, _v0, _invalid,
        method='localmean', max_iter=10, kernel_size=3,
    )

    _xs, _ys, _u3, _v3 = scaling.uniform(_x, _y, _u2, _v2, scaling_factor=100)
    _xs, _ys, _u3, _v3 = tools.transform_coordinates(_xs, _ys, _u3, _v3)
    tools.save('tmp2.txt', _xs, _ys, _u3, _v3, _invalid)
    return


@app.cell
def _(Path, plt, tools):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    tools.display_vector_field(
        Path('tmp2.txt'), ax=_ax, scaling_factor=100,
        scale=1, width=0.0035,
        on_img=True, image_name='tmp.png',
    )
    return


app._unparsable_cell(
    r"""
    # ---- Multipass PIV with window deformation (windef API) ----
    piv_settings = windef.PIVSettings()

    # geometry: coarse -> fine
    piv_settings.num_iterations = 3
    piv_settings.windowsizes = (64, 32, 32)   # last size repeated = refinement pass
    piv_settings.overlap = (32, 16, 16)       # 50% overlap each pass
    piv_settings.deformation_method = "symmetric"

    # correlation
    piv_settings.correlation_method = "circular"
    piv_settings.sig2noise_method = "peak2peak"
    piv_settings.dt = dt
    piv_settings.scaling_factor = 100

    # validation (applied inside every pass)
    piv_settings.validation_first_pass = True
    piv_settings.sig2noise_validate = True
    piv_settings.sig2noise_threshold = 1.3
    piv_settings.median_threshold = 3
    piv_settings.median_size = 1
    piv_settings.std_threshold = 7
    piv_settings.min_max_u_disp = (-40, 40)   # px/frame — widen if Cell-1 shift is larger
    piv_settings.min_max_v_disp = (-40, 40)

    # outlier replacement between passes
    piv_settings.replace_vectors = True
    piv_settings.filter_method = "localmean"
    piv_settings.max_filter_iteration = 10
    piv_settings.filter_kernel_size = 3

    # optional smoothing of the predictor field (helps deformation stability)
    piv_settings.smoothn = True
    piv_settings.smoothn_p = 0.5

    frame_a = a1.astype(np.float64)
    frame_b = a2.astype(np.float64)

    # --- pass 1: standard cross-correlation on the coarse grid ---
    x_mp, y_mp, u_mp, v_mp, s2n_mp = windef.first_pass(frame_a, frame_b, piv_settings)

    # typical_validation returns ONLY the boolean invalid-vector mask
    # (unpacking it into u, v, mask caused "too many values to unpack")
    mask_mp = validation.typical_validation(u_mp, v_mp, s2n_mp, piv_settings)
    print(f"pass 1: {mask_mp.sum()}/{mask_mp.size} invalid ({100*mask_mp.mean():.1f}%)")

    u_mp, v_mp = filters.replace_outliers(
        u_mp, v_mp, mask_mp,
        method=piv_settings.filter_method,
        max_iter=piv_settings.max_filter_iteration,
        kernel_size=piv_settings.filter_kernel_size,
    )

    # --- passes 2..N: image deformation on progressively finer grids ---
    # multipass_img_deform returns (x, y, u, v, grid_mask, flags)
    for _i in range(1, piv_settings.num_iterations):
        x_mp, y_mp, u_mp, v_mp, grid_mask_mp, mask_mp = windef.multipass_img_deform(
            frame_a, frame_b, _i, x_mp, y_mp, u_mp, v_mp, piv_settings,
        )
        print(f"pass {_i + 1}: grid {u_mp.shape}, "
              f"{np.count_nonzero(mask_mp)}/{mask_mp.size} invalid")

    # masked arrays -> plain arrays
    if np.ma.isMaskedArray(u_mp):
        u_mp = u_mp.filled(0.0)
        v_mp = v_mp.filled(0.0)

    # px/frame -> px/s -> mm/s, then flip to physical coordinates
    u_mp = u_mp / piv_settings.dt
    v_mp = v_mp / piv_settings.dt

    x_ph, y_ph, u_ph, v_ph = scaling.uniform(
        x_mp, y_mp, u_mp, v_mp, scaling_factor=piv_settings.scaling_factor,
    )
    x_ph, y_ph, u_ph, v_ph = tools.transform_coordinates(x_ph, y_ph, u_ph, v_ph)
    tools.save("multipass.txt", x_ph, y_ph, u_ph, v_ph, mask_mp)

    print(f"final grid: {u_mp.shape[0]} x {u_mp.shape[1]} vectors, "
          f"mean |V| = {np.hypot(u_ph, v_ph).mean():.2f} mm/s")
    ```

    The error came from the line `u_mp, v_mp, mask_mp = validation.typical_validation(...)`. In the current OpenPIV, this function returns only a single boolean flag array — in `windef.py` it is used as `flags = validation.typical_validation(u, v, s2n, settings)` followed by `u, v = filters.replace_outliers(u, v, flags)`. Unpacking that single 2D array into three variables iterates over its rows, hence "too many values to unpack (expected 3)".

    I also renamed the fifth return value of `multipass_img_deform` to `grid_mask_mp`, since the function returns `x, y, u, v, grid_mask, flags` — the fifth item is the grid mask, not the signal-to-noise ratio.
    """,
    name="_"
)


@app.cell
def _(Path, piv_settings, plt, tools):
    _fig_mp, _ax_mp = plt.subplots(figsize=(9, 9))
    tools.display_vector_field(
        Path("multipass.txt"),
        ax=_ax_mp,
        scaling_factor=piv_settings.scaling_factor,
        scale=100,
        width=0.0035,
        on_img=True,
        image_name="tmp.png",
    )
    _ax_mp.set_title(
        f"Multipass PIV: {piv_settings.windowsizes} px windows, "
        f"{piv_settings.deformation_method} deformation"
    )
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
