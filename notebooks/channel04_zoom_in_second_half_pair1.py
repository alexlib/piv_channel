# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "lvpyio",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import lvpyio as lv
    from scipy import ndimage as ndi
    from scipy.optimize import curve_fit

    return curve_fit, lv, mo, ndi, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Sinusoidal channel, zoom-in (second_half) - right-wall mask fitting

    Single-pair exploration to fit a sinusoidal wall model to the right
    wall visible in this near-wall zoom-in camera view (only one wall is
    in frame here, unlike the full-channel `channel04_steady_state_pair1.py`
    which fits both walls). Sliders control one wave
    (`sinusoidal_wall_bounds` with the left wall parameters set to
    never trigger, since there's no left wall in this crop).
    """)
    return


@app.cell
def _(lv, np):
    # SECOND_HALF = (
    #     r"D:\channel_flow_research\channel_04_zoom_in"
    #     r"\Project_FlowMaster_260705_110445\second_half"
    #     r"\exported_images\second_half\B0001.im7"
    # )

    SECOND_HALF = r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\second_half\ImgPreproc_01\B00001.im7"

    SECOND_HALF = r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\second_half\ImgPreproc\B00001.im7"

    FIRST_HALF_2 = (r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\first_half_case_2\ImgPreproc\B00001.im7")

    FIRST_HALF_1 = (r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\first_half_case_1\ImgPreproc_03\B00001.im7")

    second_half = np.asarray(lv.read_buffer(SECOND_HALF).as_masked_array(0).data)
    first_half_2 = np.asarray(lv.read_buffer(FIRST_HALF_2).as_masked_array(0).data)
    first_half_1 = np.asarray(lv.read_buffer(FIRST_HALF_1).as_masked_array(0).data)

    frame_a = first_half_1
    frame_b = second_half
    frame_a.shape
    return frame_a, frame_b


@app.cell
def _(mo):
    hp_sigma_slider = mo.ui.slider(1, 30, step=1, value=16, label="high_pass sigma")
    hp_pct_slider = mo.ui.slider(90.0, 99.99, step=0.1, value=99.0, label="contrast stretch percentile")
    mo.hstack([hp_sigma_slider, hp_pct_slider])
    return hp_pct_slider, hp_sigma_slider


@app.cell
def _(frame_a, frame_b, hp_pct_slider, hp_sigma_slider, np):
    import openpiv.preprocess as _pp

    def _to_uint8_highpass(frame, sigma, pct):
        hp = _pp.high_pass(frame.astype(float), sigma=sigma, clip=True)
        pmax = np.percentile(hp, pct)
        hp = np.clip(hp, 0, pmax)
        return ((255.0 / pmax) * hp).astype(np.uint8)

    frame_a_proc = _to_uint8_highpass(frame_a, hp_sigma_slider.value, hp_pct_slider.value)
    frame_b_proc = _to_uint8_highpass(frame_b, hp_sigma_slider.value, hp_pct_slider.value)
    frame_a_proc.shape
    return frame_a_proc, frame_b_proc


@app.cell
def _(mo):
    dark_threshold_slider = mo.ui.slider(1.0, 20.0, value=5.0, step=0.5, label="frame_a dark-region threshold")
    reliable_from_slider = mo.ui.slider(200, 1500, value=700, step=50, label="frame_b reliable-ridge cutoff (px into frame_b)")
    mo.vstack([dark_threshold_slider, reliable_from_slider])
    return dark_threshold_slider, reliable_from_slider


@app.cell
def _(
    curve_fit,
    dark_threshold_slider,
    frame_a_proc,
    frame_b_proc,
    mo,
    ndi,
    np,
    reliable_from_slider,
):
    # Almost-sinusoidal mask: center + drift*row + amplitude*sin(2*pi*row/wavelength + phase).
    # A pure sinusoid systematically missed both peak regions - the extra
    # `drift*row` linear term accounts for the camera not being perfectly
    # vertical relative to the channel (a few pixels of rotation per row), which
    # a real sinusoidal wall would still produce if imaged at a slight tilt.
    #
    # The trustworthy (row, column) points feeding the fit come from two
    # detectors, since frame_a and frame_b have very different image quality:
    # frame_a (first_half_1) has a genuine dark exterior region -> blur +
    # threshold + connected-component fill tracks the true bright/dark boundary
    # directly. frame_b (second_half)'s exterior and interior background are
    # statistically indistinguishable after blurring (checked directly: 0.88 vs
    # 0.89) - too low-SNR for region fill, so its own bright reflection ridge is
    # traced instead, kept only past `reliable_from_slider` px into the frame
    # (before that point the reflection isn't yet optically resolved, and
    # "brightest column" is just tracking stray particles).
    h_a, w_a = frame_a_proc.shape
    h_b, w_b = frame_b_proc.shape

    _blurred_a = ndi.gaussian_filter(frame_a_proc.astype(float), sigma=15)
    _dark_a = _blurred_a <= dark_threshold_slider.value
    _labeled_a, _ = ndi.label(_dark_a)
    _dark_region_a = _labeled_a == _labeled_a[0, -1]
    _edge_a = np.full(h_a, np.nan)
    for _r in range(h_a):
        _row_dark = _dark_region_a[_r, 700:]
        if _row_dark.any():
            _edge_a[_r] = 700 + np.argmax(_row_dark)
    edge_a = ndi.median_filter(np.nan_to_num(_edge_a, nan=w_a - 1), size=31)
    rows_a_abs = np.arange(h_a)

    _band_b = frame_b_proc[:, 700:].astype(float)
    _col_b = _band_b.argmax(axis=1) + 700
    _edge_b_raw = ndi.median_filter(_col_b.astype(float), size=51)
    reliable_from = reliable_from_slider.value
    rows_b_abs = np.arange(h_a + reliable_from, h_a + h_b)
    edge_b_reliable = _edge_b_raw[reliable_from:]

    _rows_fit = np.concatenate([rows_a_abs, rows_b_abs])
    _edge_fit = np.concatenate([edge_a, edge_b_reliable])
    _order = np.argsort(_rows_fit)
    _rows_fit, _edge_fit = _rows_fit[_order], _edge_fit[_order]

    _rolling_med = ndi.median_filter(_edge_fit, size=41)
    _keep = np.abs(_edge_fit - _rolling_med) < 60
    _rows_fit, _edge_fit = _rows_fit[_keep], _edge_fit[_keep]

    def rotated_sine(row, wavelength, phase, center, amplitude, drift):
        return center + drift * row + amplitude * np.sin(2 * np.pi * row / wavelength + phase)

    _p0 = [5010, 4.73, 1380, 566, 0.0]
    wall_fit_params, _ = curve_fit(rotated_sine, _rows_fit, _edge_fit, p0=_p0, maxfev=20000)

    n_rows_stitched = h_a + h_b
    rows_stitched = np.arange(n_rows_stitched)
    right_edge = rotated_sine(rows_stitched, *wall_fit_params)
    left_edge = np.zeros(n_rows_stitched)  # no left wall in this crop
    mo.md(
        f"fitted wavelength={wall_fit_params[0]:.1f}px, phase={wall_fit_params[1]:.3f}rad, "
        f"center={wall_fit_params[2]:.1f}px, amplitude={wall_fit_params[3]:.1f}px, "
        f"drift={wall_fit_params[4]:.4f}px/row"
    )
    return right_edge, rows_stitched


@app.cell
def _(frame_a, frame_a_proc, frame_b_proc, np, plt, right_edge, rows_stitched):
    _fig, _ax = plt.subplots(figsize=(6, 20))
    _stitched = np.vstack([frame_a_proc, frame_b_proc])
    _ax.imshow(_stitched, cmap="gray", vmin=0, vmax=255)
    _ax.plot(right_edge, rows_stitched, color="red", linewidth=1.5, label="fitted wall boundary (rotated sinusoid)")
    _ax.axhline(frame_a.shape[0], color="cyan", linewidth=0.8, linestyle="--", label="first/second half seam")
    _ax.set_title("high_pass preprocessed, stitched: frame_a (top) / frame_b (bottom)")
    _ax.legend()
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
