import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# channel_04 - raw export vs. DaVis `ImgPreproc`\n"
        "Compares the raw `.im7` this project reads "
        "(`exported_images\\...\\B0001.im7`) against DaVis's own "
        "preprocessed version of the same acquisition "
        "(`ImgPreproc\\B00001.im7`, note the extra digit in the filename) - "
        "the input DaVis's own multi-pass PIV "
        "(`PIV_MPd(4x16x16_25%ov_ImgCorr)`) actually correlates on. Useful "
        "context for `piv_pipeline.process_im7_pair(preprocess=\"high_pass\")`: "
        "if DaVis's own preprocessing is doing something similar (or "
        "different), that explains part of the density/coverage gap versus "
        "the DaVis reference image."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import lvpyio as lv

    return lv, mo, np, plt


@app.cell
def _(mo):
    RAW_PATH = (
        r"D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\exported_images"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\B0001.im7"
    )
    PREPROC_PATH = (
        r"D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547"
        r"\channel_04_Vmax_0p66_m2sec_steady_state_part_1\ImgPreproc\B00001.im7"
    )
    mo.md(f"`RAW_PATH = {RAW_PATH}`  \n`PREPROC_PATH = {PREPROC_PATH}`")
    return PREPROC_PATH, RAW_PATH


@app.cell
def _(PREPROC_PATH, RAW_PATH, lv, np):
    _raw_buf = lv.read_buffer(RAW_PATH)
    _pre_buf = lv.read_buffer(PREPROC_PATH)

    a_raw = np.asarray(_raw_buf.as_masked_array(0).data)
    a_pre = np.asarray(_pre_buf.as_masked_array(0).data)
    return a_pre, a_raw


@app.cell
def _(a_pre, a_raw, mo, np):
    def _stats(a):
        return dict(
            shape=a.shape, dtype=str(a.dtype),
            min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
            p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)),
            p99=float(np.percentile(a, 99)),
        )

    _rs, _ps = _stats(a_raw), _stats(a_pre)
    mo.md(
        f"| | raw (exported_images) | ImgPreproc |\n"
        f"|---|---|---|\n"
        f"| shape | {_rs['shape']} | {_ps['shape']} |\n"
        f"| dtype | {_rs['dtype']} | {_ps['dtype']} |\n"
        f"| min | {_rs['min']:.1f} | {_ps['min']:.1f} |\n"
        f"| max | {_rs['max']:.1f} | {_ps['max']:.1f} |\n"
        f"| mean | {_rs['mean']:.2f} | {_ps['mean']:.2f} |\n"
        f"| median (p50) | {_rs['p50']:.1f} | {_ps['p50']:.1f} |\n"
        f"| p90 | {_rs['p90']:.1f} | {_ps['p90']:.1f} |\n"
        f"| p99 | {_rs['p99']:.1f} | {_ps['p99']:.1f} |\n\n"
        f"Lower mean/median in `ImgPreproc` with a similar max suggests "
        f"background subtraction (particles keep their peak brightness, "
        f"the diffuse background around them drops) - the same idea as "
        f"this project's own `high_pass` preprocessing, worth comparing "
        f"visually below."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Side by side (shared display range)
    """)
    return


@app.cell
def _(mo):
    display_max_slider = mo.ui.slider(20, 500, step=10, value=60, label="display max")
    display_max_slider
    return (display_max_slider,)


@app.cell
def _(a_pre, a_raw, display_max_slider, np, plt):
    def _to_u8(a, dmax):
        a = np.clip(a.astype(float), 0, dmax)
        return ((255.0 / dmax) * a).astype(np.uint8)

    _dmax = display_max_slider.value
    _raw8 = _to_u8(a_raw, _dmax)
    _pre8 = _to_u8(a_pre, _dmax)

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 10))
    _axes[0].imshow(_raw8, cmap="gray", origin="upper")
    _axes[0].set_title(f"raw (exported_images), display_max={_dmax}")
    _axes[1].imshow(_pre8, cmap="gray", origin="upper")
    _axes[1].set_title(f"ImgPreproc, display_max={_dmax}")
    for _ax in _axes:
        _ax.set_xlabel("column [px]")
        _ax.set_ylabel("row [px]")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Difference (ImgPreproc - raw) and intensity histograms
    """)
    return


@app.cell
def _(a_pre, a_raw, np, plt):
    _diff = a_pre.astype(float) - a_raw.astype(float)

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))
    _im = _axes[0].imshow(_diff, cmap="RdBu_r", vmin=-50, vmax=50, origin="upper")
    _axes[0].set_title("ImgPreproc - raw (clipped ±50)")
    _fig.colorbar(_im, ax=_axes[0], shrink=0.8)

    _bins = np.linspace(0, 200, 100)
    _axes[1].hist(a_raw.ravel(), bins=_bins, alpha=0.5, label="raw", density=True)
    _axes[1].hist(a_pre.ravel(), bins=_bins, alpha=0.5, label="ImgPreproc", density=True)
    _axes[1].set_yscale("log")
    _axes[1].set_xlabel("intensity")
    _axes[1].set_ylabel("density (log)")
    _axes[1].legend()
    _axes[1].set_title("Intensity histograms (0-200 range)")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Zoomed comparison (one well-seeded patch)
    """)
    return


@app.cell
def _(a_pre, a_raw, np, plt):
    _r0, _r1, _c0, _c1 = 600, 800, 600, 800

    def _to_u8_local(a, dmax):
        a = np.clip(a.astype(float), 0, dmax)
        return ((255.0 / dmax) * a).astype(np.uint8)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))
    _axes[0].imshow(_to_u8_local(a_raw[_r0:_r1, _c0:_c1], 60), cmap="gray")
    _axes[0].set_title("raw, zoomed patch")
    _axes[1].imshow(_to_u8_local(a_pre[_r0:_r1, _c0:_c1], 60), cmap="gray")
    _axes[1].set_title("ImgPreproc, zoomed patch (same display range)")
    _fig.tight_layout()
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
