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

    return lv, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # channel_04_zoom_in - image comparison

    Side-by-side comparison of every raw/preprocessed image variant
    found across `first_half_case_1`, `first_half_case_2`, and
    `second_half`, to check whether they share the same field of view
    and to compare raw vs DaVis's own `ImgPreproc` background-subtracted
    output. Findings so far: `case_1` and `case_2` capture visibly
    different portions of the wavy wall despite identical pixel
    dimensions (2432x2048); `second_half`'s `exported`/`ImgPreproc`/
    `ImgPreproc_01` all share the same field of view, and
    `ImgPreproc`/`ImgPreproc_01`'s first frame is pixel-identical
    (`ImgPreproc_01` looks like a superseded partial run - only 1000
    files vs `ImgPreproc`'s 3000).
    """)
    return


@app.cell
def _():
    ROOT = r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445"

    IMAGE_PATHS = {
        "case1_exported": ROOT + r"\first_half_case_1\exported_images\first_half_case_1\B0001.im7",
        "case1_ImgPreproc_03": ROOT + r"\first_half_case_1\ImgPreproc_03\B00001.im7",
        "case2_exported": ROOT + r"\first_half_case_2\exported_images\first_half_case_2\B0001.im7",
        "case2_ImgPreproc": ROOT + r"\first_half_case_2\ImgPreproc\B00001.im7",
        "second_half_exported": ROOT + r"\second_half\exported_images\second_half\B0001.im7",
        "second_half_ImgPreproc": ROOT + r"\second_half\ImgPreproc\B00001.im7",
        "second_half_ImgPreproc_01": ROOT + r"\second_half\ImgPreproc_01\B00001.im7",
    }
    return (IMAGE_PATHS,)


@app.cell
def _(IMAGE_PATHS, lv, np):
    images = {
        name: np.asarray(lv.read_buffer(path).as_masked_array(0).data)
        for name, path in IMAGE_PATHS.items()
    }
    return (images,)


@app.cell
def _(mo):
    mo.md("""
    ## Stats
    """)
    return


@app.cell
def _(images, mo):
    _rows = [
        f"| {name} | {img.shape} | {img.min():.0f} | {img.max():.0f} | {img.mean():.2f} |"
        for name, img in images.items()
    ]
    mo.md(
        "| name | shape | min | max | mean |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    vmax_pct_slider = mo.ui.slider(90.0, 99.99, value=99.5, step=0.1, label="display vmax percentile")
    vmax_pct_slider
    return (vmax_pct_slider,)


@app.cell
def _(images, np, plt, vmax_pct_slider):
    _fig, _axes = plt.subplots(1, len(images), figsize=(4 * len(images), 10))
    for _ax, (_name, _img) in zip(_axes, images.items()):
        _vmax = np.percentile(_img, vmax_pct_slider.value)
        _ax.imshow(_img, cmap="gray", vmin=0, vmax=_vmax)
        _ax.set_title(f"{_name}\nshape={_img.shape}", fontsize=9)
    _fig.tight_layout()
    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
