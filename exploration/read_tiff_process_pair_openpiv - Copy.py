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


    return Path, lv, np, plt


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
def _(lv):
    lv.read_buffer('./zipped/B0001.im7')
    return


@app.cell
def _():
    return


@app.cell
def _(np):

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
def _(a1, a2, filters, np, pyprocess, scaling, tools, validation):
    winsize = 32 # pixels, interrogation window size in frame A
    searchsize = 48  # pixels, search area size in frame B
    overlap = 16 # pixels, 50% overlap
    dt = 0.02 # sec, time interval between the two frames

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        a1.astype(np.int32),
        a2.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak',
    )
    x, y = pyprocess.get_coordinates(
        image_size=a1.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )
    invalid_mask = validation.sig2noise_val(
        sig2noise,
        threshold = 1.05,
    )
    u2, v2 = filters.replace_outliers(
        u0, v0,
        invalid_mask,
        method='localmean',
        max_iter=3,
        kernel_size=3,
    )
    # convert x,y to mm
    # convert u,v to mm/sec

    x, y, u3, v3 = scaling.uniform(
        x, y, u2, v2,
        scaling_factor = 100,  # 100 pixels/millimeter
    )

    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    tools.save('tmp.txt' , x, y, u3, v3, invalid_mask)
    return


@app.cell
def _(Path, plt, tools):
    fig, ax = plt.subplots(figsize=(8,8))
    tools.display_vector_field(
        Path('tmp.txt'),
        ax=ax, scaling_factor=100,
        scale=100, # scale defines here the arrow length
        width=0.0035, # width is the thickness of the arrow
        on_img=True, # overlay on the image
        image_name= 'tmp.png',
    );
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
