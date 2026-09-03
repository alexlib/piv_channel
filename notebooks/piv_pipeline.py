"""Single-pair PIV pipeline, factored out of pair1_first_pass.py so both that
notebook and the batch notebook use the exact same tested parameters."""

import imageio.v3 as iio
import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
import numpy as np
from openpiv import tools, scaling as opiv_scaling, validation, filters
import openpiv.pyprocess as pyprocess


def process_pair(
    tif_path,
    crop_cols=(200, 1945),
    display_min=0,
    display_max=150,
    winsize=64,
    searchsize=96,
    overlap=32,
    dt=1.0,
    s2n_threshold=1.3,
    median_threshold=3,
    median_size=1,
    scaling_factor=100,
):
    """Run single-pass PIV on one dual-frame TIFF (frame A stacked on frame B).

    Same parameters as the exploratory pipeline in pair1_first_pass.py.
    Returns physical-coordinate x, y, u, v grids plus the outlier mask and
    raw signal-to-noise ratio (before outlier replacement).
    """
    img = iio.imread(tif_path).astype(float)
    np.clip(img, display_min, display_max, out=img)
    img -= display_min
    img = ((255.0 / (display_max - display_min)) * img).astype(np.uint8)

    h = img.shape[0] // 2
    a1 = img[:h, crop_cols[0]:crop_cols[1]]
    a2 = img[h:, crop_cols[0]:crop_cols[1]]

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

    mask_s2n = validation.sig2noise_val(s2n, threshold=s2n_threshold)
    mask_med = validation.local_median_val(
        u0, v0, u_threshold=median_threshold, v_threshold=median_threshold, size=median_size
    )
    invalid = mask_s2n | mask_med

    u2, v2 = filters.replace_outliers(
        u0, v0, invalid, method="localmean", max_iter=10, kernel_size=3,
    )

    xs, ys, u3, v3 = opiv_scaling.uniform(x, y, u2, v2, scaling_factor=scaling_factor)
    xs, ys, u3, v3 = tools.transform_coordinates(xs, ys, u3, v3)

    return {"x": xs, "y": ys, "u": u3, "v": v3, "invalid": invalid, "s2n": s2n}
