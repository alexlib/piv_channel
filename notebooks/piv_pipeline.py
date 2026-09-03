"""Single-pair PIV pipeline, factored out of pair1_first_pass.py so both that
notebook and the batch notebook use the exact same tested parameters."""

import imageio.v3 as iio
import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
import numpy as np
from openpiv import tools, scaling as opiv_scaling, validation, filters
import openpiv.pyprocess as pyprocess

# Real DaVis calibration for this camera/lens, from
# D:\channel_flow_research\baseline_channel\Properties\Calibration\Calibration.xml
# (PixelPerMmFactor). The 100 px/mm used in the first-attempt scripts was a
# placeholder guess and gave a 10 mm channel gap as ~17-20 mm.
PX_PER_MM = 173.419


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
    scaling_factor=PX_PER_MM,
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


def quiver_on_image(ax, image, x, y, u, v, *, color_by=None, cmap="viridis",
                     extent=None, scaling_factor=PX_PER_MM, image_alpha=0.6,
                     image_cmap="gray", arrow_width=0.004):
    """PIVlab-style plot: raw image with a quiver colored by a continuous field.

    color_by: 2D array same shape as u/v to color the arrows by (e.g. v for
    streamwise velocity, or np.hypot(u, v) for speed). None -> single color.
    cmap: colormap for the arrows when color_by is set.
    image_cmap: colormap for the background image (default "gray").
    Arrow spacing/length are auto-scaled the same way pivpy.graphics.plot()
    does, so the field stays readable instead of a solid wall of arrows.
    extent: (left, right, bottom, top) in the same physical units as x/y;
    if None, derived from the image shape and scaling_factor (px/unit).
    """
    if extent is None:
        extent = (0, image.shape[1] / scaling_factor, 0, image.shape[0] / scaling_factor)
    ax.imshow(image, cmap=image_cmap, extent=extent, origin="upper", alpha=image_alpha)

    ny, nx = u.shape
    step = max(1, round(max(nx, ny) / 16))
    dx = abs(x[0, 1] - x[0, 0]) if x.ndim == 2 else abs(x[1] - x[0])
    med_speed = np.nanmedian(np.hypot(u, v)) or 1.0
    auto_scale = med_speed / (0.85 * step * dx)

    color_args = (color_by[::step, ::step],) if color_by is not None else ()
    quiver = ax.quiver(
        x[::step, ::step], y[::step, ::step], u[::step, ::step], v[::step, ::step],
        *color_args,
        cmap=cmap if color_by is not None else None,
        angles="xy", scale_units="xy", scale=auto_scale,
        width=arrow_width, pivot="mid",
    )
    ax.set_aspect("equal")
    return quiver
