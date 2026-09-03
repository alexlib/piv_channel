"""Single-pair PIV pipeline, factored out of pair1_first_pass.py so both that
notebook and the batch notebooks use the exact same tested parameters."""

import imageio.v3 as iio
import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
import lvpyio as lv
import numpy as np
from openpiv import tools, scaling as opiv_scaling, validation, filters
import openpiv.pyprocess as pyprocess

# Real DaVis calibration for this camera/lens, from
# D:\channel_flow_research\baseline_channel\Properties\Calibration\Calibration.xml
# (PixelPerMmFactor). The 100 px/mm used in the first-attempt scripts was a
# placeholder guess and gave a 10 mm channel gap as ~17-20 mm.
PX_PER_MM = 173.419

# Channel gap columns in pixel space, same camera/session for both the
# sample_data test TIFF and the real .im7 acquisitions in
# D:\channel_flow_research\baseline_channel - verified visually (the two
# bright wall reflections sit almost exactly at these columns).
CHANNEL_CROP_COLS = (200, 1945)


def _run_piv(
    a1, a2, *,
    winsize, searchsize, overlap, dt,
    s2n_threshold, median_threshold, median_size, scaling_factor,
):
    """Shared single-pass PIV + validation + scaling core for one frame pair."""
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


def _to_uint8(frame, display_min, display_max):
    frame = np.clip(frame.astype(float), display_min, display_max)
    frame -= display_min
    return ((255.0 / (display_max - display_min)) * frame).astype(np.uint8)


def process_pair(
    tif_path,
    crop_cols=CHANNEL_CROP_COLS,
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
    img = iio.imread(tif_path)
    h = img.shape[0] // 2
    img8 = _to_uint8(img, display_min, display_max)
    a1 = img8[:h, crop_cols[0]:crop_cols[1]]
    a2 = img8[h:, crop_cols[0]:crop_cols[1]]

    return _run_piv(
        a1, a2, winsize=winsize, searchsize=searchsize, overlap=overlap, dt=dt,
        s2n_threshold=s2n_threshold, median_threshold=median_threshold,
        median_size=median_size, scaling_factor=scaling_factor,
    )


def process_im7_pair(
    im7_path,
    crop_cols=CHANNEL_CROP_COLS,
    display_min=0,
    display_max=200,
    winsize=64,
    searchsize=96,
    overlap=32,
    dt=None,
    s2n_threshold=1.3,
    median_threshold=3,
    median_size=1,
    scaling_factor=PX_PER_MM,
):
    """Run single-pass PIV on one real dual-frame LaVision .im7 buffer.

    Unlike the sample_data test TIFF, real acquisitions store frame A and
    frame B as two frames of the same buffer (no vertical stacking/splitting
    needed) and carry the true PIV pulse separation in their own metadata
    (DevDataTrace5, microseconds - the "Reference time dt" channel) - pass
    dt=None (the default) to read it from the file instead of guessing.
    """
    buffer = lv.read_buffer(str(im7_path))
    if dt is None:
        dt = float(np.asarray(buffer.attributes["DevDataTrace5"]).flat[0]) * 1e-6  # us -> s

    a1_full = np.asarray(buffer.as_masked_array(0).data)
    a2_full = np.asarray(buffer.as_masked_array(1).data)
    a1 = _to_uint8(a1_full[:, crop_cols[0]:crop_cols[1]], display_min, display_max)
    a2 = _to_uint8(a2_full[:, crop_cols[0]:crop_cols[1]], display_min, display_max)

    result = _run_piv(
        a1, a2, winsize=winsize, searchsize=searchsize, overlap=overlap, dt=dt,
        s2n_threshold=s2n_threshold, median_threshold=median_threshold,
        median_size=median_size, scaling_factor=scaling_factor,
    )
    result["dt"] = dt
    result["image"] = a1
    return result


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


def scalar_on_image(ax, image, x, y, scalar, *, cmap="RdBu_r", levels=60,
                     symmetric=True, extent=None, scaling_factor=PX_PER_MM,
                     image_alpha=0.5, image_cmap="gray", scalar_alpha=0.75):
    """Raw image with a semi-transparent scalar field contour on top.

    Companion to quiver_on_image() for scalar fields (e.g. Reynolds stress,
    vorticity) that don't fit ds.piv.plot()'s single background= slot
    together with a raw image. symmetric=True (default, matching signed
    fields like Reynolds stress) centers the colormap at zero using a 99th
    percentile clim, same convention as pivpy's own vorticity/divergence
    backgrounds.
    """
    if extent is None:
        extent = (0, image.shape[1] / scaling_factor, 0, image.shape[0] / scaling_factor)
    ax.imshow(image, cmap=image_cmap, extent=extent, origin="upper", alpha=image_alpha)

    finite = scalar[np.isfinite(scalar)]
    if symmetric:
        vmax = max(float(np.nanpercentile(np.abs(finite), 99)), 1e-12) if finite.size else 1.0
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0

    cf = ax.contourf(x, y, scalar, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
                      alpha=scalar_alpha, extend="both")
    ax.set_aspect("equal")
    return cf
