"""Single-pair PIV pipeline, factored out of pair1_first_pass.py so both that
notebook and the batch notebooks use the exact same tested parameters."""

import json

import imageio.v3 as iio
import imagecodecs  # noqa: F401  (registers LZW-compressed TIFF codec)
import lvpyio as lv
import numpy as np
from openpiv import tools, scaling as opiv_scaling, validation, filters
import openpiv.pyprocess as pyprocess


def load_run_config(path):
    """Load a per-run parameter JSON (e.g. channel04_pair1_config.json) -
    calibration source, wall-mask sinusoid params, preprocessing, PIV
    settings - hand-tuned once in a *_pair1.py exploration notebook and
    reused unchanged by the matching *_batch.py notebook, instead of
    re-tuning or hardcoding the same numbers twice.
    """
    with open(path) as f:
        return json.load(f)

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


def piv_run_metadata(
    *, px_per_mm, calibration_source, dt, winsize, searchsize, overlap,
    s2n_threshold, median_threshold, source_folder, crop_cols=None,
    wall_mask=False, history="",
):
    """Standard xarray attrs dict for a PIV Dataset - pivpy's own convention
    (units_x/y/u/v, dt, history: see skills/pivpy/SKILL.md) plus this
    project's processing provenance (calibration source, crop/mask choice,
    PIV parameters), so a saved zarr store is self-describing (`xr.open_zarr(...).attrs`)
    without needing to cross-reference README.md. Pass to `ds.attrs.update(...)`
    before `ds.to_zarr(...)`; `ds.piv.reynolds_decomposition()` carries attrs
    through automatically, so `stats` doesn't need it set again.
    """
    return dict(
        units_x="mm", units_y="mm", units_u="mm/s", units_v="mm/s",
        dt=dt,
        px_per_mm=px_per_mm,
        calibration_source=str(calibration_source),
        source_folder=str(source_folder),
        crop_cols=list(crop_cols) if crop_cols is not None else "none (wall-masked)",
        wall_mask=wall_mask,
        winsize=winsize, searchsize=searchsize, overlap=overlap,
        s2n_threshold=s2n_threshold, median_threshold=median_threshold,
        history=history,
    )


def wavy_wall_bounds(image, margin_px=350, brightness_threshold=15, smooth_window=51):
    """Per-row (left, right) channel-interior pixel-column bounds for a frame
    with sinusoidal (wavy) walls in the left/right margins.

    For each row, finds the brightest column within margin_px of each edge
    (the wall reflection is far brighter than seeded particles - but NOT
    anchored at column 0/w-1 itself, it peaks a bit inset from the true
    frame edge, hence the margin search rather than an edge-contiguity
    test). margin_px must be narrow enough to exclude interior bright
    outliers (dust, stray reflections) - too wide a margin (e.g. a fraction
    of the frame width) reliably catches them; 350 px matches this camera's
    actual wall excursion with headroom, verified visually.

    Rows where nothing in the margin clears brightness_threshold (patchy
    illumination can dim the reflection at some wave phases) are treated as
    unknown and linearly interpolated from confident neighboring rows,
    rather than defaulting to "no wall" - the wall position varies
    smoothly, so that's a much safer fallback than silently opening the
    mask. A final median filter over rows suppresses any remaining speckle.

    Compute once from a representative frame (the wall geometry is fixed
    per run) and reuse across a whole batch via process_im7_pair's
    wall_bounds= - masks wall-adjacent vectors out of the *results* without
    cropping the frame, since the wall's column position genuinely varies
    with row.
    """
    from scipy.ndimage import median_filter

    def interp_missing(edge, confident):
        rows = np.arange(len(edge))
        if not confident.any():
            return edge
        return np.interp(rows, rows[confident], edge[confident])

    h, w = image.shape
    left_band = image[:, :margin_px]
    right_band = image[:, w - margin_px:]
    left_confident = left_band.max(axis=1) > brightness_threshold
    right_confident = right_band.max(axis=1) > brightness_threshold
    left_edge = interp_missing(left_band.argmax(axis=1) + 1.0, left_confident)
    right_edge = interp_missing(w - margin_px + right_band.argmax(axis=1) - 1.0, right_confident)
    left_edge = median_filter(left_edge, size=smooth_window)
    right_edge = median_filter(right_edge, size=smooth_window)
    return left_edge, right_edge


def sinusoidal_wall_bounds(
    n_rows, *, wavelength_px, phase, left_center, left_amplitude,
    right_center, right_amplitude,
):
    """Per-row (left, right) channel-interior pixel-column bounds from an
    explicit sinusoid model, for manual/parametric masking instead of
    per-frame brightness detection (wavy_wall_bounds()).

    The wavy channel's wall shape is a single known physical sinusoid - both
    walls are cut from the same wave, so they share one wavelength_px and
    phase, differing only in their mean column position (*_center) and how
    far they swing (*_amplitude, sign encodes swing direction relative to
    the wave). Fit/tune this once from a representative frame (e.g. via
    scipy.optimize.curve_fit against wavy_wall_bounds()'s per-row trace, or
    by eye against the image) and reuse the same parameters unchanged across
    a whole batch - immune to the per-frame illumination dropouts and stray
    bright particles that make brightness-based detection fragile.
    """
    rows = np.arange(n_rows)
    wave = np.sin(2 * np.pi * rows / wavelength_px + phase)
    left_edge = left_center + left_amplitude * wave
    right_edge = right_center + right_amplitude * wave
    return left_edge, right_edge


def _run_piv(
    a1, a2, *,
    winsize, searchsize, overlap, dt,
    s2n_threshold, median_threshold, median_size, scaling_factor,
    wall_bounds=None,
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
    # median_threshold is a px/frame displacement tolerance (intuitive,
    # dt-independent), but u0/v0 are already in px/s (extended_search_area_piv
    # divides by dt internally) - local_median_val compares its threshold
    # against u0/v0 directly, in whatever units they're in. Without this
    # conversion, a tiny dt (e.g. 120us here) inflates any real px/frame
    # variation into a huge px/s number that a small threshold like 3
    # rejects almost unconditionally, regardless of actual data quality -
    # this was silently making the median check reject ~everything.
    mask_med = validation.local_median_val(
        u0, v0,
        u_threshold=median_threshold / dt, v_threshold=median_threshold / dt,
        size=median_size,
    )
    invalid = mask_s2n | mask_med

    if wall_bounds is not None:
        left_edge, right_edge = wall_bounds
        row_idx = np.clip(np.round(y[:, 0]).astype(int), 0, len(left_edge) - 1)
        invalid |= (x < left_edge[row_idx][:, None]) | (x > right_edge[row_idx][:, None])

    u2, v2 = filters.replace_outliers(
        u0, v0, invalid, method="localmean", max_iter=10, kernel_size=3,
    )

    xs, ys, u3, v3 = opiv_scaling.uniform(x, y, u2, v2, scaling_factor=scaling_factor)
    xs, ys, u3, v3 = tools.transform_coordinates(xs, ys, u3, v3)

    return {
        "x": xs, "y": ys, "u": u3, "v": v3, "invalid": invalid, "s2n": s2n,
        # Pre-validation, pre-outlier-fill: straight off the correlation
        # peak, in px/dt and pixel coordinates (not yet scaled/flipped) -
        # for inspecting what the correlation actually found, independent
        # of the validation thresholds and filters.replace_outliers() smoothing.
        "u_raw": u0, "v_raw": v0, "x_px": x, "y_px": y,
    }


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
    return_image=True,
    wall_bounds=None,
    mask_input=False,
    preprocess=None,
    hp_sigma=3,
    hp_pct=99.5,
):
    """Run single-pass PIV on one real dual-frame LaVision .im7 buffer.

    Unlike the sample_data test TIFF, real acquisitions store frame A and
    frame B as two frames of the same buffer (no vertical stacking/splitting
    needed) and carry the true PIV pulse separation in their own metadata
    (DevDataTrace5, microseconds - the "Reference time dt" channel) - pass
    dt=None (the default) to read it from the file instead of guessing.

    crop_cols: (left, right) column slice, or None to keep the full frame
    uncropped. For a wavy-wall channel, pass crop_cols=None and instead pass
    wall_bounds=wavy_wall_bounds(...)/sinusoidal_wall_bounds(...) (computed
    once from a representative frame) to mask wall-adjacent vectors out of
    the *results* - the wall's column position varies with row, so a single
    crop either clips valid interior at some rows or leaves wall pixels in
    at others.

    mask_input: also zero out wall pixels in a1/a2 themselves (statically,
    via wall_bounds) before correlation, not just the output vectors after.
    Protects windows whose search area straddles the wall boundary from
    correlating against the wall's bright reflection at all - wall_bounds
    alone only discards those windows' *output*, after the wall pixels have
    already contributed to the correlation. Requires wall_bounds.

    return_image: include the cropped frame A (uint8) in the result, for use
    as a plot background. Only needed for one representative frame per batch
    (e.g. the first) - set False for the rest to avoid holding one image per
    frame in memory (each is a few MB; adds up fast over hundreds of frames).

    preprocess: None (default) for the naive linear display_min/display_max
    clip, or "high_pass" to remove the slowly-varying background via
    openpiv.preprocess.high_pass(sigma=hp_sigma) before a percentile
    (hp_pct) contrast stretch - measurably improves correlation quality on
    dim/low-contrast footage (channel_04) versus the linear clip; tune
    hp_sigma/hp_pct on one representative pair first (see
    channel04_steady_state_pair1.py), display_min/display_max are unused
    when this is set.
    """
    buffer = lv.read_buffer(str(im7_path))
    if dt is None:
        dt = float(np.asarray(buffer.attributes["DevDataTrace5"]).flat[0]) * 1e-6  # us -> s

    a1_full = np.asarray(buffer.as_masked_array(0).data)
    a2_full = np.asarray(buffer.as_masked_array(1).data)
    cols = slice(*crop_cols) if crop_cols is not None else slice(None)
    if preprocess == "high_pass":
        import openpiv.preprocess as pp

        def _to_uint8_highpass(frame):
            hp = pp.high_pass(frame.astype(float), sigma=hp_sigma, clip=True)
            pmax = np.percentile(hp, hp_pct)
            hp = np.clip(hp, 0, pmax)
            return ((255.0 / pmax) * hp).astype(np.uint8)

        a1 = _to_uint8_highpass(a1_full[:, cols])
        a2 = _to_uint8_highpass(a2_full[:, cols])
    elif preprocess is not None:
        raise ValueError(f"unknown preprocess: {preprocess!r}")
    else:
        a1 = _to_uint8(a1_full[:, cols], display_min, display_max)
        a2 = _to_uint8(a2_full[:, cols], display_min, display_max)

    if mask_input:
        if wall_bounds is None:
            raise ValueError("mask_input=True requires wall_bounds")
        left_edge, right_edge = wall_bounds
        col_idx = np.arange(a1.shape[1])[None, :]
        outside = (col_idx < left_edge[:, None]) | (col_idx > right_edge[:, None])
        a1 = a1.copy()
        a2 = a2.copy()
        a1[outside] = 0
        a2[outside] = 0

    result = _run_piv(
        a1, a2, winsize=winsize, searchsize=searchsize, overlap=overlap, dt=dt,
        s2n_threshold=s2n_threshold, median_threshold=median_threshold,
        median_size=median_size, scaling_factor=scaling_factor,
        wall_bounds=wall_bounds,
    )
    result["dt"] = dt
    if return_image:
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


def run_steady_state_batch(im7_folder, config, out_dir, run_name, N=None):
    """Run the full steady-state PIV batch (parallel process_im7_pair over
    every pair in im7_folder, pivpy Dataset build, reynolds_decomposition,
    zarr save, mean-field/Reynolds-stress plots) for one wavy-channel run -
    the same pipeline as channel04_steady_state_batch.py, factored out so
    an orchestration notebook (e.g. running several parts of the same
    steady-state acquisition) can call it once per folder instead of
    duplicating notebook cells per part.

    config: a dict from load_run_config() - same shape as
    channel04_pair1_config.json (wall_mask, preprocessing, piv, px_per_mm,
    calibration_source). Reused unchanged across parts since it's the same
    physical channel/camera setup, just different time segments.

    Returns a small, JSON-serializable summary dict (pair count, invalid
    fraction, mean speed, output paths, wall-clock seconds) - orchestrate
    several calls and log the returned dicts for later cross-run comparison
    /averaging, since raw per-frame arrays aren't kept in memory after the
    zarr save.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)

    t0 = time.perf_counter()
    im7_files = sorted(Path(im7_folder).glob("*.im7"))
    if N is not None:
        im7_files = im7_files[:N]
    if not im7_files:
        raise ValueError(f"no .im7 files found in {im7_folder}")

    a1_raw = np.asarray(lv.read_buffer(str(im7_files[0])).as_masked_array(0).data)
    wm = config["wall_mask"]
    wall_bounds = sinusoidal_wall_bounds(
        a1_raw.shape[0],
        wavelength_px=wm["wavelength_px"], phase=wm["phase_rad"],
        left_center=wm["left_center_px"], left_amplitude=wm["left_amplitude_px"],
        right_center=wm["right_center_px"], right_amplitude=wm["right_amplitude_px"],
    )

    piv_kw = dict(
        crop_cols=None, wall_bounds=wall_bounds, mask_input=config["piv"]["mask_input"],
        preprocess="high_pass",
        hp_sigma=config["preprocessing"]["high_pass_sigma"],
        hp_pct=config["preprocessing"]["contrast_stretch_percentile"],
        winsize=config["piv"]["winsize"], searchsize=config["piv"]["searchsize"],
        overlap=config["piv"]["overlap"], s2n_threshold=config["piv"]["s2n_threshold"],
        median_threshold=config["piv"]["median_threshold_px_per_frame"],
        scaling_factor=config["px_per_mm"], return_image=False,
    )

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(partial(process_im7_pair, **piv_kw), im7_files))
    results[0] = process_im7_pair(im7_files[0], **{**piv_kw, "return_image": True})

    def _to_frame(r, t):
        return xr.Dataset(
            data_vars={
                "u": (("y", "x"), r["u"]),
                "v": (("y", "x"), r["v"]),
                "chc": (("y", "x"), (~r["invalid"]).astype(float)),
            },
            coords={"x": r["x"][0, :], "y": r["y"][:, 0], "t": t},
        )

    ds = xr.concat([_to_frame(r, t) for t, r in enumerate(results)], dim="t")
    ds.attrs.update(piv_run_metadata(
        px_per_mm=config["px_per_mm"],
        calibration_source=config["calibration_source"],
        dt=results[0]["dt"],
        winsize=config["piv"]["winsize"], searchsize=config["piv"]["searchsize"],
        overlap=config["piv"]["overlap"],
        s2n_threshold=config["piv"]["s2n_threshold"],
        median_threshold=config["piv"]["median_threshold_px_per_frame"],
        source_folder=str(im7_folder),
        crop_cols=None,
        wall_mask=True,
        history=f"run_steady_state_batch(run_name={run_name!r})",
    ))
    stats = ds.piv.reynolds_decomposition()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    ds_path = out_dir / f"{run_name}_ds.zarr"
    stats_path = out_dir / f"{run_name}_stats.zarr"
    ds.to_zarr(ds_path, mode="w")
    stats.to_zarr(stats_path, mode="w")

    image = results[0]["image"]
    u_mean, v_mean = stats["u_mean"].values, stats["v_mean"].values
    x, y = stats["x"].values, stats["y"].values
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 10))
    Q = quiver_on_image(ax, image, X, Y, u_mean, v_mean,
                         color_by=np.hypot(u_mean, v_mean), cmap="viridis",
                         scaling_factor=config["px_per_mm"])
    fig.colorbar(Q, ax=ax, pad=0.03, shrink=0.85, label="mean speed [mm/s]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{run_name} - mean flow field (N={len(results)})")
    mean_field_path = out_dir / f"{run_name}_mean_field.png"
    fig.savefig(mean_field_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 10))
    cf = scalar_on_image(ax, image, X, Y, stats["uv_prime"].values,
                          cmap="RdBu_r", scaling_factor=config["px_per_mm"])
    fig.colorbar(cf, ax=ax, pad=0.03, shrink=0.85,
                 label="Reynolds shear stress $-\\overline{u'v'}$ [mm²/s²]")
    quiver_on_image(ax, image, X, Y, u_mean, v_mean,
                     scaling_factor=config["px_per_mm"], image_alpha=0.0, arrow_width=0.003)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{run_name} - Reynolds shear stress (N={len(results)})")
    reynolds_stress_path = out_dir / f"{run_name}_reynolds_stress.png"
    fig.savefig(reynolds_stress_path, dpi=150)
    plt.close(fig)

    return {
        "run_name": run_name,
        "source_folder": str(im7_folder),
        "n_pairs": len(im7_files),
        "dt": results[0]["dt"],
        "invalid_fraction": float(np.mean([r["invalid"].mean() for r in results])),
        "mean_speed_mm_s": float((stats["u_mean"] ** 2 + stats["v_mean"] ** 2).mean() ** 0.5),
        "peak_tke_mm2_s2": float(stats["tke"].max()),
        "ds_path": str(ds_path),
        "stats_path": str(stats_path),
        "mean_field_plot": str(mean_field_path),
        "reynolds_stress_plot": str(reynolds_stress_path),
        "elapsed_seconds": time.perf_counter() - t0,
    }
