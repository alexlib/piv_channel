# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "xarray",
#     "matplotlib",
#     "pivpy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import lvpyio as lv
    import pivpy  # noqa: F401  (registers the .piv xarray accessor)
    import pivpy.io as pivpy_io

    return json, lv, mo, np, pivpy_io, plt, xr


@app.cell
def _(mo):
    mo.md("""
    # channel_04_zoom_in - stitched vectors (first_half_case_1 + second_half)

    Loads DaVis's own `.vc7` vectors for both halves (not reprocessed -
    both are DaVis-complete), maps them onto one shared absolute pixel
    row axis (validated in `channel04_zoom_in_second_half_pair1.py`'s
    wall-mask fit: `first_half_case_1` rows 0-2432, `second_half` rows
    2432-4864 of the raw images, stacked with no overlap), masks out
    vectors outside the flow region using the fitted rotated-sinusoid
    wall boundary, and interpolates both halves onto one common grid.

    **Overlap check**: DaVis's vc7 coordinates reset to an arbitrary
    local origin per crop (both report identical -0.86 to 1.57mm x and
    -1.32 to 1.11mm y ranges despite being different physical regions -
    confirmed via the identical `RealWordMapper` calibration file, same
    camera/lens, just different vertical crop). Converting both grids to
    absolute pixel rows (using the established `PX_PER_MM` and DaVis's
    y-decreases-as-row-increases convention) shows **no overlap**:
    `first_half_case_1`'s last vc7 row lands at pixel row 2418,
    `second_half`'s first row starts at 2432 - a 14px *gap*, matching
    the expected interrogation-window edge margin (24x24 window loses
    ~half a window at each border), not a real physical camera overlap.
    """)
    return


@app.cell
def _():
    RAW_H = 2390 # 2432  # raw image height per half (established in the wall-mask notebook)
    RAW_C = 80 # column offset

    PX_PER_MM = 995.28671814631889  # channel04_zoom_in camera, from channel04_zoom_in_case2_pair1.py

    # Artificial, cosmetic-only registration nudge (per user request) - not a
    # calibration fix. second_half's fitted-wall-vs-actual-reflection alignment
    # looked slightly off relative to first_half_case_1; shifting second_half
    # left by this amount visually tightens it. Applied identically to
    # second_half's vector data (below) and its portion of the raw stitched
    # image (Step 6), so vectors/mask/image all move together.
    SECOND_HALF_COSMETIC_X_SHIFT_MM = -0.1

    FIRST_HALF_1_VC7 = "D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/first_half_case_1/ImgPreproc_03/PIV_MPd(4x24x24_75%ov_ImgCorr)/B00001.vc7"
    SECOND_HALF_VC7 = "D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/second_half/ImgPreproc/PIV_MPd(4x24x24_75%ov_ImgCorr)/B00001.vc7"
    WALL_MASK_JSON = "notebooks/channel04_zoom_in_second_half_wall_mask.json"

    return (
        FIRST_HALF_1_VC7,
        PX_PER_MM,
        RAW_C,
        RAW_H,
        SECOND_HALF_COSMETIC_X_SHIFT_MM,
        SECOND_HALF_VC7,
        WALL_MASK_JSON,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Step 1 - load both halves' vc7 vectors, convert to absolute pixel coordinates
    """)
    return


@app.cell
def _(
    FIRST_HALF_1_VC7,
    PX_PER_MM,
    RAW_C,
    RAW_H,
    SECOND_HALF_VC7,
    mo,
    pivpy_io,
):
    def to_absolute_px(ds, row_offset, col_offset=0):
        """DaVis's local mm coords reset per crop (see markdown above) - only
        the *scale* (mm/px, from the shared calibration) is trustworthy, not
        the offset. Rebuild absolute pixel coordinates from that scale plus
        the known raw-image stacking offset (row_offset=0 for the top half,
        RAW_H for the bottom), flipping y since DaVis's y decreases as pixel
        row increases (row 0 = y.max()).
        """
        y = ds.y.values
        x = ds.x.values
        row_px = (y.max() - y) * PX_PER_MM + row_offset
        col_px = (x - x.min()) * PX_PER_MM + col_offset
    
        return ds.assign_coords(y=("y", row_px), x=("x", col_px))

    ds_a_raw = pivpy_io.load_vc7(FIRST_HALF_1_VC7).isel(t=0)
    ds_b_raw = pivpy_io.load_vc7(SECOND_HALF_VC7).isel(t=0)

    ds_a = to_absolute_px(ds_a_raw, row_offset=0)
    ds_b = to_absolute_px(ds_b_raw, row_offset=RAW_H, col_offset=RAW_C)


    # DaVis's v also needs the sign flip validated on channel_04's vc7-vs-openpiv
    # comparison (see load_vc7_directory in piv_pipeline.py).
    ds_a["v"] = -ds_a["v"]
    ds_b["v"] = -ds_b["v"]

    # NOTE: SECOND_HALF_COSMETIC_X_SHIFT_MM is deliberately NOT applied to
    # ds_b's coordinates here - doing that de-aligns ds_a/ds_b's native x-grids
    # into interleaved, mutually-exclusive positions, so xr.concat's outer join
    # + .interp() can barely bridge anything (broke to <1% valid data when
    # tried). The shift only needs to move the *raw background image* (Step 6)
    # to visually match the image's wall reflection against the already-correct
    # fitted mask - it does not need to move the vector data itself.

    _line1 = f"first_half_case_1: row_px {float(ds_a.y.min()):.1f} to {float(ds_a.y.max()):.1f}, col_px {float(ds_a.x.min()):.1f} to {float(ds_a.x.max()):.1f}"
    _line2 = f"second_half: row_px {float(ds_b.y.min()):.1f} to {float(ds_b.y.max()):.1f}, col_px {float(ds_b.x.min()):.1f} to {float(ds_b.x.max()):.1f}"
    _line3 = f"gap between halves: {float(ds_b.y.min()) - float(ds_a.y.max()):.1f} px (expected: interrogation-window edge margin, not a physical gap)"
    mo.md(f"""{_line1}
    {_line2}
    {_line3}""")

    return ds_a, ds_b


@app.cell
def _(mo):
    mo.md("""
    ## Step 2 - mask vectors outside the flow region (wall side)
    """)
    return


@app.cell
def _(WALL_MASK_JSON, json):
    with open(WALL_MASK_JSON) as _f:
        wall_mask = json.load(_f)
    wall_mask["wall_mask"]
    return (wall_mask,)


@app.cell
def _(ds_a, ds_b, mo, np, wall_mask):
    def right_edge_at(row_px):
        wm = wall_mask["wall_mask"]
        return (
            wm["right_center_px"]
            + wm["drift_px_per_row"] * row_px
            + wm["right_amplitude_px"] * np.sin(2 * np.pi * row_px / wm["wavelength_px"] + wm["phase_rad"])
        )

    def mask_outside_wall(ds):
        # Two independent invalidity sources, both must be excluded before
        # interpolation: (1) our own wall-position mask, and (2) DaVis's own
        # chc==0 flag - DaVis zero-fills low-confidence vectors instead of
        # leaving them NaN, so without this a band of literal zero-velocity
        # "ghost" vectors near each half's raw-frame edge (where DaVis's own
        # correlation was unreliable) gets treated as real data and smeared
        # across the seam by .interp(), showing up as a near-zero-speed band.
        row_px_2d, col_px_2d = np.meshgrid(ds.y.values, ds.x.values, indexing="ij")
        right_edge_2d = right_edge_at(row_px_2d)
        outside_wall = col_px_2d > right_edge_2d
        invalid = outside_wall | (ds["chc"].values < 0.5)
        ds = ds.copy()
        ds["u"] = ds["u"].where(~invalid)
        ds["v"] = ds["v"].where(~invalid)
        ds["chc"] = ds["chc"].where(~invalid, 0.0)
        return ds

    ds_a_masked = mask_outside_wall(ds_a)
    ds_b_masked = mask_outside_wall(ds_b)
    _frac_a = float((ds_a_masked.chc > 0.5).mean())
    _frac_b = float((ds_b_masked.chc > 0.5).mean())
    mo.md(f"""first_half_case_1 valid fraction after wall + chc mask: {_frac_a:.1%}
    second_half valid fraction after wall + chc mask: {_frac_b:.1%}""")

    return ds_a_masked, ds_b_masked


@app.cell
def _(mo):
    mo.md("""
    ## Step 3 - stitch onto one common grid

    Both halves already share the same column (`x`) axis and a
    contiguous (non-overlapping) row (`y`) axis in absolute pixels, so
    stitching is a concat along `y`, then a resample onto one regular
    grid spanning the full combined extent - `xr.concat` alone would
    leave two different native grids end-to-end; the resample is what
    actually makes it "one smooth field" rather than two fields glued
    together, and linearly bridges the ~14px inter-half gap.
    """)
    return


@app.cell
def _(ds_a_masked, ds_b_masked, np, xr):
    # Drop rows with zero valid columns before concatenating - these carry no
    # real information (every point already NaN from the wall+chc mask), and
    # leaving them in the interp source just forces .interp() to bracket the
    # gap with an all-NaN row, propagating NaN into the output instead of
    # bridging to the next row that actually has data.
    _frac_valid_a = (~np.isnan(ds_a_masked.u.values)).mean(axis=1)
    _frac_valid_b = (~np.isnan(ds_b_masked.u.values)).mean(axis=1)
    ds_a_trimmed = ds_a_masked.isel(y=_frac_valid_a > 0)
    ds_b_trimmed = ds_b_masked.isel(y=_frac_valid_b > 0)

    combined_native = xr.concat([ds_a_trimmed, ds_b_trimmed], dim="y")

    # one common regular grid at the finer of the two halves' native spacing
    _dx = float(np.diff(ds_a_masked.x.values).mean())
    _dy = float(np.diff(ds_a_masked.y.values).mean())
    common_x = np.arange(combined_native.x.min(), combined_native.x.max(), _dx)
    common_y = np.arange(combined_native.y.min(), combined_native.y.max(), _dy)

    combined_ds = combined_native.interp(x=common_x, y=common_y, method="linear")
    combined_ds

    return combined_ds, ds_a_trimmed, ds_b_trimmed


@app.cell
def _(mo):
    mo.md("""
    ## Step 4 - convert to pivpy's expected coordinate convention

    `pivpy.piv.plot()` (used from here on for arrow-size/colormap
    control) expects the standard PIV/Cartesian convention: physical
    units, y increasing *upward*. Our absolute pixel coordinates
    increase downward (image/array convention), so this builds a
    separate mm, y-up copy for plotting - `combined_ds` itself stays in
    pixel space, still useful for indexing against the raw images.
    """)
    return


@app.cell
def _(PX_PER_MM, combined_ds):
    combined_cart = combined_ds.assign_coords(
        x=("x", combined_ds.x.values / PX_PER_MM),
        y=("y", -combined_ds.y.values / PX_PER_MM),
    )
    combined_cart
    return (combined_cart,)


@app.cell
def _(SECOND_HALF_COSMETIC_X_SHIFT_MM, combined_cart, half_b_edge_y_mm, np):
    # Cosmetic-only display copy: shift second_half's chunk sideways by the
    # same amount as the raw image (SECOND_HALF_COSMETIC_X_SHIFT_MM), applied
    # to the already-gridded, already-masked data - not before interpolation.
    # Doing it there (on ds_b's coordinates, pre-concat) de-aligns the two
    # halves' native x-grids into interleaved, mutually-exclusive positions and
    # breaks .interp() almost entirely (valid data dropped from ~27% to <1%
    # when tried). Rolling the already-regular-gridded array instead just moves
    # u/v/chc sideways as one rigid block - both plots (with or without the
    # image background) now show the same shift, arrows and mask included.
    combined_cart_display = combined_cart.copy(deep=True)
    _dx = float(np.diff(combined_cart.x.values).mean())
    _shift_cols = round(SECOND_HALF_COSMETIC_X_SHIFT_MM / _dx)
    _is_second_half = combined_cart.y.values <= half_b_edge_y_mm
    for _var in ("u", "v", "chc"):
        _arr = combined_cart_display[_var].values
        _sub = np.roll(_arr[_is_second_half], _shift_cols, axis=1)
        if _shift_cols > 0:
            _sub[:, :_shift_cols] = np.nan
        elif _shift_cols < 0:
            _sub[:, _shift_cols:] = np.nan
        _arr[_is_second_half] = _sub
    combined_cart_display

    return (combined_cart_display,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 5 - pivpy plot of the stitched, masked, interpolated field
    """)
    return


@app.cell
def _(mo):
    cmap_dd = mo.ui.dropdown(
        options=["viridis", "plasma", "cool", "turbo", "coolwarm"], value="viridis", label="colormap"
    )
    background_dd = mo.ui.dropdown(
        options=["mag", "vorticity", "none"], value="mag", label="background"
    )
    arrow_width_slider = mo.ui.slider(0.002, 0.02, value=0.004, step=0.001, label="arrow width")
    arrow_length_slider = mo.ui.slider(0.5, 2.0, value=1.0, step=0.1, label="arrow length")
    skip_slider = mo.ui.slider(1, 15, value=5, step=1, label="arrow skip (density)")
    mo.vstack([cmap_dd, background_dd, arrow_width_slider, arrow_length_slider, skip_slider])
    return (
        arrow_length_slider,
        arrow_width_slider,
        background_dd,
        cmap_dd,
        skip_slider,
    )


@app.function
def add_compact_colorbar(fig, ax, label):
    """piv.plot()'s own colorbar (shrink=0.92 relative to `ax`) comes out
    absurdly tall on our very elongated (tall/narrow) figures, since the
    `aspect="equal"` letterboxing that shrinks the visible plot doesn't
    shrink the colorbar the same way. Disabling it (colorbar=False) and
    drawing our own with fraction/pad - the standard compact-colorbar
    recipe - sizes it sensibly regardless of the axes' own aspect ratio.
    """
    mappable = next(
        (c for c in ax.collections if hasattr(c, "get_array") and c.get_array() is not None),
        None,
    )
    if mappable is None:
        return None
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    return cbar


@app.cell
def _(PX_PER_MM, ds_a_trimmed, ds_b_trimmed):
    # Cartesian mm (y-up) position of each half's own true data edge (last/first
    # row with any chc-valid vector, after the wall+chc mask and the fully-
    # invalid-row trim in Step 3) - marks where one half's real measurements
    # end and interpolation into the other half's real measurements begins,
    # so it's visually clear this is a stitched image, not one continuous shot.
    half_a_edge_y_mm = -float(ds_a_trimmed.y.max()) / PX_PER_MM
    half_b_edge_y_mm = -float(ds_b_trimmed.y.min()) / PX_PER_MM

    return half_a_edge_y_mm, half_b_edge_y_mm


@app.cell
def _(
    arrow_length_slider,
    arrow_width_slider,
    background_dd,
    cmap_dd,
    combined_cart_display,
    half_a_edge_y_mm,
    half_b_edge_y_mm,
    np,
    plt,
    skip_slider,
):
    # arrow_length_slider controls arrow_scale (matplotlib quiver: SMALLER
    # scale = LONGER arrows) - reproducing pivpy's own auto-scale formula so
    # length=1.0 matches what piv.plot(arrow_scale=None) would auto-pick.
    _dx = float(np.diff(combined_cart_display.x.values).mean())
    _dy = float(np.diff(combined_cart_display.y.values).mean())
    _med_speed = float(np.nanmedian(np.hypot(combined_cart_display.u.values, combined_cart_display.v.values)))
    _target_len = 0.85 * min(skip_slider.value * abs(_dx), skip_slider.value * abs(_dy))
    _auto_scale = (_med_speed / _target_len) if _target_len > 0 else 1.0
    _arrow_scale = _auto_scale / arrow_length_slider.value

    # pivpy.piv.plot() draws a colorbar per colored artist independently
    # (background contourf AND colored quiver each gated by the same
    # colorbar=True default, with no check for redundancy) - setting both
    # background= and color_by= to a scalar draws two colorbars for the same
    # data. Not a two-halves/stitching issue - it reproduces on a single frame
    # too. Fix: only color the quiver when there's no scalar background, and
    # always draw our own single compact colorbar via colorbar=False.
    #
    # Uses combined_cart_display (not combined_cart) so the cosmetic second_half
    # shift shows here too, consistent with Step 6's image-backed plot.
    _fig, _ax = plt.subplots(figsize=(6, 13))
    _has_background = background_dd.value != "none"
    combined_cart_display.piv.plot(
        ax=_ax,
        background=background_dd.value if _has_background else None,
        quiver=True,
        streamlines=False,
        cmap=cmap_dd.value,
        color_by=None if _has_background else "mag",
        arrow_scale=_arrow_scale,
        arrow_width=arrow_width_slider.value,
        skip=skip_slider.value,
        colorbar=False,
        title="Stitched, wall-masked, interpolated field (first_half_case_1 + second_half)",
    )
    _ax.axhline(half_a_edge_y_mm, color="cyan", linewidth=1.0, linestyle="--", label="first_half_case_1 data edge")
    _ax.axhline(half_b_edge_y_mm, color="magenta", linewidth=1.0, linestyle="--", label="second_half data edge")
    _ax.set_xlim(0, 2)
    _ax.legend(fontsize=7, loc="lower right")
    add_compact_colorbar(_fig, _ax, "speed [m/s]")
    _fig.gca()

    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 6 - pivpy plot on the stitched raw image

    Same field and controls as Step 5, but with the actual stitched
    image (`first_half_case_1` raw frame on top of `second_half`'s,
    same pixel stacking used throughout) as the background instead of
    a scalar field - lets you see the vectors against the real
    particle/wall image.
    """)
    return


@app.cell
def _():
    FIRST_HALF_1_IM7 = (
        r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445"
        r"\first_half_case_1\exported_images\first_half_case_1\B0001.im7"
    )
    SECOND_HALF_IM7 = (
        r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445"
        r"\second_half\exported_images\second_half\B0001.im7"
    )
    return FIRST_HALF_1_IM7, SECOND_HALF_IM7


@app.cell
def _(
    FIRST_HALF_1_IM7,
    PX_PER_MM,
    SECOND_HALF_COSMETIC_X_SHIFT_MM,
    SECOND_HALF_IM7,
    lv,
    np,
):
    import openpiv.preprocess as _pp

    def _to_uint8_highpass(frame, sigma=16, pct=99.0):
        # first_half_case_1's raw sensor signal is ~4x brighter than
        # second_half's (mean 124 vs 31, max 4094 vs 1610) - displaying both
        # raw with one shared percentile scale makes the dim half read as a
        # dark band/gap at the seam. high_pass + per-half percentile clip
        # (same recipe validated in channel04_zoom_in_second_half_pair1.py)
        # normalizes each half's own contrast independently before stitching.
        hp = _pp.high_pass(frame.astype(float), sigma=sigma, clip=True)
        pmax = np.percentile(hp, pct)
        hp = np.clip(hp, 0, pmax)
        return ((255.0 / pmax) * hp).astype(np.uint8)

    def _shift_cols(frame, shift_px):
        # positive shift_px moves content right, negative moves it left;
        # newly-exposed edge is zero-filled (matches SECOND_HALF_COSMETIC_X_SHIFT_MM).
        out = np.zeros_like(frame)
        if shift_px == 0:
            return frame.copy()
        elif shift_px > 0:
            out[:, shift_px:] = frame[:, :-shift_px]
        else:
            out[:, :shift_px] = frame[:, -shift_px:]
        return out

    frame_a_raw = np.asarray(lv.read_buffer(FIRST_HALF_1_IM7).as_masked_array(0).data)
    frame_b_raw = np.asarray(lv.read_buffer(SECOND_HALF_IM7).as_masked_array(0).data)
    frame_a_proc = _to_uint8_highpass(frame_a_raw)
    frame_b_proc = _to_uint8_highpass(frame_b_raw)
    _shift_px = round(SECOND_HALF_COSMETIC_X_SHIFT_MM * PX_PER_MM)
    frame_b_proc = _shift_cols(frame_b_proc, _shift_px)
    stitched_image = np.vstack([frame_a_proc, frame_b_proc])
    stitched_image.shape

    return (stitched_image,)


@app.cell
def _(PX_PER_MM, stitched_image):
    # image_extent=(left, right, bottom, top) in the same mm, y-up frame as
    # combined_cart - bottom < top with both negative/zero reproduces the
    # image's own top-at-row-0 orientation once y is flipped upward.
    _h, _w = stitched_image.shape
    image_extent_mm = (0, _w / PX_PER_MM, -(_h - 1) / PX_PER_MM, 0)
    return (image_extent_mm,)


@app.cell
def _(
    arrow_width_slider,
    cmap_dd,
    combined_cart_display,
    half_a_edge_y_mm,
    half_b_edge_y_mm,
    image_extent_mm,
    plt,
    skip_slider,
    stitched_image,
):
    # pre-sized ax=, same reasoning as Step 5 - the image background doesn't
    # draw its own colorbar (it's a grayscale imshow, not a scalar mesh), so
    # color_by="mag" here is the sole (not duplicated) colorbar; still
    # disabled in favor of add_compact_colorbar() for sensible proportions.
    # Uses combined_cart_display so vectors/mask shift the same way as the
    # already-shifted background image (see the combined_cart_display cell).
    _fig, _ax = plt.subplots(figsize=(6, 13))
    combined_cart_display.piv.plot(
        ax=_ax,
        background="image",
        image=stitched_image,
        image_extent=image_extent_mm,
        image_cmap="gray",
        quiver=True,
        streamlines=False,
        cmap=cmap_dd.value,
        color_by="mag",
        arrow_width=arrow_width_slider.value,
        skip=skip_slider.value,
        colorbar=False,
        title="Stitched raw image + wall-masked, interpolated quiver",
    )
    _ax.axhline(half_a_edge_y_mm, color="cyan", linewidth=1.0, linestyle="--", label="first_half_case_1 data edge")
    _ax.axhline(half_b_edge_y_mm, color="magenta", linewidth=1.0, linestyle="--", label="second_half data edge")
    _ax.set_xlim(0, 2)
    _ax.legend(fontsize=7, loc="lower right")
    add_compact_colorbar(_fig, _ax, "speed [m/s]")
    _fig.gca()

    return


if __name__ == "__main__":
    app.run()
