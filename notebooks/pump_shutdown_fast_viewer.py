"""Fast DaVis .im7 / .vc7 viewer (readim7 backend, macOS-ARM ready).

Scroll through thousands of DaVis image pairs or vector fields with a slider.
Reads directly via readim7 (pybind11 wrapper of LaVision ReadIMX) — no lvpyio
needed, so it works on Apple Silicon where lvpyio has no wheels.

Run interactively:
    uv run marimo edit notebooks/pump_shutdown_fast_viewer.py
Smoke-test headless (runs all cells once with default control values):
    uv run python notebooks/pump_shutdown_fast_viewer.py

NOTE on the pump-shutdown bubbles: the shutdown run itself
(baseline_channel/Vmax_0p62_m2sec_pump_shutdown/Camera1-1.ims, 13.9 GB) is a
DaVis *streaming* (.ims) file, which readim7's single-buffer C++ reader
cannot parse. Select
"stream (.ims via lvpyio)" in File type and point the box at the run folder
— or directly at the Camera1-1.ims file — to stream pairs straight from disk
(1000 pairs @ 15 Hz, dt = 80 us). That mode reads through readim7.ims
(pure-numpy 12-bit unpacking, works on macOS ARM too) with an lvpyio fallback
on Windows/Linux (see notebooks/straight_channel_pump_shutdown_ims.py for the
full .ims processing pipeline).
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fast `.im7` / `.vc7` / `.ims` viewer (`readim7` + `lvpyio` backends)

    Slider-scrub through thousands of DaVis frames straight off
    `/Volumes/Yossi's_Hard_Drive/channel_flow_research`. Pulse A / B / A−B
    difference for images, quiver overlay for `.vc7` vectors, plus a 5-frame
    filmstrip for spotting transients (e.g. bubbles appearing after pump trip).

    > Pump-shutdown `.ims` streams read via `readim7.ims` (pure numpy, works on
    > Mac ARM too — point the box at the run folder or directly at
    > `Camera1-1.ims`). If that fails on Windows/Linux the notebook falls back
    > to `lvpyio`.
    """)
    return


@app.cell
def _():
    import time
    from functools import lru_cache
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")  # safe headless + marimo PNG capture
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import readim7
    from readim7 import ims as ims_stream

    @lru_cache(maxsize=4)
    def read_frame(path_str):
        """Read one .im7/.vc7 file. Returns (array, attrs dict, meta dict).

        Images: array shape (nf=2, ny, nx), [0]=pulse A, [1]=pulse B.
        Vectors: array shape (2*nf, ny, nx), pair (u, v) per frame.
        """
        buff, attrs = readim7.get_Buffer_andAttributeList(path_str)
        arr, _ = readim7.buffer_as_array(buff)
        meta = {
            "nx": buff.nx,
            "ny": buff.ny,
            "nf": buff.nf,
            "sub_type": buff.image_sub_type,
            "vector_grid": buff.vector_grid,
        }
        return np.asarray(arr), dict(attrs), meta

    def find_dt_us(attrs):
        """Best-effort laser pulse separation in microseconds.

        LaVision stores timing as DevDataAlias<N>/DevDataTrace<N> pairs; the
        channel whose alias mentions 'dt' holds the value in microseconds.
        """
        for key, name in attrs.items():
            if key.startswith("DevDataAlias") and "dt" in str(name).lower():
                num = key.replace("DevDataAlias", "")
                for prefix in ("DevDataTrace", "DevDataValue", "DevDataData"):
                    raw = attrs.get(prefix + num)
                    if raw is None:
                        continue
                    try:
                        return float(str(raw).split()[0]), str(name).strip()
                    except (ValueError, IndexError):
                        continue
        return None, None

    try:
        import lvpyio as lv

        LV_AVAILABLE = True
        LV_ERROR = ""
    except Exception as _lv_exc:  # macOS ARM has no lvpyio wheels
        lv = None
        LV_AVAILABLE = False
        LV_ERROR = str(_lv_exc)

    # Pump-shutdown stream acquisition rate (1000 dual-frame pairs @ 15 Hz).
    IMS_ACQ_HZ = 15.0

    @lru_cache(maxsize=4)
    def open_ims_set(set_path_str):
        """Open a DaVis stream set (run folder holding Camera1-1.ims)."""
        return lv.read_set(set_path_str)

    def resolve_ims_set(root_str):
        """Map user input (run folder OR direct Camera1-1.ims path) to a set folder.

        Returns (set_path_str or None, note).
        """
        _root = Path(root_str.strip())
        if not str(_root) or not _root.exists():
            return None, f"`{root_str}` does not exist."
        if _root.is_file():
            if _root.suffix.lower() != ".ims":
                return None, f"`{_root.name}` is not a `.ims` file."
            _root = _root.parent
        _ims_files = sorted(_root.glob("*.ims"))
        if not _ims_files:
            return None, f"No `*.ims` found in `{_root}`."
        return str(_root), (
            f"{len(_ims_files)} `.ims` file(s), streaming `{_ims_files[0].name}`."
        )

    @lru_cache(maxsize=4)
    def read_ims_frame(set_path_str, idx):
        """Read one dual-frame pair from an .ims set.

        Returns (pulseA, pulseB) raw arrays in sensor dtype; convert to float
        at render time. Mirrors read_frame() caching so slider scrubbing stays
        smooth (one decoded pair ~= 20 MB).
        """
        _buf = open_ims_set(set_path_str)[int(idx)]
        return (
            np.asarray(_buf.as_masked_array(0).data),
            np.asarray(_buf.as_masked_array(1).data),
        )

    return (
        IMS_ACQ_HZ,
        LV_AVAILABLE,
        LV_ERROR,
        Path,
        find_dt_us,
        ims_stream,
        mo,
        np,
        open_ims_set,
        plt,
        read_frame,
        read_ims_frame,
        resolve_ims_set,
        time,
    )


@app.cell
def _(Path, mo):
    DRIVE = Path("/Volumes/Yossi's_Hard_Drive/channel_flow_research")
    DEFAULT_FOLDER = (
        DRIVE
        / "baseline_channel/Vmax_0p62_m2sec_steady_state"
        / "exported_images/Vmax_0p62_m2sec_steady_state"
    )
    folder_box = mo.ui.text(
        value=str(DEFAULT_FOLDER),
        label="Path (used when nothing is picked in the browser below)",
        full_width=True,
    )
    mode_select = mo.ui.dropdown(
        options={
            "images (.im7)": "im7",
            "vectors (.vc7)": "vc7",
            "stream (.ims)": "ims",
        },
        value="images (.im7)",
        label="File type",
    )
    _candidates = [
        DRIVE,  # Mac mount of Yossi's drive
        Path("D:/channel_flow_research"),  # Windows acquisition box
    ]
    _start = next((str(p) for p in _candidates if p.exists()), "")
    browser = mo.ui.file_browser(
        initial_path=_start,
        selection_mode="all",
        multiple=False,
        limit=500,
        restrict_navigation=False,
        label="Browse — pick a run folder, an exported-images folder, or a .ims file (selection overrides the text box)",
    )
    mo.vstack([
        mo.hstack([folder_box, mode_select]),
        browser,
    ])
    return browser, folder_box, mode_select


@app.cell
def _(
    LV_AVAILABLE,
    LV_ERROR,
    Path,
    browser,
    folder_box,
    ims_stream,
    mo,
    mode_select,
    open_ims_set,
    read_ims_frame,
    resolve_ims_set,
):
    try:
        _picked = browser.value
        src = str(browser.path(index=0)) if _picked else ""
    except Exception:  # noqa: BLE001 — fall back to the text box
        src = ""
    if not src:
        src = folder_box.value.strip()
    files, _resolved_from = [], src
    ims_set, ims_n, ims_note, ims_backend = "", 0, "", ""
    if mode_select.value == "ims":
        try:
            _info = ims_stream.ims_info(src)
            ims_set = _info["ims_path"]
            ims_n = _info["n_pairs"]
            ims_backend = "readim7"
            _resolved_from = ims_set
            _a0, _b0 = ims_stream.read_ims_pair(_info, 0)
            ims_note = (
                f"`{Path(ims_set).name}` via **readim7** — **{ims_n}** dual-frame pairs "
                f"({_info['ny']}×{_info['nx']}px, pulse A mean {_a0.mean():.1f})"
            )
        except Exception as _e1:  # noqa: BLE001 — surfaced in the UI
            if LV_AVAILABLE:
                try:
                    _set_path, _set_note = resolve_ims_set(src)
                    if _set_path is None:
                        ims_note = f"readim7: `{_e1}`. lvpyio: {_set_note}"
                    else:
                        ims_set = _set_path
                        ims_n = len(open_ims_set(ims_set))
                        ims_backend = "lvpyio"
                        _resolved_from = ims_set
                        _a0, _b0 = read_ims_frame(ims_set, 0)
                        ims_note = (
                            f"{_set_note} via **lvpyio** — **{ims_n}** dual-frame pairs "
                            f"({_a0.shape[0]}×{_a0.shape[1]}px, pulse A mean {_a0.mean():.1f})"
                        )
                except Exception as _e2:  # noqa: BLE001 — surfaced in the UI
                    ims_note = f"readim7: `{_e1}`. lvpyio: `{_e2}`."
            else:
                ims_note = (
                    f"readim7 could not open it (`{_e1}`), and `lvpyio` is not "
                    "installed in this environment"
                    + (f" (`{LV_ERROR}`)" if LV_ERROR else "")
                    + " — check the path, or export `.im7` frames on Windows/Linux."
                )
        mo.md(
            f"**{_resolved_from}** — {ims_note if ims_note else '_enter a run folder or `Camera1-1.ims` path above_'}"
        )
    else:
        _ext = "." + mode_select.value
        _root = Path(src)
        files, _resolved_from = [], str(_root)
        if _root.is_dir():
            files = sorted(_root.glob(f"*{_ext}"))
            if not files:  # run folder? look inside exported_images/*
                for sub in sorted((_root / "exported_images").glob("*")):
                    if sub.is_dir():
                        files = sorted(sub.glob(f"*{_ext}"))
                        if files:
                            _resolved_from = str(sub)
                            break
            if not files:  # last resort: one level of recursion, capped
                files = sorted(list(_root.rglob(f"*{_ext}"))[:5000])
                if files:
                    _resolved_from = str(_root) + " (recursive)"
        mo.md(
            f"**{_resolved_from}** — **{len(files)}** `{_ext}` files"
            + ("" if files else " — ⚠️ no files found, check the path / drive mount")
        )
    return files, ims_backend, ims_n, ims_note, ims_set


@app.cell
def _(files, ims_n, mo, mode_select):
    _n = max((ims_n - 1) if mode_select.value == "ims" else (len(files) - 1), 0)
    frame_slider = mo.ui.slider(start=0, stop=_n, step=1, value=0, label="Frame")
    stride_select = mo.ui.dropdown(
        options={"1": 1, "2": 2, "5": 5, "10": 10, "50": 50},
        value="1",
        label="Stride (frames per step)",
    )
    pulse_select = mo.ui.dropdown(
        options={"pulse A": 0, "pulse B": 1, "A − B (difference)": "diff"},
        value="pulse A",
        label="Image content",
    )
    auto_contrast = mo.ui.checkbox(value=True, label="Auto contrast (p2–p99.5)")
    vmax_slider = mo.ui.slider(
        start=50, stop=4500, step=50, value=1000, label="Manual vmax (counts)"
    )
    crop_box = mo.ui.text(value="", label="Crop cols, e.g. 550:2050 (empty = full)")
    ds_select = mo.ui.dropdown(
        options={"full": 1, "½": 2, "¼": 4}, value="½", label="Preview scale"
    )
    mo.vstack(
        [
            mo.hstack([frame_slider, stride_select, pulse_select]),
            mo.hstack([auto_contrast, vmax_slider, crop_box, ds_select]),
        ]
    )
    return (
        auto_contrast,
        crop_box,
        ds_select,
        frame_slider,
        pulse_select,
        stride_select,
        vmax_slider,
    )


@app.cell(hide_code=True)
def _(
    IMS_ACQ_HZ,
    auto_contrast,
    crop_box,
    ds_select,
    files,
    find_dt_us,
    frame_slider,
    ims_backend,
    ims_n,
    ims_note,
    ims_set,
    ims_stream,
    mo,
    mode_select,
    np,
    plt,
    pulse_select,
    read_frame,
    read_ims_frame,
    stride_select,
    time,
    vmax_slider,
):
    if mode_select.value == "ims":
        if not ims_set or ims_n == 0:
            _output = mo.md(f"_No open `.ims` set. {ims_note}_")
        else:
            _idx = min(frame_slider.value * int(stride_select.value), ims_n - 1)
            _t0 = time.time()
            if ims_backend == "lvpyio":
                _a_raw, _b_raw = read_ims_frame(ims_set, _idx)
            else:
                _a_raw, _b_raw = ims_stream.read_ims_pair(ims_set, _idx)
            _sel = pulse_select.value
            if _sel == "diff":
                _img = _a_raw.astype(float) - _b_raw.astype(float)
            else:
                _img = (_a_raw if int(_sel) == 0 else _b_raw).astype(float)
            # optional column crop "a:b"
            _crop_txt = crop_box.value.strip()
            if _crop_txt:
                _a, _, _b = _crop_txt.partition(":")
                _img = _img[:, int(_a or 0) : int(_b or _img.shape[1])]
            _ds = int(ds_select.value)
            _view = _img[::_ds, ::_ds]
            if auto_contrast.value:
                _vlo, _vhi = float(np.percentile(_view, 2)), float(
                    np.percentile(_view, 99.5)
                )
            else:
                _vlo, _vhi = (0.0, float(vmax_slider.value)) if _sel != "diff" else (
                    -float(vmax_slider.value),
                    float(vmax_slider.value),
                )
            _t = _idx / IMS_ACQ_HZ
            _ms = (time.time() - _t0) * 1000.0
            _fig, _ax = plt.subplots(figsize=(12, 8))
            _ax.imshow(_view, cmap="gray", vmin=_vlo, vmax=_vhi, origin="upper")
            _ax.set_title(
                f".ims pair {_idx + 1}/{ims_n} (t ≈ {_t:.2f} s) "
                f"{_view.shape[0] * _ds}×{_view.shape[1] * _ds}px"
                f" — mean {_view.mean():.1f}, max {_view.max():.0f} "
                f"[{_ms:.0f} ms]"
            )
            _ax.set_xlabel("x [px]")
            _ax.set_ylabel("y [px]")
            _output = _fig
    elif not files:
        _output = mo.md("_No files — fix the folder above._")
    elif mode_select.value == "vc7":
        _idx = min(frame_slider.value * int(stride_select.value), len(files) - 1)
        _t0 = time.time()
        _arr, _attrs, _meta = read_frame(str(files[_idx]))
        _u, _v = _arr[0].astype(float), _arr[1].astype(float)
        _mag = np.hypot(_u, _v)
        _mmax = float(np.nanmax(_mag)) if _mag.size else 0.0
        _fig, _ax = plt.subplots(figsize=(12, 8))
        _yy, _xx = np.mgrid[0 : _u.shape[0], 0 : _u.shape[1]]
        _step = max(1, _u.shape[0] // 40, _u.shape[1] // 50)
        _q = _ax.quiver(
            _xx[::_step, ::_step],
            _yy[::_step, ::_step],
            _u[::_step, ::_step],
            _v[::_step, ::_step],
            _mag[::_step, ::_step],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=(_mmax / 15) if _mmax > 0 else 1.0,
        )
        _ax.invert_yaxis()
        _ax.set_aspect("equal")
        _ms = (time.time() - _t0) * 1000.0
        _ax.set_title(
            f"{files[_idx].name} — vectors {_meta['nx']}×{_meta['ny']} "
            f"(grid {_meta['vector_grid']}), |V|max={np.nanmax(_mag):.1f} "
            f"[{_ms:.0f} ms]"
        )
        plt.colorbar(_q, ax=_ax, label="|V| (file units)")
        _output = _fig
    else:
        _idx = min(frame_slider.value * int(stride_select.value), len(files) - 1)
        _t0 = time.time()
        _arr, _attrs, _meta = read_frame(str(files[_idx]))
        _sel = pulse_select.value
        if _sel == "diff":
            _img = _arr[0].astype(float) - _arr[1].astype(float)
        else:
            _img = _arr[int(_sel)].astype(float)
        # optional column crop "a:b"
        _crop_txt = crop_box.value.strip()
        if _crop_txt:
            _a, _, _b = _crop_txt.partition(":")
            _img = _img[:, int(_a or 0) : int(_b or _img.shape[1])]
        _ds = int(ds_select.value)
        _view = _img[::_ds, ::_ds]
        if auto_contrast.value:
            _vlo, _vhi = float(np.percentile(_view, 2)), float(
                np.percentile(_view, 99.5)
            )
        else:
            _vlo, _vhi = (0.0, float(vmax_slider.value)) if _sel != "diff" else (
                -float(vmax_slider.value),
                float(vmax_slider.value),
            )
        _dt_us, _dt_name = find_dt_us(_attrs)
        _dt_txt = f", dt≈{_dt_us:.1f} µs" if _dt_us else ""
        _ms = (time.time() - _t0) * 1000.0
        _fig, _ax = plt.subplots(figsize=(12, 8))
        _ax.imshow(_view, cmap="gray", vmin=_vlo, vmax=_vhi, origin="upper")
        _ax.set_title(
            f"{files[_idx].name} (frame {_idx + 1}/{len(files)}) "
            f"{_view.shape[0] * _ds}×{_view.shape[1] * _ds}px"
            f"{_dt_txt} — mean {_view.mean():.1f}, max {_view.max():.0f} "
            f"[{_ms:.0f} ms]"
        )
        _ax.set_xlabel("x [px]")
        _ax.set_ylabel("y [px]")
        _output = _fig
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Filmstrip — 5 frames across the folder (transient / bubble scan)

    First → 25% → 50% → 75% → last. Warms the read cache, so slider scrubbing
    afterwards is faster (OS page cache + `lru_cache`).
    """)
    return


@app.cell
def _(
    files,
    ims_backend,
    ims_n,
    ims_set,
    ims_stream,
    mo,
    mode_select,
    np,
    plt,
    read_frame,
    read_ims_frame,
):
    if mode_select.value == "ims":
        if not ims_set or ims_n == 0:
            _output = mo.md("_Filmstrip needs an open `.ims` set — fix the path above._")
        else:
            _picks = sorted({0, ims_n // 4, ims_n // 2,
                             3 * ims_n // 4, ims_n - 1})
            _ncols = len(_picks)
            _fig, _axs = plt.subplots(
                1, _ncols, figsize=(4 * _ncols, 5), constrained_layout=True
            )
            if _ncols == 1:
                _axs = [_axs]
            for _ax, _pi in zip(_axs, _picks):
                if ims_backend == "lvpyio":
                    _arr, _ = read_ims_frame(ims_set, _pi)
                else:
                    _arr = ims_stream.read_ims_frame(ims_set, _pi, pulse=0)
                _v = _arr.astype(float)[::4, ::4]
                _vlo, _vhi = float(np.percentile(_v, 2)), float(np.percentile(_v, 99.5))
                _ax.imshow(_v, cmap="gray", vmin=_vlo, vmax=_vhi, origin="upper")
                _ax.set_title(f".ims pair {_pi}\nidx {_pi}", fontsize=9)
                _ax.set_xticks([])
                _ax.set_yticks([])
            _output = _fig
    elif not files or mode_select.value == "vc7":
        _output = mo.md("_Filmstrip is images-only; switch File type to `.im7`._")
    else:
        _picks = sorted({0, len(files) // 4, len(files) // 2,
                         3 * len(files) // 4, len(files) - 1})
        _ncols = len(_picks)
        _fig, _axs = plt.subplots(
            1, _ncols, figsize=(4 * _ncols, 5), constrained_layout=True
        )
        if _ncols == 1:
            _axs = [_axs]
        for _ax, _pi in zip(_axs, _picks):
            _arr, _, _ = read_frame(str(files[_pi]))
            _v = _arr[0].astype(float)[::4, ::4]
            _vlo, _vhi = float(np.percentile(_v, 2)), float(np.percentile(_v, 99.5))
            _ax.imshow(_v, cmap="gray", vmin=_vlo, vmax=_vhi, origin="upper")
            _ax.set_title(f"{files[_pi].name}\nidx {_pi}", fontsize=9)
            _ax.set_xticks([])
            _ax.set_yticks([])
        _output = _fig
    _output
    return


if __name__ == "__main__":
    app.run()
