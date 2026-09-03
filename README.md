# piv_channel

PIV in a rectangular vertical channel, by Yossi Elimelech at Soreq.

## Goal

Analyze PIV (Particle Image Velocimetry) data from channel flow experiments.

## Experiments

Two channel geometries:

- **Baseline** — straight channel, 70 mm wide, 10 mm high. Measurement spans the full 10 mm gap, wall to wall, expecting a parabolic or flat-parabolic velocity profile.
- **Wavy** — channel with sinusoidal wavy walls.

For each geometry:

- **Steady state** measurements at constant flow rate.
- **Transient** measurements in some cases: the pump is turned off and the velocity field changes with time. The time interval between PIV maps is not currently known (not a high-speed camera) — it may be recoverable from image or experiment metadata/timestamps and is worth investigating.
- **Boundary-layer zoom-in PIV** under steady state, constant flow rate conditions.

Flow direction: top to bottom. Walls are vertical, on the left and right; the wall-to-wall distance is horizontal.

## Data

All raw and processed experiment files are stored on a single external hard drive at `D:\channel_flow_research`. Each run is a DaVis project folder (LaVision Imager `.ims` camera streams, `MetaData*.attr`, `Settings_Acquisition_*.xml`). Folders seen so far:

- `baseline_channel` — includes a `..._pump_shutdown` run (transient).
- `channel_04` — steady-state runs (`steady_state_part_1`, `_part_2`, ...).
- `channel_04_zoom_in` — boundary-layer zoom-in cases.

**Timing — solved:** the PIV frame-pair `dt` is stored per `.im7` file, not in the `.xml`/`.attr` sidecar files as first guessed. Each `.im7` buffer's own metadata has `DevDataTrace5` (paired with `DevDataAlias5 = "Reference time dt : dt 1"`), in microseconds — e.g. 80 µs for `baseline_channel/Vmax_0p62_m2sec_steady_state`. `piv_pipeline.py`'s `process_im7_pair()` reads it directly from each file rather than assuming a fixed value. Sanity-checked: at this dt, the reported `Vmax = 0.62 m/s` implies a ~8.6 px displacement (a sensible PIV window fill), and the computed mean speed over 10 frames comes out within ~10% of the reported Vmax.

## Analysis goals

- Clean, reliable statistics and visualization.
- PIV quiver maps overlaid on the original images.
- Mean and turbulence flow fields.
- Streamwise velocity profiles as a function of distance from the wall.

## Repository layout

- `notebooks/` — working marimo notebooks, run from the repo root. `pair1_first_pass.py` is the first working end-to-end pipeline (openpiv single-pass PIV on a test image pair). `baseline_steady_state_batch.py` runs the same pipeline over N real `.im7` pairs from `D:\channel_flow_research` (via `process_im7_pair()`, which reads each pair's real `dt` from its own metadata) and computes mean/turbulence statistics with PIVPy's `.piv.reynolds_decomposition()`.
- `sample_data/` — small local test dataset (one dual-frame TIFF pair, `.im7`, and LaVision manual/example data) used to prototype the pipeline before pointing it at the real data on `D:\channel_flow_research`. Not committed to git (see `.gitignore`).
- `outputs/` — generated figures and intermediate vector files. Not committed to git.
- `exploration/` — first-attempt scripts and scratch experiments (openpiv parameter tuning, image inspection, LaVision I/O probing). Kept for reference, not part of the maintained pipeline.
- `skills/pivpy/` — the PIVPy skill (see Tools below).

Known issue in the current first-pass result: most vectors fail signal-to-noise validation. `dt` is a placeholder (`1.0`, i.e. results are in px/frame) until the real pulse separation is found in the DaVis timing metadata, and the interrogation window/search-area sizing needs tuning against the real displacement.

## Tools

- [PIVPy](https://github.com/alexlib/pivpy) — used for most data analysis. See `skills/pivpy/SKILL.md`. `pyproject.toml` currently points the `pivpy` dependency at a local editable clone in `../pivpy` (not the PyPI release), because this project's PIV-on-image plotting depends on a `plot(background="image", color_by=...)` addition made there that hasn't been released yet. Anyone else working on this repo needs `../pivpy` cloned alongside it; once that feature ships in a pivpy release, switch back to a normal version dependency.
- [openpiv-python](https://github.com/alexlib/openpiv-python) (also cloned locally alongside this project) — used to analyze some image pairs directly.
- [lvpyio](https://pypi.org/project/lvpyio/) — reads LaVision `.im7`/`.vc7` buffers (the real acquisition format on `D:\channel_flow_research`), including per-file device metadata (camera exposure, laser pulse timing).
- [marimo](https://marimo.io) notebooks — analysis is done interactively in marimo (reactive, cell-based `.py` notebooks) rather than plain scripts. Use the `marimo-notebook` and `marimo-pair` skills when creating or working in these notebooks.
