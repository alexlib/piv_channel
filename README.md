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

**Timing lead:** each run folder has `Settings_Acquisition_Timing_*.xml` and `Settings_Acquisition_LightSource.EverGreenLaser_*.xml` files — these are the most likely place to find the laser pulse separation (the PIV frame-pair `dt`), which is not otherwise known.

## Analysis goals

- Clean, reliable statistics and visualization.
- PIV quiver maps overlaid on the original images.
- Mean and turbulence flow fields.
- Streamwise velocity profiles as a function of distance from the wall.

## Repository layout

- `notebooks/` — working marimo notebooks. `pair1_first_pass.py` is the first working end-to-end pipeline (openpiv single-pass PIV on a test image pair), run from the repo root.
- `sample_data/` — small local test dataset (one dual-frame TIFF pair, `.im7`, and LaVision manual/example data) used to prototype the pipeline before pointing it at the real data on `D:\channel_flow_research`. Not committed to git (see `.gitignore`).
- `outputs/` — generated figures and intermediate vector files. Not committed to git.
- `exploration/` — first-attempt scripts and scratch experiments (openpiv parameter tuning, image inspection, LaVision I/O probing). Kept for reference, not part of the maintained pipeline.
- `skills/pivpy/` — the PIVPy skill (see Tools below).

Known issue in the current first-pass result: most vectors fail signal-to-noise validation. `dt` is a placeholder (`1.0`, i.e. results are in px/frame) until the real pulse separation is found in the DaVis timing metadata, and the interrogation window/search-area sizing needs tuning against the real displacement.

## Tools

- [PIVPy](https://github.com/alexlib/pivpy) (also cloned locally alongside this project) — used for most data analysis. See `skills/pivpy/SKILL.md`.
- [openpiv-python](https://github.com/alexlib/openpiv-python) (also cloned locally alongside this project) — used to analyze some image pairs directly.
- [marimo](https://marimo.io) notebooks — analysis is done interactively in marimo (reactive, cell-based `.py` notebooks) rather than plain scripts. Use the `marimo-notebook` and `marimo-pair` skills when creating or working in these notebooks.
