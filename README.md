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

**Timing — solved:** the PIV frame-pair `dt` is stored per `.im7` file, not in the `.xml`/`.attr` sidecar files as first guessed. Each `.im7` buffer's own metadata has `DevDataTrace5` (paired with `DevDataAlias5 = "Reference time dt : dt 1"`), in microseconds — e.g. 80 µs for `baseline_channel/Vmax_0p62_m2sec_steady_state`, 120 µs for `channel_04/.../channel_04_Vmax_0p66_m2sec_steady_state_part_1`. `piv_pipeline.py`'s `process_im7_pair()` reads it directly from each file rather than assuming a fixed value. Sanity-checked for baseline: at this dt, the reported `Vmax = 0.62 m/s` implies a ~8.6 px displacement (a sensible PIV window fill), and the computed mean speed over 10 frames comes out within ~10% of the reported Vmax. Each `.im7`'s buffer also carries per-camera metadata (`DevDataAlias0/1 = "Camera 2/1 : Exposure time"`) — confirmed (investigated 2026-09-04, see `channel04_steady_state_pair1.py`) that a project's `exported_images` folder holds one camera's own two PIV exposures (labelled `1A`/`1B` in the frame metadata, `Acq.Camera.Label = "Camera 1"` for both slots of `channel_04`'s export), i.e. `as_masked_array(0)`/`(1)` are genuinely the A/B exposure pair, not two different cameras — despite the project configuring two camera types (`ImagerCX5` + `Photron`).

**Calibration (`PX_PER_MM`) — per project, not global:** each DaVis project folder has its own `Properties/Calibration/Calibration.xml` with a `PixelPerMmFactor`. `baseline_channel` uses `173.419` px/mm (`piv_pipeline.PX_PER_MM`, from `baseline_channel`'s own `Calibration.xml`). `channel_04` (`Project_FlowMaster_260630_130547`) uses `172.88323514796974` px/mm, from `channel_04\Project_FlowMaster_260630_130547\Properties\Calibration\Calibration.xml` — defined locally in `channel04_steady_state_pair1.py`, not in `piv_pipeline.py`, since it's specific to that project folder. Check the relevant project's own `Calibration.xml` before reusing either value elsewhere.

**channel_04 (wavy channel) — crop vs. mask, and a known low-validity issue (investigated 2026-09-04):**
- The wavy walls move in/out of frame with row, so a single fixed-column crop (like baseline's `CHANNEL_CROP_COLS`) either clips valid interior at some rows or leaves wall pixels in at others. Use `piv_pipeline.wavy_wall_bounds(image)` instead — traces each row's wall position (bright, but inset from the true frame edge by ~40-60 px, not anchored at column 0/w-1; a narrow ~350 px search margin excludes interior bright outliers like dust/stray reflections that a wider margin catches) and returns per-row `(left_edge, right_edge)` arrays. Pass `crop_cols=None` and `wall_bounds=wavy_wall_bounds(...)` to `process_im7_pair()` to mask wall-adjacent vectors out of the *results* while keeping the frame — and its physical x-coordinates — uncropped. Compute the bounds once from a representative frame (wall geometry is fixed per run) and reuse across a batch.
- Even with wall masking and the crop question resolved, `channel04_.../part_1/B0001.im7` (and the first 5 pairs, checked) still shows a very high invalid-vector fraction (~85-100% depending on validation settings) with the baseline's default single-pass parameters (`winsize=64`, `searchsize=96`, `s2n_threshold=1.3`). Root-caused, not just observed: median per-window `sig2noise` (peak2peak, over interior-only windows) sits at ~1.19, just under the 1.3 threshold — the correlation signal is real (consistent with the ~13 px global shift found via `phase_cross_correlation`, and individual well-seeded windows show a clean peak at the expected offset) but *weak*, not random. It is **not** a display-range/contrast bug (raising `display_max` to the sensor's full range measurably helps a single window's s2n but doesn't fix the field — interior invalid stays ~83-85% at best), **not** primarily wall contamination (masking barely moves the overall invalid fraction), and **not** a `sig2noise_method` choice (`peak2mean` looks better on paper — the ratio blows up numerically — but the median-consistency check still rejects ~99% either way, because too many individual correlation peaks land in the wrong place, not because the pass/fail threshold is miscalibrated). Raw particle-image intensity here is measurably dimmer than baseline at the same percentile comparison. Needs a decision before batching `channel_04`: confirm whether this is expected for the run, or whether single-pass PIV needs to give way to multi-pass PIV / different acquisition settings.

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
