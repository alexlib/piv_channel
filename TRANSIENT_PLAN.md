# Outstanding plan — PIV channel transient analysis

Status as of 2026-09-26. Checked against the live marimo kernel and the files on
disk, not from memory. Items marked **DONE** are listed briefly at the bottom so
the file doubles as a record of what is already in the notebooks.

## Blocking issue to resolve first

### The 100-frame "transient" run contains no transient
`baseline_channel/Vmax_after_pump_shutdown` is **quasi-steady**:
mean speed 498.6 ± 14.8 mm/s (3.0%), lag-1 autocorrelation 0.31 (noise-like),
linear trend −11 mm/s across the whole 6.6 s record, 0.85% mean reversed flow.
The time/y-averaged `v(x)` profile coincides with the 3000-frame steady-state
profile in the interior. The interior agrees with the steady run to within a few
percent; only the border columns differ, and that is a validity artefact (below).

**Consequence:** every "slowdown / phase of the slowdown" analysis currently has
no decay to resolve. Two things are needed before the transient narrative can be
written:

1. **Export the real transient to `.im7`.** `Vmax_0p62_m2sec_pump_shutdown/Camera1-1.ims`
   is 14 GB, 1000 images @ 15 Hz, and has no `.vc7` output. `lvpyio` cannot read
   `.ims` (`RuntimeError: .ims isn't an allowed set extension`) and it is not
   HDF5, so it cannot be read directly. It must be exported from DaVis to
   `exported_images/*.im7`. Once exported, `process_im7_pair()` + the tuned
   parameters in `notebooks/baseline_steady_state_batch.py`
   (`winsize=96, searchsize=96, overlap=32, s2n_threshold=1.0, median_threshold=2`,
   `preprocess="high_pass", hp_sigma=16, hp_pct=97.5`) apply unchanged, and
   `load_vc7_directory()` is not needed at all.
2. **A straight-channel zoom-in transient does not exist.** The right-boundary-layer
   folder holds only the steady run. Steps 5–6 of the notebook are resolution-limited
   for wall shear for exactly this reason; a zoom-in transient acquisition is
   required for a proper wall-shear measurement.

## Notebook 1 — straight channel (in place, needs data + finishing)

`notebooks/straight_channel_transient.py` (24 cells, all running clean).

- [ ] Point the loader at the exported transient; re-verify `dt` and the 15 Hz
      frame rate on the new run's own metadata rather than assuming.
- [ ] Re-run Steps 2–6 and confirm the audit now shows a real decay before any
      slowdown narrative is written.
- [ ] Harden the Step 5 wall-shear proxy. It currently differentiates at the
      **2nd and 3rd grid column**, which the Step 8 validity map shows is inside
      the invalid border. Use the first *reliably valid* column instead
      (threshold the time-averaged valid-fraction map), or state the wall-shear
      result as unusable at this resolution rather than reporting a number.
- [ ] Add a phase annotation to the animation (e.g. "decay", "near-stall",
      "reversed") once a real transient exists to phase.

## Notebook 2 — sinusoidal wavy channel (not started)

Planned scope, same structure as Notebook 1:

- [x] **Step 0 data probe COMPLETE.** Exhaustive temporal audit performed on 2026-09-26 across all 11 zoom-out and 3 zoom-in wavy channel folders (see `WAVY_CHANNEL_TRANSIENT_AUDIT.md`).
      **Finding**: No wavy shutdown transient exists on disk. All wavy runs are steady-state ($16,500$ frames zoom-out, $8,315$ frames zoom-in). Wavy channel analysis must focus on steady-state turbulence, wall shear, and crest/trough recirculation dynamics.
- [ ] Zoom-out steady-state analysis: `channel_04`-style processing with the existing
      `channel04_pair1_config.json` (sinusoidal wall mask, `high_pass` preprocessing)
      and `run_steady_state_batch()`.
- [ ] Zoom-in transient: the wall-shear measurement that is impossible in
      Notebook 1 becomes possible here.
- [ ] **Wave-phase binning** (the key wavy-specific analysis): bin profiles and
      diagnostics by the wall sinusoid phase from the shared
      `sinusoidal_wall_bounds()` model, then compare crest vs. trough.
- [ ] **Detachment / vortex tracking in time**: per-frame vorticity and
      Hunt's Q-criterion (`ds.piv.vorticity()`, `ds.piv.q_criterion()`), tracking
      whether the near-wall reversed region (35.5% reversed, up to +53 mm/s
      against −328 mm/s main flow in the existing steady zoom-in) grows, shrinks,
      or migrates along the wave as the flow decays.
- [ ] Same dimensional + normalized (`x/b`, `V/U_bulk`) dual-axis treatment as
      Notebook 1 so the two geometries are directly comparable.

## LaTeX report (started, incomplete)

- [x] `notebooks/straight_channel_transient_report.py` created — regenerates
      report figures from the stored zarr and emits numbers to JSON.
- [x] Figures 1–4 render to `outputs/report_figs/` (PNG + PDF):
      `fig01_timeseries`, `fig02_profiles`, `fig03_tke`, `fig04_steady_vs_shutdown`.
- [ ] **Fig 5 cell is broken and must be fixed.** Cell at
      `straight_channel_transient_report.py:272` has unresolved names: `mpl` is
      used but not in the cell signature; `step`, `X`, `Y`, `fr` are public
      loop/temp names; `snorm` is referenced but the value is bound as `_snorm`.
      Until this is fixed `fig05_maps.png/pdf` is never written.
- [ ] Run the notebook end-to-end to completion so
      `outputs/report_values.json` is actually written (it does not exist yet).
- [ ] Write `outputs/straight_channel_transient_report.tex`, pulling numbers from
      `report_values.json` rather than hand-copying them.
- [ ] Compile with `latexmk` (TinyTeX is installed: `pdflatex`, `xelatex`,
      `latexmk` all present; `pandoc` and `tectonic` are not).
- [ ] Run `uvx marimo check` on the report notebook.

## Housekeeping

- [ ] Update `README.md`: the transient-run findings (quasi-steady 100-frame run,
      the 15 Hz frame-rate discovery, the `1/(2b)` bulk-velocity convention, and
      the new `piv_pipeline` helpers `channel_center`, `bulk_velocity`,
      `profile_evolution`, `spatial_tke`).
- [ ] `README.md` still carries a stale "Known issue" paragraph claiming most
      vectors fail validation and that `dt` is a placeholder — both resolved.
- [ ] Decide whether `main.py` (still the `Hello from piv-channel!` stub) should
      be removed or pointed at something real.

## Process notes worth keeping

- `marimo check` and marimo's DAG rules are strict: `fig`, `ax`, `a`, `speed`,
  `prof` reused across cells all raise `MultipleDefinitionError`. Use `_`-prefixed
  names inside figure cells. This cost several iterations on the report notebook.
- The notebook has a PEP 723 header, so launching it needs an answer of **n** to
  the sandbox prompt (dependencies come from the project `.venv`):
  `uv run marimo edit notebooks/straight_channel_transient.py --no-token --host 127.0.0.1 --port 2718`
- Only one marimo server may hold port 2718; stale crashed servers must be killed
  or `execute-code.sh` will target the wrong kernel.

## Already done (for reference)

- [x] Timing solved: inter-frame time is 1/15 s from each run's `.set` recording-rate
      statistics; the 80 µs / 120 µs `DevDataTrace5` value is the *pulse* separation
      inside one frame pair, not the map-to-map time.
- [x] Fixed-scale quiver viewer with slider, one fixed colormap and one colorbar.
- [x] Time-colored profile evolution (`profile_evolution()`), 5-frame window,
      row-averaged, every 60 s up to 10 curves, plus a bulk-velocity-vs-time trace.
- [x] Dimensional (mm/s vs. centered mm) and normalized (`V/U_bulk` vs. `x/b`)
      variants, with `U_bulk = 1/(2b)·∫V dx` and a single reference `U_bulk`.
- [x] TKE redefined as spatial fluctuations about the instantaneous row-mean
      profile (`spatial_tke()`), with the math written above the figure.
- [x] Smoothed `coolwarm` animation, every 10th frame, static backdrop, fixed arrows.
- [x] Validity map, reversal-hotspot map, steady-vs-shutdown overlay.
- [x] Provenance markdown cell documenting source, format, timing, calibration, caveats.
- [x] HTML export: `outputs/straight_channel_transient.html` (2.87 MB, 8 embedded
      base64 PNG figures).
- [x] `b = 5.05 mm` from the grid, consistent with the 10 mm gap, in both the
      transient and steady runs.
