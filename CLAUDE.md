# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PIV (Particle Image Velocimetry) analysis of flow in a rectangular vertical channel, for Yossi Elimelech at Soreq. Python project managed with `uv`, built on `marimo` (reactive notebooks), `openpiv` (cross-correlation PIV), and `pivpy` (vector-field post-processing). See `README.md` for the full experiment scope (baseline vs. wavy channel, steady-state/transient/zoom-in cases) and the real dataset location (`D:\channel_flow_research`).

## Commands

- Install deps: `uv sync`
- Add a dependency: `uv add <package>`
- Run a notebook script (sanity-check outside the marimo UI), from the repo root: `uv run python notebooks/<name>.py`
- Edit a notebook interactively: `uv run marimo edit notebooks/<name>.py`

No test suite, linter, or formatter is configured yet.

## Repository layout

- `notebooks/` — the maintained marimo notebooks (`.py`, reactive/cell-based). Notebooks assume the repo root as the working directory (paths like `sample_data\...`, `outputs\...`), so run them from there.
- `sample_data/` — small local test dataset used to prototype pipelines before pointing them at the real data on `D:\channel_flow_research`. Gitignored.
- `outputs/` — generated figures/vector files. Gitignored.
- `exploration/` — first-attempt scripts and scratch experiments, kept for reference only; not the pipeline to build on.
- `skills/pivpy/` — the PIVPy skill (below).

## Notes

- Requires Python >=3.14 (see `.python-version`, `pyproject.toml`).
- Prefer a marimo notebook under `notebooks/` over a plain script for new analysis code — see the `marimo-notebook` skill for the correct file format.
- A `pivpy` skill is available at `skills/pivpy/SKILL.md` (sourced from github.com/alexlib/pivpy) covering the `.piv` xarray accessor: ingestion, outlier cleaning, vortex/kinematic diagnostics, turbulence statistics, and plotting/animation recipes. Consult it for any PIV vector-field analysis work.
- Raw TIFFs from this camera can be LZW-compressed; `imagecodecs` must be installed (it is, via `pyproject.toml`) for `imageio`/`tifffile` to decode them — import it even if unused directly, or reads fail with a codec error.
- `dt` (the time between PIV frame pairs) is not yet known for the real experiments. It is likely recoverable from `Settings_Acquisition_Timing_*.xml` / `Settings_Acquisition_LightSource.EverGreenLaser_*.xml` inside each DaVis run folder on `D:\channel_flow_research` — check there before assuming or asking the user.
