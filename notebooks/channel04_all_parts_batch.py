import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# Wavy channel (channel_04), steady state - all parts orchestration\n"
        "Runs the same batch pipeline as `channel04_steady_state_batch.py` "
        "(`piv_pipeline.run_steady_state_batch()`, factored out of that "
        "notebook so it isn't duplicated per part) over every "
        "`channel_04_Vmax_0p66_m2sec_steady_state_part_N` folder found under "
        "`channel_04\\Project_FlowMaster_260630_130547`, one part at a time "
        "(each part already uses every core internally - running parts "
        "concurrently would oversubscribe CPU, not speed anything up).\n\n"
        "Every part reuses the *same* `channel04_pair1_config.json` (same "
        "physical channel/camera, only the time segment differs), and each "
        "gets its own zarr store + plots, namespaced by part name. A "
        "combined run log (`channel04_all_parts_run_log.json`) records every "
        "part's pair count, invalid fraction, mean speed, output paths, and "
        "wall-clock time - the reference for later cross-part averaging."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import numpy as np

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import run_steady_state_batch, load_run_config

    return ROOT, json, load_run_config, mo, np, run_steady_state_batch


@app.cell
def _(ROOT, load_run_config, mo):
    config = load_run_config(ROOT / "notebooks" / "channel04_pair1_config.json")
    PROJECT_ROOT = (
        r"D:\channel_flow_research\channel_04\Project_FlowMaster_260630_130547"
    )
    mo.md(
        f"`PROJECT_ROOT = {PROJECT_ROOT}`  \n"
        f"config: `channel04_pair1_config.json` ({config['run_name']}) - "
        f"reused unchanged for every part below."
    )
    return PROJECT_ROOT, config


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - list the part folders
    """)
    return


@app.cell
def _(PROJECT_ROOT, mo):
    from pathlib import Path as _Path

    # Each part_N folder holds its .im7 pairs one subfolder-of-a-subfolder
    # down: part_N\exported_images\<same name as part_N> (a sibling
    # calibration_image folder lives next to it and is not PIV data).
    part_dirs = sorted(_Path(PROJECT_ROOT).glob("channel_04_Vmax_0p66_m2sec_steady_state_part_*"))
    im7_folders = []
    for _part_dir in part_dirs:
        _exported = _part_dir / "exported_images" / _part_dir.name
        if _exported.is_dir():
            im7_folders.append(_exported)

    _rows = "\n".join(
        f"| {f.parent.parent.name} | `{f}` | {len(list(f.glob('*.im7')))} pairs |"
        for f in im7_folders
    )
    mo.md(
        f"Found **{len(im7_folders)}** part folders:\n\n"
        f"| part | folder | pairs |\n|---|---|---|\n{_rows}"
    )
    return (im7_folders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - run the batch pipeline once per part
    """)
    return


@app.cell
def _(ROOT, config, im7_folders, mo, run_steady_state_batch):
    _out_dir = ROOT / "outputs" / "channel04_all_parts"

    run_log = []
    for _folder in im7_folders:
        _run_name = _folder.parent.parent.name  # e.g. channel_04_..._part_2
        _summary = run_steady_state_batch(_folder, config, _out_dir, _run_name)
        run_log.append(_summary)

    _rows = "\n".join(
        f"| {r['run_name']} | {r['n_pairs']} | {r['invalid_fraction']:.1%} | "
        f"{r['mean_speed_mm_s']:.0f} | {r['elapsed_seconds']/60:.1f} |"
        for r in run_log
    )
    mo.md(
        f"Ran {len(run_log)} parts:\n\n"
        f"| part | pairs | invalid | mean speed [mm/s] | elapsed [min] |\n"
        f"|---|---|---|---|---|\n{_rows}"
    )
    return (run_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3 - save the combined run log
    """)
    return


@app.cell
def _(ROOT, json, mo, run_log):
    _log_path = ROOT / "outputs" / "channel04_all_parts" / "channel04_all_parts_run_log.json"
    _log_path.parent.mkdir(exist_ok=True, parents=True)
    with open(_log_path, "w") as _f:
        json.dump(run_log, _f, indent=2)
    mo.md(
        f"Saved `{_log_path.name}` - {len(run_log)} runs logged "
        f"(pair count, invalid fraction, mean speed, peak TKE, zarr/plot "
        f"paths, wall-clock seconds per part). Load it back with "
        f"`json.load(open(...))` when averaging across parts."
    )
    return


if __name__ == "__main__":
    app.run()
