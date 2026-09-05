import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# channel_flow_research - full data inventory\n"
        "Scans every dataset root under `D:\\channel_flow_research` "
        "(`baseline_channel`, `channel_04`, `channel_04_zoom_in`) and builds "
        "one dataframe of every **case** (a distinct measurement) x "
        "**processing variant** (a DaVis PIV run over that case's raw "
        "images, or none if still unprocessed).\n\n"
        "The three roots don't share one folder layout, so the scanner "
        "handles both seen so far:\n\n"
        "- `channel_04`/`channel_04_zoom_in`: a `Project_FlowMaster_*` "
        "layer wraps the cases; PIV output sits inside an `ImgPreproc*` "
        "wrapper folder (`case/ImgPreproc_01/PIV_...`) - a case can have "
        "*several* such variants (different PIV settings tried on the same "
        "raw images).\n"
        "- `baseline_channel`: no `Project_FlowMaster_*` layer, cases sit "
        "directly under the root; PIV output sits directly under the case "
        "(`case/PIV_...`, no `ImgPreproc` wrapper); some cases have only "
        "dark-frame calibration images and no real acquisition, others have "
        "`.vc7` output but no `exported_images` folder at all.\n\n"
        "The PIV method itself is encoded in the `PIV_<method>(<passes>x<w>x<h>"
        "_<overlap>%ov[_<correction>])` folder name - e.g. "
        "`PIV_MPd(4x24x24_75%ov_ImgCorr)` = multi-pass deformation, 4 passes "
        "down to a 24x24 final window, 75% overlap, with image correction.\n\n"
        "Camera/timing metadata (pixel size, exposure, `dt`) comes from one "
        "representative `.im7` per case rather than DaVis's own "
        "`Settings_Acquisition_*.xml` - those store nested, DaVis-internal "
        "structures that don't reduce to simple scalars, whereas the same "
        "information sits as plain attributes on every `.im7` buffer "
        "(confirmed in earlier investigation - see README.md's Timing note)."
    )
    mo.md(_text)
    return


@app.cell
def _():
    import marimo as mo
    import re
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import lvpyio as lv

    ROOT = Path(__file__).resolve().parent.parent
    return ET, Path, ROOT, lv, mo, np, pd, re


@app.cell
def _(mo):
    DATA_ROOT = r"D:\channel_flow_research"
    DATASET_ROOTS = ["baseline_channel", "channel_04", "channel_04_zoom_in"]
    mo.md(f"`DATA_ROOT = {DATA_ROOT}`  \ndatasets: {', '.join(DATASET_ROOTS)}")
    return DATASET_ROOTS, DATA_ROOT


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 - discover projects, cases, and processing variants
    """)
    return


@app.cell
def _(ET, Path, re):
    PIV_FOLDER_RE = re.compile(
        r"PIV_(?P<method>\w+)\((?P<passes>\d+)x(?P<win_w>\d+)x(?P<win_h>\d+)"
        r"_(?P<overlap_pct>\d+)%ov(?:_(?P<correction>\w+))?\)"
    )


    def parse_calibration(project_dir):
        """PixelPerMmFactor from Properties/Calibration/Calibration.xml -
        shared across every case in a project.
        """
        cal_path = project_dir / "Properties" / "Calibration" / "Calibration.xml"
        if not cal_path.is_file():
            return None
        tree = ET.parse(cal_path)
        node = tree.getroot().find(".//PixelPerMmFactor")
        return float(node.get("Value")) if node is not None else None


    def parse_piv_folder_name(piv_dir):
        m = PIV_FOLDER_RE.match(piv_dir.name)
        if not m:
            return {}
        g = m.groupdict()
        return dict(
            piv_method=g["method"],
            piv_passes=int(g["passes"]),
            piv_final_window=f"{g['win_w']}x{g['win_h']}",
            piv_overlap_pct=int(g["overlap_pct"]),
            piv_correction=g["correction"],
        )


    def find_piv_variants(case_dir):
        """Yield (variant_label, piv_dir) for every PIV_* folder under a
        case - either directly under it (baseline_channel's convention) or
        one level down inside an ImgPreproc* wrapper (channel_04's
        convention). variant_label is the case's own name in the former
        case (no separate variant identity), or the ImgPreproc* folder's
        name in the latter.
        """
        for child in sorted(case_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("PIV_"):
                yield case_dir.name, child
            elif child.name.startswith("ImgPreproc"):
                for grandchild in sorted(child.iterdir()):
                    if grandchild.is_dir() and grandchild.name.startswith("PIV_"):
                        yield child.name, grandchild


    def find_raw_im7(case_dir):
        """(folder, count) of this case's raw .im7 files - prefers the
        exported_images/<case_name> convention, falls back to files
        directly in the case folder (some cases, e.g. dark-frame-only
        captures, never used the exported_images wrapper at all).
        """
        exported = case_dir / "exported_images" / case_dir.name
        if exported.is_dir():
            files = list(exported.glob("*.im7"))
            if files:
                return exported, len(files)
        files = list(case_dir.glob("*.im7"))
        return (case_dir, len(files)) if files else (None, 0)


    def discover_project(project_dir, dataset):
        """One dict per (case, processing variant) under one project
        folder - or one dict with variant=None for a case that has no
        PIV_* output yet (raw images only).
        """
        rows = []
        px_per_mm = parse_calibration(project_dir)

        for case_dir in sorted(project_dir.iterdir()):
            if not case_dir.is_dir() or case_dir.name == "Properties":
                continue
            im7_folder, n_im7 = find_raw_im7(case_dir)
            variants = list(find_piv_variants(case_dir))
            if n_im7 == 0 and not variants:
                continue  # nothing PIV-relevant in this folder

            base_row = dict(
                dataset=dataset,
                project=project_dir.name,
                case=case_dir.name,
                im7_folder=str(im7_folder) if im7_folder else None,
                n_im7_files=n_im7,
                px_per_mm=px_per_mm,
            )

            if not variants:
                rows.append({**base_row, "processing_variant": None})
                continue

            for variant_label, piv_dir in variants:
                row = {
                    **base_row,
                    "processing_variant": variant_label,
                    "piv_folder": str(piv_dir),
                    "n_vc7_files": len(list(piv_dir.glob("*.vc7"))),
                    "has_avg_stddev": (piv_dir / "Avg_StdDev").is_dir(),
                    **parse_piv_folder_name(piv_dir),
                }
                rows.append(row)
        return rows


    def discover_dataset(dataset_root):
        """A dataset root either wraps its cases in one or more
        Project_FlowMaster_* folders (channel_04, channel_04_zoom_in), or
        - if none exist - is itself the single implicit project
        (baseline_channel).
        """
        projects = sorted(p for p in dataset_root.glob("Project_FlowMaster_*") if p.is_dir())
        if not projects:
            projects = [dataset_root]
        rows = []
        for project_dir in projects:
            rows.extend(discover_project(project_dir, dataset_root.name))
        return rows

    return (discover_dataset,)


@app.cell
def _(DATASET_ROOTS, DATA_ROOT, Path, discover_dataset, mo, pd):
    rows = []
    for _name in DATASET_ROOTS:
        rows.extend(discover_dataset(Path(DATA_ROOT) / _name))

    inventory = pd.DataFrame(rows)
    mo.md(
        f"Found **{len(inventory)}** (case, processing variant) rows across "
        f"**{inventory['case'].nunique()}** cases in {len(DATASET_ROOTS)} datasets."
    )
    return (inventory,)


@app.cell
def _(inventory, mo):
    mo.ui.table(inventory, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 - camera/timing metadata from one representative pair per case
    """)
    return


@app.cell
def _(Path, inventory, lv, mo, np, pd):
    def case_im7_metadata(im7_folder):
        if not im7_folder:
            return {}
        files = sorted(Path(im7_folder).glob("*.im7"))
        if not files:
            return {}
        buffer = lv.read_buffer(str(files[0]))
        attrs = buffer.attributes
        dt = (
            float(np.asarray(attrs["DevDataTrace5"]).flat[0]) * 1e-6
            if "DevDataTrace5" in attrs else None
        )
        a1 = np.asarray(buffer.as_masked_array(0).data)
        frame_attrs = buffer.frames[0].attributes
        return dict(
            dt=dt,
            frame_shape=str(a1.shape),
            camera_name=frame_attrs.get("CameraName"),
            cam_pixel_size_um=frame_attrs.get("CamPixelSize"),
            ccd_exposure_time=frame_attrs.get("CCDExposureTime"),
        )

    _meta_by_folder = {
        folder: case_im7_metadata(folder)
        for folder in inventory["im7_folder"].dropna().unique()
    }
    _meta_df = pd.DataFrame(_meta_by_folder).T
    _meta_df.index.name = "im7_folder"

    inventory_full = inventory.merge(_meta_df, on="im7_folder", how="left")
    mo.ui.table(inventory_full, selection=None)
    return (inventory_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3 - save the inventory (CSV + markdown doc)
    """)
    return


@app.cell
def _(ROOT, inventory_full, mo):
    _csv_path = ROOT / "outputs" / "channel_flow_research_inventory.csv"
    _csv_path.parent.mkdir(exist_ok=True)
    inventory_full.to_csv(_csv_path, index=False)
    mo.md(f"Saved `{_csv_path.name}` - {len(inventory_full)} rows.")
    return


@app.cell
def _(ROOT, inventory_full, mo, pd):
    def _to_markdown_table(df):
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = "\n".join(
            "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)
        )
        return f"{header}\n{sep}\n{body}"

    _doc_lines = [
        "# channel_flow_research - data inventory",
        "",
        f"Scanned {inventory_full['dataset'].nunique()} dataset roots, "
        f"{inventory_full['case'].nunique()} cases, "
        f"{len(inventory_full)} (case, processing variant) rows.",
        "",
        "Generated by `notebooks/channel_flow_research_inventory.py` - "
        "see that notebook for how each column is derived (folder-layout "
        "notes, PIV-method-name parsing, calibration/timing sources).",
        "",
        "## Full inventory",
        "",
        _to_markdown_table(inventory_full),
        "",
    ]
    _doc_path = ROOT / "outputs" / "channel_flow_research_inventory.md"
    _doc_path.write_text("\n".join(_doc_lines), encoding="utf-8")
    mo.md(f"Saved `{_doc_path.name}`.")
    return


if __name__ == "__main__":
    app.run()
