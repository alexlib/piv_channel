import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import xarray as xr

    ROOT = Path(__file__).resolve().parent.parent
    from piv_pipeline import process_pair

    return ROOT, mo, np, process_pair, xr


@app.cell(hide_code=True)
def _(mo):
    _text = (
        "# Batch PIV over a folder of image pairs\n"
        "Runs `process_pair()` (same pipeline as `pair1_first_pass.py`) on every "
        "TIFF in a folder and stores the result as a PIVPy-compatible "
        "`xarray.Dataset` (dims `y, x, t`; vars `u, v, chc`) for post-analysis."
    )
    mo.md(_text)
    return


@app.cell
def _(ROOT, mo):
    pair_dir = ROOT / "sample_data" / "tiff"
    pair_files = sorted(pair_dir.glob("*.tif"))
    mo.md(f"Found **{len(pair_files)}** image pairs in `{pair_dir.relative_to(ROOT)}`: "
          f"{', '.join(f.name for f in pair_files)}")
    return (pair_files,)


@app.cell
def _(np, pair_files, process_pair, xr):
    def to_dataset(result, t):
        """One process_pair() result -> a PIVPy-shaped xarray.Dataset for a single frame."""
        x1d = result["x"][0, :]
        y1d = result["y"][:, 0]
        chc = (~result["invalid"]).astype(float)  # PIVPy convention: 1.0 valid, 0.0 spurious
        return xr.Dataset(
            data_vars={
                "u": (("y", "x"), result["u"]),
                "v": (("y", "x"), result["v"]),
                "chc": (("y", "x"), chc),
            },
            coords={"x": x1d, "y": y1d, "t": t},
        )

    datasets = [
        to_dataset(process_pair(f), t) for t, f in enumerate(pair_files)
    ]
    ds = xr.concat(datasets, dim="t")
    ds.attrs.update(
        units_x="mm",
        units_y="mm",
        units_u="mm/frame",  # dt is a 1.0 placeholder; see README "Timing"
        units_v="mm/frame",
        dt=1.0,
        history="notebooks/batch_process.py using piv_pipeline.process_pair",
    )
    return (ds,)


@app.cell
def _(ROOT, ds):
    out_path = ROOT / "outputs" / "pairs_dataset.nc"
    (ROOT / "outputs").mkdir(exist_ok=True)
    ds.to_netcdf(out_path)
    return (out_path,)


@app.cell
def _(ds, mo, out_path):
    mo.md(
        f"Saved `{out_path.name}` ({ds.sizes['t']} frames, {ds.sizes['y']}x{ds.sizes['x']} grid) "
        f"for post-analysis with PIVPy's `.piv` accessor (see `skills/pivpy/SKILL.md`)."
    )
    return


if __name__ == "__main__":
    app.run()
