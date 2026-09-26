# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "xarray",
#     "zarr",
#     "scipy",
#     "lvpyio",
#     "tqdm",
# ]
# ///

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

from lvpyio import read_buffer
import marimo
import numpy as np
import xarray as xr

# ---------------------------------------------------------
# Static geometry and wall mask configuration
# ---------------------------------------------------------
RAW_H = 2390
PX_PER_MM = 995.28671814631889

WM_RIGHT_CENTER_PX = 1310.6908790170396
WM_RIGHT_AMPLITUDE_PX = 555.2436724529737
WM_WAVELENGTH_PX = 4862.161913832164
WM_PHASE_RAD = 4.746677044475907
WM_DRIFT_PX_PER_ROW = 0.03800152249355379

ROW_START = 3
ROW_END = 400  # 397 rows per half

TOP_DIR = Path(
    r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\first_half_case_1\ImgPreproc_03\PIV_MPd(4x24x24_75%ov_ImgCorr)"
)
BOT_DIR = Path(
    r"D:\channel_flow_research\channel_04_zoom_in\Project_FlowMaster_260705_110445\second_half\ImgPreproc\PIV_MPd(4x24x24_75%ov_ImgCorr)"
)
OUT_DIR = Path("outputs/channel04_zoom_in_stitched")


def right_edge_at(row_px):
    return (
        WM_RIGHT_CENTER_PX
        + WM_DRIFT_PX_PER_ROW * row_px
        + WM_RIGHT_AMPLITUDE_PX * np.sin(2.0 * np.pi * row_px / WM_WAVELENGTH_PX + WM_PHASE_RAD)
    )


def read_half(fpath, row_offset):
    from lvpyio import read_buffer  # import inside worker
    buf = read_buffer(str(fpath))[0]
    u = buf.components["U0"][0].astype(np.float32)
    v = -buf.components["V0"][0].astype(np.float32)  # DaVis v sign flip
    mask = (buf.masks[0] & buf.enabled[0]).astype(np.float32)

    u[mask == 0] = np.nan
    v[mask == 0] = np.nan
    u = buf.scales.i.offset + u * buf.scales.i.slope
    v = buf.scales.i.offset + v * buf.scales.i.slope

    x_mm = buf.scales.x.offset + (np.arange(u.shape[1]) + 0.5) * buf.scales.x.slope * buf.grid.x
    y_mm = buf.scales.y.offset + (np.arange(u.shape[0]) + 0.5) * buf.scales.y.slope * buf.grid.y
    col_px = (x_mm - x_mm.min()) * PX_PER_MM
    row_px = (y_mm.max() - y_mm) * PX_PER_MM + row_offset

    # Wall mask
    r2d, c2d = np.meshgrid(row_px, col_px, indexing="ij")
    invalid = (c2d > right_edge_at(r2d)) | (mask < 0.5)
    u[invalid] = np.nan
    v[invalid] = np.nan
    mask[invalid] = 0.0

    return row_px, col_px, u, v, mask


def process_single_pair(args):
    """Processes a single (top, bottom) vc7 pair into stitched arrays."""
    fpath_a, fpath_b, i_y, w_y = args
    r_a, c_a, u_a, v_a, m_a = read_half(fpath_a, 0)
    r_b, c_b, u_b, v_b, m_b = read_half(fpath_b, RAW_H)

    u_concat = np.concatenate([u_a[ROW_START:ROW_END], u_b[ROW_START:ROW_END]], axis=0)
    v_concat = np.concatenate([v_a[ROW_START:ROW_END], v_b[ROW_START:ROW_END]], axis=0)
    m_concat = np.concatenate([m_a[ROW_START:ROW_END], m_b[ROW_START:ROW_END]], axis=0)

    u_stitched = (1.0 - w_y) * u_concat[i_y] + w_y * u_concat[i_y + 1]
    v_stitched = (1.0 - w_y) * v_concat[i_y] + w_y * v_concat[i_y + 1]
    m_stitched = (1.0 - w_y) * m_concat[i_y] + w_y * m_concat[i_y + 1]

    chc_stitched = (m_stitched > 0.5).astype(np.float32)
    u_stitched[chc_stitched == 0] = np.nan
    v_stitched[chc_stitched == 0] = np.nan

    return u_stitched, v_stitched, chc_stitched


def run_full_batch(
    max_frames=None,
    chunk_size=50,
    n_workers=6,
    overwrite=True,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top_files = sorted(TOP_DIR.glob("*.vc7"))
    bot_files = sorted(BOT_DIR.glob("*.vc7"))
    n_pairs = min(len(top_files), len(bot_files))
    target_n = n_pairs if max_frames is None else min(max_frames, n_pairs)

    ds_zarr_path = OUT_DIR / "channel04_zoom_in_stitched_ds.zarr"
    stats_zarr_path = OUT_DIR / "channel04_zoom_in_stitched_stats.zarr"

    if overwrite:
        if ds_zarr_path.exists():
            shutil.rmtree(ds_zarr_path)
        if stats_zarr_path.exists():
            shutil.rmtree(stats_zarr_path)

    t_start = time.perf_counter()

    # 1. Establish coordinate grids from first frame
    r_a, c_a, _, _, _ = read_half(top_files[0], 0)
    r_b, c_b, _, _, _ = read_half(bot_files[0], RAW_H)

    row_concat = np.concatenate([r_a[ROW_START:ROW_END], r_b[ROW_START:ROW_END]])
    dy = 6.0
    common_y = np.arange(row_concat.min(), row_concat.max() + 1e-5, dy)
    common_x = c_a.copy()  # exactly 405 columns

    x_mm = (common_x / PX_PER_MM).astype(np.float32)
    y_mm = (-common_y / PX_PER_MM).astype(np.float32)

    ny = len(common_y)
    nx = len(common_x)

    i_y = np.clip(np.searchsorted(row_concat, common_y) - 1, 0, len(row_concat) - 2)
    w_y = ((common_y - row_concat[i_y]) / (row_concat[i_y + 1] - row_concat[i_y]))[:, None].astype(np.float32)

    print(f"Grid established: ny={ny}, nx={nx}")
    print(f"X: {x_mm.min():.3f} to {x_mm.max():.3f} mm, Y: {y_mm.min():.3f} to {y_mm.max():.3f} mm")
    print(f"Processing {target_n} pairs in chunks of {chunk_size} with {n_workers} workers...")

    # Online accumulators
    sum_u = np.zeros((ny, nx), dtype=np.float64)
    sum_v = np.zeros((ny, nx), dtype=np.float64)
    sum_uu = np.zeros((ny, nx), dtype=np.float64)
    sum_vv = np.zeros((ny, nx), dtype=np.float64)
    sum_uv = np.zeros((ny, nx), dtype=np.float64)
    count_valid = np.zeros((ny, nx), dtype=np.int64)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, target_n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, target_n)
            current_chunk_len = chunk_end - chunk_start

            tasks = [
                (top_files[i], bot_files[i], i_y, w_y)
                for i in range(chunk_start, chunk_end)
            ]

            results = list(executor.map(process_single_pair, tasks))

            chunk_u = np.stack([res[0] for res in results], axis=0)
            chunk_v = np.stack([res[1] for res in results], axis=0)
            chunk_chc = np.stack([res[2] for res in results], axis=0)

            for i in range(current_chunk_len):
                valid = chunk_chc[i] > 0.5
                u_val = np.nan_to_num(chunk_u[i], nan=0.0)
                v_val = np.nan_to_num(chunk_v[i], nan=0.0)

                sum_u[valid] += u_val[valid]
                sum_v[valid] += v_val[valid]
                sum_uu[valid] += (u_val[valid] ** 2)
                sum_vv[valid] += (v_val[valid] ** 2)
                sum_uv[valid] += (u_val[valid] * v_val[valid])
                count_valid[valid] += 1

            t_coords = np.arange(chunk_start, chunk_end, dtype=np.int64)
            time_coords = (t_coords / 15.0).astype(np.float32)

            ds_chunk = xr.Dataset(
                data_vars={
                    "u": (("t", "y", "x"), chunk_u),
                    "v": (("t", "y", "x"), chunk_v),
                    "chc": (("t", "y", "x"), chunk_chc),
                },
                coords={
                    "t": t_coords,
                    "time_s": ("t", time_coords),
                    "y": y_mm,
                    "x": x_mm,
                },
                attrs={
                    "title": "Sinusoidal channel zoom-in stitched flow field",
                    "fs_hz": 15.0,
                    "px_per_mm": float(PX_PER_MM),
                },
            )

            if chunk_start == 0:
                ds_chunk.to_zarr(ds_zarr_path, mode="w")
            else:
                ds_chunk.to_zarr(ds_zarr_path, append_dim="t")

            elapsed = time.perf_counter() - t_start
            rate = chunk_end / elapsed
            eta = (target_n - chunk_end) / rate if rate > 0 else 0
            print(
                f"[{chunk_end}/{target_n}] ({chunk_end/target_n*100:5.1f}%) | "
                f"Rate: {rate:4.1f} fps | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s"
            )

    # Finalize stats
    valid_mask = count_valid > 0
    u_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    v_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    uu = np.full((ny, nx), np.nan, dtype=np.float32)
    vv = np.full((ny, nx), np.nan, dtype=np.float32)
    uv = np.full((ny, nx), np.nan, dtype=np.float32)
    tke = np.full((ny, nx), np.nan, dtype=np.float32)

    u_mean[valid_mask] = (sum_u[valid_mask] / count_valid[valid_mask]).astype(np.float32)
    v_mean[valid_mask] = (sum_v[valid_mask] / count_valid[valid_mask]).astype(np.float32)

    uu[valid_mask] = (sum_uu[valid_mask] / count_valid[valid_mask] - u_mean[valid_mask] ** 2).astype(np.float32)
    vv[valid_mask] = (sum_vv[valid_mask] / count_valid[valid_mask] - v_mean[valid_mask] ** 2).astype(np.float32)
    uv[valid_mask] = (sum_uv[valid_mask] / count_valid[valid_mask] - u_mean[valid_mask] * v_mean[valid_mask]).astype(np.float32)
    tke[valid_mask] = (0.5 * (uu[valid_mask] + vv[valid_mask])).astype(np.float32)

    dx_m = float(np.abs(np.diff(x_mm).mean())) * 1e-3
    dy_m = float(np.abs(np.diff(y_mm).mean())) * 1e-3
    dv_dx = np.gradient(v_mean, dx_m, axis=1)
    du_dy = np.gradient(u_mean, dy_m, axis=0) * (-1.0)
    vorticity = (dv_dx - du_dy).astype(np.float32)

    ds_stats = xr.Dataset(
        data_vars={
            "u_mean": (("y", "x"), u_mean),
            "v_mean": (("y", "x"), v_mean),
            "uu": (("y", "x"), uu),
            "vv": (("y", "x"), vv),
            "uv": (("y", "x"), uv),
            "tke": (("y", "x"), tke),
            "vorticity": (("y", "x"), vorticity),
            "valid_count": (("y", "x"), count_valid.astype(np.int32)),
        },
        coords={"y": y_mm, "x": x_mm},
        attrs={
            "title": "Sinusoidal channel zoom-in ensemble statistics",
            "n_samples": int(target_n),
            "fs_hz": 15.0,
        },
    )
    ds_stats.to_zarr(stats_zarr_path, mode="w")

    total_time = time.perf_counter() - t_start
    print(f"\nBatch processing complete in {total_time:.2f} s ({target_n/total_time:.1f} frames/s)!")
    print(f"Saved time-series store: {ds_zarr_path}")
    print(f"Saved ensemble stats store: {stats_zarr_path}")

    run_log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames_processed": target_n,
        "ny": ny,
        "nx": nx,
        "total_time_s": total_time,
        "frames_per_sec": target_n / total_time,
        "ds_zarr": str(ds_zarr_path.resolve()),
        "stats_zarr": str(stats_zarr_path.resolve()),
    }
    with open(OUT_DIR / "batch_stitch_run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    return run_log


# ---------------------------------------------------------
# Marimo app definitions
# ---------------------------------------------------------
__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Sinusoidal Channel Zoom-In Full Dataset Stitching Engine

    This notebook stitches all 2,315 dual-half frames (`first_half_case_1` and `second_half`)
    from the sinusoidal channel zoom-in dataset into a unified, chunked **PivPy Zarr** dataset:
    - **`outputs/channel04_zoom_in_stitched/channel04_zoom_in_stitched_ds.zarr`**: Full time-series $(t, y, x)$
    - **`outputs/channel04_zoom_in_stitched/channel04_zoom_in_stitched_stats.zarr`**: Ensemble statistics
    """)
    return


@app.cell
def _(mo):
    top_files = sorted(TOP_DIR.glob("*.vc7"))
    bot_files = sorted(BOT_DIR.glob("*.vc7"))
    n_pairs = min(len(top_files), len(bot_files))

    mo.md(f"""
    - **Top Half Directory**: `{TOP_DIR}` ({len(top_files)} files)
    - **Bottom Half Directory**: `{BOT_DIR}` ({len(bot_files)} files)
    - **Coincident Frames to Process**: **{n_pairs}** pairs
    - **Target Output Directory**: `{OUT_DIR}`
    """)
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Execute Full 2,315 Frame Batch")
    mo.vstack([
        mo.md("Click below to run the batch processor across all 2,315 frames:"),
        run_btn,
    ])
    return (run_btn,)


@app.cell
def _(mo, run_btn):
    res = None
    if run_btn.value:
        res = run_full_batch()
    mo.ui.table([res] if res else [])
    return (res,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sinusoidal channel zoom-in batch stitching")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size for Zarr append")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker processes")
    args = parser.parse_args()

    run_full_batch(max_frames=args.max_frames, chunk_size=args.chunk_size, n_workers=args.workers)
