# PIV Channel Flow Research — Dataset Inventory

Comprehensive audit of all `.im7` image pairs, `.vc7` vector fields, and `.ims` high-speed camera streams across `D:\channel_flow_research`.

Generated: 2026-09-26

---

## 1. Straight Baseline Channel (`baseline_channel`)

| Run / Directory | `.im7` (Images) | `.vc7` (Vectors) | `.ims` (Stream) | Acquisition Details & Description |
|---|---|---|---|---|
| [`Vmax_0p62_m2sec_steady_state`](file:///D:/channel_flow_research/baseline_channel/Vmax_0p62_m2sec_steady_state) | **3,000** in `exported_images` | **1,000** in `PIV_MPd(4x16x16_25%ov_ImgCorr)` (+2 avg) | None | Zoom-out full channel steady state ($173 \times 189$ grid, $\Delta V = +0.7\%$) |
| [`Vmax_0p62_m2sec_steady_state_right_boundary_layer`](file:///D:/channel_flow_research/baseline_channel/Vmax_0p62_m2sec_steady_state_right_boundary_layer) | **3,000** in `exported_images` | **3,000** in `PIV_MPd(4x16x16_25%ov_ImgCorr)` (+2 avg) | None | High-resolution right boundary layer steady state ($171 \times 203$ grid, $\tau_w \approx 0.335\text{ Pa}$) |
| [`Vmax_after_pump_shutdown`](file:///D:/channel_flow_research/baseline_channel/Vmax_after_pump_shutdown) | *0 particle images* (2 dark) | **100** in `PIV_MPd(4x16x16_25%ov_ImgCorr)` (+2 avg) | None | 100-frame quasi-steady run ($173 \times 189$ grid, $\Delta V = -3.9\%$, no transient decay) |
| [`Vmax_0p62_m2sec_pump_shutdown`](file:///D:/channel_flow_research/baseline_channel/Vmax_0p62_m2sec_pump_shutdown) | *0 exported* (2 dark) | *0* | **13.92 GB** (`Camera1-1.ims`) | **The real pump shutdown transient**: 1,000 dual-frame pairs @ 15 Hz ($66.7\text{ s}$), $\Delta t = 80\,\mu\text{s}$, direct streaming via `lvpyio.read_set()` |
| `Properties/Calibration...` | 6 calibration frames | *0* | None | Geometric PIV calibration target images |

---

## 2. Wavy Channel Zoom-Out (`channel_04\Project_FlowMaster_260630_130547`)

| Run / Directory | `.im7` (Images) | `.vc7` (Vectors) | Description |
|---|---|---|---|
| [`channel_04_Vmax_0p66_m2sec_steady_state_part_1`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_1) | **3,000** in `ImgPreproc`<br>**3,000** in `exported_images` | **3,000** in `ImgPreproc\PIV_MPd(4x16x16_25%ov_ImgCorr)` (+2 avg) | Steady state part 1 ($206 \times 205$ grid) |
| [`channel_04_Vmax_0p66_m2sec_steady_state_part_2`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_2) | **3,000** in `exported_images` | *0* | Steady state part 2 (images exported, uncomputed in DaVis) |
| [`channel_04_Vmax_0p66_m2sec_steady_state_part_3`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_3) | **3,000** in `exported_images` | *0* | Steady state part 3 (images exported, uncomputed in DaVis) |
| [`channel_04_Vmax_0p66_m2sec_steady_state_part_4`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_4) | **3,000** in `exported_images` | *0* | Steady state part 4 (images exported, uncomputed in DaVis) |
| [`Recording_Date=260630_Time=133312`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260630_Time=133312) | *0 particle images* (2 dark) | **200** in `PIV_MPd(4x16x16_25%ov)` (+2 avg) | 200-frame recording from stream |
| [`Recording_Date=260630_Time=144257`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260630_Time=144257) | **500** in `Subtract` | **500** in `PIV_MPd(4x16x16_25%ov)`<br>**500** in `Subtract\PIV_MPd...` | 500-frame recording with background subtraction |
| [`Recording_Date=260701_Time=111428`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=111428) | *0 particle images* (2 dark) | **200** in `PIV_MPd(4x24x24_25%ov)` (+2 avg) | 200-frame recording from stream |
| [`Recording_Date=260701_Time=123040`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=123040) | **21** in `ImgPreproc` | **500** in `PIV_MPd(4x24x24_25%ov)`<br>**21** in `ImgPreproc\PIV_MPd...` | 500-frame recording + 21-frame preprocessed sample |
| [`Recording_Date=260701_Time=142456`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=142456) | **21** in `ImgPreproc` | **21** in `ImgPreproc\PIV_MPd...` | Short test clip |
| [`Recording_Date=260701_Time=160339`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=160339) | **50** in `ImgPreproc` | **50** in `ImgPreproc\PIV_MPd...` | Short test clip |
| [`Recording_Date=260701_Time=161847`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=161847) | **21** in `ImgPreproc` | **21** in `ImgPreproc\PIV_MPd...` | Short test clip |

---

## 3. Wavy Channel Zoom-In (`channel_04_zoom_in\Project_FlowMaster_260705_110445`)

| Run / Directory | `.im7` (Images) | `.vc7` (Vectors) | Description |
|---|---|---|---|
| [`first_half_case_1`](file:///D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/first_half_case_1) | **2,315** in `ImgPreproc_03`<br>**2,315** in `exported_images` | **2,315** in `ImgPreproc_03\PIV_MPd(4x24x24_75%ov_ImgCorr)` (+2 avg) | First half (trough to crest), $404 \times 405$ grid, steady state |
| [`first_half_case_2`](file:///D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/first_half_case_2) | **1,000** in `ImgPreproc`<br>**3,000** in `exported_images` | **1,000** in `ImgPreproc\PIV_MPd(4x24x24_75%ov_ImgCorr)` (+2 avg) | High quality repetition, steady state |
| [`second_half`](file:///D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/second_half) | **3,000** in `ImgPreproc`<br>**1,000** in `ImgPreproc_01`<br>**3,000** in `exported_images` | **3,000** in `ImgPreproc\PIV_MPd(4x24x24_75%ov_ImgCorr)` (+2 avg)<br>**1,000** in `ImgPreproc_01\PIV_MPd...` | Second half (crest to trough), $404 \times 405$ grid, fluctuating flow |

---

## 4. Recommended OpenPIV Settings for `.ims` Stream Processing

Tuned in [`notebooks/straight_channel_pump_shutdown_ims.py`](file:///C:/Users/alex/Github/piv_channel/notebooks/straight_channel_pump_shutdown_ims.py):

| Parameter | Value | Rationale |
|---|---|---|
| **Loader** | `lvpyio.read_set(folder_path)` | Direct reading of `.ims` (no DaVis `.im7` export required) |
| **Preprocessing** | High-Pass (`sigma=16`, 97.5% clip) | Removes laser sheet non-uniformity and ambient light |
| **Interrogation Window** | $64 \times 64\text{ px}$ | Balances spatial resolution ($\sim 4.3\text{ mm}$) with particle count |
| **Search Area** | $96 \times 96\text{ px}$ | Accommodates large initial steady displacements ($\sim 5-9\text{ px}$) |
| **Overlap** | $32\text{ px}$ ($50\%$) | Spatial vector density |
| **Subpixel Method** | `gaussian` | Standard 3-point Gaussian peak centroid fit |
| **S/N Ratio Threshold** | $1.05 - 1.10$ | Yields $96-98\%$ valid vector coverage |
| **Median Filter** | $2.0$ | Normalized median test with local mean interpolation |
| **Channel Crop** | Columns $550 \to 2050$ | Crops unilluminated dead borders outside physical channel walls |
| **Scaling** | $14.6702\text{ px/mm}$, $\Delta t = 80\,\mu\text{s}$ | Calibrated baseline channel physical units |
