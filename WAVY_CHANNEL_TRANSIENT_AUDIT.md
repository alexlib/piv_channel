# Wavy Channel Transient Audit Report

Comprehensive investigation of all wavy channel acquisitions to determine whether any dataset contains a pump shutdown or deceleration transient.

**Audit Date**: 2026-09-26  
**Auditor**: Antigravity Automated PIV Pipeline  
**Finding**: **No wavy channel shutdown transient exists on disk.** All wavy channel recordings across both zoom-out and zoom-in campaigns are steady-state runs (or short optical setup clips).

---

## 1. Wavy Channel Zoom-Out (`channel_04\Project_FlowMaster_260630_130547`)

Every recording in `channel_04` was audited across its full temporal record using `pivpy.io.load_vc7()` or direct raw cross-correlation via `lvpyio.read_set()`:

| Recording Folder | Frames | Sampled Velocity / Displacements | Variation ($\sigma/\mu$) | Regime Determination |
|---|---|---|---|---|
| [`channel_04_Vmax_0p66...part_1`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_1) | 3,000 | $[0.519, 0.497, 0.509, 0.490, 0.487]\text{ m/s}$ | $\pm 2.4\%$ | **Steady State** (Part 1 of 4) |
| [`channel_04_Vmax_0p66...part_2`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_2) | 3,000 | Mean disp: $17.7\text{ px} \to 15.6\text{ px}$ | Flat | **Steady State** (Part 2 of 4) |
| [`channel_04_Vmax_0p66...part_3`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_3) | 3,000 | Mean disp: $15.8\text{ px} \to 18.2\text{ px}$ | Flat | **Steady State** (Part 3 of 4) |
| [`channel_04_Vmax_0p66...part_4`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/channel_04_Vmax_0p66_m2sec_steady_state_part_4) | 3,000 | Mean disp: $14.9\text{ px} \to 16.1\text{ px}$ | Flat | **Steady State** (Part 4 of 4) |
| [`Recording_Date=260630_Time=133312`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260630_Time=133312) | 200 | $[5.586, 5.807, 5.877, 5.410, 6.051]\text{ m/s}$ | $\pm 4.1\%$ | **Steady State** (Early setup run) |
| [`Recording_Date=260630_Time=144257`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260630_Time=144257) | 500 | $[4.457, 4.931, 4.869, 4.820, 5.043]\text{ m/s}$ | $\pm 3.3\%$ | **Steady State** (Setup with background sub) |
| [`Recording_Date=260701_Time=111428`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=111428) | 200 | $[9.102, 9.341, 8.889, 9.189, 9.337]\text{ m/s}$ | $\pm 1.9\%$ | **Steady State** (Early calibration run) |
| [`Recording_Date=260701_Time=123040`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=123040) | 500 | $[8.418, 8.403, 8.271, 8.531, 8.228]\text{ m/s}$ | $\pm 1.5\%$ | **Steady State** (Early calibration run) |
| [`Recording_Date=260701_Time=142456`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=142456) | 21 | $[0.4610 \to 0.4617]\text{ m/s}$ | $\pm 2.1\%$ | **Short Test Clip** |
| [`Recording_Date=260701_Time=160339`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=160339) | 50 | $[0.5046 \to 0.4919]\text{ m/s}$ | $\pm 2.5\%$ | **Short Test Clip** |
| [`Recording_Date=260701_Time=161847`](file:///D:/channel_flow_research/channel_04/Project_FlowMaster_260630_130547/Recording_Date=260701_Time=161847) | 21 | $[0.5196 \to 0.4856]\text{ m/s}$ | $\pm 2.4\%$ | **Short Test Clip** (Pre-steady check) |

*Total Zoom-Out Volume*: 16,500 dual frames ($> 18\text{ minutes}$ of recording). None shows a shutdown decay.

---

## 2. Wavy Channel Zoom-In (`channel_04_zoom_in\Project_FlowMaster_260705_110445`)

Audited at 5% intervals throughout the full recording duration:

| Recording Folder | Frames | Sampled Mean Velocity ($V_m$) | Min / Max ($V_m$) | Max / Min Ratio | Regime Determination |
|---|---|---|---|---|---|
| [`first_half_case_1`](file:///D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/first_half_case_1) | 2,315 | $0.246 \to 0.288 \to 0.245 \to 0.227\text{ m/s}$ | $0.215 / 0.288\text{ m/s}$ | 1.34 | **Steady Turbulent Flow** (Trough to crest) |
| [`first_half_case_2`](file:///D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/first_half_case_2) | 3,000 | $0.283 \to 0.356 \to 0.263 \to 0.319\text{ m/s}$ | $0.213 / 0.356\text{ m/s}$ | 1.67 | **Steady Turbulent Flow** (Repetition run) |
| [`second_half`](file:///D:/channel_flow_research/channel_04_zoom_in/Project_FlowMaster_260705_110445/second_half) | 3,000 | $0.211 \to 0.254 \to 0.272 \to 0.225\text{ m/s}$ | $0.201 / 0.319\text{ m/s}$ | 1.58 | **Steady Turbulent Flow** (Crest to trough) |

*Total Zoom-In Volume*: 8,315 dual frames. Flow exhibits natural turbulent boundary layer fluctuations and vortex shedding, but does not decelerate toward rest.

---

## 3. Comparison with the Confirmed Shutdown Case

For benchmark comparison, the only genuine pump shutdown captured across the entire research campaign is:

- **Folder**: [`D:\channel_flow_research\baseline_channel\Vmax_0p62_m2sec_pump_shutdown`](file:///D:/channel_flow_research/baseline_channel/Vmax_0p62_m2sec_pump_shutdown)
- **Container**: `Camera1-1.ims` (13.92 GB, 1,000 dual-frame pairs @ 15 Hz, $\Delta t = 80.0\,\mu\text{s}$)
- **Behavior**:
  - $t = 0.0\text{ s}$ (Frame 0): $V = -581.5\text{ mm/s}$ (97.6% valid vectors)
  - $t \approx 11.3 - 12.0\text{ s}$ (Frames 170–180): Pump trip onset with rapid flow deceleration
  - $t = 26.7\text{ s}$ (Frame 400): Decelerated intermediate flow
  - $t > 50\text{ s}$ (Frames 750–999): Creeping / quiescent flow ($< 50\text{ mm/s}$)
  - Velocity drop ratio: $> 90\%$ drop.

In contrast, none of the wavy channel recordings exhibits a $> 10\%$ persistent drop in velocity.

---

## 4. Scientific Conclusion & Recommendation

1. **Transient analysis is strictly confined to the Baseline Straight Channel**:
   - Notebook [`notebooks/straight_channel_pump_shutdown_ims.py`](file:///C:/Users/alex/Github/piv_channel/notebooks/straight_channel_pump_shutdown_ims.py) processes the complete 1,000-frame transient directly from `Camera1-1.ims`.
2. **Wavy Channel notebooks should focus exclusively on Steady-State Turbulence**:
   - Mean flow, Reynolds stresses $\langle u'v' \rangle$, wall shear distributions along the wavy profile, and flow separation/recirculation dynamics across the crests and troughs.
   - Do not attempt transient decay analysis on `channel_04` or `channel_04_zoom_in` as no pump trip was performed during those acquisitions.
