# Presentation — Filter CV predict

Directory: `figures/presentation/filter_predict/`

EKF constant-velocity predict only; more realistic dynamics stress.

## Monte Carlo — ELFO (Filter CV predict)

EKF predict: **filter CV** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs**

_EKF constant-velocity predict only; more realistic dynamics stress._

Blackout fraction: **64.1%**

_Timing: |b_rx−b_truth| (m) on GNSS/LunaNet pseudorange epochs; XNAV-only / MSP-only blackout: clock not in H._

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady μ (km) | Steady RMS (km) | Blackout μ (km) | Non-blackout μ (km) | |b| mean (m) | |b| p95 (m) |
|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|
_Steady μ / Steady RMS: last 10% of arc epochs (excludes epoch-0 init spike)._

| xnav_only | 7.39 | 7.59 | 3.50 | 2.33 | 2.91 | 1.24 | 2.80 | — | — |
| gnss_only | 10.44 | 10.64 | 680.18 | 2.36 | 3.03 | 1.23 | 808.91 | 532626.48 | 1873227.87 |
| hybrid | 3.09 | 3.17 | 2.78 | 2.27 | 2.81 | 1.23 | 1.04 | 578.01 | 2201.28 |


Spreadsheet: `results/monte_carlo_filter_predict.xlsx`


Slide tables: `presentation/tables/filter_predict/`

## Figures

| File | Description |
|------|-------------|
| `mc_elfo_all_policies_envelope.png` | MC / propagation (filter_predict) |
| `mc_elfo_all_policies_propagation.png` | MC / propagation (filter_predict) |
| `mc_elfo_boxplot.png` | MC / propagation (filter_predict) |
| `mc_elfo_final_cdf.png` | MC / propagation (filter_predict) |
| `mc_elfo_gnss_only_clock_timing.png` | MC / propagation (filter_predict) |
| `mc_elfo_gnss_only_envelope.png` | MC / propagation (filter_predict) |
| `mc_elfo_gnss_only_propagation.png` | MC / propagation (filter_predict) |
| `mc_elfo_hybrid_clock_timing.png` | MC / propagation (filter_predict) |
| `mc_elfo_hybrid_envelope.png` | MC / propagation (filter_predict) |
| `mc_elfo_hybrid_propagation.png` | MC / propagation (filter_predict) |
| `mc_elfo_policy_bars.png` | MC / propagation (filter_predict) |
| `mc_elfo_pulsar_sweep.png` | MC / propagation (filter_predict) |
| `mc_elfo_toa_sweep.png` | MC / propagation (filter_predict) |
| `mc_elfo_xnav_only_envelope.png` | MC / propagation (filter_predict) |
| `mc_elfo_xnav_only_propagation.png` | MC / propagation (filter_predict) |