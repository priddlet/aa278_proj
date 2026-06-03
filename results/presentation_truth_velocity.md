# Presentation — Truth-velocity predict

Directory: `figures/presentation/truth_velocity/`

Oracle motion between measurements (default sim; optimistic absolute errors).

## Monte Carlo — ELFO (Truth-velocity predict)

EKF predict: **truth velocity** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs**

_Oracle motion between measurements (default sim; optimistic absolute errors)._

Blackout fraction: **64.1%**

_Timing: |b_rx−b_truth| (m) on GNSS/LunaNet pseudorange epochs; XNAV-only / MSP-only blackout: clock not in H._

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady μ (km) | Steady RMS (km) | Blackout μ (km) | Non-blackout μ (km) | |b| mean (m) | |b| p95 (m) |
|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|
_Steady μ / Steady RMS: last 10% of arc epochs (excludes epoch-0 init spike)._

| xnav_only | 3.74 | 3.86 | 2.46 | 0.77 | 0.97 | 0.46 | 0.99 | — | — |
| gnss_only | 4.89 | 5.02 | 16.09 | 0.78 | 1.03 | 0.45 | 21.10 | 13774.06 | 49407.02 |
| hybrid | 1.13 | 1.23 | 2.39 | 0.73 | 0.88 | 0.45 | 0.56 | 227.54 | 812.77 |


Spreadsheet: `results/monte_carlo_truth_velocity.xlsx`


Slide tables: `presentation/tables/truth_velocity/`

## Figures

| File | Description |
|------|-------------|
| `mc_elfo_all_policies_envelope.png` | MC / propagation (truth_velocity) |
| `mc_elfo_all_policies_propagation.png` | MC / propagation (truth_velocity) |
| `mc_elfo_boxplot.png` | MC / propagation (truth_velocity) |
| `mc_elfo_final_cdf.png` | MC / propagation (truth_velocity) |
| `mc_elfo_gnss_only_clock_timing.png` | MC / propagation (truth_velocity) |
| `mc_elfo_gnss_only_envelope.png` | MC / propagation (truth_velocity) |
| `mc_elfo_gnss_only_propagation.png` | MC / propagation (truth_velocity) |
| `mc_elfo_hybrid_clock_timing.png` | MC / propagation (truth_velocity) |
| `mc_elfo_hybrid_envelope.png` | MC / propagation (truth_velocity) |
| `mc_elfo_hybrid_propagation.png` | MC / propagation (truth_velocity) |
| `mc_elfo_policy_bars.png` | MC / propagation (truth_velocity) |
| `mc_elfo_pulsar_sweep.png` | MC / propagation (truth_velocity) |
| `mc_elfo_toa_sweep.png` | MC / propagation (truth_velocity) |
| `mc_elfo_xnav_only_envelope.png` | MC / propagation (truth_velocity) |
| `mc_elfo_xnav_only_propagation.png` | MC / propagation (truth_velocity) |