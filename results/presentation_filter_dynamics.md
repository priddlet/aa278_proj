# Presentation — Filter dynamics predict

Directory: `figures/presentation/filter_dynamics/`

EKF RK45+STM predict (HW2 Tier A); σ_acc km/s²/√s process noise.

## Monte Carlo — ELFO (Filter dynamics predict)

EKF predict: **filter dynamics (MCI force model)** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs**

_EKF RK45+STM predict (HW2 Tier A); σ_acc km/s²/√s process noise._

Blackout fraction: **64.1%**

_Timing: |b_rx−b_truth| (m) on GNSS/LunaNet pseudorange epochs; XNAV-only / MSP-only blackout: clock not in H._

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady μ (km) | Steady RMS (km) | Blackout μ (km) | Non-blackout μ (km) | |b| mean (m) | |b| p95 (m) |
|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|
_Steady μ / Steady RMS: last 10% of arc epochs (excludes epoch-0 init spike)._

| xnav_only | 0.02 | 0.02 | 2.33 | 0.06 | 0.07 | 0.09 | 0.35 | — | — |
| gnss_only | 0.01 | 0.03 | 2.36 | 0.05 | 0.07 | 0.09 | 0.48 | 119.91 | 452.73 |
| hybrid | 0.01 | 0.02 | 2.33 | 0.05 | 0.06 | 0.09 | 0.29 | 13.06 | 45.16 |


Spreadsheet: `results/monte_carlo_filter_dynamics.xlsx`


Slide tables: `presentation/tables/filter_dynamics/`

## Figures

| File | Description |
|------|-------------|
| `mc_elfo_all_policies_envelope.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_all_policies_propagation.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_boxplot.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_final_cdf.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_gnss_only_clock_timing.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_gnss_only_envelope.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_gnss_only_propagation.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_hybrid_clock_timing.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_hybrid_envelope.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_hybrid_propagation.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_policy_bars.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_pulsar_sweep.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_toa_sweep.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_xnav_only_envelope.png` | MC / propagation (filter_dynamics) |
| `mc_elfo_xnav_only_propagation.png` | MC / propagation (filter_dynamics) |