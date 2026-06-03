# Presentation assets

Generated from `scripts/build_presentation_assets.py` (full mode).

Figures directory: `figures/presentation`

## Visibility

- **ELFO** arc: 30 hr
- Blackout fraction: **56.4%**
- Windows: **2**

## Monte Carlo — ELFO

Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs**

Blackout fraction: **64.1%**

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|-----------------|----------------|----------|-----------------|---------------------|
| xnav_only | 2.44 | 2.62 | 2.48 | 0.53 | 0.97 |
| gnss_only | 6.02 | 6.26 | 15.45 | 0.53 | 20.56 |
| hybrid | 1.10 | 1.29 | 4.38 | 0.53 | 3.72 |


Spreadsheet: `results/monte_carlo_presentation.xlsx`

## Figures

| File | Description |
|------|-------------|
| `elfo_orbit_blackout_3d.png` | 3D orbit colored by GNSS blackout |
| `elfo_orbit_blackout_xy.png` | XY ground track — blackout in red |
| `elfo_orbit_gnss_only.png` | Monte Carlo / sweep figure |
| `elfo_orbit_hybrid.png` | Monte Carlo / sweep figure |
| `elfo_orbit_xnav_only.png` | Monte Carlo / sweep figure |
| `elfo_segments_gnss_only.png` | Monte Carlo / sweep figure |
| `elfo_segments_hybrid.png` | Monte Carlo / sweep figure |
| `elfo_segments_xnav_only.png` | Monte Carlo / sweep figure |
| `elfo_truth_propagation.png` | Truth propagation verification panels |
| `elfo_visibility_timeline.png` | Earth elevation, GNSS/LunaNet flags, nav mode |
| `mc_elfo_all_policies_envelope.png` | MC mean error — all policies |
| `mc_elfo_all_policies_propagation.png` | All policies overlaid |
| `mc_elfo_boxplot.png` | Final error distribution by policy |
| `mc_elfo_final_cdf.png` | Final error empirical CDF |
| `mc_elfo_gnss_only_envelope.png` | Monte Carlo / sweep figure |
| `mc_elfo_gnss_only_propagation.png` | GNSS when visible / XNAV in blackout |
| `mc_elfo_hybrid_envelope.png` | Monte Carlo / sweep figure |
| `mc_elfo_hybrid_propagation.png` | Hybrid error vs time (trial 0) |
| `mc_elfo_policy_bars.png` | Segment error bar chart |
| `mc_elfo_pulsar_sweep.png` | Monte Carlo / sweep figure |
| `mc_elfo_toa_sweep.png` | Monte Carlo / sweep figure |
| `mc_elfo_xnav_only_envelope.png` | Monte Carlo / sweep figure |
| `mc_elfo_xnav_only_propagation.png` | XNAV-only error vs time |