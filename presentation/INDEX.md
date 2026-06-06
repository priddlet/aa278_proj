# Presentation assets

**Primary campaign:** `filter_dynamics` (EKF RK45+STM + HW2 process noise)

Regenerate: `python scripts/build_presentation_assets.py --pipelines filter_dynamics`

## Model assumptions (filter dynamics - primary campaign)

| Item | Value |
|------|--------|
| **Truth orbit** | HW2 ELFO case 1: a = 6541.4 km, e = 0.6, **T approx 13.2 h** (not LCRNS 30 h) |
| **Truth dynamics** | Moon + Earth + Sun indirect (HW2 P2), SPICE DE440 |
| **EKF predict** | MCI RK45 + analytic STM; HW2 CWNA Q, `dynamics_sigma_acc_km = 1e-6` km/s^2/sqrt(s) |
| **Measurements** | Synthetic from **truth position** + noise (optimistic sensing) |
| **XNAV** | All 5 SEXTANT MSPs, sigma_TOA = **1 us** (~300 m range sigma), linear `n_hat dot r` |
| **GNSS / LunaNet** | Sidelobe PRNs, sigma = 15 m; LunaNet supplemental in blackout when relay visible |
| **MC arc** | **26.4 h** (2x orbital period), step **120 s**, 20 trials |
| **Blackout** | Earth < 5 deg elev.; **~65.7% one orbit**, **~64.1%** on MC arc |
| **Position metrics** | Report **Steady RMS** (last 10% of arc); full-arc RMS includes epoch-0 spike |
| **Timing metrics** | |b_rx - b_truth| (m) on GNSS/LunaNet pseudorange epochs only - not pulsar TOA |

---

## Geometry & truth (ELFO)

Orbit: HW2 ELFO (a=6541 km, e=0.6, Tapprox13.2 h; rpapprox2617 km, raapprox10466 km)

Use **one-orbit blackout (~65.7%)** for per-revolution geometry; **MC arc (26.4 h, ~64.1%)** for navigation statistics.

### Figures - `figures/presentation/common/`

| File | Description |
|------|-------------|
| `figures/presentation/common/elfo_blackout_fraction_specs.png` | Blackout % - one orbit (T approx 13.2 h) vs 30 h sim vs MC arc |
| `figures/presentation/common/elfo_orbit_blackout_3d_one_orbit.png` | Orbit colored by blackout - one revolution |
| `figures/presentation/common/elfo_orbit_blackout_3d.png` | Orbit colored by blackout - 30 h visibility arc |
| `figures/presentation/common/elfo_orbit_blackout_xy.png` | Blackout map (Moon xy) |
| `figures/presentation/common/elfo_truth_propagation.png` | Truth trajectory (30 h arc) |
| `figures/presentation/common/elfo_visibility_one_orbit.png` | Visibility timeline - one orbit |
| `figures/presentation/common/elfo_visibility_timeline.png` | Visibility timeline - 30 h (approx 2.3 rev) |
| `figures/presentation/common/elfo_orbit_{policy}.png` | Orbit colored by planned policy phase |
| `figures/presentation/common/elfo_segments_{policy}.png` | Policy segment timeline |
| `figures/presentation/common/predict_mode_comparison_{policy}.png` | Predict-mode bar charts (hybrid / gnss_only / xnav_only) |

### Tables - `presentation/tables/common/`

| File | Description |
|------|-------------|
| `presentation/tables/common/blackout_specs.md` | One-orbit vs MC blackout numbers |
| `presentation/tables/common/pulsar_catalog.md` | SEXTANT MSP catalog (Hz, roles) |
| `presentation/tables/common/predict_mode_comparison.md` | Cross-mode comparison |
| `presentation/tables/common/predict_mode_comparison.csv` | Same data (CSV) |

Per-mode detailed tables exist only under **`filter_dynamics/`** (primary campaign).

---

## Filter dynamics - results

### Tables - `presentation/tables/filter_dynamics/`

| File | Description |
|------|-------------|
| `presentation/tables/filter_dynamics/main_summary.md` | Quantitative results - position + timing |
| `presentation/tables/filter_dynamics/main_policy_summary.csv` | Aggregated policy stats (CSV) |
| `presentation/tables/filter_dynamics/main_trials.csv` | Per-trial rows |
| `presentation/tables/filter_dynamics/timing_conclusion.md` | Timing conclusions (clock on PR epochs) |
| `presentation/tables/filter_dynamics/pulsar_sweep.md` | Pulsar count sweep table |
| `presentation/tables/filter_dynamics/toa_sweep.md` | TOA sigma sweep table |
| `presentation/tables/filter_dynamics/q_sweep.md` | Process-noise / sigma_acc sweep (Q diagnostics) |

### Figures - `figures/presentation/filter_dynamics/`

| File | Description |
|------|-------------|
| `figures/presentation/filter_dynamics/mc_elfo_all_policies_propagation.png` | Position error propagation - all policies |
| `figures/presentation/filter_dynamics/mc_elfo_{policy}_propagation.png` | Position error propagation per policy |
| `figures/presentation/filter_dynamics/mc_elfo_{policy}_envelope.png` | MC error envelope (mean +/- band) |
| `figures/presentation/filter_dynamics/mc_elfo_all_policies_envelope.png` | MC envelope - all policies |
| `figures/presentation/filter_dynamics/mc_elfo_hybrid_clock_timing.png` | Clock timing error trace (hybrid) |
| `figures/presentation/filter_dynamics/mc_elfo_gnss_only_clock_timing.png` | Clock timing trace (gnss_only) |
| `figures/presentation/filter_dynamics/mc_elfo_policy_bars.png` | Bar chart - final / blackout / steady metrics |
| `figures/presentation/filter_dynamics/mc_elfo_boxplot.png` | Boxplot - final position error by policy |
| `figures/presentation/filter_dynamics/mc_elfo_final_cdf.png` | CDF - final position error |
| `figures/presentation/filter_dynamics/mc_elfo_pulsar_sweep.png` | Pulsar count sweep figure |
| `figures/presentation/filter_dynamics/mc_elfo_toa_sweep.png` | TOA noise sweep figure |
| `figures/presentation/filter_dynamics/mc_elfo_q_diagnostics.png` | Q diagnostics - median NIS/df vs sigma_acc |

### Spreadsheet

`results/monte_carlo_filter_dynamics.xlsx`


---

## Predict-mode comparison

Other predict modes (`truth_velocity`, `filter_predict`) run in MC for comparison only - no separate figure/table trees.

- `presentation/tables/common/predict_mode_comparison.md`
- `figures/presentation/common/predict_mode_comparison_*.png`


---

## Docs

- [docs/PRESENTATION.md](../docs/PRESENTATION.md) - slide order & talking points
- [docs/SIMULATION_LIMITATIONS.md](../docs/SIMULATION_LIMITATIONS.md) - what not to over-claim
