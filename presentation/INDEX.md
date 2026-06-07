# Presentation assets

**Primary campaign:** `filter_dynamics` (EKF RK45+STM + process noise)

Regenerate: `python scripts/build_presentation_assets.py --pipelines filter_dynamics`

## Model assumptions (filter dynamics - primary campaign)

| Item | Value |
|------|--------|
| **Truth orbit** | ELFO case 1: a = 6541.4 km, e = 0.6, **T approx 13.2 h** (not LCRNS 30 h) |
| **Truth dynamics** | Moon + Earth + Sun indirect, SPICE DE440 |
| **EKF predict** | MCI RK45 + analytic STM; CWNA Q, `dynamics_sigma_acc_km = 1e-6` km/s^2/sqrt(s) |
| **Measurements** | Synthetic from **truth position** + noise (optimistic sensing) |
| **XNAV** | All 5 SEXTANT MSPs, sigma_TOA = **1 us** (~300 m range sigma), linear `n_hat dot r` |
| **GNSS / LunaNet** | Sidelobe PRNs, sigma = 15 m; LunaNet supplemental in blackout when relay visible |
| **MC arc** | **26.4 h** (2x orbital period), step **120 s**, 20 trials |
| **Blackout** | Earth < 5 deg elev.; **~65.7% one orbit**, **~64.1%** on MC arc |
| **Position metrics** | Report **Steady RMS** (last 10% of arc); full-arc RMS includes epoch-0 spike |
| **Timing metrics** | |b_rx - b_truth| (m) on GNSS/LunaNet pseudorange epochs only - not pulsar TOA |

---

## Geometry & truth (ELFO)

Orbit: ELFO (a=6541 km, e=0.6, Tapprox13.2 h; rpapprox2617 km, raapprox10466 km)

Use **one-orbit blackout (~65.7%)** for per-revolution geometry; **MC arc (26.4 h, ~64.1%)** for navigation statistics.

### Figures - `figures/presentation/common/`

| File | Description |
|------|-------------|
| `figures/presentation/common/blackout_specs.png` | Blackout fraction by arc |
| `figures/presentation/common/orbit_blackout_3d_orbit.png` | Orbit by blackout (one orbit) |
| `figures/presentation/common/orbit_blackout_3d.png` | Orbit by blackout |
| `figures/presentation/common/orbit_blackout_xy.png` | Blackout map (xy) |
| `figures/presentation/common/truth_orbit.png` | Truth orbit |
| `figures/presentation/common/visibility_orbit.png` | Visibility (one orbit) |
| `figures/presentation/common/visibility_timeline.png` | Visibility timeline |
| `figures/presentation/common/orbit_{policy}.png` | Orbit by policy |
| `figures/presentation/common/segments_{policy}.png` | Policy segments |
| `figures/presentation/common/predict_{policy}.png` | Predict mode comparison |

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
| `figures/presentation/filter_dynamics/errors_all.png` | Position error (all policies) |
| `figures/presentation/filter_dynamics/errors_{policy}.png` | Position error |
| `figures/presentation/filter_dynamics/envelope_{policy}.png` | Error envelope |
| `figures/presentation/filter_dynamics/envelope_all.png` | Envelopes (all policies) |
| `figures/presentation/filter_dynamics/clock_{policy}.png` | Clock error |
| `figures/presentation/filter_dynamics/policy_bars.png` | Mean error by segment |
| `figures/presentation/filter_dynamics/final_boxplot.png` | Final error boxplot |
| `figures/presentation/filter_dynamics/final_cdf.png` | Final error CDF |
| `figures/presentation/filter_dynamics/pulsar_sweep.png` | Pulsar count sweep |
| `figures/presentation/filter_dynamics/toa_sweep.png` | TOA noise sweep |
| `figures/presentation/filter_dynamics/q_diagnostics.png` | Process noise diagnostics |

### Spreadsheet

`results/monte_carlo_filter_dynamics.xlsx`


---

## Predict-mode comparison

Other predict modes (`truth_velocity`, `filter_predict`) run in MC for comparison only - no separate figure/table trees.

- `presentation/tables/common/predict_mode_comparison.md`
- `figures/presentation/common/predict_*.png`


---

## Docs

- [docs/PRESENTATION.md](../docs/PRESENTATION.md) - slide order & talking points
- [docs/SIMULATION_LIMITATIONS.md](../docs/SIMULATION_LIMITATIONS.md) - what not to over-claim
