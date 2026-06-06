# Presentation tables

**Unified asset index (figures + tables):** [../INDEX.md](../INDEX.md)

| Predict mode | Directory | EKF time update |
|--------------|-----------|-----------------|
| **Filter dynamics predict** | `filter_dynamics/` | Filter dynamics (RK45 + STM, HW2 Q) |

## Primary campaign (`filter_dynamics/`)

- `main_summary.md` - policy comparison (km)
- `main_policy_summary.csv` - aggregated stats
- `main_trials.csv` - per-trial rows
- `pulsar_sweep.md` / `.csv` - MSP count sensitivity
- `toa_sweep.md` / `.csv` - TOA sigma sensitivity
- `q_sweep.md` / `.csv` - process-noise / sigma_acc sweep
- `timing_conclusion.md` - clock timing on PR epochs

## Shared tables (`common/`)

- `blackout_specs.md` - one-orbit vs MC arc blackout %
- `pulsar_catalog.md` / `.csv` - SEXTANT MSP set
- `predict_mode_comparison.md` / `.csv` - truth-velocity vs CV vs dynamics (all policies)

Other predict modes run in MC for comparison only; they do not have separate table folders.
