# Presentation guide

Slide-ready figures for **Pulsar Hybrid Navigation for the Lunar Far Side**.

## Two navigation pipelines + shared geometry

| Set | Path | What it shows |
|-----|------|----------------|
| **Common** | `figures/presentation/common/` | Blackout geometry, policy segments, truth propagation — **no EKF** |
| **Truth velocity** | `figures/presentation/truth_velocity/` | MC + errors with **oracle** motion between updates |
| **Filter predict** | `figures/presentation/filter_predict/` | MC + errors with **EKF CV** predict only (more realistic) |

Tables / Excel per navigation pipeline:

- `results/presentation_truth_velocity.md` · `results/monte_carlo_truth_velocity.xlsx`
- `results/presentation_filter_predict.md` · `results/monte_carlo_filter_predict.xlsx`
- `results/presentation_INDEX.md` — master index

See [SIMULATION_LIMITATIONS.md](./SIMULATION_LIMITATIONS.md) before citing **absolute** km errors from the truth-velocity set.

## Build commands

```bash
pip install -e ".[dev,spice,viz,export]"

# Everything (common + both nav pipelines; ~15–25 min full MC)
python scripts/build_presentation_assets.py

# Quick smoke test (5 trials, no envelopes/sweeps)
python scripts/build_presentation_assets.py --quick

# Only shared geometry (fast)
python scripts/build_presentation_assets.py --pipelines common

# Only filter-dynamics MC (for honest error magnitudes)
python scripts/build_presentation_assets.py --pipelines filter_predict

# Only truth-velocity MC (policy comparison, optimistic absolutes)
python scripts/build_presentation_assets.py --pipelines truth_velocity
```

CLI Monte Carlo (either predict mode):

```bash
python scripts/demo_monte_carlo.py --trials 20 --no-show
python scripts/demo_monte_carlo.py --trials 20 --no-truth-velocity --no-show
```

## Recommended slide order

1. **Problem** — `common/elfo_orbit_blackout_3d.png` or `elfo_orbit_blackout_xy.png`
2. **Three scenarios** — `common/elfo_orbit_{xnav_only,gnss_only,hybrid}.png` + `elfo_segments_*.png`
3. **Truth orbit** — `common/elfo_truth_propagation.png`
4. **Policy comparison (pick one pipeline)**  
   - Optimistic / clean: `truth_velocity/mc_elfo_all_policies_propagation.png`  
   - Realistic dynamics: `filter_predict/mc_elfo_all_policies_propagation.png`
5. **Statistics** — `mc_elfo_boxplot.png`, `mc_elfo_policy_bars.png` (same subfolder)
6. **Sensitivity** — `mc_elfo_pulsar_sweep.png`, `mc_elfo_toa_sweep.png` (full build)

## Policy semantics (switching)

| Policy | GNSS visible | Blackout |
|--------|--------------|----------|
| `xnav_only` | Pulsars | Pulsars |
| `gnss_only` | GNSS only | Pulsars |
| `hybrid` | GNSS + LunaNet (if relay) | Pulsars |

## Talking points

- Use **common/** for geometry and **why** hybrid exists.
- Use **truth_velocity/** for “hybrid vs XNAV vs GNSS” **trends** at 1 µs TOA.
- Use **filter_predict/** when discussing **achievable** error scales without oracle propagation.
- Do not mix plots from the two nav folders on one slide without labeling the predict mode.
