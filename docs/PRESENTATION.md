# Presentation guide

Slide-ready assets for **Pulsar Hybrid Navigation for the Lunar Far Side**.

**Single index (figures + tables):** [presentation/INDEX.md](../presentation/INDEX.md)

**Primary campaign:** `filter_dynamics` - EKF RK45+STM predict with HW2 process noise.

See [SIMULATION_LIMITATIONS.md](./SIMULATION_LIMITATIONS.md) before citing absolute km errors.

## Build commands

```bash
pip install -e ".[dev,spice,viz,export]"

# Full campaign: common geometry + all three predict pipelines (~1-2 hr MC)
python scripts/build_presentation_assets.py --pipelines nav

# Filter dynamics only (recommended for slides)
python scripts/build_presentation_assets.py --pipelines filter_dynamics

# Geometry + tables only (fast)
python scripts/build_presentation_assets.py --pipelines common

# Refresh INDEX / timing conclusion from existing CSVs (no MC)
python scripts/refresh_presentation_manifest.py
```

## Orbit identity (do not confuse with LCRNS)

Truth preset **`elfo`** is **AA278 HW2 case 1**: **a = 6541.4 km**, **e = 0.6**, **T approx 13.2 h** (two-body at lunar GM). Not the LCRNS 30 h ELFO.

| Arc | Duration | Blackout % | Use on slides |
|-----|----------|------------|---------------|
| **One orbit** | T approx 13.2 h | **~65.7%** | Per-revolution geometry, blackout specs bar |
| **MC campaign** | 26.4 h (2xT) | **~64.1%** | Navigation statistics, policy comparison |
| **30 h visibility** | approx 2.3 rev | ~56% | Timeline figure only (non-integer revolutions) |

Table: `presentation/tables/common/blackout_specs.md` | Figure: `figures/presentation/common/elfo_blackout_fraction_specs.png`

## Recommended slide order (filter dynamics)

1. **Problem / geometry** - `common/elfo_blackout_fraction_specs.png` or `elfo_orbit_blackout_3d_one_orbit.png`
2. **Truth orbit** - `common/elfo_truth_propagation.png`
3. **Three policies** - `common/elfo_orbit_{xnav_only,gnss_only,hybrid}.png`
4. **Quantitative results** - `presentation/tables/filter_dynamics/main_summary.md`
5. **Error propagation** - `filter_dynamics/mc_elfo_all_policies_propagation.png`
6. **Bar / box stats** - `mc_elfo_policy_bars.png`, `mc_elfo_boxplot.png`
7. **Timing** - `mc_elfo_hybrid_clock_timing.png` + `filter_dynamics/timing_conclusion.md`
8. **Sensitivity** - `mc_elfo_pulsar_sweep.png`, `mc_elfo_toa_sweep.png`
9. **Q tuning** - `mc_elfo_q_diagnostics.png` + `filter_dynamics/q_sweep.md`
10. **Optional baselines** - `common/predict_mode_comparison_hybrid.png`

## Policy semantics

| Policy | Non-blackout | Blackout |
|--------|--------------|----------|
| `xnav_only` | Pulsars | Pulsars (+ LunaNet if relay) |
| `gnss_only` | GNSS sidelobe | Pulsars + LunaNet if relay |
| `hybrid` | **Fuse** GNSS + pulsars (+ LunaNet) | Pulsars + supplemental LunaNet |

## Metrics to report

| Metric | Meaning |
|--------|---------|
| **Steady RMS** | Last 10% of arc - fair position compare (excludes epoch-0 spike) |
| **Full-arc RMS** | Dominated by initial offset at t=0 (no EKF update) |
| **\|b\| mean / p95** | Clock timing error on **GNSS/LunaNet pseudorange** epochs only |
| **sigma_TOA = 1 us** | ~300 m range sigma per pulsar (not clock timing sigma) |

Pulsar catalog (Hz, nav vs timing roles): `presentation/tables/common/pulsar_catalog.md`

## Talking points

- Lead with **filter_dynamics** for propagation honesty; use **`common/predict_mode_comparison.md`** for cross-mode trends.
- Cite **one-orbit blackout (~66%)** for geometry; **64%** for MC arc stats.
- Position can look similar across policies; **hybrid** wins on **clock** (|b| ~10 m vs gnss_only ~120 m on PR epochs).
- Sub-100 m steady position under dynamics predict is still **optimistic** (truth-synthesized measurements).
