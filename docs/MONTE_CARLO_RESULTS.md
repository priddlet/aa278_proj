# Monte Carlo results

> **Important:** Sub-km XNAV/hybrid blackout numbers are **not** flight-realistic with default settings (`use_truth_velocity_predict=True`, synthetic LOS from truth). See **[SIMULATION_LIMITATIONS.md](./SIMULATION_LIMITATIONS.md)** before citing absolute errors. Pulsar/TOA sweeps must use a **fixed seed** across sweep points (fixed in code May 2026).

**Orbit:** ELFO truth (HW2 case 1, Earth-OP frame) · **Epoch:** 2026-01-15 12:00 UTC  
**Arc:** 26.4 hr (2× orbital period, T ≈ 13.2 hr at a = 6541.4 km)  
**GNSS:** broadcast ephemeris + corrected sidelobe gate (far-side clear LOS, limb annulus)  
**Policies (switching):** `xnav_only` (pulsars all arc); `gnss_only` (GNSS if visible, pulsars in blackout); `hybrid` (GNSS+LunaNet if visible, pulsars in blackout)  
**Noise:** TOA σ = **1 µs** (~300 m range), GNSS/LunaNet σ = 15 m, offset uniform 30–100 km

Run: `python scripts/demo_monte_carlo.py --trials 20 --no-show`

**Blackout fraction (26.4 hr):** ~64%  
**LunaNet 13.43 m:** LCRNS steady-state relay target — not the bar for XNAV/hybrid at km-scale init; success = **graceful degradation** on the far side.

---

## Current campaign — three policies (20 trials, 26.4 hr, 1 µs TOA)

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|-----------------|----------------|----------|-----------------|---------------------|
| **hybrid** | **1.15** | 1.29 | **2.39** | **0.45** | **0.56** |
| **xnav_only** | 2.42 | 2.64 | 2.48 | 0.53 | 0.97 |
| **gnss_only** | 65.8 | 153.0 | 52.0 | 52.7 | **16.7** |

**Takeaways**

- **Hybrid beats XNAV-only** on final mean, RMS, and blackout segment in **20/20 trials** (was corrupted by inverted sidelobe gate + 100 µs TOA floor).
- **GNSS split is correct:** non-blackout **16.7 km** vs blackout **52.7 km** (GNSS-only coasts with no measurements in blackout).
- **Fusion headline:** hybrid non-blackout **0.56 km** vs XNAV **0.97 km** vs GNSS **16.7 km**; hybrid RMS **2.4 km** beats both components.
- **XNAV periapsis gap** persists mildly (non-blackout 0.97 vs blackout 0.53 km) — CV predict vs e = 0.6 dynamics; two-body predict would flatten this.
- **GNSS-only** still km-scale: sparse 0–4 sidelobe PRNs, poor DOP, 30–100 km init — not a 15 m steady-state fix.
- Do **not** expect 13.43 m at these init errors; frame XNAV value as **far-side continuity** while GNSS-only blows up in blackout.

---

## Superseded — old campaign (6 hr, 100 µs TOA, pre-fix sidelobe)

> Kept for history. Used wrong arc length (<0.5 orbit), 30 km TOA range floor, and inverted occultation gate.

**Blackout on 6 hr (old):** ~94% (MOON_PA frame) → ~26% (OP frame, still undersampled)

| Policy | Final mean (km) | RMS (km) | Blackout μ | Non-blackout μ |
|--------|-----------------|----------|------------|----------------|
| hybrid | 11.71 | 9.95 | 7.70 | 7.13 |
| xnav_only | 10.13 | 14.56 | 11.69 | 26.12 |
| gnss_only | 119.8 | 94.97 | 94.28 | 10.70 |

Old table had **backwards GNSS split** (better in blackout) — signature of through-Earth satellites.

---

## Legacy sections (4 hr / sweep — pre-fix, stale numbers)

Same offset every trial (matches sensitivity to a single deployment case).

| Policy | Final mean ± σ (km) | Final p95 (km) | Mean error (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|---------------------|----------------|-----------------|-----------------|---------------------|
| **hybrid** | 10.69 ± 4.84 | 18.52 | **8.58** | **8.62** | 8.17 |
| **xnav_only** | 12.31 ± 4.45 | 20.13 | 17.76 | 16.37 | 32.40 |

Blackout ~91.4% on 4 hr arc.

---

## 3. Hybrid vs GNSS-only — blackout stress (20 trials, 4 hr)

| Policy | Final mean ± σ (km) | Final p95 (km) | Mean error (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|---------------------|----------------|-----------------|-----------------|---------------------|
| **hybrid** | 10.96 ± 5.05 | 18.42 | **8.87** | **8.82** | 9.39 |
| **gnss_only** | 168.95 ± 2.37 | 172.72 | 118.25 | 128.16 | 13.48 |

Hybrid mean blackout error is **~15× lower** than GNSS-only.

---

## 4. Pulsar count sweep (20 trials each, hybrid + XNAV-only)

| # MSPs | Policy | Final mean (km) | Final p95 (km) | Mean error (km) | Blackout μ (km) |
|--------|--------|-----------------|----------------|-----------------|-----------------|
| **1** | hybrid | 41.94 | 43.61 | 16.68 | 17.30 |
| **1** | xnav_only | 138.00 | 192.85 | 121.18 | 125.24 |
| **3** | hybrid | 14.03 | 22.89 | 8.85 | 8.92 |
| **3** | xnav_only | 14.61 | 23.13 | 18.70 | 17.23 |
| **5** | hybrid | **9.71** | 14.95 | **7.16** | **7.12** |
| **5** | xnav_only | 8.80 | 14.79 | 12.61 | 11.50 |

More pulsars strongly help hybrid in blackout; with 5 MSPs hybrid and XNAV-only are similar on finals, hybrid still wins on **mean arc error**.

---

## 5. TOA noise sweep (20 trials, 50 / 100 / 200 µs)

| TOA σ (µs) | Policy | Final mean (km) | Final p95 (km) | Mean error (km) | Blackout μ (km) |
|------------|--------|-----------------|----------------|-----------------|-----------------|
| **50** | hybrid | **5.83** | 9.50 | **5.02** | **4.87** |
| **50** | xnav_only | 5.43 | 7.92 | 8.20 | 7.50 |
| **50** | gnss_only | 120.09 | 123.82 | 89.43 | 94.52 |
| **100** | hybrid | 10.10 | 13.39 | 7.13 | 7.15 |
| **100** | xnav_only | 8.57 | 15.55 | 12.37 | 11.48 |
| **100** | gnss_only | 120.38 | 123.02 | 89.58 | 94.70 |
| **200** | hybrid | 16.95 | 33.35 | 10.66 | 10.90 |
| **200** | xnav_only | 17.96 | 35.16 | 20.02 | 18.62 |
| **200** | gnss_only | 120.70 | 124.61 | 89.81 | 94.92 |

GNSS-only is insensitive to pulsar TOA (no pulsars). Hybrid and XNAV degrade similarly as TOA noise increases.

---

## 6. Per-trial sample — main campaign (trial 0–4)

Each trial uses one random offset; all three policies share the same offset per trial.

| Trial | Offset (km) | Hybrid final | XNAV final | GNSS final | Hybrid mean | XNAV mean |
|-------|-------------|--------------|------------|------------|-------------|-----------|
| 0 | 74.59 | 20.65 | 9.21 | 121.72 | 8.48 | 12.46 |
| 1 | 86.93 | 7.91 | **4.16** | 120.88 | 9.67 | 12.63 |
| 2 | 68.05 | 18.79 | 9.30 | 119.01 | 8.02 | 12.90 |
| 3 | 90.02 | 9.22 | 11.95 | 119.11 | 7.27 | 13.10 |
| 4 | 90.42 | 20.00 | 12.20 | 119.64 | 8.17 | 14.14 |

Full per-trial CSV: 60 rows (20 trials × 3 policies) — regenerate with `run_monte_carlo()` and export from `result.trials`.

---

## 7. Pytest checks on Monte Carlo (logic, not full 20-trial runs)

| Test | What it validates |
|------|-------------------|
| `test_select_pulsars_subset` | Pulsar count 1 / 3 / 5 selection |
| `test_aggregate_policy_stats` | Mean, p95, LunaNet flag aggregation |
| `test_monte_carlo_runs_all_policies` | 2 trials × 3 policies completes; errors < 500 km |
| `test_hybrid_beats_xnav_with_broadcast_gps` | 2 trials, 4 hr: hybrid final mean ≤ 1.05 × XNAV |
| `test_hybrid_beats_gnss_only_in_blackout_heavy_arc` | 3 trials: hybrid blackout mean < GNSS-only |

---

## Configuration reference

| Parameter | Main campaign | Pulsar sweep | TOA sweep |
|-----------|---------------|--------------|-----------|
| Duration | 6 hr | 6 hr | 6 hr |
| Step | 120 s | 120 s | 120 s |
| Trials | 20 | 20 per n | 20 per σ |
| Seed | 0 | 10 + n | 20 + index |
| Offset | 30–100 km random | 30–100 km random | 30–100 km random |
| Pulsars | 5 (all) | 1, 3, 5 | 5 |
| TOA σ | 100 µs | 100 µs | 50, 100, 200 µs |
| GNSS σ | 15 m | 15 m | 15 m |
| Process noise | 1×10⁻⁴ m²/s³ | same | same |
| Predict | truth velocity | same | same |
