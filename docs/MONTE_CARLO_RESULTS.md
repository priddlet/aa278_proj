# Monte Carlo results

> **Note (2026-05):** Truth-orbit initial state now uses the HW2 Earth-OP frame (`mci_to_op_rotation`), not MOON_PA. Blackout fractions and navigation errors below were generated with the old frame and should be re-run: `python scripts/demo_monte_carlo.py --trials 20 --sweep-pulsars --sweep-toa --no-show`

**Orbit:** ELFO truth (DE440 + lunar dynamics, HW2 case 1 COE in Earth-OP frame)  
**Epoch:** 2026-01-15 12:00 UTC  
**GNSS:** HW2 broadcast ephemeris (`brdc_data.npz`)  
**Hybrid policy:** XNAV every epoch + GNSS (non-blackout) + LunaNet when visible  
**LunaNet reference (pitch):** 13.43 m final position (p95 check)

**Blackout fraction on 6 hr ELFO (correct OP frame):** ~26% (was ~94% with MOON_PA bug)  
**Default noise:** TOA σ = 100 µs, GNSS/LunaNet σ = 15 m, initial offset uniform 30–100 km

Run: `python scripts/demo_monte_carlo.py --trials 20 --sweep-pulsars --sweep-toa --no-show`

---

## 1. Main campaign — three policies (20 trials)

Compares **hybrid**, **XNAV-only**, and **GNSS-only** on the same truth arc and random offsets.

| Policy | Trials | Init offset | Final mean ± σ (km) | Final p95 (km) | Mean error (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) | Meets 13.43 m p95? |
|--------|--------|-------------|---------------------|----------------|-----------------|----------|-----------------|---------------------|--------------------|
| **hybrid** | 20 | 30–100 km | 11.71 ± 5.01 | 20.03 | **7.67** | 9.95 | **7.70** | 7.13 | No |
| **xnav_only** | 20 | 30–100 km | 10.13 ± 3.04 | 14.61 | 12.57 | 14.56 | 11.69 | 26.12 | No |
| **gnss_only** | 20 | 30–100 km | 119.8 ± 1.49 | 122.13 | 89.2 | 94.97 | 94.28 | **10.70** | No |

**Takeaways**

- **Hybrid has the lowest mean error over the arc** (7.7 km vs 12.6 km XNAV-only) because GNSS helps near-side while pulsars run always.
- **XNAV-only can beat hybrid on final-error mean/p95** in this draw (10.1 vs 11.7 km final mean) — finals are dominated by long blackout; near-side GNSS is not always a better final constraint.
- **GNSS-only fails in blackout** (~94% of arc): mean ~89 km, final ~120 km (coast with no measurements).
- **Blackout segment:** hybrid **7.7 km** vs XNAV **11.7 km** vs GNSS **94.3 km**.
- None meet the **13.43 m** LunaNet pitch target at km-scale initial errors.

---

## 2. Hybrid vs XNAV-only — fixed 50 km offset (20 trials, 4 hr)

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
