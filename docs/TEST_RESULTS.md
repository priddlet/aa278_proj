# Unit test results (`pytest`)

**Suite status:** 25 / 25 passed (≈2.9 s)

For **Monte Carlo navigation campaigns** (policy comparison, pulsar/TOA sweeps, per-trial stats), see **[MONTE_CARLO_RESULTS.md](./MONTE_CARLO_RESULTS.md)**.

---

## Unit tests (`pytest`)

| # | Test | File | Status | Dependencies | What it checks | Key setup | Pass criterion |
|---|------|------|--------|--------------|----------------|-----------|----------------|
| 1 | `test_bundled_count` | `test_catalog.py` | PASS | none | SEXTANT MSP catalog load | bundled JSON | exactly **5** pulsars |
| 2 | `test_load_by_name` | `test_catalog.py` | PASS | none | catalog lookup by name | `J0437-4715` | `f0_hz > 170` |
| 3 | `test_get_gps_posclk_mci` | `test_gps_broadcast.py` | PASS | SPICE + `brdc_data.npz` + Earth PCK | HW2 broadcast GPS → MCI | PRN 5, fixed ET | ‖r‖ > 100,000 km; \|clock\| < 1000 km |
| 4 | `test_visible_gps_prns_on_lunar_orbit` | `test_gps_broadcast.py` | PASS | SPICE + brdc | visible GPS PRNs from ELFO | 1 min ELFO arc | ≥1 PRN; range to SC > 1000 km |
| 5 | `test_hybrid_uses_gnss_during_near_side` | `test_hybrid.py` | PASS | SPICE + brdc + GPS frames | hybrid applies GNSS + always XNAV | 6 hr ELFO, 30 km offset | `n_gnss > 0`; pulsar meas = epochs × 5 |
| 6 | `test_hybrid_beats_xnav_only_on_elfo` | `test_hybrid.py` | PASS | SPICE + brdc + GPS frames | hybrid filter sanity on ELFO | 6 hr, 50 km offset | final error < 50 km; GNSS epochs < 5 km |
| 7 | `test_select_pulsars_subset` | `test_monte_carlo.py` | PASS | none | pulsar count selector | n = 1, 3 | lengths 1 and 3 |
| 8 | `test_aggregate_policy_stats` | `test_monte_carlo.py` | PASS | none | MC statistics aggregation | 2 synthetic trials | mean final = 15 m; LunaNet p95 flag false |
| 9 | `test_monte_carlo_runs_all_policies` | `test_monte_carlo.py` | PASS | SPICE + brdc + GPS frames | full MC campaign structure | 2 trials × 3 policies, 30 min | 6 trials; final mean < 500 km |
| 10 | `test_hybrid_beats_xnav_with_broadcast_gps` | `test_monte_carlo.py` | PASS | SPICE + brdc + GPS frames | hybrid mean final ≤ XNAV-only | 4 hr, 2 trials, 50 km offset | hybrid final mean ≤ 1.05 × XNAV |
| 11 | `test_hybrid_beats_gnss_only_in_blackout_heavy_arc` | `test_monte_carlo.py` | PASS | SPICE + brdc + GPS frames | hybrid vs GNSS-only in blackout | 4 hr, 3 trials, random offset | hybrid blackout mean < GNSS-only |
| 12 | `test_coe_to_cart_circular` | `test_propagation.py` | PASS | SPICE | COE → Cartesian | a = 2000 km, e = 0 | ‖r‖ ≈ a; ‖v‖ ≈ a·n |
| 13 | `test_elfo_propagation_radii` | `test_propagation.py` | PASS | SPICE | ELFO truth propagation | 1 hr, 300 s step | radius 1500–12000 km; ICRS shape OK |
| 14 | `test_acceleration_finite` | `test_propagation.py` | PASS | SPICE | lunar dynamics acceleration | r = 7000 km at epoch | finite, nonzero **a** |
| 15 | `test_pseudorange_zero_innovation_at_truth` | `test_pseudorange.py` | PASS | SPICE | ρ model at truth state | fixed r, r_tx, ET | \|innovation\| < 1 m |
| 16 | `test_pseudorange_jacobian_position` | `test_pseudorange.py` | PASS | SPICE | analytic vs numeric ∂ρ/∂r | finite difference, ε = 1 m | H_pos matches numeric |
| 17 | `test_gps_constellation_offsets` | `test_pseudorange.py` | PASS | none | legacy analytic GPS shell (deprecated path) | 6 sats, Earth at 380 Mm | shape (6,3); range > 20 Mm |
| 18 | `test_earth_elevation_sign` | `test_visibility.py` | PASS | none | Earth elevation geometry | toy MCI positions | below horizon < 0; above > 5° |
| 19 | `test_elfo_has_blackout_windows` | `test_visibility.py` | PASS | SPICE | far-side blackout on 30 hr ELFO | 300 s step | blackout fraction > 15%; ≥1 window; > 1 hr total |
| 20 | `test_blackout_implies_xnav_mode` | `test_visibility.py` | PASS | SPICE | mode labels in blackout | 6 hr ELFO | in blackout → XNAV or LONET; no GNSS |
| 21 | `test_los_projection` | `test_xnav_geometry.py` | PASS | none | LOS range = n̂·r | B1937+21 | error < 1 µm |
| 22 | `test_jacobian_is_los` | `test_xnav_geometry.py` | PASS | none | XNAV H row = unit LOS | J0437-4715 | H[0:3] = n̂; rest zero |
| 23 | `test_single_pulsar_ekf_reduces_los_error` | `test_xnav_geometry.py` | PASS | none | 1-MSP EKF constrains LOS | 80 km ⊥ LOS, 30 steps | LOS error ↓ 10× or < 500 m |
| 24 | `test_multi_pulsar_full_rank` | `test_xnav_geometry.py` | PASS | none | 5-MSP geometry full rank | SEXTANT set | rank(A) = 3 |
| 25 | `test_multi_pulsar_beats_single_on_propagated_truth` | `test_xnav_truth.py` | PASS | SPICE | multi vs single on ELFO truth | 1 hr, 30 km offset | multi final < single; multi converges |

---

## Dependencies by test class

| Dependency | Tests using it |
|------------|----------------|
| None (pure Python) | 10 tests (catalog, geometry, MC stats, elevation toy, legacy GPS shell) |
| SPICE kernels (`de440`, Moon PCK/FK, LSK) | 15 tests |
| `brdc_data.npz` + Earth `earth_*` kernels | 7 tests (GPS broadcast + hybrid + MC) |

---

## How to reproduce

```bash
cd /Users/tanispriddle/Downloads/278proj
source .venv/bin/activate
pytest -v
python scripts/demo_hybrid_elo.py --preset elfo --duration 6 --no-show
python scripts/demo_monte_carlo.py --trials 5 --no-show
```
