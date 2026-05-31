# Unit test results (`pytest`)

**Suite status:** 30 / 30 passed

For **Monte Carlo navigation campaigns** (policy comparison, pulsar/TOA sweeps, per-trial stats), see **[MONTE_CARLO_RESULTS.md](./MONTE_CARLO_RESULTS.md)**. Monte Carlo tables there are stale until re-run after the HW2 Earth-OP frame fix.

---

## Unit tests (`pytest`)

New regression tests for the Earth-OP frame fix:

| Test | File | What it checks |
|------|------|----------------|
| `test_mci_to_op_rotation_matches_hw_p23` | `test_propagation.py` | OP rotation matches HW2 P2.3 autograder |
| `test_initial_state_mci_from_coe_op_roundtrip` | `test_propagation.py` | `R.T @ x_op` ↔ MCI roundtrip |
| `test_elfo_presets_differ_in_visibility` | `test_monte_carlo.py` | `elfo` vs `elfo_nav` differ in GNSS geometry |

Run `pytest -v` for the full list (30 tests).

---

## Dependencies by test class

| Dependency | Tests using it |
|------------|----------------|
| None (pure Python) | 10 tests (catalog, geometry, MC stats, elevation toy, legacy GPS shell) |
| SPICE kernels (`de440`, Moon PCK/FK, LSK) | 17 tests |
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
