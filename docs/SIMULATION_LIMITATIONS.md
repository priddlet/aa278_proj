# Simulation limitations (read before interpreting MC / figures)

Monte Carlo and envelope plots are useful for **relative** policy comparison in this codebase, not for claiming sub-kilometer absolute XNAV accuracy on orbit.

## Why XNAV-only can look “too good” (sub-km in blackout)

Several choices make the filter optimistic:

| Setting | Default | Effect |
|---------|---------|--------|
| **`use_truth_velocity_predict`** | `True` | Propagation uses **true velocity** each step. The state follows the reference orbit between updates, so pulsar updates only need to correct offset, not integrate lunar dynamics error. |
| **Synthetic XNAV** | truth position + noise | Measurements are `n̂·r_true + ν`, not a full timing / delta-TOA pipeline with clock states tied to barycentric TOA. |
| **LOS model** | linear `n̂·r` | Sheikh linearized scalar; no explicit range ambiguity, no separate timing arc. |
| **TOA σ = 1 µs** | ~300 m range σ per pulsar | With **5 MSPs every 120 s** and oracle propagation, sub-km means over the arc are possible in simulation. |

Quick check (26.4 hr ELFO, 5 MSPs, 1 µs, 5 trials):

- `truth_vel=True` → XNAV-only final mean **~2.5 km**, blackout segment **~0.5 km**
- `truth_vel=False` → XNAV-only final mean **~9 km**, blackout **~1.6 km**

Real far-side XNAV (SEXTANT-class) is not modeled to sub-km with 30–100 km initial errors; frame results as **continuity vs GNSS coast**, not NICER steady-state meters.

To stress-test more honestly:

```bash
python scripts/demo_monte_carlo.py --trials 20 --no-show
# (after setting use_truth_velocity_predict=False in MonteCarloConfig or adding a CLI flag)
```

## Pulsar count sweep — what to expect

- **More MSPs should help** when geometry is diverse (better observability).
- With **1 pulsar**, hybrid blackout error is often **very large** (weak geometry); **3–5** usually improve blackout and final mean.
- Sweeps now use the **same RNG seed** for all `n` so trial offsets are paired; older runs used `seed + n`, which confounded pulsar count with different random trials.

`select_pulsars(n)` takes the **first n** catalog entries, not an optimal subset — adding a pulsar can hurt if the extra LOS is nearly degenerate.

## TOA noise sweep — direction of trend

Sweep points: **0.1, 1, 10 µs** (not “larger is better”).

Typical pattern (switching policies, 26.4 hr):

| TOA σ | Hybrid final mean (km) | Comment |
|-------|------------------------|---------|
| 0.1 µs | ~0.15 | Unrealistically tight; filter over-trusts pulsar geometry |
| 1 µs | ~1.2 | Main campaign default |
| 10 µs | ~1.4–1.5 | Worse than 1 µs; XNAV-only much worse (~11 km) |

**Smaller TOA σ → lower error** in simulation. The plot uses a **log x-axis** (0.1, 1, 10): the left side is the smallest noise, not the largest.

Sweeps use a **common seed** across σ so each trial’s offset is the same when comparing noise levels.

## Policy semantics (current)

| Policy | Non-blackout | Blackout |
|--------|--------------|----------|
| `xnav_only` | Pulsars | Pulsars |
| `gnss_only` | GNSS sidelobe PRNs (0 PRN → pulsar fallback) | Pulsars + LunaNet if relay |
| `hybrid` | **Fuse** GNSS + pulsars + LunaNet if relay | Pulsars + **supplemental LunaNet** if relay |

LunaNet is **not** a fourth standalone phase. Plots label measured segments (`xnav + LunaNet supplemental (blackout)`), not geometric `NavMode.lonet`. Run `gnss_sidelobe_coverage_stats` to see how often non-blackout epochs have ≥4 trackable PRNs.

`docs/MONTE_CARLO_RESULTS.md` sections marked **legacy / 6 hr / pre-switching** may not match current figures until regenerated.

## Process noise (constant CWNA)

Monte Carlo uses fixed **`process_noise_accel`** (default **1e-4 m²/s³**) on the CV position/velocity block — no periapsis scaling. Filter CV may show **NIS/df ≫ 1**; check with `python scripts/check_nis.py --filter-predict`. Do not treat **p95** as calibrated when NIS/df is far from 1.

**`gnss_only`** under filter CV may still diverge in position (poor sidelobe geometry).

Lead with **`filter_predict/`** for dynamics stress; **`truth_velocity/`** for optimistic trends.

## Stale figures

Regenerate after policy or sweep fixes:

```bash
python scripts/build_presentation_assets.py
python scripts/demo_monte_carlo.py --sweep-pulsars --sweep-toa --trials 20 --no-show
```
