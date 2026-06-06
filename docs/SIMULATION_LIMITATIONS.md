# Simulation limitations (read before interpreting MC / figures)

Monte Carlo and envelope plots are useful for **relative** policy comparison in this codebase, not for claiming sub-kilometer absolute XNAV accuracy on orbit.

**Primary presentation campaign:** `filter_dynamics` (RK45+STM EKF predict, HW2 sigma_acc Q). Asset index: [presentation/INDEX.md](../presentation/INDEX.md).

## Primary campaign assumptions (`filter_dynamics`)

| Item | Default |
|------|---------|
| **Truth orbit** | HW2 ELFO case 1, T approx **13.2 h**, MC arc **26.4 h** (2x period) |
| **Blackout** | Earth < 5 deg mask; **~65.7% one orbit**, **~64.1%** on MC arc |
| **EKF predict** | MCI RK45 + analytic STM; `dynamics_sigma_acc_km = **1e-6**` km/s^2/sqrt(s) |
| **Process noise** | HW2 CWNA on [r,v] + RAFS clock PSDs (`hw2_process_noise.py`) |
| **Measurements** | Synthetic from **truth position** + noise (optimistic) |
| **XNAV** | 5 SEXTANT MSPs, sigma_TOA = **1 us** (~300 m range sigma), linear `n_hat dot r` |
| **GNSS / LunaNet** | sigma = 15 m; sidelobe PRN gate uses truth SC position |
| **Position metrics** | Report **Steady RMS** (last 10%); full-arc RMS includes epoch-0 spike |
| **Timing metrics** | **\|b_rx - b_truth\|** on GNSS/LunaNet PR epochs - not pulsar TOA |

Under dynamics predict with truth-synthesized measurements, **steady position** can reach ~0.06 km while still **not** representing flight ODTS. **Hybrid |b| ~10 m** on PR epochs vs **gnss_only ~120 m** is a meaningful policy split even when position means look similar.

## Force model (truth vs filter)

| Layer | Default | With `include_disturbances=True` / `--disturbed-dynamics` |
|-------|---------|-------------------------------------------------------------|
| **Truth arc** | HW2 P2: mu\_m + Earth + Sun (`solve_ivp` RK45) | + Moon **J2** + **SRP** (HW2 P3 gamma = C\_R | A/m) |
| **Filter CV / truth_velocity** | Same truth arc; predict uses CV or true **v** | Same disturbed truth; predict unchanged |
| **Filter dynamics** | RK45+STM on estimate; same `DynamicsConfig` as truth when disturbances on | Same |

Disturbances are **not** on by default (preserves existing presentation tables). Regenerate all pipelines with the same flag so truth and `filter_dynamics` stay matched.

## Why XNAV-only can look “too good” (sub-km in blackout)

Several choices make the filter optimistic:

| Setting | Default | Effect |
|---------|---------|--------|
| **`use_truth_velocity_predict`** | `True` | Propagation uses **true velocity** each step. The state follows the reference orbit between updates, so pulsar updates only need to correct offset, not integrate lunar dynamics error. |
| **Synthetic XNAV** | truth position + noise | Measurements are `n_hat dot r_true + nu`, not a full timing / delta-TOA pipeline with clock states tied to barycentric TOA. |
| **Clock timing metrics** | `|b_rx - 0|` on PR epochs | Reported for `hybrid` / `gnss_only` only. XNAV-only and MSP-only blackout epochs do not observe `b` in **H** (LOS rows). |
| **LOS model** | linear `n_hat dot r` | Sheikh linearized scalar; no explicit range ambiguity, no separate timing arc. |
| **TOA sigma = 1 us** | ~300 m range sigma per pulsar | With **5 MSPs every 120 s** and oracle propagation, sub-km means over the arc are possible in simulation. |

Quick check (26.4 hr ELFO, 5 MSPs, 1 us, 5 trials):

- `truth_vel=True` -> XNAV-only final mean **~2.5 km**, blackout segment **~0.5 km**
- `truth_vel=False` -> XNAV-only final mean **~9 km**, blackout **~1.6 km**

Real far-side XNAV (SEXTANT-class) is not modeled to sub-km with 30-100 km initial errors; frame results as **continuity vs GNSS coast**, not NICER steady-state meters.

**TOA / pulsar sweeps:** Sweep helpers now copy `predict_mode` from `base_config` via `dataclasses.replace`. Regenerate with `python scripts/build_presentation_assets.py --pipelines filter_dynamics`.

**Audit checklist (not bugs, but easy to misread):**

| Item | Status |
|------|--------|
| Measurements synthesized from **truth position** | By design (`hybrid_run`); optimistic |
| **Epoch 0** no EKF update -> dominates full-arc RMS | Use **Steady mu / Steady RMS** (last 10%) in tables |
| `build_presentation_assets.py` | Full MC + assets for `filter_dynamics`; comparison MC for other predict modes |
| `meets_lunanet_p95` uses **final p95**, not steady-state | Pitch target is relay timing, not XNAV final |
| `select_pulsars(n)` = first **n** catalog entries | Not geometry-optimized subset |
| GNSS visibility / PRN gate uses **truth** SC position | Same as measurements |
| Regenerate sweeps/plots after May 2026 config fixes | Required |

**Arc metrics:** Full-arc **RMS** is dominated by **epoch 0** (no EKF update; error = initial offset). Monte Carlo tables also report **Steady mu** and **Steady RMS** over the **last 10%** of epochs - a fairer compare across predict modes. Regenerate with `python scripts/build_presentation_assets.py`.

To stress-test CV predict without oracle velocity, set `use_truth_velocity_predict=False` in `MonteCarloConfig` or use `build_presentation_assets.py --pipelines compare`.

## Pulsar count sweep - what to expect

- **More MSPs should help** when geometry is diverse (better observability).
- With **1 pulsar**, hybrid blackout error is often **very large** (weak geometry); **3-5** usually improve blackout and final mean.
- Sweeps now use the **same RNG seed** for all `n` so trial offsets are paired; older runs used `seed + n`, which confounded pulsar count with different random trials.

`select_pulsars(n)` takes the **first n** catalog entries, not an optimal subset - adding a pulsar can hurt if the extra LOS is nearly degenerate.

## TOA noise sweep - direction of trend

Sweep points: **0.1, 1, 10 us** (not “larger is better”).

Typical pattern (switching policies, 26.4 hr):

| TOA sigma | Hybrid final mean (km) | Comment |
|-------|------------------------|---------|
| 0.1 us | ~0.15 | Unrealistically tight; filter over-trusts pulsar geometry |
| 1 us | ~1.2 | Main campaign default |
| 10 us | ~1.4-1.5 | Worse than 1 us; XNAV-only much worse (~11 km) |

**Smaller TOA sigma -> lower error** in simulation. The plot uses a **log x-axis** (0.1, 1, 10): the left side is the smallest noise, not the largest.

Sweeps use a **common seed** across sigma so each trial’s offset is the same when comparing noise levels.

## Policy semantics (current)

| Policy | Non-blackout | Blackout |
|--------|--------------|----------|
| `xnav_only` | Pulsars | Pulsars |
| `gnss_only` | GNSS sidelobe PRNs (0 PRN -> pulsar fallback) | Pulsars + LunaNet if relay |
| `hybrid` | **Fuse** GNSS + pulsars + LunaNet if relay | Pulsars + **supplemental LunaNet** if relay |

LunaNet is **not** a fourth standalone phase. Plots label measured segments (`xnav + LunaNet supplemental (blackout)`), not geometric `NavMode.lonet`. Run `gnss_sidelobe_coverage_stats` to see how often non-blackout epochs have >=4 trackable PRNs.

`docs/MONTE_CARLO_RESULTS.md` sections marked **legacy / 6 hr / pre-switching** may not match current figures until regenerated.

## Process noise

| Predict mode | Q model | Tuning |
|--------------|---------|--------|
| **filter_dynamics** (primary) | HW2 sigma_acc CWNA + clock PSDs | `dynamics_sigma_acc_km`; sweep: `filter_dynamics/q_sweep.md` |
| **filter_predict** | Constant `process_noise_accel` (1e-4 m^2/s^3 default) | `sweep_process_noise.py --filter-predict` |
| **truth_velocity** | CV Q between oracle steps | Same constant Q as CV |

Filter CV may show **NIS/df >> 1** with undertuned constant Q. Dynamics predict: median NIS/df approx **0.84-0.94** at default sigma_acc = 1e-6; **1e-7-1e-8** nearer unity.

Lead with **`filter_dynamics/`** for slides; use **`common/predict_mode_comparison.md`** for cross-mode CV vs truth-velocity vs dynamics.

## Stale figures

Regenerate after policy or sweep fixes:

```bash
python scripts/build_presentation_assets.py
python scripts/refresh_presentation_manifest.py
```
