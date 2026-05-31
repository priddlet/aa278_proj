# Policy comparison — honest framing

Use this paragraph (or shorten for slides) when comparing navigation policies on the 26.4 hr ELFO arc. It matches **current** `NavPolicy` semantics in `src/pulsar_nav/simulation/policy.py`.

---

## Report paragraph (drop-in)

We compare three **switching** policies on a 26.4 hr ELFO truth arc (~64% geometric GNSS blackout): **XNAV-only** (five millisecond pulsars every epoch), **GNSS-only** (sidelobe GNSS when not in blackout, with pulsar fallback when zero PRNs are trackable; **pulsars only** in blackout, no LunaNet), and **hybrid** (non-blackout **joint fusion** of sidelobe GNSS, five pulsars, and LunaNet when a relay is visible; blackout uses pulsars plus supplemental LunaNet when the relay is visible). The headline far-side failure mode is **not** `gnss_only` under this definition—its blackout segment uses the same pulsar measurements as XNAV-only, so blackout errors stay near the sub-km scale in the truth-velocity demonstration mode. The stress baseline for “GNSS when visible, **no measurements** in blackout” is the separate **`gnss_coast`** policy (legacy coasting). Against that baseline, hybrid and XNAV-only preserve far-side continuity while `gnss_coast` diverges (~50 km mean error in blackout in our Monte Carlo). Near-side differences among the pulsar-backed policies come mainly from **GNSS sidelobe geometry** (sparse 0–4 PRNs, poor DOP) and optional **LunaNet** aiding—not from coasting through blackout. Segment means can show **lower error in blackout than outside blackout** because non-blackout epochs correlate with perilune, where constant-velocity prediction between 120 s updates injects up to kilometer-scale error on an e ≈ 0.6 orbit; we therefore emphasize **final mean error** and **blackout vs non-blackout segment means**, not RMS, which is dominated by the shared 30–100 km initial offset transient.

---

## What *not* to claim

| Claim | Problem |
|-------|---------|
| “`gnss_only` blows up to ~50 km in blackout” | Stale: that was **coasting** (`gnss_coast`). Current `gnss_only` uses pulsars in blackout (~0.5 km blackout μ with truth-velocity predict). |
| “Hybrid RMS clearly wins” | RMS ≈ 2.4 km for hybrid and XNAV-only; dominated by init transient, not steady state. |
| “Blackout segment is harder” (always) | With truth-velocity predict, non-blackout μ can exceed blackout μ near periapsis. |

---

## Recommended comparison axes

1. **Far-side continuity:** `hybrid` / `xnav_only` / `gnss_only` vs **`gnss_coast`** (blackout segment means, final mean).
2. **Near-side GNSS value:** `hybrid` vs `gnss_only` vs `xnav_only` on **non-blackout** segment means and finals. Pure **switching** (GNSS without pulsars near periapsis) inflates hybrid non-blackout error (~4 km) because collinear sidelobe PRNs have poor PDOP; **fusion** keeps pulsars in the update and should beat XNAV-only there.
3. **LunaNet supplement:** `hybrid` vs `gnss_only` where relay visibility applies (blackout + near side).
4. **Realistic dynamics:** repeat with `use_truth_velocity_predict=False` (`filter_predict` pipeline); absolute km grow, **trends** on (1)–(3) remain.

Regenerate numbers: `python scripts/demo_monte_carlo.py --trials 20 --no-show` and `python scripts/build_presentation_assets.py`.
