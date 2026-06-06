## Timing conclusions - ELFO (Filter dynamics predict)

**Metric:** |b_rx - b_truth| (m) on GNSS/LunaNet pseudorange epochs.

| Policy | |b| mean (m) | |b| p95 (m) |
|--------|------------|-----------|
| **xnav_only** | - | - |
| **gnss_only** | 119.91 | 452.73 |
| **hybrid** | 13.06 | 45.16 |

See `presentation/INDEX.md` for interpretation. Hybrid maintains clock observability on PR epochs; gnss_only drift is larger.
