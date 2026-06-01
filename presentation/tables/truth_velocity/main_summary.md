## Monte Carlo — ELFO (Truth-velocity predict)

EKF predict: **Truth-velocity predict** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs** · pulsars: **all (5)**

_Oracle motion between measurements (default sim; optimistic absolute errors)._

Blackout fraction: **64.1%**

_Timing: |b_rx−b_truth| (m) averaged over GNSS/LunaNet pseudorange epochs only. XNAV-only and MSP-only blackout do not constrain b in H._

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) | |b| mean (m) | |b| p95 (m) |
|--------|-----------------|----------------|----------|-----------------|---------------------|------------|-----------|
| **xnav_only** | 3.74 | 3.86 | 2.46 | 0.46 | 0.99 | — | — |
| **gnss_only** | 4.89 | 5.02 | 16.09 | 0.45 | 21.10 | 13774.09 | 49407.06 |
| **hybrid** | 1.13 | 1.23 | 2.39 | 0.45 | 0.56 | 227.54 | 812.77 |
