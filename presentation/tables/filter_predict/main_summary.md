## Monte Carlo — ELFO (Filter CV predict)

EKF predict: **Filter CV predict** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs** · pulsars: **all (5)**

_EKF constant-velocity predict only; more realistic dynamics stress._

Blackout fraction: **64.1%**

_Timing: |b_rx−b_truth| (m) averaged over GNSS/LunaNet pseudorange epochs only. XNAV-only and MSP-only blackout do not constrain b in H._

_Steady μ / Steady RMS: mean and RMS over the last 10% of arc epochs (post-convergence; excludes epoch-0 init spike)._

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady μ (km) | Steady RMS (km) | Blackout μ (km) | Non-blackout μ (km) | |b| mean (m) | |b| p95 (m) |
|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|
| **xnav_only** | 7.39 | 7.59 | 3.50 | 2.33 | 2.91 | 1.24 | 2.80 | — | — |
| **gnss_only** | 10.44 | 10.64 | 680.18 | 2.36 | 3.03 | 1.23 | 808.91 | 532626.51 | 1873228.14 |
| **hybrid** | 3.09 | 3.17 | 2.78 | 2.27 | 2.81 | 1.23 | 1.04 | 578.01 | 2201.28 |
