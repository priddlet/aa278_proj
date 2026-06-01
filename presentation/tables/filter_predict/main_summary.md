## Monte Carlo — ELFO (Filter CV predict)

EKF predict: **Filter CV predict** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs** · pulsars: **all (5)**

_EKF constant-velocity predict only; more realistic dynamics stress._

Blackout fraction: **64.1%**

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|-----------------|----------------|----------|-----------------|---------------------|
| **xnav_only** | 8.78 | 8.98 | 3.74 | 1.60 | 2.81 |
| **gnss_only** | 21.88 | 22.19 | 674.35 | 1.59 | 805.47 |
| **hybrid** | 3.09 | 3.17 | 2.78 | 1.23 | 1.04 |
