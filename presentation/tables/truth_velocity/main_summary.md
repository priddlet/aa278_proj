## Monte Carlo — ELFO (Truth-velocity predict)

EKF predict: **Truth-velocity predict** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs** · pulsars: **all (5)**

_Oracle motion between measurements (default sim; optimistic absolute errors)._

Blackout fraction: **64.1%**

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|-----------------|----------------|----------|-----------------|---------------------|
| **xnav_only** | 2.44 | 2.62 | 2.48 | 0.53 | 0.97 |
| **gnss_only** | 6.02 | 6.26 | 15.45 | 0.53 | 20.56 |
| **hybrid** | 1.13 | 1.23 | 2.39 | 0.45 | 0.56 |
