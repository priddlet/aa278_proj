## Monte Carlo — ELFO (Filter dynamics predict)

EKF predict: **Filter dynamics predict** · Arc: **26.4 hr** · Trials: **20** · TOA σ: **1.0 µs** · pulsars: **all (5)**

_EKF RK45+STM predict (HW2 Tier A); σ_acc km/s²/√s process noise._

Blackout fraction: **64.1%**

_Timing: |b_rx−b_truth| (m) averaged over GNSS/LunaNet pseudorange epochs only. XNAV-only and MSP-only blackout do not constrain b in H._

_Steady μ / Steady RMS: mean and RMS over the last 10% of arc epochs (post-convergence; excludes epoch-0 init spike)._

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady μ (km) | Steady RMS (km) | Blackout μ (km) | Non-blackout μ (km) | |b| mean (m) | |b| p95 (m) |
|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|
| **xnav_only** | 0.02 | 0.02 | 2.33 | 0.06 | 0.07 | 0.09 | 0.35 | — | — |
| **gnss_only** | 0.01 | 0.03 | 2.36 | 0.05 | 0.07 | 0.09 | 0.48 | 119.91 | 452.73 |
| **hybrid** | 0.01 | 0.02 | 2.33 | 0.05 | 0.06 | 0.09 | 0.29 | 13.06 | 45.16 |
