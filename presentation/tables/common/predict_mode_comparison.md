## Predict-mode comparison - ELFO (hybrid policy)

_Same truth arc, measurements, and TOA sigma; only the EKF time-update differs._

| Predict mode | Final mean (km) | Final p95 (km) | Full-arc RMS (km) | Steady RMS (km) | Blackout mu (km) | |b| mean (m) |
|--------------|-----------------|----------------|-------------------|-----------------|-----------------|------------|
| Truth-velocity predict | 1.13 | 1.23 | 2.39 | 0.88 | 0.45 | 227.54 |
| Filter CV predict | 3.09 | 3.17 | 2.78 | 2.81 | 1.23 | 578.01 |
| Filter dynamics (RK45+STM) | 0.01 | 0.02 | 2.33 | 0.06 | 0.09 | 13.06 |

**Full-arc RMS** includes epoch-0 initial offset (no update at t=0). **Steady RMS** is last 10% of arc.

See per-mode policy tables in `filter_dynamics/`; other modes appear in this file only.
