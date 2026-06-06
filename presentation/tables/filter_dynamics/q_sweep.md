## Q diagnostics - HW2 sigma_acc sweep

**Trials:** 10 | **Predict:** filter dynamics (RK45+STM)

Target **median NIS/df approx 1**. Default MC uses sigma_acc = **1e-6** km/s^2/sqrt(s).

| sigma_acc (km/s^2/sqrt(s)) | Policy | final mu (km) | med NIS/df | |b| mean (m) |
|------------------|--------|--------------|------------|------------|
| 1e-08 | hybrid | 0.01 | 0.94 | 13.69 |
| 1e-08 | gnss_only | 0.00 | 0.87 | 82.36 |
| 1e-08 | xnav_only | 0.01 | 0.87 |  |
| 1e-07 | hybrid | 0.01 | 0.92 | 13.42 |
| 1e-07 | gnss_only | 0.01 | 0.86 | 87.04 |
| 1e-07 | xnav_only | 0.01 | 0.87 |  |
| 1e-06 | hybrid | 0.01 | 0.89 | 14.75 |
| 1e-06 | gnss_only | 0.01 | 0.84 | 107.95 |
| 1e-06 | xnav_only | 0.01 | 0.86 |  |
| 1e-05 | hybrid | 0.02 | 0.85 | 18.85 |
| 1e-05 | gnss_only | 0.02 | 0.79 | 353.28 |
| 1e-05 | xnav_only | 0.03 | 0.82 |  |
| 0.0001 | hybrid | 0.02 | 0.74 | 20.8 |
| 0.0001 | gnss_only | 0.05 | 0.65 | 1666.1 |
| 0.0001 | xnav_only | 0.09 | 0.71 |  |
| 0.001 | hybrid | 0.02 | 0.54 | 21.26 |
| 0.001 | gnss_only | 0.47 | 0.42 | 2188.28 |
| 0.001 | xnav_only | 0.38 | 0.45 |  |
