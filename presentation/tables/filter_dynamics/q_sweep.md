# Process-noise sweep — ELFO

**EKF predict:** Filter dynamics predict (RK45+STM) · **trials:** 10

| Mode | Q setting | note | policy | final μ (km) | RMS (km) | blk μ (km) | non-blk μ (km) | med NIS/df | |b| mean (m) |
|------|-----------|------|--------|--------------|----------|------------|----------------|------------|-------------|

**HW2 CWNA:** `dynamics_sigma_acc_km` (km/s²/√s) on RK45+STM predict; clock Q from HW2 RAFS PSDs. 
Truth-radius range: **2617–10460 km**. Gravity-scaled q_a at truth radii (scale=1): **0.00201–0.513** m²/s³. Scalar RMS reference q_a ≈ **8.05e-06** m²/s³.

Target **med NIS/df ≈ 1** under filter CV / dynamics predict. XNAV-only: timing blank (MSP-only H). **|b| mean** = mean |b_rx−b_truth| (m) on GNSS/LunaNet pseudorange epochs.

| hw2_sigma_acc | 1e-8 |  | hybrid | 0.00 | 2.15 | 0.04 | 0.26 | 0.94 | 13.67 |
| hw2_sigma_acc | 1e-8 |  | gnss_only | 0.01 | 2.17 | 0.03 | 0.37 | 0.87 | 91.48 |
| hw2_sigma_acc | 1e-8 |  | xnav_only | 0.01 | 2.15 | 0.03 | 0.28 | 0.88 | — |
| hw2_sigma_acc | 1e-7 |  | hybrid | 0.01 | 2.15 | 0.04 | 0.25 | 0.92 | 11.1 |
| hw2_sigma_acc | 1e-7 |  | gnss_only | 0.01 | 2.17 | 0.05 | 0.39 | 0.86 | 96.03 |
| hw2_sigma_acc | 1e-7 |  | xnav_only | 0.01 | 2.15 | 0.05 | 0.29 | 0.87 | — |
| hw2_sigma_acc | 1e-6 | MC default | hybrid | 0.01 | 2.15 | 0.09 | 0.26 | 0.9 | 10.74 |
| hw2_sigma_acc | 1e-6 | MC default | gnss_only | 0.01 | 2.17 | 0.09 | 0.45 | 0.84 | 123.02 |
| hw2_sigma_acc | 1e-6 | MC default | xnav_only | 0.01 | 2.15 | 0.09 | 0.32 | 0.86 | — |
| hw2_sigma_acc | 1e-5 |  | hybrid | 0.01 | 2.15 | 0.15 | 0.30 | 0.86 | 14.14 |
| hw2_sigma_acc | 1e-5 |  | gnss_only | 0.03 | 2.23 | 0.15 | 0.83 | 0.78 | 337.44 |
| hw2_sigma_acc | 1e-5 |  | xnav_only | 0.03 | 2.15 | 0.16 | 0.40 | 0.81 | — |
| hw2_sigma_acc | 1e-4 |  | hybrid | 0.01 | 2.16 | 0.24 | 0.35 | 0.74 | 16.13 |
| hw2_sigma_acc | 1e-4 |  | gnss_only | 0.05 | 4.08 | 0.24 | 3.66 | 0.65 | 2159.98 |
| hw2_sigma_acc | 1e-4 |  | xnav_only | 0.08 | 2.17 | 0.24 | 0.50 | 0.72 | — |
| hw2_sigma_acc | 1e-3 |  | hybrid | 0.01 | 2.17 | 0.32 | 0.38 | 0.55 | 16.78 |
| hw2_sigma_acc | 1e-3 |  | gnss_only | 0.43 | 5.64 | 0.32 | 5.37 | 0.42 | 3224.82 |
| hw2_sigma_acc | 1e-3 |  | xnav_only | 0.32 | 2.18 | 0.32 | 0.59 | 0.46 | — |
