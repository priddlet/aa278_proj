# Process-noise sweep — ELFO

**EKF predict:** Filter CV predict · **trials:** 3

| Mode | Q setting | note | policy | final μ (km) | RMS (km) | blk μ (km) | non-blk μ (km) | med NIS/df | |b| mean (m) |
|------|-----------|------|--------|--------------|----------|------------|----------------|------------|-------------|

**Constant CWNA:** fixed `process_noise_accel` (m²/s³). **Gravity-scaled:** q_a(r) ≈ (scale·GM/r²)² each step (periapsis-aware). Truth-radius range: **2617–10460 km**. Gravity-scaled q_a at truth radii (scale=1): **0.00201–0.513** m²/s³. Scalar RMS reference q_a ≈ **8.05e-06** m²/s³.

Target **med NIS/df ≈ 1** under filter CV. XNAV-only: timing blank (MSP-only H).

| constant | 1e-5 |  | hybrid | 13.13 | 6.41 | 4.68 | 3.67 | 75.15 | 2342.98 |
| constant | 1e-5 |  | gnss_only | 26.64 | 1491.41 | 4.69 | 1977.80 | 133.54 | 1302527.38 |
| constant | 1e-4 |  | hybrid | 3.05 | 3.27 | 1.23 | 1.11 | 8.75 | 584.73 |
| constant | 1e-4 |  | gnss_only | 10.41 | 679.08 | 1.24 | 808.79 | 12.46 | 532515.52 |
| constant | 1e-3 |  | hybrid | 0.55 | 2.95 | 0.43 | 0.55 | 1.69 | 174.47 |
| constant | 1e-3 |  | gnss_only | 4.94 | 260.59 | 0.43 | 285.48 | 2.01 | 187845.68 |
| gravity_scaled | scale=0.5 | q_a∈[0.000502,0.128] | hybrid | 0.04 | 2.94 | 0.35 | 0.51 | 1.2 | 116.02 |
| gravity_scaled | scale=0.5 | q_a∈[0.000502,0.128] | gnss_only | 1.37 | 53.41 | 0.34 | 65.51 | 1.34 | 42959.89 |
| gravity_scaled | scale=1 | q_a∈[0.00201,0.513] | hybrid | 0.05 | 2.94 | 0.34 | 0.50 | 0.71 | 116.97 |
| gravity_scaled | scale=1 | q_a∈[0.00201,0.513] | gnss_only | 1.43 | 24.95 | 0.33 | 28.99 | 0.68 | 18911.2 |
| gravity_scaled | scale=2 | q_a∈[0.00803,2.05] | hybrid | 0.05 | 2.94 | 0.34 | 0.51 | 0.55 | 118.41 |
| gravity_scaled | scale=2 | q_a∈[0.00803,2.05] | gnss_only | 1.47 | 16.59 | 0.34 | 20.44 | 0.47 | 13284.61 |
