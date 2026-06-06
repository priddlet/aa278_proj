## SEXTANT MSP catalog - ELFO simulation

All five millisecond pulsars are used for **navigation** (LOS range, sigma = c*sigma_TOA).
Catalog **f0** supports **timing** / phase models; the MC EKF uses linear n_hat dot r only.

Default campaign: **all 5 MSPs** every **120 s**; sigma_TOA = **1 us** (~300 m range sigma).

| Name | J-name | RA (hms) | Dec (dms) | f0 (Hz) | f1 (Hz/s) | DM (pc/cm^3) | Navigation | Timing model |
|------|--------|----------|-----------|---------|-----------|-------------|------------|--------------|
| B1937+21 | J1939+2134 | 19:39:38.561297 | +21:34:59.12971 | 641.928 | -1.03e-14 | 71.02 | LOS XNAV | f0, f1, pepoch |
| B1821-24 | J1824-2452A | 18:24:32.0084 | -24:52:11.009 | 327.406 | -1.72e-13 | 21.57 | LOS XNAV | f0, f1, pepoch |
| J0437-4715 | J0437-4715 | 04:37:15.81476 | -47:15:08.6242 | 173.677 | -1.73e-14 | 2.64476 | LOS XNAV | f0, f1, pepoch |
| J0218+4232 | J0218+4232 | 02:18:06.3117 | +42:32:22.36 | 430.461 | -1.40e-14 | 30.25 | LOS XNAV | f0, f1, pepoch |
| J0030+0451 | J0030+0451 | 00:30:27.428 | +04:51:39.71 | 205.530 | -3.30e-15 | 4.333 | LOS XNAV | f0, f1, pepoch |
