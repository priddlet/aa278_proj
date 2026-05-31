# Presentation tables

## SEXTANT / NICER MSP catalog (bundled)

| Name | J-name | F0 (Hz) | DM (pc cm⁻³) |
|------|--------|---------|--------------|
| B1937+21 | J1939+2134 | 641.9282 | 71.02 |
| B1821-24 | J1824-2452A | 327.4056 | 21.57 |
| J0437-4715 | J0437-4715 | 173.6775 | 2.64 |
| J0218+4232 | J0218+4232 | 430.4610 | 30.25 |
| J0030+0451 | J0030+0451 | 205.5305 | 4.33 |

## Monte Carlo — ELFO (26.4 hr, n=20)

> **Stale CSVs below** (`main_*.csv`) label **coast-in-blackout** stats as `gnss_only`. Current code uses **`gnss_coast`** for that. Regenerate: `python scripts/demo_monte_carlo.py --trials 20 --no-show` and see **`docs/MONTE_CARLO_RESULTS.md`**.

TOA σ = 1 µs · pulsars = all (5) · truth-velocity predict  
Blackout fraction: **64.1%**

| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Blackout μ (km) | Non-blackout μ (km) |
|--------|-----------------|----------------|----------|-----------------|---------------------|
| **hybrid** | 1.13 | 1.29 | 2.39 | 0.45 | 0.56 |
| **xnav_only** | 2.44 | 2.62 | 2.48 | 0.53 | 0.97 |
| **gnss_only** | 6.02 | 6.26 | 15.45 | 0.53 | 20.56 |
| **gnss_coast** (stress) | 65.79 | 153.02 | 52.02 | 52.70 | 16.58 |

## Pulsar count sweep

| MSPs | Policy | Final mean (km) | Final p95 (km) | Blackout μ (km) |
|---|--------|-----------------|----------------|-----------------|
| 1 | hybrid | 1.39 | 1.56 | 20.87 |
| 1 | xnav_only | 50.53 | 85.14 | 91.85 |
| 3 | hybrid | 1.35 | 1.57 | 0.67 |
| 3 | xnav_only | 4.38 | 4.85 | 0.79 |
| 5 | hybrid | 1.16 | 1.32 | 0.45 |
| 5 | xnav_only | 2.43 | 2.68 | 0.53 |

## TOA noise sweep

| TOA µs | Policy | Final mean (km) | Final p95 (km) | Blackout μ (km) |
|---|--------|-----------------|----------------|-----------------|
| 0.1 | hybrid | 0.15 | 0.17 | 0.05 |
| 0.1 | xnav_only | 0.22 | 0.25 | 0.06 |
| 0.1 | gnss_only | 63.86 | 188.21 | 53.18 |
| 1.0 | hybrid | 1.18 | 1.38 | 0.45 |
| 1.0 | xnav_only | 2.50 | 2.68 | 0.53 |
| 1.0 | gnss_only | 77.31 | 179.28 | 51.76 |
| 10.0 | hybrid | 1.43 | 1.59 | 2.23 |
| 10.0 | xnav_only | 10.86 | 12.29 | 2.62 |
| 10.0 | gnss_only | 54.99 | 120.80 | 52.20 |
