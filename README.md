# Pulsar Hybrid Navigation (AA278)

Lunar far-side navigation simulator: pulsar XNAV, GNSS sidelobe, LunaNet relays, and hybrid EKF policies on an ELFO truth arc.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,spice,viz,export]"
```

**SPICE kernels** — copy into `data/kernels/` (or set `PULSAR_NAV_KERNEL_DIR`):

- `naif0012.tls`, `de440.bsp`
- `moon_pa_de440_200625.bpc`, `moon_de440_250416.tf`

If you have AA278 HW2 data, `HW2/data/` is searched automatically (`brdc_data.npz` + Earth frame kernels for GNSS).

## Run

```bash
pytest

# Figures + tables -> figures/presentation/, presentation/, results/ (local, gitignored)
python scripts/build_presentation_assets.py --pipelines filter_dynamics
python scripts/refresh_presentation_manifest.py

# Diagnostics
python scripts/verify_pipeline.py
python scripts/check_nis.py --filter-predict
python scripts/sweep_process_noise.py --dynamics-predict --trials 10
```

Primary Monte Carlo campaign: **filter_dynamics** (EKF RK45+STM predict). Other predict modes (`truth_velocity`, `filter_predict`) remain in code for comparison.

## Layout

```
src/pulsar_nav/     catalog, measurements, filter, spice, propagation, visibility, simulation
scripts/            build assets, NIS check, process-noise sweep, pipeline verify
tests/
data/catalog/       bundled SEXTANT MSP catalog
```

## References

- Sheikh & Pines (2006) — X-ray pulsar navigation
- [NICER/SEXTANT](https://heasarc.gsfc.nasa.gov/docs/nicer/) — on-orbit XNAV demo
