"""Load pulsars from bundled catalog or ATNF PSRCAT (when available)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import requests

from pulsar_nav.catalog.pulsar import Pulsar

# Historical CGI endpoints (may be offline); kept for future use.
PSRCAT_CGI_CANDIDATES = (
    "https://www.atnf.csiro.au/cgi-bin/psrcat/psrcat.cgi",
    "https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat.cgi",
)

DEFAULT_CATALOG = Path(__file__).resolve().parents[3] / "data" / "catalog" / "sextant_msp.json"


def load_bundled_catalog(path: Path | None = None) -> list[Pulsar]:
    path = path or DEFAULT_CATALOG
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [Pulsar.from_dict(r) for r in records]


def load_catalog(names: Sequence[str] | None = None) -> list[Pulsar]:
    """Load pulsars by name from bundled SEXTANT MSP set."""
    catalog = load_bundled_catalog()
    if names is None:
        return catalog
    lookup: dict[str, Pulsar] = {}
    for p in catalog:
        lookup[p.name.upper()] = p
        if p.jname:
            lookup[p.jname.upper()] = p
    out: list[Pulsar] = []
    for n in names:
        key = n.upper().replace("PSR ", "").strip()
        if key not in lookup:
            raise KeyError(f"Unknown pulsar {n!r} in bundled catalog")
        out.append(lookup[key])
    return out


def _parse_psrcat_table(text: str) -> list[dict]:
    """Parse short-format PSRCAT text table into dict records."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    header = re.split(r"\s+", lines[0].strip())
    records = []
    for ln in lines[1:]:
        if ln.startswith("---") or "NAME" in ln.upper():
            continue
        parts = re.split(r"\s+", ln.strip(), maxsplit=len(header) - 1)
        if len(parts) < len(header):
            continue
        records.append(dict(zip(header, parts)))
    return records


def fetch_psrcat(
    names: Iterable[str],
    columns: Sequence[str] = ("name", "raj", "decj", "f0", "f1", "pepoch", "dm"),
    timeout_s: float = 30.0,
) -> list[Pulsar]:
    """
    Query ATNF PSRCAT web interface.

    Falls back to bundled catalog if all CGI endpoints return errors.
    See https://www.atnf.csiro.au/research/pulsar/psrcat/
    """
    col_args = "".join(f"&-c&{c}" for c in columns)
    pulsars: list[Pulsar] = []
    for name in names:
        last_err: Exception | None = None
        for base in PSRCAT_CGI_CANDIDATES:
            url = f"{base}?-q&{name}{col_args}&-format&2&-short&-null&"
            try:
                resp = requests.get(url, timeout=timeout_s)
                resp.raise_for_status()
                if "404" in resp.text[:200] or "<html" in resp.text[:50].lower():
                    continue
                rows = _parse_psrcat_table(resp.text)
                if not rows:
                    continue
                row = rows[0]
                pulsars.append(
                    Pulsar(
                        name=row.get("NAME", name),
                        raj=row["RAJ"],
                        decj=row["DECJ"],
                        f0_hz=float(row["F0"]),
                        f1_hz_s=float(row.get("F1", 0.0)),
                        pepoch_mjd=float(row.get("PEPOCH", 51544.0)),
                        dm=float(row["DM"]) if row.get("DM") else None,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            bundled = {p.name: p for p in load_bundled_catalog()}
            if name in bundled or name.upper() in {k.upper() for k in bundled}:
                for p in bundled.values():
                    if p.name == name or (p.jname and p.jname == name):
                        pulsars.append(p)
                        break
            else:
                raise RuntimeError(
                    f"PSRCAT query failed for {name}; install bundled catalog or "
                    f"download PSRCAT locally. Last error: {last_err}"
                )
    return pulsars
