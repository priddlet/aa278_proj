"""Fetch satellite TLEs from CelesTrak (optional LunaNet / GPS groups)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import requests

# CelesTrak GP group URLs (https://celestrak.org/NORAD/elements/)
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
GROUP_URLS = {
    "gps": "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
    "gnss": "https://celestrak.org/NORAD/elements/gp.php?GROUP=gnss&FORMAT=tle",
    "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
}


def fetch_tle_group(group: str = "gnss", timeout_s: float = 30.0) -> list[tuple[str, str, str]]:
    """
    Download TLE triplets (name, line1, line2) from CelesTrak.

    Falls back to empty list on network failure; use Walker LunaNet model instead.
    """
    url = GROUP_URLS.get(group, f"{CELESTRAK_GP_URL}?GROUP={group}&FORMAT=tle")
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    return parse_tle_text(resp.text)


def parse_tle_text(text: str) -> list[tuple[str, str, str]]:
    """Parse raw TLE text into (name, line1, line2) records."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    records: list[tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(lines):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            records.append((lines[i], lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return records


def save_tle_file(records: Iterable[tuple[str, str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name, l1, l2 in records:
            f.write(f"{name}\n{l1}\n{l2}\n")
    return path
