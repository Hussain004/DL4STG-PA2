"""Inject the result tables into FINDINGS.md at the <!--NAME_TABLE--> markers."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables as mt          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "FINDINGS.md"

TABLES = {
    "ARCH": mt.table_arch(),
    "LOOKBACK": mt.table_generic("lookback", label="history window L (steps of 10 min)"),
    "ROUNDS": mt.table_generic("rounds", label="spatial message-passing rounds"),
    "INPUTS": mt.table_generic("inputs", label="input variant"),
    "LOSS": mt.table_generic("loss", label="training objective"),
    "ROBUST": mt.table_generic("robust", label="degradation"),
    "FLOOD": mt.table_flood(),
    "INTERVENTION": mt.table_intervention(),
    "REVERSE": mt.table_reverse(),
    "PROBE": mt.table_probe(),
    "INTERVENTION": mt.table_intervention(),
}


def main():
    text = MD.read_text()
    for key, table in TABLES.items():
        marker = f"<!--{key}_TABLE-->"
        if marker not in text:
            continue
        if not table:
            print(f"{key}: no data yet, marker left in place")
            continue
        text = text.replace(marker, table)
        print(f"{key}: injected")
    MD.write_text(text)
    print("wrote", MD)


if __name__ == "__main__":
    main()
