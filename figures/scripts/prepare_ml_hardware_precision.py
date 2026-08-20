#!/usr/bin/env python3
"""Prepare precision-specific peak-performance series for the ML trend plot."""

import csv
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "ml_hardware.csv"

FIELDS = {
    "fp32": "FP32 (single precision) performance (FLOP/s)",
    "tf32": "TF32 (TensorFloat-32) performance (FLOP/s)",
    "fp16": "FP16 (half precision) performance (FLOP/s)",
    "fp8": "FP8 performance (FLOP/s)",
    "int8": "INT8 performance (OP/s)",
}


def decimal_year(value: str) -> float:
    year, month, day = (int(part) for part in value[:10].split("-"))
    start = date(year, 1, 1)
    current = date(year, month, day)
    return year + (current - start).days / (date(year + 1, 1, 1) - start).days


def main() -> None:
    rows = {name: [] for name in FIELDS}
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            release_date = row.get("Release date", "")
            if not release_date or len(release_date) < 10:
                continue
            try:
                year = decimal_year(release_date)
            except ValueError:
                continue
            for name, field in FIELDS.items():
                try:
                    value = float(row.get(field, ""))
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    # Plot all series in TOP/s, preserving the source's OP/s
                    # convention for INT8.
                    rows[name].append((year, value / 1e12))

    for name, values in rows.items():
        output = ROOT / "data" / f"ml_hardware_{name}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["year", "tops"])
            writer.writerows(sorted(values))


if __name__ == "__main__":
    main()
