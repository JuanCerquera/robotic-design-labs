"""Section 2.2 - Calculate Kt from the calibration Kv."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
KV_DATASHEET_RPM_PER_V = 330.0


def main():
    logs = list(HERE.glob("moteus-cal-*.log"))
    logs += list(HERE.parent.glob("moteus-cal-*.log"))
    if not logs:
        raise FileNotFoundError("Place moteus-cal-*.log beside this script.")

    log_path = max(logs, key=lambda path: path.stat().st_mtime)
    calibration = json.loads(log_path.read_text(encoding="utf-8"))
    kv = calibration["kv"]
    kt = 60.0 / (2.0 * np.pi * kv)
    difference = 100.0 * (kv - KV_DATASHEET_RPM_PER_V) / KV_DATASHEET_RPM_PER_V

    print("Calibration Kv [rpm/V]:", kv)
    print("Calibration Kt [N m/A]:", kt)
    print("Kv difference from 330 rpm/V [%]:", difference)


if __name__ == "__main__":
    main()

