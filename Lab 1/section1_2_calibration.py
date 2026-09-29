"""Section 1.2 - Read the saved moteus calibration REPORT log.

Run calibration manually first:
  python -m moteus.moteus_tool --target 1 --calibrate

Place the resulting moteus-cal-*.log beside this script.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    logs = list(HERE.glob("moteus-cal-*.log"))
    logs += list(HERE.parent.glob("moteus-cal-*.log"))
    if not logs:
        raise FileNotFoundError("Place moteus-cal-*.log beside this script.")

    log_path = max(logs, key=lambda path: path.stat().st_mtime)
    calibration = json.loads(log_path.read_text(encoding="utf-8"))

    resistance = calibration["winding_resistance"]
    inductance = calibration["inductance"]
    inductance_d = calibration["inductance_d"]
    inductance_q = calibration["inductance_q"]
    kv = calibration["kv"]
    poles = calibration["calibration"]["poles"]

    print("Calibration file:", log_path)
    print("Resistance [ohm]:", resistance)
    print("Inductance [H]:", inductance)
    print("Inductance d [H]:", inductance_d)
    print("Inductance q [H]:", inductance_q)
    print("Saliency ratio Lq/Ld:", inductance_q / inductance_d)
    print("Kv [rpm/V]:", kv)
    print("Pole count:", poles)
    print("Pole pairs:", poles // 2)

    for name, value in [("inductance", inductance),
                        ("inductance_q", inductance_q)]:
        time_constant = value / resistance
        corner_hz = 1.0 / (2.0 * np.pi * time_constant)
        print(f"{name}: L/R={time_constant:.8g} s, fc={corner_hz:.8g} Hz")


if __name__ == "__main__":
    main()

