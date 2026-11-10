"""Section 5 / Q5.1 - Parse moteus calibration and compare with the RI60 datasheet.

Run calibration first (shaft free to turn):
  python -m moteus.moteus_tool --target 1 --calibrate

Place the resulting moteus-cal-*.log beside this script, then:
  python section5_1_calibration.py
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

# CubeMars RI60 KV120 datasheet (Lab 2 §1.1) — phase-to-phase electricals.
DS_KV_RPM_PER_V = 120.0
DS_POLE_PAIRS = 14
DS_R_PHASE_PHASE_OHM = 0.900
DS_L_PHASE_PHASE_H = 877.5e-6


def pct_diff(measured, datasheet):
    return 100.0 * (measured - datasheet) / datasheet


def main():
    logs = list(HERE.glob("moteus-cal-*.log"))
    logs += list(HERE.parent.glob("moteus-cal-*.log"))
    if not logs:
        raise FileNotFoundError(
            "Place moteus-cal-*.log beside this script "
            "(run: python -m moteus.moteus_tool --target 1 --calibrate)."
        )

    log_path = max(logs, key=lambda path: path.stat().st_mtime)
    calibration = json.loads(log_path.read_text(encoding="utf-8"))

    resistance = calibration["winding_resistance"]
    inductance = calibration["inductance"]
    inductance_d = calibration["inductance_d"]
    inductance_q = calibration["inductance_q"]
    kv = calibration["kv"]
    poles = calibration["calibration"]["poles"]
    pole_pairs = poles // 2

    print("Calibration file:", log_path)
    print()
    print("=== moteus calibration REPORT ===")
    print(f"Resistance [ohm]:     {resistance}")
    print(f"Inductance [H]:       {inductance}")
    print(f"Inductance d [H]:     {inductance_d}")
    print(f"Inductance q [H]:     {inductance_q}")
    print(f"Kv [rpm/V]:           {kv}")
    print(f"Pole count:           {poles}")
    print(f"Pole pairs:           {pole_pairs}")

    # moteus winding_resistance / inductance are the FOC phase values
    # (≈ half of phase-to-phase for a Y-wound motor).
    r_pp_from_moteus = 2.0 * resistance
    l_pp_from_moteus = 2.0 * inductance

    print()
    print("=== Q5.1 vs RI60 datasheet (phase-to-phase) ===")
    print(f"{'Quantity':<28} {'moteus':>14} {'datasheet':>14} {'diff %':>10}")
    print("-" * 70)
    print(
        f"{'Kv [rpm/V]':<28} {kv:14.4g} {DS_KV_RPM_PER_V:14.4g} "
        f"{pct_diff(kv, DS_KV_RPM_PER_V):10.2f}"
    )
    print(
        f"{'Pole pairs':<28} {pole_pairs:14d} {DS_POLE_PAIRS:14d} "
        f"{pct_diff(pole_pairs, DS_POLE_PAIRS):10.2f}"
    )
    print(
        f"{'R phase-phase [ohm]':<28} {r_pp_from_moteus:14.4g} "
        f"{DS_R_PHASE_PHASE_OHM:14.4g} "
        f"{pct_diff(r_pp_from_moteus, DS_R_PHASE_PHASE_OHM):10.2f}"
    )
    print(
        f"{'L phase-phase [H]':<28} {l_pp_from_moteus:14.4g} "
        f"{DS_L_PHASE_PHASE_H:14.4g} "
        f"{pct_diff(l_pp_from_moteus, DS_L_PHASE_PHASE_H):10.2f}"
    )
    print()
    print("Which does moteus report?")
    print(
        "  winding_resistance and inductance are the *phase* FOC values "
        "(≈ half of the datasheet phase-to-phase numbers for a Y wind)."
    )
    print(
        f"  So moteus R≈{resistance:.4g} Ω vs datasheet R_pp/2="
        f"{DS_R_PHASE_PHASE_OHM / 2.0:.4g} Ω; "
        f"moteus L≈{inductance:.4g} H vs L_pp/2={DS_L_PHASE_PHASE_H / 2.0:.4g} H."
    )

    for name, value in [("inductance", inductance), ("inductance_q", inductance_q)]:
        time_constant = value / resistance
        corner_hz = 1.0 / (2.0 * np.pi * time_constant)
        print(f"{name}: L/R={time_constant:.8g} s, fc={corner_hz:.8g} Hz")


if __name__ == "__main__":
    main()
