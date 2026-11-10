"""Section 5 / Q5.2 - Kt from calibrated Kv; compare RI60 vs Lab 1 mj5208.

Uses Lab 1 P2.1:  Kt = 60 / (2 π Kv)   with Kv in rpm/V, Kt in N·m/A.

  python section5_2_kt_comparison.py
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

# RI60 datasheet
DS_KV_RPM_PER_V = 120.0
DS_KT_NM_PER_A = 0.100
DS_R_PHASE_PHASE_OHM = 0.900
DS_L_PHASE_PHASE_H = 877.5e-6

# Lab 1 mj5208 calibration snapshot (same board / Lab 1 log in this repo).
# Used so Q5.2 works even if Lab 1 files are elsewhere.
MJ5208_KV_RPM_PER_V = 310.76850941349124
MJ5208_R_PHASE_OHM = 0.06412671977061232
MJ5208_L_PHASE_H = 2.749872812945687e-05


def kt_from_kv(kv_rpm_per_v: float) -> float:
    return 60.0 / (2.0 * np.pi * kv_rpm_per_v)


def load_ri60_calibration():
    logs = list(HERE.glob("moteus-cal-*.log"))
    logs += list(HERE.parent.glob("moteus-cal-*.log"))
    # Prefer a Lab 2 local log; ignore the Lab 1 log if both exist by path name.
    lab2_logs = [p for p in logs if "Lab 1" not in str(p)]
    candidates = lab2_logs if lab2_logs else logs
    if not candidates:
        return None, None
    log_path = max(candidates, key=lambda path: path.stat().st_mtime)
    calibration = json.loads(log_path.read_text(encoding="utf-8"))
    return log_path, calibration


def main():
    log_path, calibration = load_ri60_calibration()

    print("=== Ideal Kt from datasheet Kv (Lab 1 P2.1) ===")
    kt_ds = kt_from_kv(DS_KV_RPM_PER_V)
    print(f"RI60 datasheet Kv:  {DS_KV_RPM_PER_V} rpm/V")
    print(f"Ideal Kt from Kv:   {kt_ds:.6f} N·m/A")
    print(f"Datasheet Kt:       {DS_KT_NM_PER_A:.6f} N·m/A")
    print(f"Difference:         {100.0 * (kt_ds - DS_KT_NM_PER_A) / DS_KT_NM_PER_A:.2f} %")

    if calibration is None:
        print()
        print("No moteus-cal-*.log found yet — using datasheet Kv for RI60 side.")
        kv_ri60 = DS_KV_RPM_PER_V
        r_ri60 = DS_R_PHASE_PHASE_OHM / 2.0
        l_ri60 = DS_L_PHASE_PHASE_H / 2.0
        source = "datasheet (no cal log)"
    else:
        kv_ri60 = calibration["kv"]
        r_ri60 = calibration["winding_resistance"]
        l_ri60 = calibration["inductance"]
        source = str(log_path)
        print()
        print(f"RI60 calibration file: {log_path}")

    kt_ri60 = kt_from_kv(kv_ri60)
    kt_mj = kt_from_kv(MJ5208_KV_RPM_PER_V)
    tau_ri60 = l_ri60 / r_ri60
    tau_mj = MJ5208_L_PHASE_H / MJ5208_R_PHASE_OHM

    print()
    print("=== Q5.2 comparison ===")
    print(f"RI60 source:              {source}")
    print(f"RI60 Kv [rpm/V]:          {kv_ri60:.4g}")
    print(f"RI60 Kt from Kv [N·m/A]:  {kt_ri60:.6f}")
    print(f"RI60 datasheet Kt:        {DS_KT_NM_PER_A:.6f}")
    print(
        f"RI60 Kt vs datasheet:     "
        f"{100.0 * (kt_ri60 - DS_KT_NM_PER_A) / DS_KT_NM_PER_A:+.2f} %"
    )
    print()
    print(f"mj5208 (Lab 1) Kv:        {MJ5208_KV_RPM_PER_V:.4g} rpm/V")
    print(f"mj5208 Kt from Kv:        {kt_mj:.6f} N·m/A")
    print()
    higher = "RI60" if kt_ri60 > kt_mj else "mj5208"
    print(f"Higher torque constant:   {higher}")
    print(f"RI60 electrical τ=L/R:    {tau_ri60:.6e} s")
    print(f"mj5208 electrical τ=L/R:  {tau_mj:.6e} s")
    shorter = "RI60" if tau_ri60 < tau_mj else "mj5208"
    print(f"Shorter electrical τ:     {shorter}")


if __name__ == "__main__":
    main()
