"""Section 6 / Q6.2 - Extract rotor inertia J from a step response.

Lab 1 §5.2 method: underdamped PD step → overshoot / rise time → ζ, ωn →
J = kp_rad / ωn². Compare with your CAD prediction from Q2.4.

Runs one small position step (no hand disturbance). Use underdamped PD
(ki=0, small kd) so the response rings.

Close tview before running.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
MAX_TORQUE_NM = 1.63

# Underdamped PD for inertia ID (ki=0; keep kd small so it rings).
KP = 5
KD = 0.03
KI = 0.0
ILIMIT = 0.0

STEP_REV = 0.01
PRE_STEP_S = 0.5
RUN_S = 3.0
SAMPLE_DT_S = 0.005

# Q2.4 CAD mass-property inertia about the shaft axis [kg·m²].
# Fill this in after you measure it in SolidWorks / CAD.
J_CAD_KG_M2 = np.nan

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section6_4_gains.cfg"
    cfg_path.write_text(
        "\n".join([
            f"conf set servo.pid_position.kp {kp}",
            f"conf set servo.pid_position.kd {kd}",
            f"conf set servo.pid_position.ki {ki}",
            f"conf set servo.pid_position.ilimit {ilimit}",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Writing gains via moteus_tool from {cfg_path.name}...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "moteus.moteus_tool",
                "--target", str(MOTOR_ID),
                "--write-config", str(cfg_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "moteus_tool --write-config hung (timeout). Close tview and retry."
        ) from exc
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError("moteus_tool --write-config failed. Close tview and retry.")
    print(f"Gains set (RAM): kp={kp}, kd={kd}, ki={ki}, ilimit={ilimit}")


async def collect_step(controller):
    first = await controller.query()
    start_position = first.values[moteus.Register.POSITION]
    target = start_position + STEP_REV
    rows = []
    start = time.perf_counter()

    print(f"Step {STEP_REV} rev from {start_position:.6f} to {target:.6f}")
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= PRE_STEP_S + RUN_S:
            break
        command = start_position if elapsed < PRE_STEP_S else target
        result = await controller.set_position(
            position=command,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=1.0,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "phase": "step",
            "time_s": elapsed,
            "command_rev": command,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)

    return pd.DataFrame(rows), start_position, target


def analyze_step(data, start_position, target):
    step_data = data[data["time_s"] >= PRE_STEP_S].copy().reset_index(drop=True)
    t0 = PRE_STEP_S
    y0 = float(start_position)
    y_cmd = float(target)
    step_size = y_cmd - y0
    if abs(step_size) < 1e-9:
        raise RuntimeError("Step size is zero.")

    t_end = float(step_data["time_s"].iloc[-1])
    settled = step_data[step_data["time_s"] >= t_end - 0.4]
    y_final = float(settled["position_rev"].mean())

    progress = (step_data["position_rev"] - y0) * np.sign(step_size)
    reached_cmd = bool((progress >= abs(step_size)).any())
    y_ref = y_cmd if reached_cmd else y_final
    ref_size = y_ref - y0
    if abs(ref_size) < 1e-6:
        raise RuntimeError(
            "Motor barely moved after the step. Increase KP or MAX_TORQUE_NM."
        )
    if not reached_cmd:
        print(
            f"Note: command target not reached "
            f"(got {progress.max():.5f} of {abs(step_size):.5f} rev). "
            f"Using settled final position {y_final:.6f} rev as reference."
        )

    crossed = step_data[
        ((step_data["position_rev"] - y_ref) * np.sign(ref_size)) >= 0
    ]
    if crossed.empty:
        level = y0 + 0.9 * ref_size
        crossed = step_data[
            ((step_data["position_rev"] - level) * np.sign(ref_size)) >= 0
        ]
        if crossed.empty:
            raise RuntimeError(
                "Position never crossed the step reference — increase RUN_S or KP."
            )
        print("Note: using 90% rise time (never fully crossed reference).")
    t_cross = float(crossed["time_s"].iloc[0])
    rise_time = t_cross - t0

    if ref_size > 0:
        y_peak = float(step_data["position_rev"].max())
    else:
        y_peak = float(step_data["position_rev"].min())
    overshoot_frac = (y_peak - y_ref) / ref_size
    overshoot_percent = 100.0 * overshoot_frac

    if overshoot_frac <= 1e-4:
        raise RuntimeError(
            f"No overshoot measured (peak-ref)/step = {overshoot_frac:.4f}. "
            "Increase KP (or set a small KD>0 and raise KP) so the step rings."
        )

    zeta = -np.log(overshoot_frac) / np.sqrt(np.pi**2 + np.log(overshoot_frac)**2)
    theta = np.arccos(np.clip(zeta, -1.0, 1.0))
    omega_d = (np.pi - theta) / rise_time
    omega_n = omega_d / np.sqrt(1.0 - zeta**2)

    kp_rad = KP / (2.0 * np.pi)
    inertia = kp_rad / omega_n**2

    diff_pct = np.nan
    if np.isfinite(J_CAD_KG_M2) and J_CAD_KG_M2 > 0:
        diff_pct = 100.0 * (inertia - J_CAD_KG_M2) / J_CAD_KG_M2

    return {
        "rise_time_s": rise_time,
        "t_cross_s": t_cross,
        "y_peak_rev": y_peak,
        "y_ref_rev": y_ref,
        "reached_command": reached_cmd,
        "overshoot_frac": overshoot_frac,
        "overshoot_percent": overshoot_percent,
        "zeta": zeta,
        "omega_d_rad_s": omega_d,
        "omega_n_rad_s": omega_n,
        "kp_Nm_per_rev": KP,
        "kp_Nm_per_rad": kp_rad,
        "inertia_kg_m2": inertia,
        "j_cad_kg_m2": J_CAD_KG_M2,
        "diff_vs_cad_percent": diff_pct,
    }


def plot_step(data, start_position, target, metrics):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], data["command_rev"], label="Command")
    ax.plot(data["time_s"], data["position_rev"], label="Measured")
    ax.axvline(metrics["t_cross_s"], color="C2", linestyle="--", label="First crossing")
    ax.axhline(metrics["y_peak_rev"], color="C3", linestyle=":", label="Peak")
    ax.axhline(metrics["y_ref_rev"], color="C4", linestyle="-.", label="Reference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [rev]")
    ax.set_title(
        f"Q6.2 Step response (kp={KP}, kd={KD})  "
        f"OS={metrics['overshoot_percent']:.1f}%  "
        f"tr={metrics['rise_time_s']*1e3:.1f} ms  "
        f"J={metrics['inertia_kg_m2']:.4e} kg m^2"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


async def main():
    print(f"Setting gains: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    print("Inertia ID needs an underdamped step (ki=0, small kd).")
    if not np.isfinite(J_CAD_KG_M2):
        print("Tip: set J_CAD_KG_M2 from your Q2.4 CAD mass properties before comparing.")
    apply_gains(KP, KD, KI, ILIMIT)

    controller = moteus.Controller(id=MOTOR_ID)
    try:
        step_data, start_position, target = await collect_step(controller)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    step_path = DATA_DIR / "section6_4_step.csv"
    step_data.to_csv(step_path, index=False)
    print("Saved:", step_path)

    metrics = analyze_step(step_data, start_position, target)
    print("\n=== Q6.2 Step-response identification ===")
    print(f"Overshoot:              {metrics['overshoot_percent']:.2f} %")
    print(f"First-crossing rise:    {metrics['rise_time_s']:.4f} s")
    print(f"Damping ratio zeta:     {metrics['zeta']:.4f}")
    print(f"Natural freq omega_n:   {metrics['omega_n_rad_s']:.2f} rad/s")
    print(f"Rotor inertia J:        {metrics['inertia_kg_m2']:.6e} kg m^2")
    if np.isfinite(metrics["diff_vs_cad_percent"]):
        print(f"CAD J (Q2.4):           {metrics['j_cad_kg_m2']:.6e} kg m^2")
        print(f"Difference vs CAD:      {metrics['diff_vs_cad_percent']:+.1f} %")
        print(
            "Likely sources of difference: bearing/encoder mass not in CAD rotor, "
            "glue, wave spring/shims, and PLA coupling compliance."
        )
    else:
        print("CAD J not set — edit J_CAD_KG_M2 at top of this script.")

    pd.DataFrame([metrics]).to_csv(DATA_DIR / "section6_4_inertia_metrics.csv", index=False)
    print("Saved:", DATA_DIR / "section6_4_inertia_metrics.csv")
    plot_step(step_data, start_position, target, metrics)


if __name__ == "__main__":
    asyncio.run(main())
