"""Section 5.2 - Manual disturbance, then step response to estimate rotor inertia.

Sequence:
  1. Set gains in the script
  2. Hold position while you disturb by hand (like 5.1)
  3. Stop, wait PAUSE_BEFORE_STEP_S
  4. Run a small position step, extract overshoot / rise time, compute J

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
MAX_TORQUE_NM = 0.1

# PD gains used for the inertia identification step (edit as needed).
KP = 1.6          # N m / rev
KD = 0.0         # N m / (rev/s) — use a bit of D so the 2nd-order formulas apply
KI = 0.0
ILIMIT = 0.0

STEP_REV = 0.01
PRE_STEP_S = 0.5
RUN_S = 3.0
SAMPLE_DT_S = 0.005

# Phase 1: hold position while you disturb by hand (like 5.1).
HOLD_DISTURB_S = 10.0
# Pause after stop, before the automatic step.
PAUSE_BEFORE_STEP_S = 5.0

# Optional geometric ring comparison (leave nan to skip).
OUTER_DIAMETER_M = np.nan
INNER_DIAMETER_M = np.nan
AXIAL_WIDTH_M = np.nan
STEEL_DENSITY_KG_M3 = 7800.0

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    # --write-config needs diagnostic commands. Skip "conf write" (can hang).
    cfg_path = DATA_DIR / "section5_2_gains.cfg"
    cfg_path.write_text(
        "\n".join([
            f"conf set servo.pid_position.kp {kp}",
            f"conf set servo.pid_position.kd {kd}",
            f"conf set servo.pid_position.ki {ki}",
            f"conf set servo.pid_position.ilimit {ilimit}",
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


async def hold_and_log(controller, duration_s, label="hold"):
    first = await controller.query()
    hold_position = first.values[moteus.Register.POSITION]
    rows = []
    start = time.perf_counter()
    print(
        f"{label}: holding {hold_position:.6f} rev "
        f"({hold_position * 360:.2f} deg) for {duration_s:.0f} s — disturb by hand"
    )
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration_s:
            break
        result = await controller.set_position(
            position=hold_position,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=1.0,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "phase": label,
            "time_s": elapsed,
            "command_rev": hold_position,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)
    return pd.DataFrame(rows), hold_position


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
    """Extract overshoot and first-crossing rise time after the step instant."""
    step_data = data[data["time_s"] >= PRE_STEP_S].copy().reset_index(drop=True)
    t0 = PRE_STEP_S
    y0 = float(start_position)
    y_cmd = float(target)
    step_size = y_cmd - y0
    if abs(step_size) < 1e-9:
        raise RuntimeError("Step size is zero.")

    # Settled value from the last 0.4 s (handles P-only steady-state error).
    t_end = float(step_data["time_s"].iloc[-1])
    settled = step_data[step_data["time_s"] >= t_end - 0.4]
    y_final = float(settled["position_rev"].mean())

    # Prefer the commanded target if it was actually reached; otherwise use settle.
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

    # First crossing of the reference level after the step.
    crossed = step_data[
        ((step_data["position_rev"] - y_ref) * np.sign(ref_size)) >= 0
    ]
    if crossed.empty:
        # Fall back to 90% rise if it never quite hits the reference.
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

    # Peak excursion beyond the reference (same direction as the step).
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

    # moteus kp is N m / rev; convert to N m / rad for J = kp_rad / omega_n^2.
    kp_rad = KP / (2.0 * np.pi)
    inertia = kp_rad / omega_n**2

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
    }


def geometric_ring_inertia():
    if any(np.isnan(v) for v in (OUTER_DIAMETER_M, INNER_DIAMETER_M, AXIAL_WIDTH_M)):
        return None
    outer_r = OUTER_DIAMETER_M / 2.0
    inner_r = INNER_DIAMETER_M / 2.0
    volume = np.pi * (outer_r**2 - inner_r**2) * AXIAL_WIDTH_M
    mass = STEEL_DENSITY_KG_M3 * volume
    inertia = 0.5 * mass * (outer_r**2 + inner_r**2)
    return {"mass_kg": mass, "inertia_kg_m2": inertia}


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
        f"Q5.5 Step response (kp={KP}, kd={KD})  "
        f"OS={metrics['overshoot_percent']:.1f}%  "
        f"tr={metrics['rise_time_s']*1e3:.1f} ms  "
        f"J={metrics['inertia_kg_m2']:.4e} kg m^2"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_hold(data):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], data["command_rev"] * 360.0, label="Hold command")
    ax.plot(data["time_s"], data["position_rev"] * 360.0, label="Measured")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [deg]")
    ax.set_title(f"Q5.2 Manual disturbance before step (kp={KP}, kd={KD})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


async def main():
    print(f"Setting gains: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    apply_gains(KP, KD, KI, ILIMIT)

    controller = moteus.Controller(id=MOTOR_ID)
    try:
        # 1) Hold + manual disturbance (like 5.1)
        hold_data, _ = await hold_and_log(controller, HOLD_DISTURB_S, label="hold")
        hold_path = DATA_DIR / "section5_2_hold_disturbance.csv"
        hold_data.to_csv(hold_path, index=False)
        print("Saved:", hold_path)

        # 2) Stop, then wait before the step
        await controller.set_stop()
        print(f"Motor stopped. Waiting {PAUSE_BEFORE_STEP_S:.0f} s before step...")
        await asyncio.sleep(PAUSE_BEFORE_STEP_S)

        # 3) Automatic position step for inertia ID
        step_data, start_position, target = await collect_step(controller)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    step_path = DATA_DIR / "section5_2_step.csv"
    step_data.to_csv(step_path, index=False)
    print("Saved:", step_path)

    plot_hold(hold_data)

    metrics = analyze_step(step_data, start_position, target)
    print("\n=== Step-response identification ===")
    print(f"Overshoot:              {metrics['overshoot_percent']:.2f} %")
    print(f"First-crossing rise:    {metrics['rise_time_s']:.4f} s")
    print(f"Damping ratio zeta:     {metrics['zeta']:.4f}")
    print(f"Natural freq omega_n:   {metrics['omega_n_rad_s']:.2f} rad/s")
    print(f"Rotor inertia J:        {metrics['inertia_kg_m2']:.6e} kg m^2")

    geo = geometric_ring_inertia()
    if geo is not None:
        print("\n=== Geometric annular-ring estimate ===")
        print(f"Ring mass:              {geo['mass_kg']:.4f} kg")
        print(f"Ring inertia:           {geo['inertia_kg_m2']:.6e} kg m^2")
        print(
            f"Ratio step/geometric:   "
            f"{metrics['inertia_kg_m2'] / geo['inertia_kg_m2']:.3f}"
        )

    pd.DataFrame([metrics]).to_csv(DATA_DIR / "section5_2_inertia_metrics.csv", index=False)
    plot_step(step_data, start_position, target, metrics)


if __name__ == "__main__":
    asyncio.run(main())
