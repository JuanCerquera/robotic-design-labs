"""Section 6.1 - Raise kd until the step is critically damped; suggest ki.

Same Lab 1 §5.3 method: keep kp from §6.1, raise kd until overshoot is gone,
then suggest ki ≈ 0.1 * kp * e0 / Ie from the error integral.

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

# Keep KP from your earlier tuning. Raise KD until overshoot is ~0.
KP = 7
KD = 0.05
KI = 0.0
ILIMIT = 0.0

STEP_REV = 0.05
HOME_POSITION_REV = 0.0
HOME_SETTLE_S = 3.0
PRE_STEP_S = 0.5
RUN_S = 1.0
SAMPLE_DT_S = 0.005
OVERSHOOT_EPS = 0.002

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section6_2_gains.cfg"
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


async def run_home_and_step(controller):
    hold_cmd = HOME_POSITION_REV
    rows = []
    start = time.perf_counter()
    total_pre = HOME_SETTLE_S + PRE_STEP_S
    print(f"Hold command={hold_cmd:.6f} for {total_pre:.1f} s, then step +{STEP_REV}")

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= total_pre:
            break
        result = await controller.set_position(
            position=hold_cmd,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=1.0,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "time_s": elapsed,
            "command_rev": hold_cmd,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)

    start_position = float(rows[-1]["position_rev"])
    step_cmd = start_position + STEP_REV
    print(f"Step to {step_cmd:.6f}  (measured start={start_position:.6f})")

    step_t0 = time.perf_counter()
    while True:
        elapsed_step = time.perf_counter() - step_t0
        if elapsed_step >= RUN_S:
            break
        result = await controller.set_position(
            position=step_cmd,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=1.0,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "time_s": total_pre + elapsed_step,
            "command_rev": step_cmd,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)

    data = pd.DataFrame(rows)
    data = data[data["time_s"] >= HOME_SETTLE_S].copy().reset_index(drop=True)
    data["time_s"] = data["time_s"] - HOME_SETTLE_S
    return data, start_position, step_cmd


def analyze_step(data, start_position, target):
    step_data = data[data["time_s"] >= PRE_STEP_S].copy().reset_index(drop=True)
    t = step_data["time_s"].to_numpy() - PRE_STEP_S
    y = step_data["position_rev"].to_numpy()

    y0 = float(start_position)
    y_cmd = float(target)
    e0 = y_cmd - y0
    if abs(e0) < 1e-9:
        raise RuntimeError("Step size is zero.")

    error = y_cmd - y
    progress = (y - y0) * np.sign(e0)
    reached_cmd = bool((progress >= abs(e0) - 1e-6).any())

    if e0 > 0:
        y_peak = float(np.max(y))
    else:
        y_peak = float(np.min(y))
    overshoot_frac = (y_peak - y_cmd) / e0
    overshoot_percent = 100.0 * max(overshoot_frac, 0.0)

    crossed_idx = np.where(progress >= abs(e0) - 1e-6)[0]
    if crossed_idx.size == 0:
        t_cross = np.nan
        rise_time = np.nan
        Ie = np.nan
        ki_suggest = np.nan
        print("Note: response never reached the command target — cannot compute Ie / ki.")
    else:
        i1 = int(crossed_idx[0])
        t_cross = float(step_data["time_s"].iloc[i1])
        rise_time = t_cross - PRE_STEP_S
        Ie = float(np.trapezoid(error[: i1 + 1], t[: i1 + 1]))
        if abs(Ie) < 1e-12:
            ki_suggest = np.nan
        else:
            ki_suggest = float(0.1 * KP * e0 / Ie)

    t_abs = step_data["time_s"].to_numpy()
    late = y[t_abs >= t_abs[-1] - 0.4]
    y_ss = float(np.mean(late)) if late.size else float(y[-1])
    ess = abs(y_cmd - y_ss)
    ilimit_min = KP * ess
    # Keep ilimit modest vs rated torque so PLA / continuous heating stay sane.
    ilimit_suggest = min(0.57, max(3.0 * ilimit_min, 1e-4))

    critically_damped = reached_cmd and overshoot_frac <= OVERSHOOT_EPS
    return {
        "kp": KP,
        "kd": KD,
        "e0_rev": e0,
        "overshoot_frac": overshoot_frac,
        "overshoot_percent": overshoot_percent,
        "rise_time_s": rise_time,
        "t_cross_s": t_cross,
        "y_peak_rev": y_peak,
        "y_ref_rev": y_cmd,
        "y0_rev": y0,
        "y_ss_rev": y_ss,
        "ess_rev": ess,
        "Ie_rev_s": Ie,
        "ki_suggest": ki_suggest,
        "ilimit_suggest_nm": ilimit_suggest,
        "reached_command": reached_cmd,
        "critically_damped": critically_damped,
    }


def plot_step(data, metrics):
    y0 = metrics["y0_rev"]
    y_ref = metrics["y_ref_rev"]
    pos_rel = data["position_rev"] - y0
    cmd_rel = np.where(
        data["time_s"].to_numpy() < PRE_STEP_S,
        0.0,
        float(y_ref - y0),
    )
    peak_rel = metrics["y_peak_rev"] - y0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], cmd_rel, label="Command / target")
    ax.plot(data["time_s"], pos_rel, label="Measured")
    ax.axhline(peak_rel, color="C3", linestyle=":", label="Peak")
    if not np.isnan(metrics["t_cross_s"]):
        ax.axvline(metrics["t_cross_s"], color="C2", linestyle="--", label="First crossing")

    status = "critically damped (~no OS)" if metrics["critically_damped"] else "still overshooting"
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    ax.set_title(
        f"§6.1 Step (kp={KP}, kd={KD})  OS={metrics['overshoot_percent']:.2f}%  [{status}]"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


async def main():
    print(f"Setting gains: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    print("Goal: raise KD until overshoot is just barely gone.")
    apply_gains(KP, KD, KI, ILIMIT)

    controller = moteus.Controller(id=MOTOR_ID)
    try:
        data, start_position, target = await run_home_and_step(controller)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    csv_path = DATA_DIR / "section6_2_step.csv"
    data.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    metrics = analyze_step(data, start_position, target)
    print("\n=== Critical-damping check ===")
    print(f"kp:                    {metrics['kp']}")
    print(f"kd:                    {metrics['kd']}")
    print(f"Overshoot:             {metrics['overshoot_percent']:.3f} %")
    print(f"First-crossing rise:   {metrics['rise_time_s']}")
    if metrics["critically_damped"]:
        print("Status:                critically damped (barely no overshoot)")
        print("Report these as your final kp and kd.")
    else:
        print("Status:                still overshooting — increase KD and re-run")

    print("\n=== Choosing ki (Lab 1 method) ===")
    print(f"e0 (step size) [rev]:  {metrics['e0_rev']:.6f}")
    print(f"Ie = ∫e dt [rev·s]:    {metrics['Ie_rev_s']}")
    print(f"Suggested ki ≈ 0.1*kp*e0/Ie:  {metrics['ki_suggest']}")
    print(f"Steady-state |ess| [rev]:     {metrics['ess_rev']:.6f}")
    print(f"Suggested ilimit (~3*kp*ess, ≤0.57):  {metrics['ilimit_suggest_nm']:.4f} N·m")
    print("Copy into section6_3_integral.py, then re-test.")

    metrics_path = DATA_DIR / "section6_2_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)
    plot_step(data, metrics)


if __name__ == "__main__":
    asyncio.run(main())
