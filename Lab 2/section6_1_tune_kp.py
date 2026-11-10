"""Section 6.1 - Tune kp with a small position step (kd = ki = 0).

Lab 1 §5.1 idea, but a step is used instead of a hand disturbance — the RI60
hold is too stiff to nudge by hand once kp is useful.

Edit KP below. Raise until the step rings / goes unstable, then back off ~30–50%.
Close tview before running. Peak torque capped at 1.63 N·m.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import pandas as pd


MOTOR_ID = 1
MAX_TORQUE_NM = 1.63  # RI60 peak; keep RMS ≤ 0.57 N·m

# Edit these gains (start with kd = ki = 0).
KP = 17.5
KD = 0.0
KI = 0.0
ILIMIT = 0.0

STEP_REV = 0.05
PRE_STEP_S = 0.5
RUN_S = 3.0
SAMPLE_DT_S = 0.005
FILENAME = "section6_1_step.csv"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section6_1_gains.cfg"
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
        raise RuntimeError(
            "moteus_tool --write-config failed. Close tview and retry."
        )
    print(f"Gains set (RAM): kp={kp}, kd={kd}, ki={ki}, ilimit={ilimit}")


async def run_step(controller):
    first = await controller.query()
    start_position = float(first.values[moteus.Register.POSITION])
    hold_cmd = start_position
    step_cmd = start_position + STEP_REV
    print(f"Start = {start_position:.6f} rev  ({start_position * 360:.2f} deg)")
    print(f"Hold {PRE_STEP_S:.1f} s, then step +{STEP_REV} rev → {step_cmd:.6f}")

    rows = []
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= PRE_STEP_S + RUN_S:
            break
        cmd = hold_cmd if elapsed < PRE_STEP_S else step_cmd
        result = await controller.set_position(
            position=cmd,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=1.0,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "time_s": elapsed,
            "command_rev": cmd,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
            "command_deg": cmd * 360.0,
            "position_deg": result.values[moteus.Register.POSITION] * 360.0,
            "kp": KP,
            "kd": KD,
            "ki": KI,
        })
        await asyncio.sleep(SAMPLE_DT_S)

    return pd.DataFrame(rows), start_position, step_cmd


async def main():
    print(f"Setting gains in script: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    print("Procedure: raise kp until the step rings / goes unstable, then back off ~30–50%.")
    apply_gains(KP, KD, KI, ILIMIT)

    controller = moteus.Controller(id=MOTOR_ID)
    try:
        data, start_position, step_cmd = await run_step(controller)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    csv_path = DATA_DIR / FILENAME
    data.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    print(f"Samples: {len(data)}")
    print("Look for ringing / sustained oscillation → kp too high.")
    print("Next: copy your settled KP into section6_2_critical_damping.py")

    pos_rel_deg = (data["position_rev"] - start_position) * 360.0
    cmd_rel_deg = (data["command_rev"] - start_position) * 360.0

    plt.figure(figsize=(9, 4))
    plt.plot(data["time_s"], cmd_rel_deg, label="Command")
    plt.plot(data["time_s"], pos_rel_deg, label="Measured")
    plt.axvline(PRE_STEP_S, color="0.5", linestyle=":", label="Step")
    plt.xlabel("Time [s]")
    plt.ylabel("Position relative to start [deg]")
    plt.title(f"§6.1 Step (kp={KP}, kd={KD}, ki={KI})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
