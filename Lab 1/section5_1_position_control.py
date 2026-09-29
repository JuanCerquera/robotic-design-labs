"""Section 5.1 - Hold a position (kp-only) while you disturb the shaft by hand.

Edit KP below. This script sets kp/kd/ki/ilimit via moteus_tool, holds the
current position, logs your manual disturbance, and plots the response.

Close tview before running. Maximum torque is limited to 0.1 N m.
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
MAX_TORQUE_NM = 0.1

# Edit these gains in the script (Q5.1: start with kd = ki = 0).
KP = 1.7
KD = 0.0
KI = 0.0
ILIMIT = 0.0

DURATION_S = 30.0
SAMPLE_DT_S = 0.005
FILENAME = "section5_1_hold_disturbance.csv"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    # --write-config needs diagnostic commands. Skip "conf write" (can hang).
    cfg_path = DATA_DIR / "section5_1_gains.cfg"
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


async def main():
    print(f"Setting gains in script: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    apply_gains(KP, KD, KI, ILIMIT)

    controller = moteus.Controller(id=MOTOR_ID)
    first = await controller.query()
    hold_position = first.values[moteus.Register.POSITION]
    print(f"Holding position = {hold_position:.6f} rev  ({hold_position * 360:.2f} deg)")
    print(f"Disturb the shaft by hand for {DURATION_S:.0f} s — logging now...")

    rows = []
    start = time.perf_counter()
    try:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= DURATION_S:
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
                "time_s": elapsed,
                "command_rev": hold_position,
                "position_rev": result.values[moteus.Register.POSITION],
                "velocity_rev_s": result.values[moteus.Register.VELOCITY],
                "command_deg": hold_position * 360.0,
                "position_deg": result.values[moteus.Register.POSITION] * 360.0,
                "kp": KP,
                "kd": KD,
                "ki": KI,
            })
            await asyncio.sleep(SAMPLE_DT_S)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    data = pd.DataFrame(rows)
    csv_path = DATA_DIR / FILENAME
    data.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    print(f"Samples: {len(data)}")

    plt.figure(figsize=(9, 4))
    plt.plot(data["time_s"], data["command_deg"], label="Hold command")
    plt.plot(data["time_s"], data["position_deg"], label="Measured")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [deg]")
    plt.title(f"Q5.1 Manual disturbance (kp={KP}, kd={KD}, ki={KI})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
