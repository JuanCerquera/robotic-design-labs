"""Section 3.1 - Sweep electrical phase and record mechanical position."""

import asyncio
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
PWM_VOLTAGE_V = 1
SETTLE_S = 1
PHASES_RAD = np.arange(0.0, 2.0 * np.pi + 1e-9, 0.5)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


async def main():
    controller = moteus.Controller(id=MOTOR_ID)
    positions_deg = []

    try:
        for phase in PHASES_RAD:
            start = time.perf_counter()
            result = None
            while time.perf_counter() - start < SETTLE_S:
                result = await controller.set_vfoc(
                    theta=float(phase),
                    voltage=float(PWM_VOLTAGE_V),
                    query=True,
                )
                await asyncio.sleep(0.02)

            position_deg = result.values[moteus.Register.POSITION] * 360.0
            positions_deg.append(position_deg)
            print(f"phase={phase:.3f} rad, position={position_deg:.3f} deg")
    finally:
        await controller.set_stop()
        print("Motor stopped")

    positions_deg = np.asarray(positions_deg)
    data = pd.DataFrame({
        "commanded_phase_rad": PHASES_RAD,
        "voltage_V": np.full(len(PHASES_RAD), PWM_VOLTAGE_V),
        "rotor_position_deg": positions_deg,
    })
    data.to_csv(DATA_DIR / "section3_1_phase_sweep.csv", index=False)
    print(data.to_string(index=False))
    print("Saved:", DATA_DIR / "section3_1_phase_sweep.csv")

    rotor_angle_deg = positions_deg - positions_deg[0]
    slope, intercept = np.polyfit(PHASES_RAD, rotor_angle_deg, 1)
    pole_pairs = 180.0 / (np.pi * abs(slope))
    print("Slope [mechanical deg/electrical rad]:", slope)
    print("Pole pairs from slope:", pole_pairs)

    plt.figure(figsize=(7, 4))
    plt.scatter(PHASES_RAD, rotor_angle_deg, label="Measured")
    plt.plot(PHASES_RAD, slope * PHASES_RAD + intercept, label="Fit")
    plt.xlabel("Commanded electrical phase [rad]")
    plt.ylabel("Mechanical rotor angle [deg]")
    plt.title("Q3.1/Q3.2 Electrical phase versus mechanical angle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())

