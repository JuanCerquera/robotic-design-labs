"""Section 1.3 - Collect and analyze 30 seconds of encoder noise."""

import asyncio
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
ENCODER_COUNTS_PER_REV = 16384
DURATION_S = 30.0
SAMPLE_DELAY_S = 0.01

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


async def main():
    controller = moteus.Controller(id=MOTOR_ID)
    rows = []
    start = time.perf_counter()

    while time.perf_counter() - start < DURATION_S:
        result = await controller.query()
        rows.append({
            "time_s": time.perf_counter() - start,
            "position_rev": result.values[moteus.Register.POSITION],
        })
        await asyncio.sleep(SAMPLE_DELAY_S)

    data = pd.DataFrame(rows)
    csv_path = DATA_DIR / "section1_3_encoder_noise.csv"
    data.to_csv(csv_path, index=False)

    position_rev = data["position_rev"].to_numpy()
    position_counts = position_rev * ENCODER_COUNTS_PER_REV
    position_deg = position_rev * 360.0

    print("Saved:", csv_path)
    print("Mean sample rate [Hz]:", (len(data)-1) / data["time_s"].iloc[-1])
    print("Peak-to-peak [counts]:", np.ptp(position_counts))
    print("Standard deviation [counts]:", np.std(position_counts, ddof=1))
    print("Peak-to-peak [deg]:", np.ptp(position_deg))
    print("Standard deviation [deg]:", np.std(position_deg, ddof=1))

    plt.figure(figsize=(8, 4))
    plt.plot(data["time_s"], position_deg)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [deg]")
    plt.title("Q1.3 Encoder noise floor")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())

