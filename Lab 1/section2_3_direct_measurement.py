"""Section 2.3 - Analyze the manual lever-arm/scale measurement.

Use tview manually for each point:
  d dq 0 <amps>
  d stop
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RADIUS_M = 0.080

# Measurements copied from the current notebook. These are current magnitudes.
CURRENT_RISING_A = np.arange(1.0, 5.01, 0.5)
CURRENT_FALLING_A = CURRENT_RISING_A[::-1]
SCALE_RISING_G = np.array([33.6, 49.4, 65.3, 81.6, 97.7,
                           113.8, 130.2, 146.5, 163.0])
SCALE_FALLING_G = np.array([163.7, 149.4, 134.6, 119.9, 104.2,
                            88.6, 72.7, 56.7, 37.9])

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def torque_from_scale(scale_g):
    return scale_g / 1000.0 * 9.81 * RADIUS_M


def main():
    torque_rising = torque_from_scale(SCALE_RISING_G)
    torque_falling = torque_from_scale(SCALE_FALLING_G)
    kt_rise, intercept_rise = np.polyfit(CURRENT_RISING_A, torque_rising, 1)
    kt_fall, intercept_fall = np.polyfit(CURRENT_FALLING_A, torque_falling, 1)
    kt_average = 0.5 * (kt_rise + kt_fall)

    print("Rising Kt and intercept:", kt_rise, intercept_rise)
    print("Falling Kt and intercept:", kt_fall, intercept_fall)
    print("Average measured Kt [N m/A]:", kt_average)
    print("Peak torque at 20 A [N m]:", kt_average * 20.0)
    print("Current needed for 1.7 N m [A]:", 1.7 / kt_average)

    plt.figure(figsize=(7, 4))
    plt.scatter(CURRENT_RISING_A, torque_rising, label="Rising")
    plt.scatter(CURRENT_FALLING_A, torque_falling, label="Falling")
    plt.plot(CURRENT_RISING_A,
             kt_rise * CURRENT_RISING_A + intercept_rise)
    plt.plot(CURRENT_FALLING_A,
             kt_fall * CURRENT_FALLING_A + intercept_fall)
    plt.xlabel("Commanded q-axis current magnitude [A]")
    plt.ylabel("Torque magnitude [N m]")
    plt.title("Q2.2 Direct torque-constant measurement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

