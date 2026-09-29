"""Section 3.2 - Run Park-transform cases, log telemetry, and plot.

Commands run directly (close tview first):
  fixed stator field  -> set_vfoc  (like d pwm <phase> <magnitude>)
  q-axis current      -> set_current(d=0, q=I)
  d-axis current      -> set_current(d=I, q=0)

A d stop is issued between cases so the motor returns to rest.
"""

import asyncio
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
PWM_PHASE_RAD = 0.0
# Keep PWM small: winding R ~ 0.064 ohm, so I ~ V/R. At 1 V that is ~15 A and
# hits the controller current limit (~12 A here) mostly on the d-axis after align.
PWM_VOLTAGE_V = 0.3
CURRENT_A = 1.0
SETTLE_S = 3.0
HOLD_DT_S = 0.02
STOP_TIMEOUT_S = 5.0
STOP_VEL_EPS_REV_S = 0.05
SETTLE_AVG_S = 0.5  # average this many trailing seconds for the summary row
PLOT_SMOOTH_S = 0.15  # rolling-mean window for cleaner plots

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def make_controller():
    query = moteus.QueryResolution()
    query.q_current = moteus.F32
    query.d_current = moteus.F32
    return moteus.Controller(id=MOTOR_ID, query_resolution=query)


def sample_from_result(t_s, label, tview_command, result):
    values = result.values
    return {
        "time_s": t_s,
        "label": label,
        "command": tview_command,
        "position_deg": values[moteus.Register.POSITION] * 360.0,
        "velocity_rev_s": values[moteus.Register.VELOCITY],
        "measured_id_A": values[moteus.Register.D_CURRENT],
        "measured_iq_A": values[moteus.Register.Q_CURRENT],
        "mode": int(values[moteus.Register.MODE]),
        "fault": int(values[moteus.Register.FAULT]),
    }


async def hold_and_log(controller, label, tview_command, send_once, settle_s=SETTLE_S):
    """Keep sending a command, logging registers until settle_s."""
    rows = []
    start = time.perf_counter()
    while True:
        t = time.perf_counter() - start
        result = await send_once()
        rows.append(sample_from_result(t, label, tview_command, result))
        if t >= settle_s:
            return result, rows
        await asyncio.sleep(HOLD_DT_S)


async def full_stop(controller, timeout_s=STOP_TIMEOUT_S):
    """Issue d stop and wait until velocity is near zero."""
    print("d stop — waiting for rest...")
    start = time.perf_counter()
    while True:
        result = await controller.set_stop(query=True)
        velocity = result.values[moteus.Register.VELOCITY]
        if abs(velocity) < STOP_VEL_EPS_REV_S:
            print(f"  stopped (vel={velocity:.4f} rev/s)")
            return result
        if time.perf_counter() - start > timeout_s:
            print(f"  stop timeout (vel={velocity:.4f} rev/s) — continuing")
            return result
        await asyncio.sleep(HOLD_DT_S)


METRIC_COLS = [
    "position_deg",
    "velocity_rev_s",
    "measured_id_A",
    "measured_iq_A",
]


def average_row_from_samples(samples, window="settle"):
    """Average registers over the full run or the trailing settle window."""
    frame = pd.DataFrame(samples)
    if window == "settle":
        t_end = frame["time_s"].iloc[-1]
        used = frame[frame["time_s"] >= t_end - SETTLE_AVG_S]
        suffix = f"last_{SETTLE_AVG_S:.1f}s"
    else:
        used = frame
        suffix = "full_run"

    row = {
        "label": frame["label"].iloc[-1],
        "command": frame["command"].iloc[-1],
        "avg_window": suffix,
        "n_samples": int(len(used)),
    }
    for col in METRIC_COLS:
        row[col] = float(used[col].mean())
        row[f"{col}_std"] = float(used[col].std(ddof=1)) if len(used) > 1 else 0.0
    row["id_peak_A"] = float(frame["measured_id_A"].abs().max())
    row["iq_peak_A"] = float(frame["measured_iq_A"].abs().max())
    return row


def print_averages(avg_table: pd.DataFrame):
    print("\n=== Averages ===")
    show = avg_table[
        [
            "command",
            "avg_window",
            "position_deg",
            "velocity_rev_s",
            "measured_id_A",
            "measured_iq_A",
        ]
    ].copy()
    show = show.rename(
        columns={
            "position_deg": "pos_deg_avg",
            "velocity_rev_s": "vel_avg",
            "measured_id_A": "id_avg",
            "measured_iq_A": "iq_avg",
        }
    )
    print(show.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))


def plot_timeseries(timeseries: pd.DataFrame, averages: pd.DataFrame):
    commands = list(timeseries["command"].unique())
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    smooth_n = max(1, int(round(PLOT_SMOOTH_S / HOLD_DT_S)))

    series = [
        ("position_deg", "Position [deg]"),
        ("velocity_rev_s", "Velocity [rev/s]"),
        ("measured_id_A", "Measured id [A]"),
        ("measured_iq_A", "Measured iq [A]"),
    ]

    for ax, (col, ylabel) in zip(axes, series):
        for command in commands:
            chunk = timeseries[timeseries["command"] == command].copy()
            ax.plot(chunk["time_s"], chunk[col], alpha=0.2, linewidth=1)
            smoothed = chunk[col].rolling(smooth_n, min_periods=1, center=True).mean()
            ax.plot(chunk["time_s"], smoothed, label=command, linewidth=2)

            settle = averages[
                (averages["command"] == command)
                & (averages["avg_window"].str.startswith("last_"))
            ]
            if not settle.empty:
                avg_val = float(settle.iloc[0][col])
                ax.axhline(
                    avg_val,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{command} avg={avg_val:.3f}",
                )

        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time within command [s]")
    fig.suptitle("Q3.3 traces + settle averages (dashed)")
    fig.tight_layout()
    plt.show()


def plot_average_bars(averages: pd.DataFrame):
    settle = averages[averages["avg_window"].str.startswith("last_")].reset_index(drop=True)
    full = averages[averages["avg_window"] == "full_run"].reset_index(drop=True)
    x = np.arange(len(settle))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(x - 1.5 * width, settle["measured_id_A"], width, label="id settle-avg")
    axes[0].bar(x - 0.5 * width, settle["measured_iq_A"], width, label="iq settle-avg")
    axes[0].bar(x + 0.5 * width, full["measured_id_A"], width, label="id full-avg")
    axes[0].bar(x + 1.5 * width, full["measured_iq_A"], width, label="iq full-avg")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settle["command"], rotation=15, ha="right")
    axes[0].set_ylabel("Current [A]")
    axes[0].set_title("Average id / iq")
    axes[0].grid(True, axis="y")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 1.5 * width, settle["position_deg"], width, label="pos settle-avg")
    axes[1].bar(x - 0.5 * width, settle["velocity_rev_s"], width, label="vel settle-avg")
    axes[1].bar(x + 0.5 * width, full["position_deg"], width, label="pos full-avg")
    axes[1].bar(x + 1.5 * width, full["velocity_rev_s"], width, label="vel full-avg")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settle["command"], rotation=15, ha="right")
    axes[1].set_ylabel("Position [deg] / Velocity [rev/s]")
    axes[1].set_title("Average position / velocity")
    axes[1].grid(True, axis="y")
    axes[1].legend(fontsize=8)

    fig.suptitle("Q3.3 Averaged Park-transform measurements")
    fig.tight_layout()
    plt.show()


async def main():
    controller = make_controller()
    average_rows = []
    timeseries_rows = []

    cases = [
        (
            "fixed stator field",
            f"d pwm {PWM_PHASE_RAD} {PWM_VOLTAGE_V}",
            lambda: controller.set_vfoc(
                theta=float(PWM_PHASE_RAD),
                voltage=float(PWM_VOLTAGE_V),
                query=True,
            ),
        ),
        (
            "q-axis current",
            f"d dq 0 {CURRENT_A}",
            lambda: controller.set_current(
                d_A=0.0,
                q_A=float(CURRENT_A),
                query=True,
            ),
        ),
        (
            "d-axis current",
            f"d dq {CURRENT_A} 0",
            lambda: controller.set_current(
                d_A=float(CURRENT_A),
                q_A=0.0,
                query=True,
            ),
        ),
    ]

    try:
        await full_stop(controller)

        for label, tview_command, send_once in cases:
            print(f"\nRunning: {tview_command}")
            _, samples = await hold_and_log(
                controller, label, tview_command, send_once
            )
            timeseries_rows.extend(samples)

            settle_avg = average_row_from_samples(samples, window="settle")
            full_avg = average_row_from_samples(samples, window="full")
            average_rows.extend([settle_avg, full_avg])

            print(
                f"  settle-avg ({SETTLE_AVG_S:.1f}s): "
                f"pos={settle_avg['position_deg']:.2f} deg  "
                f"vel={settle_avg['velocity_rev_s']:.3f} rev/s  "
                f"id={settle_avg['measured_id_A']:.3f} A  "
                f"iq={settle_avg['measured_iq_A']:.3f} A"
            )
            print(
                f"  full-run avg: "
                f"pos={full_avg['position_deg']:.2f} deg  "
                f"vel={full_avg['velocity_rev_s']:.3f} rev/s  "
                f"id={full_avg['measured_id_A']:.3f} A  "
                f"iq={full_avg['measured_iq_A']:.3f} A"
            )
            print(
                f"  peaks |id|={settle_avg['id_peak_A']:.2f} A  "
                f"|iq|={settle_avg['iq_peak_A']:.2f} A"
            )

            await full_stop(controller)
    finally:
        await controller.set_stop()
        print("\nMotor stopped")

    averages = pd.DataFrame(average_rows)
    timeseries = pd.DataFrame(timeseries_rows)
    settle_summary = averages[averages["avg_window"].str.startswith("last_")].copy()

    summary_path = DATA_DIR / "section3_2_park_transform.csv"
    averages_path = DATA_DIR / "section3_2_park_transform_averages.csv"
    timeseries_path = DATA_DIR / "section3_2_park_transform_timeseries.csv"
    settle_summary.to_csv(summary_path, index=False)
    averages.to_csv(averages_path, index=False)
    timeseries.to_csv(timeseries_path, index=False)

    print_averages(averages)
    print("Saved:", summary_path)
    print("Saved:", averages_path)
    print("Saved:", timeseries_path)

    plot_timeseries(timeseries, averages)
    plot_average_bars(averages)


if __name__ == "__main__":
    asyncio.run(main())
