"""Section 5.3 - Backdrive torque at the output (motor drive off).

Type `d stop` (or let this script stop the drive). Point the lever arm
straight up so its weight adds no torque. Zero the scale in the push
orientation. Press the scale against the arm tip, increase slowly, record
the highest reading before the output turns. Five pushes each direction.

  python section5_3_backdrive.py
  python section5_3_backdrive.py --analyze --no-show
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
LEVER_RADIUS_M = 0.100
G = 9.81
N_PUSHES = 5

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "section5_3_backdrive.csv"
SUMMARY_CSV = DATA_DIR / "section5_3_backdrive_summary.csv"
PLOT_PNG = PLOTS_DIR / "section5_3_backdrive.png"


def parse_scale_reading(text: str) -> float:
    s = text.strip().lower().replace(" ", "")
    if s.endswith("kg"):
        return float(s[:-2]) * 1000.0
    if s.endswith("g"):
        return float(s[:-1])
    value = float(s)
    if value < 5.0:
        return value * 1000.0
    return value


def torque_from_scale_g(scale_g: float) -> float:
    return (scale_g / 1000.0) * G * LEVER_RADIUS_M


async def ensure_stopped():
    controller = moteus.Controller(id=MOTOR_ID)
    await controller.set_stop()
    print("Issued set_stop() — drive off. (Also type `d stop` in tview if open.)")


def collect_pushes(direction: str) -> list[dict]:
    rows = []
    print(f"\n=== {direction} pushes ({N_PUSHES}) ===")
    print("Arm straight up. Zero the scale in the push orientation.")
    print("Press the platform against the tip, normal to the arm. Slow push.")
    for i in range(1, N_PUSHES + 1):
        while True:
            raw = input(f"  {direction} push {i}/{N_PUSHES} peak scale [g or kg]: ").strip()
            try:
                scale_g = parse_scale_reading(raw)
                break
            except ValueError:
                print("  Could not parse — try e.g. 850 or 0.85kg")
        tau = torque_from_scale_g(scale_g)
        print(f"    → τ_backdrive = {tau:.4f} N·m")
        rows.append({
            "direction": direction,
            "push": i,
            "scale_g": scale_g,
            "backdrive_torque_nm": tau,
            "lever_radius_m": LEVER_RADIUS_M,
        })
    return rows


async def run_capture():
    await ensure_stopped()
    print("\nQ5.3 backdrive measurement")
    print(f"Lever radius = {LEVER_RADIUS_M * 1000:.0f} mm")
    input("Press Enter to record +direction pushes...")
    rows = collect_pushes("+")
    input("\nPress Enter to record −direction pushes...")
    rows.extend(collect_pushes("-"))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print("Saved:", CSV_PATH)
    return df


def analyze(df: pd.DataFrame | None = None, show: bool = True):
    if df is None:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Missing {CSV_PATH}. Run without --analyze first.")
        df = pd.read_csv(CSV_PATH)

    summary_rows = []
    print("\nQ5.3 Backdrive torque at output:")
    for direction, group in df.groupby("direction"):
        vals = group["backdrive_torque_nm"]
        summary_rows.append({
            "direction": direction,
            "mean_nm": float(vals.mean()),
            "std_nm": float(vals.std(ddof=0)),
            "min_nm": float(vals.min()),
            "max_nm": float(vals.max()),
            "n": int(len(vals)),
        })
        print(
            f"  {direction}: mean={vals.mean():.4f} N·m, "
            f"std={vals.std(ddof=0):.4f}, "
            f"range=[{vals.min():.4f}, {vals.max():.4f}]"
        )

    # Prediction from §3.2 breakaway × gear ratio (ideal, no stage friction).
    br_path = DATA_DIR / "section3_2_breakaway.csv"
    if br_path.exists():
        br = pd.read_csv(br_path)
        moved = br[br["moved"] == True]  # noqa: E712
        if not moved.empty:
            mean_ba = float(moved["breakaway_torque_nm"].mean())
            pred = mean_ba * 6.0
            print(f"\nP5.3-style prediction from Q3.2: "
                  f"τ_ba_motor×6 = {mean_ba:.4f}×6 = {pred:.4f} N·m")
            print("Difference vs measured ≈ stage friction + efficiency losses.")
            for row in summary_rows:
                row["prediction_from_q32_nm"] = pred
                row["measured_minus_prediction_nm"] = row["mean_nm"] - pred

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("Saved:", SUMMARY_CSV)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = []
    data = []
    labels = []
    for i, (direction, group) in enumerate(df.groupby("direction")):
        positions.append(i)
        data.append(group["backdrive_torque_nm"].to_numpy())
        labels.append(direction)
    ax.boxplot(data, positions=positions, widths=0.5)
    for pos, vals in zip(positions, data):
        ax.scatter(np.full_like(vals, pos, dtype=float), vals, alpha=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Backdrive torque [N·m]")
    ax.set_title("Q5.3 Backdrive torque at output")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    print("Saved plot:", PLOT_PNG)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df, summary


async def main_async(args):
    if args.analyze:
        analyze(show=not args.no_show)
        return
    df = await run_capture()
    analyze(df, show=not args.no_show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
