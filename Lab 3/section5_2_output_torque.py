"""Section 5.2 - Output torque vs scale (lever arm), rising and falling sweeps.

Bolt the lever arm to the carrier. Rest the 100 mm point on the scale with
the arm horizontal. Zero the scale with the arm resting on it.

Commands torque the same way as §3.2 (kp_scale=kd_scale=0, feedforward_torque).
Torque is kept streaming while you type the scale reading (avoids command timeout).

With the 100 mm arm, the scale reads ~1.02 kg at 1 N·m.

Requires gear ratio set. Close tview. Wear safety glasses.

  python section5_2_output_torque.py
  python section5_2_output_torque.py --analyze --no-show
"""

from __future__ import annotations

import argparse
import asyncio
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
# Cap near the test range — not the 9.8 peak — so a runaway is limited.
MAX_TORQUE_NM = 1.2
GEAR_RATIO = 6.0
LEVER_RADIUS_M = 0.100
G = 9.81

# Positive command must press the arm onto the scale. Flip if your arm is mirrored.
TORQUE_SIGN = +1.0

# Output torques [N·m] — rising then falling, one direction only within each sweep.
TORQUE_STEPS_NM = [round(x, 1) for x in np.arange(0.2, 1.01, 0.1)]
HOLD_S = 2.0
SAMPLE_DT_S = 0.02
MAX_VEL_ABORT_REV_S = 0.3
MAX_POS_ABORT_REV = 0.05

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "section5_2_output_torque.csv"
FIT_CSV = DATA_DIR / "section5_2_efficiency.csv"
PLOT_PNG = PLOTS_DIR / "section5_2_output_torque.png"


def ensure_gear_ratio():
    cfg_path = DATA_DIR / "section5_2_ratio.cfg"
    cfg_path.write_text(
        "\n".join([
            f"conf set motor_position.rotor_to_output_ratio {1.0 / GEAR_RATIO:.8f}",
            "conf set servo.default_timeout_s 0.1",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Ensuring gear ratio via {cfg_path.name}...")
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
        raise RuntimeError("moteus_tool hung. Close tview and retry.") from exc
    if result.returncode != 0:
        raise RuntimeError("moteus_tool --write-config failed.")
    print("Ratio OK (output units).")


def parse_scale_reading(text: str) -> float:
    """Accept grams ('123.4', '123.4g') or kilograms ('0.123', '0.123kg')."""
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


async def stream_torque(controller, torque_nm: float, start_pos: float, stop_event: asyncio.Event):
    """Keep commanding torque until stop_event is set. Abort on runaway motion."""
    last = None
    while not stop_event.is_set():
        last = await controller.set_position(
            position=math.nan,
            velocity=0.0,
            kp_scale=0.0,
            kd_scale=0.0,
            ilimit_scale=0.0,
            feedforward_torque=torque_nm,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        fault = last.values.get(moteus.Register.FAULT, 0)
        if fault:
            raise RuntimeError(f"Fault {fault} while holding {torque_nm} N·m.")
        vel = float(last.values[moteus.Register.VELOCITY])
        pos = float(last.values[moteus.Register.POSITION])
        if abs(vel) > MAX_VEL_ABORT_REV_S or abs(pos - start_pos) > MAX_POS_ABORT_REV:
            await controller.set_stop()
            raise RuntimeError(
                f"Abort: arm moved (v={vel:.3f} rev/s, "
                f"Δpos={pos - start_pos:+.4f} rev). "
                "Check TORQUE_SIGN and that the arm is on the scale."
            )
        await asyncio.sleep(SAMPLE_DT_S)
    return last


async def hold_torque_and_read_scale(controller, torque_nm: float, start_pos: float, prompt: str):
    """Stream torque through settle + scale prompt; return (last_result, scale_g)."""
    stop_event = asyncio.Event()
    task = asyncio.create_task(stream_torque(controller, torque_nm, start_pos, stop_event))
    try:
        await asyncio.sleep(HOLD_S)
        while True:
            raw = (await asyncio.to_thread(input, prompt)).strip()
            try:
                scale_g = parse_scale_reading(raw)
                break
            except ValueError:
                print("  Could not parse — try e.g. 510 or 0.510kg")
        stop_event.set()
        last = await task
        return last, scale_g
    except Exception:
        stop_event.set()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        raise


async def run_sweep(controller, torques, sweep_name: str, start_pos: float):
    rows = []
    print(f"\n=== {sweep_name} sweep ===")
    print("Enter scale reading when prompted (grams, or append kg).")
    print("Torque stays commanded while you type (timeout-safe).")
    for tau_cmd in torques:
        signed = TORQUE_SIGN * tau_cmd
        print(f"\nCommanding {signed:+.1f} N·m at output "
              f"(motor ≈ {signed / GEAR_RATIO:.3f} N·m)...")
        result, scale_g = await hold_torque_and_read_scale(
            controller,
            signed,
            start_pos,
            f"  Scale reading for {tau_cmd:.1f} N·m [g or kg]: ",
        )
        tau_reg = float(result.values[moteus.Register.TORQUE])
        pos = float(result.values[moteus.Register.POSITION])
        tau_meas = torque_from_scale_g(scale_g)
        print(f"  scale={scale_g:.1f} g → τ_meas={tau_meas:.4f} N·m "
              f"(register τ={tau_reg:.4f} N·m)")
        rows.append({
            "sweep": sweep_name,
            "command_torque_nm": tau_cmd,
            "signed_command_torque_nm": signed,
            "register_torque_nm": tau_reg,
            "scale_g": scale_g,
            "measured_torque_nm": tau_meas,
            "position_rev": pos,
            "lever_radius_m": LEVER_RADIUS_M,
            "torque_sign": TORQUE_SIGN,
        })
    return rows


async def run_capture():
    ensure_gear_ratio()
    print("Setup check:")
    print(f"  Lever radius = {LEVER_RADIUS_M * 1000:.0f} mm")
    print(f"  At 1 N·m, scale ≈ {1.0 / (G * LEVER_RADIUS_M):.2f} kg")
    print(f"  TORQUE_SIGN = {TORQUE_SIGN:+.0f}  "
          "(flip at top of script if positive lifts the arm off the scale)")
    print(f"  MAX_TORQUE_NM = {MAX_TORQUE_NM} (test cap, not 9.8 peak)")
    print("  Zero the scale with the arm resting on it, arm horizontal.")
    await asyncio.to_thread(input, "Press Enter when ready to start the rising sweep...")

    controller = moteus.Controller(id=MOTOR_ID)
    await controller.set_stop()
    first = await controller.query()
    start_pos = float(first.values[moteus.Register.POSITION])
    rows = []
    try:
        rows.extend(await run_sweep(controller, TORQUE_STEPS_NM, "rising", start_pos))
        print("\nNow the falling sweep — do not reverse mid-sweep.")
        # Keep last rising torque streaming? Better hold last rising value until falling starts.
        last_tau = TORQUE_SIGN * TORQUE_STEPS_NM[-1]
        stop_event = asyncio.Event()
        hold_task = asyncio.create_task(
            stream_torque(controller, last_tau, start_pos, stop_event)
        )
        try:
            await asyncio.to_thread(input, "Press Enter to start falling...")
        finally:
            stop_event.set()
            try:
                await hold_task
            except Exception:
                pass
        rows.extend(
            await run_sweep(
                controller, list(reversed(TORQUE_STEPS_NM)), "falling", start_pos
            )
        )
    finally:
        await controller.set_stop()
        print("Motor stopped.")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print("Saved:", CSV_PATH)
    return df


def analyze(df: pd.DataFrame | None = None, show: bool = True):
    if df is None:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Missing {CSV_PATH}. Run without --analyze first.")
        df = pd.read_csv(CSV_PATH)

    results = []
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for sweep, color in [("rising", "C0"), ("falling", "C1")]:
        sub = df[df["sweep"] == sweep]
        if sub.empty:
            continue
        x = sub["command_torque_nm"].to_numpy(dtype=float)
        y = sub["measured_torque_nm"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.scatter(x, y, label=f"{sweep} data", color=color)
        ax.plot(
            x_line, slope * x_line + intercept, color=color,
            label=f"{sweep} fit: slope={slope:.3f}",
        )
        results.append({
            "sweep": sweep,
            "slope": float(slope),
            "intercept_nm": float(intercept),
        })
        print(f"{sweep}: slope={slope:.4f}, intercept={intercept:.4f} N·m")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal slope=1")
    ax.set_xlabel("Commanded output torque [N·m]")
    ax.set_ylabel("Measured output torque [N·m]")
    ax.set_title("Q5.2 Output torque: rising vs falling")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    print("Saved plot:", PLOT_PNG)
    if show:
        plt.show()
    else:
        plt.close(fig)

    by_sweep = {r["sweep"]: r for r in results}
    if "rising" in by_sweep and "falling" in by_sweep:
        eta_forward = by_sweep["rising"]["slope"]
        eta_backdrive = (
            1.0 / by_sweep["falling"]["slope"]
            if by_sweep["falling"]["slope"] else float("nan")
        )
        print("\nEfficiency from fitted slopes:")
        print(f"  Rising (motor → output):  η = slope_rise = {eta_forward:.4f}")
        print(f"  Falling (output → motor): η = 1/slope_fall = {eta_backdrive:.4f}")
        print("  Rising sweep: motor drives the output (gears wind up into the scale).")
        print("  Falling sweep: output drives the motor (scale + compliance push back).")
        results.append({
            "sweep": "efficiency",
            "eta_motor_drives_output": eta_forward,
            "eta_output_drives_motor": eta_backdrive,
            "slope_rising": by_sweep["rising"]["slope"],
            "slope_falling": by_sweep["falling"]["slope"],
            "slope_ratio_rise_over_fall": (
                by_sweep["rising"]["slope"] / by_sweep["falling"]["slope"]
                if by_sweep["falling"]["slope"] else float("nan")
            ),
        })
        print(
            "\nKt note: moteus commanded torque assumes a Kt. "
            "Relative slope ratio is less sensitive to Kt than absolute η."
        )

    pd.DataFrame(results).to_csv(FIT_CSV, index=False)
    print("Saved:", FIT_CSV)
    return df, results


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
