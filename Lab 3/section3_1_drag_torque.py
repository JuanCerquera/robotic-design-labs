"""Section 3.1 - Baseline motor drag torque (no gear stage).

Command constant velocity with position=NaN. Step through speeds in both
directions, hold each ~3 s, average torque over the steady part.

Before mounting the gear stage. Ratio must be 1.0. Close tview first.

  python section3_1_drag_torque.py
  python section3_1_drag_torque.py --analyze --no-show
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
MAX_TORQUE_NM = 1.63  # motor peak; need headroom while accelerating to 10 rev/s

# Speeds at the motor shaft [rev/s]. Run all + then all − (no flip each step).
SPEEDS_REV_S = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
HOLD_S = 4.0
SETTLE_S = 1.5
SAMPLE_DT_S = 0.01
ACCEL_LIMIT = 5.0  # gentler than 20 — less bang into each new speed

# Post-cal / moteus-c1 kits often ship with max_velocity=10. With accel_limit set,
# that value also becomes the trajectory velocity_limit — and any command gap
# longer than default_timeout_s stops the drive. Raise both for this sweep.
MAX_VELOCITY_REV_S = 50.0
COMMAND_TIMEOUT_S = 1.0

# Lab 2 §6.3 / §6.5 final tuned gains (edit if yours differ).
# Note: config_after_cal.cfg used kp=1, ki=2.14, kd=0.032, ilimit=0 instead.
KP = 5.0
KD = 0.03
KI = 7.0
ILIMIT = 1.0

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "section3_1_drag_raw.csv"
SUMMARY_CSV = DATA_DIR / "section3_1_drag_summary.csv"
FIT_CSV = DATA_DIR / "section3_1_drag_fit.csv"
PLOT_PNG = PLOTS_DIR / "section3_1_drag.png"


def release_transport():
    """Drop the COM port so a moteus_tool subprocess can open it."""
    try:
        import moteus.transport_factory as tf
        if tf.GLOBAL_TRANSPORT is not None:
            tf.GLOBAL_TRANSPORT.close()
            tf.GLOBAL_TRANSPORT = None
    except Exception as exc:
        print(f"(transport release: {exc})")


def write_config(lines: list[str], label: str):
    cfg_path = DATA_DIR / f"section3_1_{label}.cfg"
    cfg_path.write_text("\n".join(lines + [""]), encoding="utf-8")
    print(f"Writing {cfg_path.name}...")
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
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "moteus_tool --write-config failed. Close tview and retry.\n" + err
        )


def setup_velocity_mode():
    """Unity ratio + Lab 2 gains. Raise max_velocity past the 10 rev/s kit default."""
    write_config(
        [
            "conf set motor_position.rotor_to_output_ratio 1.0",
            f"conf set servo.default_timeout_s {COMMAND_TIMEOUT_S}",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "conf set servo.max_position_slip nan",
            f"conf set servo.max_velocity {MAX_VELOCITY_REV_S}",
            "conf set servo.max_velocity_derate 2",
            f"conf set servo.pid_position.kp {KP}",
            f"conf set servo.pid_position.kd {KD}",
            f"conf set servo.pid_position.ki {KI}",
            f"conf set servo.pid_position.ilimit {ILIMIT}",
        ],
        "velocity_setup",
    )
    print(
        f"Lab 2 gains (RAM): kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}; "
        f"max_velocity={MAX_VELOCITY_REV_S}, timeout={COMMAND_TIMEOUT_S}s."
    )


def restore_kit_limits():
    """Put back typical post-cal safety limits (RAM)."""
    write_config(
        [
            "conf set servo.max_velocity 10",
            "conf set servo.max_velocity_derate 2",
            "conf set servo.default_timeout_s 0.1",
        ],
        "restore_limits",
    )
    print("Restored max_velocity=10, timeout=0.1s (RAM).")


async def hold_velocity(controller, velocity_rev_s, duration_s, phase):
    rows = []
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration_s:
            break
        result = await controller.set_position(
            position=math.nan,
            velocity=velocity_rev_s,
            velocity_limit=MAX_VELOCITY_REV_S,
            accel_limit=ACCEL_LIMIT,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        fault = result.values.get(moteus.Register.FAULT, 0)
        if fault:
            hint = ""
            if int(fault) == 39:
                hint = " (fault 39 = position limit — clear servopos min/max)"
            raise RuntimeError(
                f"Fault {fault} during '{phase}' at t={elapsed:.2f}s{hint}. "
                "Run d stop in tview and retry."
            )
        rows.append({
            "phase": phase,
            "cmd_velocity_rev_s": velocity_rev_s,
            "time_s": elapsed,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
            "torque_nm": result.values[moteus.Register.TORQUE],
        })
        await asyncio.sleep(SAMPLE_DT_S)
    return pd.DataFrame(rows)


async def coast_stop(controller, duration_s=0.8):
    """Ramp toward zero velocity, then set_stop — quieter than hard reversals."""
    await hold_velocity(controller, 0.0, duration_s, phase="coast")
    await controller.set_stop()
    await asyncio.sleep(0.3)


def summarize_holds(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, group in raw.groupby("phase", sort=False):
        steady = group[group["time_s"] >= SETTLE_S]
        if steady.empty:
            steady = group
        cmd = float(group["cmd_velocity_rev_s"].iloc[0])
        rows.append({
            "phase": phase,
            "cmd_velocity_rev_s": cmd,
            "velocity_mean_rev_s": float(steady["velocity_rev_s"].mean()),
            "torque_mean_nm": float(steady["torque_nm"].mean()),
            "torque_std_nm": float(steady["torque_nm"].std(ddof=0)),
            "n_samples": int(len(steady)),
        })
    return pd.DataFrame(rows)


def fit_coulomb_viscous(summary: pd.DataFrame):
    """τ ≈ τ_c * sign(v) + b * v  (fit on measured velocity and mean torque)."""
    v = summary["velocity_mean_rev_s"].to_numpy(dtype=float)
    tau = summary["torque_mean_nm"].to_numpy(dtype=float)
    # Linear model: tau = b * v + tau_c * sign(v)
    # Build design matrix [v, sign(v)]
    sign_v = np.sign(v)
    sign_v[sign_v == 0] = 1.0
    A = np.column_stack([v, sign_v])
    coeffs, *_ = np.linalg.lstsq(A, tau, rcond=None)
    b, tau_c = float(coeffs[0]), float(coeffs[1])
    tau_hat = A @ coeffs
    resid = tau - tau_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "viscous_nm_per_rev_s": b,
        "coulomb_nm": tau_c,
        "rmse_nm": rmse,
    }


def plot_and_report(summary: pd.DataFrame, fit: dict, show: bool):
    v = summary["velocity_mean_rev_s"].to_numpy(dtype=float)
    tau = summary["torque_mean_nm"].to_numpy(dtype=float)
    v_line = np.linspace(v.min(), v.max(), 200)
    sign = np.sign(v_line)
    sign[sign == 0] = 1.0
    tau_fit = (
        fit["viscous_nm_per_rev_s"] * v_line
        + fit["coulomb_nm"] * sign
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(v, tau, label="Measured (steady avg)", zorder=3)
    ax.plot(v_line, tau_fit, label=(
        f"Fit: τ = {fit['coulomb_nm']:.4f}·sgn(v) "
        f"+ {fit['viscous_nm_per_rev_s']:.5f}·v"
    ))
    ax.set_xlabel("Velocity [rev/s]")
    ax.set_ylabel("Torque [N·m]")
    ax.set_title("Q3.1 Baseline motor drag torque")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    print("Saved plot:", PLOT_PNG)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\nQ3.1 fit (motor shaft):")
    print(f"  Coulomb τ_c = {fit['coulomb_nm']:.5f} N·m")
    print(f"  Viscous b   = {fit['viscous_nm_per_rev_s']:.6f} N·m /(rev/s)")
    print(f"  RMSE        = {fit['rmse_nm']:.5f} N·m")


async def run_capture():
    setup_velocity_mode()
    controller = moteus.Controller(id=MOTOR_ID)
    all_rows = []
    # One direction at a time — avoids slamming through zero every step.
    speeds = list(SPEEDS_REV_S) + [-s for s in SPEEDS_REV_S]

    try:
        await controller.set_stop()
        for i, vel in enumerate(speeds):
            if i == len(SPEEDS_REV_S):
                print("Coasting to stop before negative speeds...")
                await coast_stop(controller)

            phase = f"v{vel:+.1f}"
            print(f"[{i + 1}/{len(speeds)}] Hold {vel:+.1f} rev/s for {HOLD_S:.1f} s...")
            df = await hold_velocity(controller, vel, HOLD_S, phase)
            all_rows.append(df)
            steady = df[df["time_s"] >= SETTLE_S]
            mean_tau = float(steady["torque_nm"].mean())
            std_tau = float(steady["torque_nm"].std(ddof=0))
            mean_v = float(steady["velocity_rev_s"].mean())
            print(
                f"    steady: cmd={vel:+.1f} → v={mean_v:+.3f} rev/s, "
                f"τ={mean_tau:.5f}±{std_tau:.5f} N·m"
            )
            if abs(mean_v) < 0.2 * abs(vel) and abs(vel) > 0.5:
                print(
                    "    WARNING: measured speed << command. "
                    "Check max_velocity / fault / timeout."
                )
            if std_tau > 0.05:
                print("    (high τ scatter — check for mechanical rub / resonance)")
            # Checkpoint without leaving a long command gap: append this hold only.
            header = not RAW_CSV.exists() or i == 0
            df.to_csv(RAW_CSV, mode="w" if i == 0 else "a", header=header, index=False)
    finally:
        try:
            await controller.set_stop()
        except Exception:
            pass
        print("Motor stopped.")
        # Controller holds COM7; release before moteus_tool restore.
        release_transport()
        try:
            restore_kit_limits()
        except Exception as exc:
            print(f"Could not restore kit limits: {exc}")
            print("  In tview: conf set servo.max_velocity 10")
            print("            conf set servo.default_timeout_s 0.1")
        if all_rows:
            raw = pd.concat(all_rows, ignore_index=True)
            raw.to_csv(RAW_CSV, index=False)
            print("Saved:", RAW_CSV)
        else:
            raw = pd.DataFrame()

    return raw


def analyze(raw: pd.DataFrame | None = None, show: bool = True):
    if raw is None:
        if not RAW_CSV.exists():
            raise FileNotFoundError(f"Missing {RAW_CSV}. Run without --analyze first.")
        raw = pd.read_csv(RAW_CSV)
    summary = summarize_holds(raw)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("Saved:", SUMMARY_CSV)
    fit = fit_coulomb_viscous(summary)
    pd.DataFrame([fit]).to_csv(FIT_CSV, index=False)
    print("Saved:", FIT_CSV)
    plot_and_report(summary, fit, show=show)
    return summary, fit


async def main_async(args):
    if args.analyze:
        analyze(show=not args.no_show)
        return
    raw = await run_capture()
    analyze(raw, show=not args.no_show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analyze", action="store_true",
        help="Re-fit / re-plot from saved CSV (no motor).",
    )
    parser.add_argument("--no-show", action="store_true", help="Save plots without plt.show().")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
