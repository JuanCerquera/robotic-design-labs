"""Section 3.2 - Baseline breakaway torque (no gear stage).

Open-loop torque ramp (kp_scale = kd_scale = 0, feedforward_torque).
Hold each step while streaming commands; breakaway = the commanded torque
when the shaft first moves a small angle from the start of that ramp.

Before mounting the gear stage. Ratio must be 1.0. Close tview first.

  python section3_2_breakaway.py
  python section3_2_breakaway.py --analyze --no-show
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
import pandas as pd


MOTOR_ID = 1
MAX_TORQUE_NM = 1.63

# Six starts evenly around one turn: 0°, 60°, 120°, 180°, 240°, 300°.
START_OFFSETS_REV = [i / 6.0 for i in range(6)]
DIRECTIONS = [+1.0, -1.0]

TORQUE_STEP_NM = 0.005
TORQUE_MAX_NM = 0.25  # abort if still stuck
STEP_HOLD_S = 0.25    # dwell at each torque before stepping up
SAMPLE_DT_S = 0.01
# Trip only after clear motion in the commanded direction (not encoder creep).
MOVE_POS_REV = 0.01    # 3.6 deg
# After breakaway: hold torque (+ margin) for 1 s and require real spin.
CONFIRM_HOLD_S = 1.0
CONFIRM_TORQUE_MARGIN_NM = 0.01  # a bit above trip so kinetic friction wins
CONFIRM_VEL_REV_S = 0.02         # |mean v| in command direction
CONFIRM_POS_REV = 0.02           # or ≥7.2 deg of travel during the hold

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "section3_2_breakaway.csv"
PLOT_PNG = PLOTS_DIR / "section3_2_breakaway.png"


def ensure_unity_ratio():
    # Do NOT touch pid_position.* here — a later `conf write` (section4_1)
    # would otherwise flash whatever temporary gains we set into RAM.
    cfg_path = DATA_DIR / "section3_2_unity_ratio.cfg"
    cfg_path.write_text(
        "\n".join([
            "conf set motor_position.rotor_to_output_ratio 1.0",
            "conf set servo.default_timeout_s 0.1",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Ensuring ratio=1.0 via {cfg_path.name}...")
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
        raise RuntimeError("moteus_tool --write-config failed. Close tview and retry.")
    print("Ratio = 1.0 (existing Lab 2 position gains used for start-angle moves).")


async def move_to(controller, position_rev, settle_s=1.0):
    start = time.perf_counter()
    while time.perf_counter() - start < settle_s:
        await controller.set_position(
            position=position_rev,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        await asyncio.sleep(SAMPLE_DT_S)
    await controller.set_stop()
    await asyncio.sleep(0.4)


async def apply_torque(controller, feedforward_nm: float):
    result = await controller.set_position(
        position=math.nan,
        velocity=0.0,
        kp_scale=0.0,
        kd_scale=0.0,
        ilimit_scale=0.0,
        feedforward_torque=feedforward_nm,
        maximum_torque=MAX_TORQUE_NM,
        query=True,
    )
    fault = result.values.get(moteus.Register.FAULT, 0)
    if fault:
        raise RuntimeError(f"Fault {fault} during breakaway ramp.")
    return result


async def confirm_spin(controller, feedforward_nm: float, direction: float, duration_s: float):
    """Hold torque for duration_s; confirm directed travel + speed."""
    first = await apply_torque(controller, feedforward_nm)
    pos0 = float(first.values[moteus.Register.POSITION])
    rows = [{
        "time_s": 0.0,
        "velocity_rev_s": float(first.values[moteus.Register.VELOCITY]),
        "position_rev": pos0,
        "torque_nm": float(first.values[moteus.Register.TORQUE]),
    }]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        result = await apply_torque(controller, feedforward_nm)
        rows.append({
            "time_s": time.perf_counter() - t0,
            "velocity_rev_s": float(result.values[moteus.Register.VELOCITY]),
            "position_rev": float(result.values[moteus.Register.POSITION]),
            "torque_nm": float(result.values[moteus.Register.TORQUE]),
        })
        await asyncio.sleep(SAMPLE_DT_S)
    df = pd.DataFrame(rows)
    v = df["velocity_rev_s"].to_numpy(dtype=float)
    peak_v = float(v[abs(v).argmax()]) if len(v) else 0.0
    mean_v = float(v.mean()) if len(v) else 0.0
    delta_pos = float(df["position_rev"].iloc[-1] - pos0)
    # Motion must be in the commanded torque direction.
    directed_v = direction * mean_v
    directed_pos = direction * delta_pos
    confirmed = (
        directed_v >= CONFIRM_VEL_REV_S
        or directed_pos >= CONFIRM_POS_REV
    )
    return {
        "confirm_peak_velocity_rev_s": peak_v,
        "confirm_mean_velocity_rev_s": mean_v,
        "confirm_delta_position_rev": delta_pos,
        "confirm_mean_torque_nm": float(df["torque_nm"].mean()) if len(df) else float("nan"),
        "confirm_hold_torque_nm": abs(feedforward_nm),
        "spin_confirmed": confirmed,
        "confirm_samples": int(len(df)),
    }


async def ramp_until_move(controller, direction: float, start_position: float):
    """Ramp τ_ff slowly; on breakaway, hold torque ~1 s and check speed."""
    # Settle at zero torque so the start pose is well defined.
    t_settle0 = time.perf_counter()
    while time.perf_counter() - t_settle0 < 0.3:
        await apply_torque(controller, 0.0)
        await asyncio.sleep(SAMPLE_DT_S)

    first = await apply_torque(controller, 0.0)
    ramp_start_pos = float(first.values[moteus.Register.POSITION])
    torque = 0.0
    t0 = time.perf_counter()

    while torque <= TORQUE_MAX_NM:
        torque = round(torque + TORQUE_STEP_NM, 6)
        ff = direction * torque
        step_start = time.perf_counter()
        while time.perf_counter() - step_start < STEP_HOLD_S:
            last = await apply_torque(controller, ff)
            vel = float(last.values[moteus.Register.VELOCITY])
            pos = float(last.values[moteus.Register.POSITION])
            delta = pos - ramp_start_pos
            # Require travel in the commanded direction (sign of feedforward).
            if direction * delta >= MOVE_POS_REV:
                tau_reg = float(last.values[moteus.Register.TORQUE])
                hold_ff = direction * (torque + CONFIRM_TORQUE_MARGIN_NM)
                print(
                    f"    tripped at τ_cmd={abs(ff):.4f} N·m  "
                    f"Δpos={delta * 360:+.2f} deg  v={vel:+.3f} — "
                    f"holding τ={abs(hold_ff):.4f} for {CONFIRM_HOLD_S:.1f}s..."
                )
                confirm = await confirm_spin(
                    controller, hold_ff, direction, CONFIRM_HOLD_S
                )
                await controller.set_stop()
                ok = confirm["spin_confirmed"]
                print(
                    f"    confirm: peak_v={confirm['confirm_peak_velocity_rev_s']:+.3f} "
                    f"mean_v={confirm['confirm_mean_velocity_rev_s']:+.3f} rev/s  "
                    f"Δpos={confirm['confirm_delta_position_rev'] * 360:+.1f} deg  "
                    f"mean_τ_reg={confirm['confirm_mean_torque_nm']:+.4f} N·m  "
                    f"{'OK' if ok else 'FAIL — no sustained speed'}"
                )
                return {
                    "direction": direction,
                    "start_position_rev": start_position,
                    "start_position_deg": start_position * 360.0,
                    "ramp_start_position_rev": ramp_start_pos,
                    "breakaway_torque_nm": abs(ff),
                    "signed_torque_nm": ff,
                    "register_torque_nm": tau_reg,
                    "velocity_at_break_rev_s": vel,
                    "position_at_break_rev": pos,
                    "delta_position_rev": delta,
                    "ramp_time_s": time.perf_counter() - t0,
                    "moved": True,
                    **confirm,
                }
            await asyncio.sleep(SAMPLE_DT_S)

    await controller.set_stop()
    return {
        "direction": direction,
        "start_position_rev": start_position,
        "start_position_deg": start_position * 360.0,
        "ramp_start_position_rev": ramp_start_pos,
        "breakaway_torque_nm": float("nan"),
        "signed_torque_nm": float("nan"),
        "register_torque_nm": float("nan"),
        "velocity_at_break_rev_s": float("nan"),
        "position_at_break_rev": float("nan"),
        "delta_position_rev": float("nan"),
        "ramp_time_s": time.perf_counter() - t0,
        "moved": False,
        "confirm_peak_velocity_rev_s": float("nan"),
        "confirm_mean_velocity_rev_s": float("nan"),
        "confirm_delta_position_rev": float("nan"),
        "confirm_mean_torque_nm": float("nan"),
        "confirm_hold_torque_nm": float("nan"),
        "spin_confirmed": False,
        "confirm_samples": 0,
    }


async def run_capture():
    ensure_unity_ratio()
    controller = moteus.Controller(id=MOTOR_ID)
    await controller.set_stop()
    first = await controller.query()
    home = float(first.values[moteus.Register.POSITION])
    print(f"Home position = {home:.6f} rev ({home * 360:.2f} deg)")
    print(
        f"Starts at 0/60/120/180/240/300 deg from home. "
        f"Trip when directed Δpos ≥ {MOVE_POS_REV * 360:.1f} deg; "
        f"confirm {CONFIRM_HOLD_S:.0f}s at trip+{CONFIRM_TORQUE_MARGIN_NM} N·m."
    )

    rows = []
    try:
        for offset in START_OFFSETS_REV:
            target = home + offset
            print(
                f"\nMove to start angle +{offset * 360:.0f} deg "
                f"(abs ≈ {(home + offset) * 360:.1f} deg)..."
            )
            await move_to(controller, target, settle_s=1.5)
            settled = await controller.query()
            start_pos = float(settled.values[moteus.Register.POSITION])

            for direction in DIRECTIONS:
                label = "CW" if direction > 0 else "CCW"
                print(f"  Ramp {label} until motion...")
                row = await ramp_until_move(controller, direction, start_pos)
                rows.append(row)
                if not row["moved"]:
                    print(f"    no motion by {TORQUE_MAX_NM:.3f} N·m — check shaft free?")
                await asyncio.sleep(0.5)
    finally:
        await controller.set_stop()
        print("\nMotor stopped.")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print("Saved:", CSV_PATH)
    return df


def analyze(df: pd.DataFrame | None = None, show: bool = True):
    if df is None:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Missing {CSV_PATH}. Run without --analyze first.")
        df = pd.read_csv(CSV_PATH)

    moved = df[df["moved"] == True]  # noqa: E712
    if moved.empty:
        print("No successful breakaway events recorded.")
        return df

    if "spin_confirmed" in moved.columns:
        n_fail = int((~moved["spin_confirmed"].astype(bool)).sum())
        if n_fail:
            print(f"WARNING: {n_fail} trip(s) failed the {CONFIRM_HOLD_S:.0f}s spin confirm.")
        confirmed = moved[moved["spin_confirmed"] == True]  # noqa: E712
        if not confirmed.empty:
            moved = confirmed

    plot_df = moved.copy()
    plot_df["angle_wrapped_deg"] = plot_df["start_position_deg"] % 360.0
    plot_df["signed_breakaway_nm"] = (
        plot_df["direction"] * plot_df["breakaway_torque_nm"]
    )

    print("\nQ3.2 Breakaway torque [N·m] (commanded feedforward; spin confirmed):")
    for direction, group in plot_df.groupby("direction"):
        label = "+ (same sign as +torque)" if direction > 0 else "−"
        vals = group["breakaway_torque_nm"]
        print(
            f"  direction {label}: "
            f"mean={vals.mean():.4f}, min={vals.min():.4f}, "
            f"max={vals.max():.4f}, n={len(vals)}"
        )
        print("    by start angle [deg] (wrapped 0–360):")
        for _, row in group.sort_values("angle_wrapped_deg").iterrows():
            peak = row.get("confirm_peak_velocity_rev_s", float("nan"))
            print(
                f"      {row['angle_wrapped_deg']:7.2f} deg → "
                f"{row['signed_breakaway_nm']:+.4f} N·m  "
                f"(confirm peak v={peak:+.3f} rev/s)"
            )

    fit_path = DATA_DIR / "section3_1_drag_fit.csv"
    if fit_path.exists():
        coulomb = float(pd.read_csv(fit_path).iloc[0]["coulomb_nm"])
        print(f"\nCompare with Q3.1 Coulomb |τ_c| = {abs(coulomb):.4f} N·m")

    # Signed torque so +/− don’t hide each other when |τ| matches.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for direction, group in plot_df.groupby("direction"):
        label = "+torque" if direction > 0 else "−torque"
        ax.scatter(
            group["angle_wrapped_deg"],
            group["signed_breakaway_nm"],
            marker="o" if direction > 0 else "s",
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Starting position [deg] (wrapped 0–360)")
    ax.set_ylabel("Signed breakaway torque [N·m]")
    ax.set_title("Q3.2 Breakaway torque vs starting angle")
    ax.set_xlim(0, 360)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    # Also keep a copy under lab3_data for convenience.
    fig.savefig(DATA_DIR / PLOT_PNG.name, dpi=150)
    print("Saved plot:", PLOT_PNG)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df


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
