"""Section 6.3 - Enable integral action; step + windup recovery tests.

Uses critically-damped kp/kd from section6_2 plus your tuned ki/ilimit.

Automated sequence (no hand disturbance):
  1. Step with chosen ilimit
  2. Offset hold then return (chosen ilimit) — windup recovery
  3. Offset hold then return (max ilimit) — compare windup
  4. Final step with chosen ilimit

Close tview before running.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import moteus
import numpy as np
import pandas as pd


MOTOR_ID = 1
MAX_TORQUE_NM = 1.63

# Tuned gains (edit if needed).
KP = 5
KD = 0.03
KI = 7
ILIMIT = 1

HOLD_OFFSET_S = 5.0
RELEASE_SETTLE_S = 8.0
HOME_SETTLE_S = 2.0
PRE_STEP_S = 0.5
STEP_REV = 0.05
OFFSET_REV = 20.0 / 360.0  # ~20 deg
STEP_RUN_S = 3.0
SAMPLE_DT_S = 0.005

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
DATA_DIR.mkdir(exist_ok=True)


def print_metrics_suggestions():
    path = DATA_DIR / "section6_2_metrics.csv"
    if not path.exists():
        print("No section6_2_metrics.csv — using gains set at top of script.")
        return
    row = pd.read_csv(path).iloc[0]
    print(f"section6_2 suggestions ({path.name}):")
    print(
        f"  kp={row.get('kp')}, kd={row.get('kd')}, "
        f"ki≈{row.get('ki_suggest')}, ilimit≈{row.get('ilimit_suggest_nm')}"
    )
    print("Using KP/KD/KI/ILIMIT from top of this script.")


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section6_3_gains.cfg"
    cfg_path.write_text(
        "\n".join([
            f"conf set servo.pid_position.kp {kp}",
            f"conf set servo.pid_position.kd {kd}",
            f"conf set servo.pid_position.ki {ki}",
            f"conf set servo.pid_position.ilimit {ilimit}",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Writing gains via moteus_tool (kp={kp}, kd={kd}, ki={ki}, ilimit={ilimit})...")
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
        raise RuntimeError("moteus_tool --write-config failed. Close tview and retry.")
    print("Gains set (RAM).")


def ilimit_scale_for(desired_ilimit_nm):
    if MAX_TORQUE_NM <= 0:
        raise RuntimeError("MAX_TORQUE_NM must be > 0")
    return float(desired_ilimit_nm) / float(MAX_TORQUE_NM)


async def clear_timeout(controller):
    await controller.set_stop()


async def set_hold(controller, hold_position, ilimit_scale=1.0):
    return await controller.set_position(
        position=hold_position,
        velocity=0.0,
        kp_scale=1.0,
        kd_scale=1.0,
        ilimit_scale=ilimit_scale,
        maximum_torque=MAX_TORQUE_NM,
        query=True,
    )


async def command_and_log(controller, command_rev, duration_s, phase, ilimit_scale=1.0):
    rows = []
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration_s:
            break
        result = await set_hold(controller, command_rev, ilimit_scale=ilimit_scale)
        fault = result.values.get(moteus.Register.FAULT, 0)
        if fault:
            raise RuntimeError(
                f"Fault {fault} during '{phase}' at t={elapsed:.2f}s. "
                "Run d stop in tview, clear servopos limits if fault 39, retry."
            )
        rows.append({
            "phase": phase,
            "time_s": elapsed,
            "command_rev": command_rev,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
            "command_deg": command_rev * 360.0,
            "position_deg": result.values[moteus.Register.POSITION] * 360.0,
            "ilimit_scale": ilimit_scale,
            "ilimit_effective_nm": ilimit_scale * MAX_TORQUE_NM,
        })
        await asyncio.sleep(SAMPLE_DT_S)
    return pd.DataFrame(rows)


async def run_step_from_current(controller, step_rev, ilimit_scale=1.0, settle_s=HOME_SETTLE_S):
    first = await controller.query()
    y0_cmd = float(first.values[moteus.Register.POSITION])
    print(
        f"Settle at current ({y0_cmd:.6f} rev) for {settle_s:.1f} s, "
        f"then step +{step_rev:.4f} rev"
    )
    pre = await command_and_log(
        controller, y0_cmd, settle_s + PRE_STEP_S, phase="pre_step", ilimit_scale=ilimit_scale
    )
    y0 = float(pre["position_rev"].iloc[-1])
    step_cmd = y0 + step_rev
    print(f"Step to {step_cmd:.6f} (measured start={y0:.6f})")
    step = await command_and_log(
        controller, step_cmd, STEP_RUN_S, phase="step", ilimit_scale=ilimit_scale
    )
    step = step.copy()
    step["time_s"] = step["time_s"] + PRE_STEP_S
    pre = pre.copy()
    pre["time_s"] = pre["time_s"] - settle_s
    data = pd.concat([pre[pre["time_s"] >= 0], step], ignore_index=True)
    return data, y0, step_cmd


async def offset_hold_then_return(controller, phase, ilimit_scale):
    """Hold ~20 deg away to wind integral, then command back and watch recovery."""
    first = await controller.query()
    home = float(first.values[moteus.Register.POSITION])
    offset_cmd = home + OFFSET_REV
    print(
        f"\n[{phase}] Move to offset {offset_cmd:.6f} rev "
        f"(+{OFFSET_REV * 360:.1f} deg), hold {HOLD_OFFSET_S:.0f} s, "
        f"then return (ilimit_eff={ilimit_scale * MAX_TORQUE_NM:.4f} N·m)"
    )

    await command_and_log(
        controller, home, HOME_SETTLE_S, phase=f"{phase}_home", ilimit_scale=ilimit_scale
    )
    go = await command_and_log(
        controller, offset_cmd, HOLD_OFFSET_S, phase=f"{phase}_hold", ilimit_scale=ilimit_scale
    )
    print("Commanding return to home — logging recovery...")
    back = await command_and_log(
        controller, home, RELEASE_SETTLE_S, phase=f"{phase}_release", ilimit_scale=ilimit_scale
    )
    back = back.copy()
    back["time_s"] = back["time_s"] + HOLD_OFFSET_S
    return pd.concat([go, back], ignore_index=True), home, offset_cmd


def plot_traj(data, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], data["command_deg"], label="Command")
    ax.plot(data["time_s"], data["position_deg"], label="Measured")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [deg]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_step(data, y0, y_cmd, title):
    pos_rel = data["position_rev"] - y0
    cmd_rel = np.where(data["time_s"].to_numpy() < PRE_STEP_S, 0.0, y_cmd - y0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], cmd_rel, label="Command / target")
    ax.plot(data["time_s"], pos_rel, label="Measured")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def ess_summary(data, label):
    late = data[data["time_s"] >= float(data["time_s"].iloc[-1]) - 0.5]
    cmd = float(late["command_rev"].mean())
    pos = float(late["position_rev"].mean())
    ess = abs(cmd - pos)
    print(f"{label}: |ess| ≈ {ess:.6f} rev ({ess * 360:.3f} deg)")
    return ess


async def main():
    print_metrics_suggestions()
    scale_chosen = ilimit_scale_for(ILIMIT)
    scale_max = 1.0
    print("\n=== §6.3 integral gains (from top of script) ===")
    print(f"kp={KP}, kd={KD}, ki={KI}")
    print(
        f"Configured ilimit={MAX_TORQUE_NM} N·m; "
        f"trial scales → chosen {ILIMIT} N·m (scale={scale_chosen:.4f}), "
        f"max {MAX_TORQUE_NM} N·m (scale=1)"
    )
    print("Close tview if open, then applying gains...")

    # Store max in config; scale per-command for chosen vs max comparison.
    apply_gains(KP, KD, KI, MAX_TORQUE_NM)
    controller = moteus.Controller(id=MOTOR_ID)

    try:
        await clear_timeout(controller)

        print("\n[1] Step with chosen ilimit")
        step1, y0, y_cmd = await run_step_from_current(
            controller, STEP_REV, ilimit_scale=scale_chosen
        )
        step1.to_csv(DATA_DIR / "section6_3_step_ilimit.csv", index=False)
        print("Saved:", DATA_DIR / "section6_3_step_ilimit.csv")
        ess_summary(step1[step1["time_s"] >= PRE_STEP_S], "Step settle (chosen ilimit)")
        plot_step(
            step1,
            y0,
            y_cmd,
            f"§6.3 Step (ki={KI}, ilimit={ILIMIT} N·m)",
        )

        await clear_timeout(controller)
        trial_a, _, _ = await offset_hold_then_return(
            controller, phase=f"ilimit_{ILIMIT}", ilimit_scale=scale_chosen
        )
        trial_a.to_csv(DATA_DIR / "section6_3_hold_release_ilimit.csv", index=False)
        print("Saved:", DATA_DIR / "section6_3_hold_release_ilimit.csv")
        ess_summary(
            trial_a[trial_a["phase"].str.endswith("_release")],
            f"Recovery after offset (ilimit={ILIMIT})",
        )
        plot_traj(trial_a, f"§6.3 Offset→return (ilimit={ILIMIT} N·m)")

        print(
            f"\nSwitching to ilimit={MAX_TORQUE_NM} N·m via ilimit_scale=1 "
            "(no config rewrite)..."
        )
        await clear_timeout(controller)
        trial_b, _, _ = await offset_hold_then_return(
            controller, phase=f"ilimit_{MAX_TORQUE_NM}", ilimit_scale=scale_max
        )
        trial_b.to_csv(DATA_DIR / "section6_3_hold_release_ilimit_max.csv", index=False)
        print("Saved:", DATA_DIR / "section6_3_hold_release_ilimit_max.csv")
        ess_summary(
            trial_b[trial_b["phase"].str.endswith("_release")],
            f"Recovery after offset (ilimit={MAX_TORQUE_NM})",
        )
        plot_traj(trial_b, f"§6.3 Offset→return (ilimit={MAX_TORQUE_NM} N·m)")

        print("\n[4] Final step with chosen ilimit")
        await clear_timeout(controller)
        step, y0, y_cmd = await run_step_from_current(
            controller, STEP_REV, ilimit_scale=scale_chosen
        )
        step.to_csv(DATA_DIR / "section6_3_step.csv", index=False)
        print("Saved:", DATA_DIR / "section6_3_step.csv")
        ess_summary(step[step["time_s"] >= PRE_STEP_S], "Final step settle")
        plot_step(
            step,
            y0,
            y_cmd,
            f"§6.3 Final step (kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT})",
        )

        summary = pd.DataFrame([{
            "kp": KP,
            "kd": KD,
            "ki": KI,
            "ilimit_nm": ILIMIT,
            "ilimit_max_nm": MAX_TORQUE_NM,
            "ilimit_scale_chosen": scale_chosen,
            "offset_rev": OFFSET_REV,
            "step_y0_rev": y0,
            "step_target_rev": y_cmd,
        }])
        summary.to_csv(DATA_DIR / "section6_3_summary.csv", index=False)
        print("Saved:", DATA_DIR / "section6_3_summary.csv")
        print("\nCompare chosen vs max ilimit recovery plots for windup.")
        print("Copy final kp/kd/ki/ilimit into section6_4 and section6_5.")
    finally:
        await controller.set_stop()
        print("Motor stopped")


if __name__ == "__main__":
    asyncio.run(main())
