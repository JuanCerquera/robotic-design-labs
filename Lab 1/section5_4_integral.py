"""Section 5.4 / Q5.4 - Enable integral action and compare responses.

Uses your critically-damped kp/kd from Q5.3 plus ki / ilimit from the
error-integral calculation (defaults load from section5_3_metrics.csv).

Sequence:
  1. Gentle displace-and-release (like Q5.2) — does it still park short?
  2. Hold ~20 deg off target for 5 s, then release — at your ilimit, then
     again with ilimit = MAX_TORQUE_NM (0.1). Compare windup recovery.
  3. Step response with integral on.

Close tview before running. Outputs overwrite lab1_data/section5_4_*.
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
MAX_TORQUE_NM = 0.1

# From Q5.3 critical damping + ki/ilimit suggestions (edit if needed).
KP = 1.6
KD = 0.075
KI = 2.68
ILIMIT = 0.0022

# Test timing
DISPLACE_S = 8.0
HOLD_OFFSET_S = 5.0
RELEASE_SETTLE_S = 8.0
HOME_SETTLE_S = 3.0
PRE_STEP_S = 0.5
STEP_REV = 0.05
STEP_RUN_S = 3.0
SAMPLE_DT_S = 0.005
HOME_POSITION_REV = 0.0
# Brief settle under closed-loop hold before an interactive trial.
PRE_HOLD_SETTLE_S = 1.5

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def print_metrics_suggestions():
    path = DATA_DIR / "section5_3_metrics.csv"
    if not path.exists():
        print("No section5_3_metrics.csv — using gains set at top of script.")
        return
    row = pd.read_csv(path).iloc[0]
    print(f"Q5.3 metrics suggestions ({path.name}):")
    print(
        f"  kp={row.get('kp')}, kd={row.get('kd')}, "
        f"ki≈{row.get('ki_suggest')}, ilimit≈{row.get('ilimit_suggest_nm')}"
    )
    print("Edit KP/KD/KI/ILIMIT at top of this script if you want different values.")


def apply_gains(kp, kd, ki, ilimit):
    """One-shot config write. Call only before opening the Controller (port exclusive)."""
    cfg_path = DATA_DIR / "section5_4_gains.cfg"
    cfg_path.write_text(
        "\n".join([
            f"conf set servo.pid_position.kp {kp}",
            f"conf set servo.pid_position.kd {kd}",
            f"conf set servo.pid_position.ki {ki}",
            f"conf set servo.pid_position.ilimit {ilimit}",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
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
    """Configured ilimit is MAX_TORQUE_NM; scale commands to the desired clamp."""
    if MAX_TORQUE_NM <= 0:
        raise RuntimeError("MAX_TORQUE_NM must be > 0")
    return float(desired_ilimit_nm) / float(MAX_TORQUE_NM)


async def clear_timeout(controller):
    """Position mode latches on command timeout; stop clears it so holds work again."""
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


async def hold_and_log(
    controller, hold_position, duration_s, phase, ilimit_scale=1.0, prompt=None
):
    if prompt:
        print(prompt)
    rows = []
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration_s:
            break
        result = await set_hold(controller, hold_position, ilimit_scale=ilimit_scale)
        rows.append({
            "phase": phase,
            "time_s": elapsed,
            "command_rev": hold_position,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
            "command_deg": hold_position * 360.0,
            "position_deg": result.values[moteus.Register.POSITION] * 360.0,
            "ilimit_scale": ilimit_scale,
            "ilimit_effective_nm": ilimit_scale * MAX_TORQUE_NM,
        })
        await asyncio.sleep(SAMPLE_DT_S)
    return pd.DataFrame(rows)


async def hold_during_input(controller, hold_position, prompt, ilimit_scale=1.0):
    """Keep streaming the hold command while waiting for Enter (avoids timeout latch)."""
    stop = asyncio.Event()

    async def _hold_loop():
        while not stop.is_set():
            await set_hold(controller, hold_position, ilimit_scale=ilimit_scale)
            await asyncio.sleep(SAMPLE_DT_S)

    task = asyncio.create_task(_hold_loop())
    try:
        await asyncio.to_thread(input, prompt)
    finally:
        stop.set()
        await task


async def begin_hold(
    controller, hold_position, ilimit_scale=1.0, settle_s=PRE_HOLD_SETTLE_S
):
    """Clear timeout, then actively hold so the setpoint is live before the user moves."""
    await clear_timeout(controller)
    print(
        f"Establishing hold at {hold_position:.6f} rev "
        f"({hold_position * 360:.1f} deg), "
        f"ilimit_eff={ilimit_scale * MAX_TORQUE_NM:.4f} N·m..."
    )
    await hold_and_log(
        controller,
        hold_position,
        settle_s,
        phase="pre_hold_settle",
        ilimit_scale=ilimit_scale,
    )


async def hold_offset_then_release(controller, hold_position, phase, ilimit_scale):
    """User holds ~20 deg off target for HOLD_OFFSET_S, then releases."""
    await begin_hold(controller, hold_position, ilimit_scale=ilimit_scale)
    print(
        f"\n[{phase}] Holding target = {hold_position:.6f} rev "
        f"({hold_position * 360:.1f} deg), "
        f"ilimit_eff={ilimit_scale * MAX_TORQUE_NM:.4f} N·m."
    )
    print(
        f"When the 5 s timer starts: twist shaft ~20 deg off target, HOLD for "
        f"{HOLD_OFFSET_S:.0f} s, then LET GO and watch it recover."
    )
    # Must keep commanding through this prompt or moteus latches position-timeout.
    await hold_during_input(
        controller,
        hold_position,
        "Press Enter to start the 5 s offset-hold timer (keep holding the shaft off)...",
        ilimit_scale=ilimit_scale,
    )

    hold_rows = await hold_and_log(
        controller,
        hold_position,
        HOLD_OFFSET_S,
        phase=f"{phase}_hold",
        ilimit_scale=ilimit_scale,
        prompt=f"HOLD ~20 deg off NOW for {HOLD_OFFSET_S:.0f} s...",
    )
    print("Release the shaft NOW — logging recovery...")
    release_rows = await hold_and_log(
        controller,
        hold_position,
        RELEASE_SETTLE_S,
        phase=f"{phase}_release",
        ilimit_scale=ilimit_scale,
    )
    release_rows = release_rows.copy()
    release_rows["time_s"] = release_rows["time_s"] + HOLD_OFFSET_S
    return pd.concat([hold_rows, release_rows], ignore_index=True)


async def run_step(controller, ilimit_scale=1.0):
    """Home toward 0, then step +STEP_REV from measured (same idea as 5.3)."""
    hold_cmd = HOME_POSITION_REV
    rows = []
    start = time.perf_counter()
    total_pre = HOME_SETTLE_S + PRE_STEP_S
    print(f"\n[step] Homing/hold at {hold_cmd} for {total_pre:.1f} s, then step +{STEP_REV}")

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= total_pre:
            break
        result = await set_hold(controller, hold_cmd, ilimit_scale=ilimit_scale)
        rows.append({
            "time_s": elapsed,
            "command_rev": hold_cmd,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)

    y0 = float(rows[-1]["position_rev"])
    step_cmd = y0 + STEP_REV
    print(f"Step to {step_cmd:.6f} (start={y0:.6f})")

    step_t0 = time.perf_counter()
    while True:
        elapsed_step = time.perf_counter() - step_t0
        if elapsed_step >= STEP_RUN_S:
            break
        result = await controller.set_position(
            position=step_cmd,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=ilimit_scale,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "time_s": total_pre + elapsed_step,
            "command_rev": step_cmd,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)

    data = pd.DataFrame(rows)
    data = data[data["time_s"] >= HOME_SETTLE_S].copy().reset_index(drop=True)
    data["time_s"] = data["time_s"] - HOME_SETTLE_S
    return data, y0, step_cmd


def plot_hold(data, title):
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
    """Rough steady-state error from last 0.5 s of a hold log."""
    late = data[data["time_s"] >= float(data["time_s"].iloc[-1]) - 0.5]
    cmd = float(late["command_rev"].mean())
    pos = float(late["position_rev"].mean())
    ess = abs(cmd - pos)
    print(f"{label}: |ess| ≈ {ess:.6f} rev ({ess * 360:.3f} deg)")
    return ess


async def main():
    print_metrics_suggestions()
    scale_chosen = ilimit_scale_for(ILIMIT)
    scale_max = 1.0  # configured ilimit = MAX_TORQUE_NM
    print("\n=== Q5.4 gains (from top of script) ===")
    print(f"kp={KP}, kd={KD}, ki={KI}")
    print(
        f"Configured ilimit={MAX_TORQUE_NM} N·m; "
        f"trial scales → chosen {ILIMIT} N·m (scale={scale_chosen:.4f}), "
        f"max {MAX_TORQUE_NM} N·m (scale=1)"
    )
    input("Close tview, then press Enter to apply gains and start...")

    # Write once before opening the port — mid-run moteus_tool can't share COM.
    apply_gains(KP, KD, KI, MAX_TORQUE_NM)
    controller = moteus.Controller(id=MOTOR_ID)

    try:
        # --- 1. Displace and release ---
        await clear_timeout(controller)
        first = await controller.query()
        hold = float(first.values[moteus.Register.POSITION])
        await begin_hold(controller, hold, ilimit_scale=scale_chosen)
        print(
            f"\n[1] Displace-and-release at {hold:.6f} rev. "
            f"Gently displace and release a few times for {DISPLACE_S:.0f} s."
        )
        await hold_during_input(
            controller, hold, "Press Enter to start logging...", ilimit_scale=scale_chosen
        )
        displace = await hold_and_log(
            controller,
            hold,
            DISPLACE_S,
            phase="displace_release",
            ilimit_scale=scale_chosen,
        )
        displace.to_csv(DATA_DIR / "section5_4_displace_release.csv", index=False)
        print("Saved:", DATA_DIR / "section5_4_displace_release.csv")
        ess_summary(displace, "After displace/release settle")
        plot_hold(
            displace,
            f"Q5.4 Displace-release (kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT})",
        )

        # --- 2a. Hold 20 deg off, release (chosen ilimit) ---
        await clear_timeout(controller)
        first = await controller.query()
        hold = float(first.values[moteus.Register.POSITION])
        trial_a = await hold_offset_then_release(
            controller, hold, phase=f"ilimit_{ILIMIT}", ilimit_scale=scale_chosen
        )
        trial_a.to_csv(DATA_DIR / "section5_4_hold_release_ilimit.csv", index=False)
        print("Saved:", DATA_DIR / "section5_4_hold_release_ilimit.csv")
        plot_hold(
            trial_a,
            f"Q5.4 Hold-off then release (ilimit={ILIMIT} N·m)",
        )

        # --- 2b. Same with ilimit = max torque (scale only — no moteus_tool) ---
        print(
            f"\nSwitching to ilimit={MAX_TORQUE_NM} N·m via ilimit_scale=1 "
            "(no config rewrite; port stays open)..."
        )
        await clear_timeout(controller)
        first = await controller.query()
        hold = float(first.values[moteus.Register.POSITION])
        trial_b = await hold_offset_then_release(
            controller, hold, phase=f"ilimit_{MAX_TORQUE_NM}", ilimit_scale=scale_max
        )
        trial_b.to_csv(DATA_DIR / "section5_4_hold_release_ilimit_max.csv", index=False)
        print("Saved:", DATA_DIR / "section5_4_hold_release_ilimit_max.csv")
        plot_hold(
            trial_b,
            f"Q5.4 Hold-off then release (ilimit={MAX_TORQUE_NM} N·m)",
        )

        # --- 3. Step response ---
        input("\nPress Enter to run the step response...")
        await clear_timeout(controller)
        step, y0, y_cmd = await run_step(controller, ilimit_scale=scale_chosen)
        step.to_csv(DATA_DIR / "section5_4_step.csv", index=False)
        print("Saved:", DATA_DIR / "section5_4_step.csv")
        plot_step(
            step,
            y0,
            y_cmd,
            f"Q5.4 Step with integral (kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT})",
        )

        summary = pd.DataFrame([{
            "kp": KP,
            "kd": KD,
            "ki": KI,
            "ilimit_nm": ILIMIT,
            "ilimit_max_nm": MAX_TORQUE_NM,
            "ilimit_scale_chosen": scale_chosen,
            "step_y0_rev": y0,
            "step_target_rev": y_cmd,
        }])
        summary.to_csv(DATA_DIR / "section5_4_summary.csv", index=False)
        print("Saved:", DATA_DIR / "section5_4_summary.csv")
        print(
            "\nQ5.4 done. Compare park-short vs integral, name the release "
            "behaviour (integrator windup), and note what ilimit protects vs costs."
        )
    finally:
        await controller.set_stop()
        print("Motor stopped")


if __name__ == "__main__":
    asyncio.run(main())
