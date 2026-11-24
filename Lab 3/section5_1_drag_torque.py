"""Section 5.1 - Output drag torque with the gear stage installed.

Same method as Section 3.1, but speeds are motor speeds / 6 (output units)
and torque is already at the output once the ratio is set.

Plots measured output drag vs Section 3.1 data referred to the output
(τ_out ≈ τ_motor × 6, ω_out = ω_motor / 6). Stage friction = difference.

Requires section4_1_set_ratio.py. Close tview first.

  python section5_1_drag_torque.py
  python section5_1_drag_torque.py --analyze --no-show
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
# Continuous drag sweep: use RMS cap, not the 9.8 peak.
MAX_TORQUE_NM = 3.4
GEAR_RATIO = 6.0

# Same motor shaft speeds as §3.1, expressed at the output.
MOTOR_SPEEDS_REV_S = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
SPEEDS_REV_S = [s / GEAR_RATIO for s in MOTOR_SPEEDS_REV_S]
HOLD_S = 3.0
SETTLE_S = 1.0
SAMPLE_DT_S = 0.01
ACCEL_LIMIT = 5.0

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "section5_1_drag_raw.csv"
SUMMARY_CSV = DATA_DIR / "section5_1_drag_summary.csv"
FIT_CSV = DATA_DIR / "section5_1_drag_fit.csv"
STAGE_FIT_CSV = DATA_DIR / "section5_1_stage_fit.csv"
PLOT_PNG = PLOTS_DIR / "section5_1_drag.png"
BASELINE_SUMMARY = DATA_DIR / "section3_1_drag_summary.csv"


def ensure_gear_ratio():
    cfg_path = DATA_DIR / "section5_1_ratio.cfg"
    cfg_path.write_text(
        "\n".join([
            f"conf set motor_position.rotor_to_output_ratio {1.0 / GEAR_RATIO:.8f}",
            "conf set servo.default_timeout_s 0.1",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
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
        raise RuntimeError("moteus_tool --write-config failed. Close tview and retry.")
    print(f"Ratio = 1/{GEAR_RATIO:.0f} (output units).")


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
            accel_limit=ACCEL_LIMIT,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        fault = result.values.get(moteus.Register.FAULT, 0)
        if fault:
            raise RuntimeError(
                f"Fault {fault} during '{phase}' at t={elapsed:.2f}s."
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
    v = summary["velocity_mean_rev_s"].to_numpy(dtype=float)
    tau = summary["torque_mean_nm"].to_numpy(dtype=float)
    sign_v = np.sign(v)
    sign_v[sign_v == 0] = 1.0
    A = np.column_stack([v, sign_v])
    coeffs, *_ = np.linalg.lstsq(A, tau, rcond=None)
    b, tau_c = float(coeffs[0]), float(coeffs[1])
    resid = tau - (A @ coeffs)
    return {
        "viscous_nm_per_rev_s": b,
        "coulomb_nm": tau_c,
        "rmse_nm": float(np.sqrt(np.mean(resid ** 2))),
    }


def baseline_referred_to_output() -> pd.DataFrame | None:
    if not BASELINE_SUMMARY.exists():
        print(f"No {BASELINE_SUMMARY.name} — skipping §3.1 overlay.")
        return None
    base = pd.read_csv(BASELINE_SUMMARY)
    out = pd.DataFrame({
        "velocity_mean_rev_s": base["velocity_mean_rev_s"] / GEAR_RATIO,
        "torque_mean_nm": base["torque_mean_nm"] * GEAR_RATIO,
        "source": "section3_1_referred",
    })
    return out


def plot_and_report(summary: pd.DataFrame, fit: dict, show: bool):
    v = summary["velocity_mean_rev_s"].to_numpy(dtype=float)
    tau = summary["torque_mean_nm"].to_numpy(dtype=float)
    v_line = np.linspace(min(v.min(), -v.max()), max(v.max(), -v.min()), 200)
    sign = np.sign(v_line)
    sign[sign == 0] = 1.0
    tau_fit = fit["viscous_nm_per_rev_s"] * v_line + fit["coulomb_nm"] * sign

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(v, tau, label="Measured output (stage + motor)", zorder=3)
    ax.plot(
        v_line, tau_fit,
        label=(
            f"Total fit: τ = {fit['coulomb_nm']:.4f}·sgn(v) "
            f"+ {fit['viscous_nm_per_rev_s']:.5f}·v"
        ),
    )

    referred = baseline_referred_to_output()
    stage_fit = None
    if referred is not None:
        ax.scatter(
            referred["velocity_mean_rev_s"],
            referred["torque_mean_nm"],
            marker="x",
            label="§3.1 motor drag ×6 (referred to output)",
            zorder=3,
        )
        # Stage-only ≈ total − referred, matched by commanded output speed sign/mag.
        # Merge on nearest velocity.
        stage_rows = []
        for _, row in summary.iterrows():
            v_i = row["velocity_mean_rev_s"]
            j = (referred["velocity_mean_rev_s"] - v_i).abs().idxmin()
            tau_ref = float(referred.loc[j, "torque_mean_nm"])
            # Preserve sign consistently: both should share direction.
            stage_rows.append({
                "velocity_mean_rev_s": v_i,
                "torque_total_nm": row["torque_mean_nm"],
                "torque_motor_ref_nm": tau_ref,
                "torque_stage_nm": row["torque_mean_nm"] - tau_ref,
            })
        stage_df = pd.DataFrame(stage_rows)
        stage_df.to_csv(DATA_DIR / "section5_1_stage_breakdown.csv", index=False)
        ax.scatter(
            stage_df["velocity_mean_rev_s"],
            stage_df["torque_stage_nm"],
            marker="^",
            label="Stage alone (total − referred motor)",
            zorder=3,
        )
        stage_fit = fit_coulomb_viscous(pd.DataFrame({
            "velocity_mean_rev_s": stage_df["velocity_mean_rev_s"],
            "torque_mean_nm": stage_df["torque_stage_nm"],
        }))
        pd.DataFrame([stage_fit]).to_csv(STAGE_FIT_CSV, index=False)
        print("Saved:", STAGE_FIT_CSV)
        tau_stage_fit = (
            stage_fit["viscous_nm_per_rev_s"] * v_line
            + stage_fit["coulomb_nm"] * sign
        )
        ax.plot(
            v_line, tau_stage_fit, linestyle="--",
            label=(
                f"Stage fit: τ = {stage_fit['coulomb_nm']:.4f}·sgn(v) "
                f"+ {stage_fit['viscous_nm_per_rev_s']:.5f}·v"
            ),
        )

    ax.set_xlabel("Output velocity [rev/s]")
    ax.set_ylabel("Output torque [N·m]")
    ax.set_title("Q5.1 Output drag torque")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    print("Saved plot:", PLOT_PNG)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\nQ5.1 total output drag fit:")
    print(f"  Coulomb τ_c = {fit['coulomb_nm']:.5f} N·m")
    print(f"  Viscous b   = {fit['viscous_nm_per_rev_s']:.6f} N·m /(rev/s)")
    print(f"  RMSE        = {fit['rmse_nm']:.5f} N·m")
    if stage_fit is not None:
        print("\nGear-stage-only fit (total − §3.1×6):")
        print(f"  Coulomb τ_c = {stage_fit['coulomb_nm']:.5f} N·m")
        print(f"  Viscous b   = {stage_fit['viscous_nm_per_rev_s']:.6f} N·m /(rev/s)")
        print(f"  RMSE        = {stage_fit['rmse_nm']:.5f} N·m")
        print("Compare with your P5.1 prediction.")


async def run_capture():
    ensure_gear_ratio()
    controller = moteus.Controller(id=MOTOR_ID)
    all_rows = []
    speeds = []
    for s in SPEEDS_REV_S:
        speeds.append(s)
        speeds.append(-s)

    try:
        await controller.set_stop()
        for i, vel in enumerate(speeds):
            phase = f"v{vel:+.4f}"
            print(
                f"[{i + 1}/{len(speeds)}] Hold {vel:+.4f} rev/s output "
                f"(motor ≈ {vel * GEAR_RATIO:+.1f} rev/s) for {HOLD_S:.1f} s..."
            )
            df = await hold_velocity(controller, vel, HOLD_S, phase)
            all_rows.append(df)
            # Checkpoint after each hold so a fault/Ctrl+C keeps prior data.
            pd.concat(all_rows, ignore_index=True).to_csv(RAW_CSV, index=False)
            mean_tau = df.loc[df["time_s"] >= SETTLE_S, "torque_nm"].mean()
            mean_v = df.loc[df["time_s"] >= SETTLE_S, "velocity_rev_s"].mean()
            print(f"    steady: v={mean_v:.4f} rev/s, τ={mean_tau:.5f} N·m")
    finally:
        await controller.set_stop()
        print("Motor stopped.")
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
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
