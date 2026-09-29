"""Section 7.2 - Retune kp/kd with the inertia disk mounted.

Scale the Q7.1 gains by the inertia ratio so wn and zeta roughly match bare:

  alpha = (J_rotor + J_disk) / J_rotor
        = (wn_bare / wn_loaded)^2     (same gains, before retune)
  kp' = alpha * kp
  kd' = alpha * kd

Then tweak RETUNED_KP / RETUNED_KD if needed and re-run the step.

  python section7_2_retune.py              # print alpha + predicted gains, then live step
  python section7_2_retune.py --predict    # numbers only (no motor)
  python section7_2_retune.py --keep-gains # live step without moteus_tool
  python section7_2_retune.py --analyze    # re-plot existing CSV
"""

import argparse
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

# Original (Section 5 / Q7.1) gains — match section7_1 (incl. ki from Q5.4).
ORIGINAL_KP = 1.6
ORIGINAL_KD = 0.05   # same as Q7.1 (slightly below Q5.3 critical 0.075)
ORIGINAL_KI = 2.68
ORIGINAL_ILIMIT = 0.0022

# After first predict, set these to alpha*original (or edit after experimental retune).
# Leave as None to auto-fill from Q7.1 summary on each run.
RETUNED_KP = 1.6*0.935
RETUNED_KD = 0.05*(1.065)
RETUNED_KI = 2.68
RETUNED_ILIMIT = 0.0022

STEP_REV = 0.05
HOME_POSITION_REV = 0.0
HOME_SETTLE_S = 3.0
PRE_STEP_S = 0.5
RUN_S = 5.0  # longer so the integral can kill steady-state error
SAMPLE_DT_S = 0.005

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def load_q71_summary():
    path = DATA_DIR / "section7_1_summary.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Finish Q7.1 bare+loaded first "
            "(python section7_1_identify_inertia.py)."
        )
    return pd.read_csv(path).iloc[0]


def predict_gains(summary=None):
    s = summary if summary is not None else load_q71_summary()
    j_rotor = float(s["j_rotor_kg_m2"])
    j_disk = float(s["j_disk_identified_kg_m2"])
    wn_bare = float(s["omega_n_bare_rad_s"])
    wn_loaded = float(s["omega_n_loaded_rad_s"])
    j_cad = float(s.get("j_disk_cad_kg_m2", np.nan))

    if np.isnan(wn_bare) or np.isnan(wn_loaded) or wn_loaded <= 0:
        raise SystemExit(
            "Q7.1 summary is incomplete (need both wn). "
            "Re-run bare and loaded steps."
        )

    alpha_wn = (wn_bare / wn_loaded) ** 2
    alpha_j = (
        (j_rotor + j_disk) / j_rotor
        if np.isfinite(j_disk) and j_disk > 0
        else np.nan
    )
    alpha_cad = (
        (j_rotor + j_cad) / j_rotor
        if np.isfinite(j_cad) and j_cad > 0
        else np.nan
    )

    # Adding mass must increase inertia => alpha > 1. If wn_loaded > wn_bare,
    # the summary is swapped/bad (would scale gains *down* and cause huge ess).
    note = ""
    if alpha_wn >= 1.0 + 1e-6:
        alpha = alpha_wn
        note = "alpha from wn ratio"
    elif np.isfinite(alpha_j) and alpha_j >= 1.0 + 1e-6:
        alpha = alpha_j
        note = "alpha from J_disk (wn ratio was < 1 — check bare/loaded labels)"
    elif np.isfinite(alpha_cad):
        alpha = alpha_cad
        note = (
            "alpha from CAD J_disk (Q7.1 wn ratio < 1 — re-run bare/loaded; "
            "do not trust swapped metrics)"
        )
    else:
        raise SystemExit(
            f"Cannot form alpha>1 (wn_bare={wn_bare:.2f}, wn_loaded={wn_loaded:.2f}). "
            "Re-run Q7.1 bare then loaded with the same gains."
        )

    kp_pred = ORIGINAL_KP * alpha
    kd_pred = ORIGINAL_KD * alpha
    ki_pred = ORIGINAL_KI * alpha
    # Keep at least the Q5.4 ilimit; scaling down starves the integrator.
    ilimit_pred = max(ORIGINAL_ILIMIT * alpha, ORIGINAL_ILIMIT)
    return {
        "j_rotor_kg_m2": j_rotor,
        "j_disk_kg_m2": j_disk,
        "j_total_kg_m2": j_rotor + max(j_disk, 0.0) if np.isfinite(j_disk) else np.nan,
        "omega_n_bare_rad_s": wn_bare,
        "omega_n_loaded_rad_s": wn_loaded,
        "alpha_from_J": alpha_j,
        "alpha_from_wn": alpha_wn,
        "alpha_from_cad": alpha_cad,
        "alpha": alpha,
        "alpha_note": note,
        "original_kp": ORIGINAL_KP,
        "original_kd": ORIGINAL_KD,
        "original_ki": ORIGINAL_KI,
        "original_ilimit": ORIGINAL_ILIMIT,
        "kp_predicted": kp_pred,
        "kd_predicted": kd_pred,
        "ki_predicted": ki_pred,
        "ilimit_predicted": ilimit_pred,
    }


def print_prediction(pred):
    print("\n=== Q7.2 gain scaling ===")
    print(f"J_rotor:                {pred['j_rotor_kg_m2']:.6e} kg·m²")
    print(f"J_disk (Q7.1):          {pred['j_disk_kg_m2']}")
    print(f"alpha from wn ratio:    {pred['alpha_from_wn']:.4f}")
    print(f"alpha from J:           {pred['alpha_from_J']}")
    print(f"Using alpha:            {pred['alpha']:.4f}  ({pred['alpha_note']})")
    print(
        f"Original kp, kd, ki:    {pred['original_kp']}, "
        f"{pred['original_kd']}, {pred['original_ki']}"
    )
    print(f"Predicted kp' = a*kp:   {pred['kp_predicted']:.4f}")
    print(f"Predicted kd' = a*kd:   {pred['kd_predicted']:.5f}")
    print(f"Predicted ki' = a*ki:   {pred['ki_predicted']:.4f}")
    print(f"Predicted ilimit':      {pred['ilimit_predicted']:.5f}")


def resolve_retuned_gains(pred):
    kp = RETUNED_KP if RETUNED_KP is not None else pred["kp_predicted"]
    kd = RETUNED_KD if RETUNED_KD is not None else pred["kd_predicted"]
    ki = RETUNED_KI if RETUNED_KI is not None else pred["ki_predicted"]
    ilimit = (
        RETUNED_ILIMIT if RETUNED_ILIMIT is not None else pred["ilimit_predicted"]
    )
    return float(kp), float(kd), float(ki), float(ilimit)


def ilimit_scale_for(desired_ilimit_nm):
    """Match Q5.4: conf ilimit = MAX_TORQUE; commands use this scale."""
    if MAX_TORQUE_NM <= 0:
        raise RuntimeError("MAX_TORQUE_NM must be > 0")
    return float(desired_ilimit_nm) / float(MAX_TORQUE_NM)


def apply_gains(kp, kd, ki, ilimit):
    """Write kp/kd/ki; set conf ilimit to MAX_TORQUE (scale on each command)."""
    cfg_path = DATA_DIR / "section7_2_gains.cfg"
    # Same pattern as Q5.4: park the real clamp in ilimit_scale, not conf.
    cfg_path.write_text(
        "\n".join([
            f"conf set servo.pid_position.kp {kp}",
            f"conf set servo.pid_position.kd {kd}",
            f"conf set servo.pid_position.ki {ki}",
            f"conf set servo.pid_position.ilimit {MAX_TORQUE_NM}",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
            "",
        ]),
        encoding="utf-8",
    )
    scale = ilimit_scale_for(ilimit)
    print(
        f"Writing gains (kp={kp}, kd={kd}, ki={ki}, "
        f"conf_ilimit={MAX_TORQUE_NM}, cmd_ilimit={ilimit} via scale={scale:.4f})..."
    )
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
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "moteus_tool --write-config failed. Close tview and retry, "
            "or use --keep-gains if you already set gains."
        )
    print("Gains set (RAM).")
    return scale


async def run_home_and_step(controller, ilimit_scale=1.0):
    hold_cmd = HOME_POSITION_REV
    rows = []
    start = time.perf_counter()
    total_pre = HOME_SETTLE_S + PRE_STEP_S
    print(
        f"  Holding at {hold_cmd:.3f} rev for {total_pre:.1f} s, "
        f"then step +{STEP_REV:.3f} rev "
        f"(ilimit_scale={ilimit_scale:.4f}) ..."
    )
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= total_pre:
            break
        result = await controller.set_position(
            position=hold_cmd,
            velocity=0.0,
            kp_scale=1.0,
            kd_scale=1.0,
            ilimit_scale=ilimit_scale,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        rows.append({
            "time_s": elapsed,
            "command_rev": hold_cmd,
            "position_rev": result.values[moteus.Register.POSITION],
            "velocity_rev_s": result.values[moteus.Register.VELOCITY],
        })
        await asyncio.sleep(SAMPLE_DT_S)

    y0 = float(rows[-1]["position_rev"])
    step_cmd = y0 + STEP_REV
    print(f"  STEPPING now: {y0:.6f} -> {step_cmd:.6f} rev")

    step_t0 = time.perf_counter()
    while True:
        elapsed_step = time.perf_counter() - step_t0
        if elapsed_step >= RUN_S:
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

    print("  Step capture done.")
    data = pd.DataFrame(rows)
    data = data[data["time_s"] >= HOME_SETTLE_S].copy().reset_index(drop=True)
    data["time_s"] = data["time_s"] - HOME_SETTLE_S

    late = data[data["time_s"] >= float(data["time_s"].iloc[-1]) - 0.5]
    ess = abs(float(late["command_rev"].mean()) - float(late["position_rev"].mean()))
    print(f"  Steady-state |error| ≈ {ess:.6f} rev ({ess * 360:.3f} deg)")
    return data, y0, step_cmd, ess


def analyze_step(data, start_position, target, kp, kd):
    step_data = data[data["time_s"] >= PRE_STEP_S].copy().reset_index(drop=True)
    t = step_data["time_s"].to_numpy() - PRE_STEP_S
    y = step_data["position_rev"].to_numpy()
    y0 = float(start_position)
    y_cmd = float(target)
    e0 = y_cmd - y0
    if abs(e0) < 1e-9:
        raise RuntimeError("Step size is zero.")

    i_peak = int(np.argmax(y) if e0 > 0 else np.argmin(y))
    y_peak = float(y[i_peak])
    mp = (y_peak - y_cmd) / e0
    t_peak = float(t[i_peak])

    zeta = np.nan
    omega_n = np.nan
    note = ""
    if mp > 1e-4:
        zeta = -np.log(mp) / np.sqrt(np.pi**2 + np.log(mp) ** 2)
        if zeta < 1.0:
            omega_n = np.pi / (t_peak * np.sqrt(1.0 - zeta**2))
        else:
            note = "zeta >= 1 from Mp"
    else:
        note = "no overshoot — compare rise shape to bare; lower kd a bit if needed"
        # Still report peak time of max excursion.
        mp = max(mp, 0.0)

    return {
        "kp": kp,
        "kd": kd,
        "y0_rev": y0,
        "y_ref_rev": y_cmd,
        "y_peak_rev": y_peak,
        "overshoot_percent": 100.0 * mp,
        "t_peak_from_step_s": t_peak,
        "t_peak_s": t_peak + PRE_STEP_S,
        "zeta": zeta,
        "omega_n_rad_s": omega_n,
        "note": note,
    }


def infer_step_endpoints(data: pd.DataFrame):
    cmd = data["command_rev"].to_numpy()
    changes = np.where(np.diff(cmd) != 0)[0]
    if changes.size == 0:
        raise RuntimeError("No command step found in CSV.")
    i_step = int(changes[0]) + 1
    y0 = float(data["position_rev"].iloc[i_step - 1])
    target = float(cmd[i_step])
    t_step = float(data["time_s"].iloc[i_step])
    out = data.copy()
    out["time_s"] = out["time_s"] - (t_step - PRE_STEP_S)
    out = out[out["time_s"] >= 0].reset_index(drop=True)
    return out, y0, target


def plot_step(data, metrics, bare_metrics=None):
    y0 = metrics["y0_rev"]
    y_ref = metrics["y_ref_rev"]
    pos_rel = data["position_rev"] - y0
    cmd_rel = np.where(data["time_s"].to_numpy() < PRE_STEP_S, 0.0, y_ref - y0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], cmd_rel, label="Command / target")
    ax.plot(data["time_s"], pos_rel, label="Retuned (disk)")
    if bare_metrics is not None:
        # Overlay not possible without bare CSV time series alignment; annotate target.
        ax.axhline(
            0.0, color="0.7", linewidth=0.5
        )
    ax.axhline(metrics["y_peak_rev"] - y0, color="C3", linestyle=":", label="Peak")
    ax.axvline(metrics["t_peak_s"], color="C1", linestyle="--", label="Peak time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    zeta_s = f"{metrics['zeta']:.3f}" if np.isfinite(metrics["zeta"]) else "n/a"
    wn_s = (
        f"{metrics['omega_n_rad_s']:.1f}"
        if np.isfinite(metrics["omega_n_rad_s"]) else "n/a"
    )
    ax.set_title(
        f"Q7.2 Retuned step (kp={metrics['kp']:.3f}, kd={metrics['kd']:.4f})  "
        f"OS={metrics['overshoot_percent']:.1f}%  ζ={zeta_s}  ωn={wn_s} rad/s"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    for folder in (PLOTS_DIR, DATA_DIR):
        out = folder / "section7_2_step_retuned.png"
        fig.savefig(out)
        print("Saved:", out)
    plt.show()


def compare_to_bare(metrics, pred):
    bare_path = DATA_DIR / "section7_1_metrics_bare.csv"
    print("\n=== Compare retuned (disk) vs bare ===")
    bare_zeta = np.nan
    bare_os = np.nan
    print(
        f"Bare:    wn={pred['omega_n_bare_rad_s']:.2f} rad/s  "
        f"(from Q7.1 bare step)"
    )
    if bare_path.exists():
        b = pd.read_csv(bare_path).iloc[0]
        bare_zeta = float(b["zeta"])
        bare_os = float(b["overshoot_percent"])
        print(f"Bare:    OS={bare_os:.2f}%  zeta={bare_zeta:.3f}")
    wn = metrics["omega_n_rad_s"]
    zeta = metrics["zeta"]
    if np.isfinite(wn):
        print(
            f"Retuned: wn={wn:.2f} rad/s  OS={metrics['overshoot_percent']:.2f}%  "
            f"zeta={zeta}"
        )
        print(
            f"wn error vs bare:   "
            f"{100.0 * (wn - pred['omega_n_bare_rad_s']) / pred['omega_n_bare_rad_s']:+.1f} %"
        )
    else:
        print(
            f"Retuned: OS={metrics['overshoot_percent']:.2f}%  "
            f"(wn n/a — {metrics.get('note', '')})"
        )
    if np.isfinite(zeta) and np.isfinite(bare_zeta) and abs(bare_zeta) > 1e-12:
        print(
            f"zeta error vs bare: "
            f"{100.0 * (zeta - bare_zeta) / bare_zeta:+.1f} %"
        )
    print(
        "Goal: match bare bandwidth (wn) and damping (overshoot/zeta) as closely as you can."
    )


async def run_live(keep_gains: bool = False):
    pred = predict_gains()
    print_prediction(pred)
    kp, kd, ki, ilimit = resolve_retuned_gains(pred)
    print(
        f"\nUsing retuned gains: kp={kp:.4f}, kd={kd:.5f}, "
        f"ki={ki:.4f}, ilimit={ilimit:.5f}"
    )
    print(
        "(Edit RETUNED_KP / RETUNED_KD / RETUNED_KI / RETUNED_ILIMIT "
        "at top of script to tweak, then re-run.)"
    )

    input(
        "\nDisk MOUNTED and stand clamped.\n"
        "Close tview, then press Enter to apply gains and run the step..."
    )
    if keep_gains:
        print("Skipping moteus_tool (--keep-gains).")
        scale = ilimit_scale_for(ilimit)
    else:
        scale = apply_gains(kp, kd, ki, ilimit)

    controller = moteus.Controller(id=MOTOR_ID)
    try:
        await controller.set_stop()
        data, y0, target, ess = await run_home_and_step(
            controller, ilimit_scale=scale
        )
    finally:
        await controller.set_stop()
        print("Motor stopped")

    csv_path = DATA_DIR / "section7_2_step_retuned.csv"
    data.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    metrics = analyze_step(data, y0, target, kp, kd)
    metrics.update({
        "ki": ki,
        "ilimit": ilimit,
        "ilimit_scale": scale,
        "ess_rev": ess,
        "alpha": pred["alpha"],
        "alpha_note": pred["alpha_note"],
        "kp_predicted": pred["kp_predicted"],
        "kd_predicted": pred["kd_predicted"],
        "ki_predicted": pred["ki_predicted"],
        "ilimit_predicted": pred["ilimit_predicted"],
        "omega_n_bare_rad_s": pred["omega_n_bare_rad_s"],
    })
    mpath = DATA_DIR / "section7_2_metrics.csv"
    pd.DataFrame([metrics]).to_csv(mpath, index=False)
    print("Saved:", mpath)
    compare_to_bare(metrics, pred)
    plot_step(data, metrics)

    print(
        "\nIf OS/wn are still off vs bare: raise/lower kd for damping, "
        "kp for speed, update RETUNED_KP/KD/KI, re-run."
    )
    print(
        "Adding inertia slows the plant: without retuning, wn drops and the "
        "motor feels more sluggish; restoring wn needs higher kp (and kd)."
    )


def analyze_existing():
    pred = predict_gains()
    print_prediction(pred)
    csv_path = DATA_DIR / "section7_2_step_retuned.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run without --analyze first.")
    kp, kd, ki, ilimit = resolve_retuned_gains(pred)
    # Prefer gains stored in metrics if present.
    mpath = DATA_DIR / "section7_2_metrics.csv"
    if mpath.exists():
        m0 = pd.read_csv(mpath).iloc[0]
        kp, kd = float(m0["kp"]), float(m0["kd"])
        if "ki" in m0.index:
            ki = float(m0["ki"])
        if "ilimit" in m0.index:
            ilimit = float(m0["ilimit"])
    data, y0, target = infer_step_endpoints(pd.read_csv(csv_path))
    metrics = analyze_step(data, y0, target, kp, kd)
    metrics.update({
        "ki": ki,
        "ilimit": ilimit,
        "alpha": pred["alpha"],
        "kp_predicted": pred["kp_predicted"],
        "kd_predicted": pred["kd_predicted"],
        "ki_predicted": pred["ki_predicted"],
        "ilimit_predicted": pred["ilimit_predicted"],
        "omega_n_bare_rad_s": pred["omega_n_bare_rad_s"],
    })
    pd.DataFrame([metrics]).to_csv(mpath, index=False)
    compare_to_bare(metrics, pred)
    plot_step(data, metrics)


def main():
    parser = argparse.ArgumentParser(description="Q7.2 retune gains with disk mounted")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Only print alpha and predicted kp'/kd' (no motor)",
    )
    parser.add_argument("--analyze", action="store_true", help="Re-analyze saved step CSV")
    parser.add_argument(
        "--keep-gains",
        action="store_true",
        help="Skip moteus_tool; assume gains already set",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        plt.show = lambda *a, **k: None

    if args.predict:
        pred = predict_gains()
        print_prediction(pred)
        return

    if args.analyze:
        analyze_existing()
        return

    asyncio.run(run_live(keep_gains=args.keep_gains))


if __name__ == "__main__":
    main()
