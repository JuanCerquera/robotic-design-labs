"""Section 7.1 - Disk inertia: CAD mass scaling + step before/after mount.

Handout reference (scale CAD by your measured mass):
  J_disk_CAD = 6.6185e-5 * (m_g / 76.85)

Identified from natural frequencies at the *same* gains:
  J_disk = J_rotor * ((wn_bare / wn_loaded)^2 - 1)

Default run is a guided live sequence:
  1) bare-shaft step  -> plot
  2) prompt to mount disk
  3) loaded step      -> plot
  4) print J_disk vs CAD

  python section7_1_identify_inertia.py
  python section7_1_identify_inertia.py --bare
  python section7_1_identify_inertia.py --loaded
  python section7_1_identify_inertia.py --loaded --keep-gains
  python section7_1_identify_inertia.py --analyze
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

# Section 5 settled gains (Q5.4), with KD slightly below critical so both
# bare and loaded steps still show measurable overshoot (needed for wn from tp).
# Keep these identical for bare and loaded.
KP = 1.6
KD = 0.05      # Q5.3 critical was ~0.075; a bit lower for Mp > 0
KI = 2.68      # from Q5.4 — not zero
ILIMIT = 0.0022

STEP_REV = 0.05
HOME_POSITION_REV = 0.0
HOME_SETTLE_S = 3.0
PRE_STEP_S = 0.5
RUN_S = 3.0
SAMPLE_DT_S = 0.005

# Your printed disk mass [grams].
DISK_MASS_G = 72.4
CAD_INERTIA_AT_76_85_G = 6.6185e-5  # kg m^2 at 76.85 g
CAD_REF_MASS_G = 76.85

# Rotor inertia from Q5.5 step (kg m^2).
ROTOR_INERTIA_KG_M2 = 1.114095e-4

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def cad_disk_inertia(mass_g=DISK_MASS_G):
    return CAD_INERTIA_AT_76_85_G * (mass_g / CAD_REF_MASS_G)


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section7_1_gains.cfg"
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
    print(f"Writing gains (kp={kp}, kd={kd}, ki={ki}, ilimit={ilimit})...")
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
            "moteus_tool --write-config failed. Close tview (and any other "
            "moteus script), wait a second, then retry. Or use "
            "--loaded --keep-gains if gains are already set."
        )
    print("Gains set (RAM).")


async def run_home_and_step(controller):
    hold_cmd = HOME_POSITION_REV
    rows = []
    start = time.perf_counter()
    total_pre = HOME_SETTLE_S + PRE_STEP_S
    print(
        f"  Holding at {hold_cmd:.3f} rev for {total_pre:.1f} s, "
        f"then step +{STEP_REV:.3f} rev ..."
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
            ilimit_scale=1.0,
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
    print(f"  STEPPING now: {y0:.6f} -> {step_cmd:.6f} rev  (watch the shaft)")

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
            ilimit_scale=1.0,
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
    return data, y0, step_cmd


def analyze_step(data, start_position, target):
    """zeta from Mp, omega_n from peak time (same as Q5.5)."""
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
    if mp <= 1e-4:
        raise RuntimeError(
            f"No overshoot (Mp={mp:.4f}). Q7.1 needs a bit of overshoot to get "
            "wn from the peak. Slightly lower KD (e.g. 0.05) and re-run both "
            "bare and loaded with the SAME kd."
        )
    t_peak = float(t[i_peak])
    zeta = -np.log(mp) / np.sqrt(np.pi**2 + np.log(mp) ** 2)
    if zeta >= 1.0:
        raise RuntimeError(f"zeta={zeta:.3f} >= 1; need more overshoot.")
    omega_n = np.pi / (t_peak * np.sqrt(1.0 - zeta**2))
    return {
        "kp": KP,
        "kd": KD,
        "y0_rev": y0,
        "y_ref_rev": y_cmd,
        "y_peak_rev": y_peak,
        "overshoot_percent": 100.0 * mp,
        "t_peak_from_step_s": t_peak,
        "t_peak_s": t_peak + PRE_STEP_S,
        "zeta": zeta,
        "omega_n_rad_s": omega_n,
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


def plot_step(data, metrics, title):
    y0 = metrics["y0_rev"]
    y_ref = metrics["y_ref_rev"]
    pos_rel = data["position_rev"] - y0
    cmd_rel = np.where(data["time_s"].to_numpy() < PRE_STEP_S, 0.0, y_ref - y0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], cmd_rel, label="Command / target")
    ax.plot(data["time_s"], pos_rel, label="Measured")
    ax.axhline(metrics["y_peak_rev"] - y0, color="C3", linestyle=":", label="Peak")
    ax.axvline(metrics["t_peak_s"], color="C1", linestyle="--", label="Peak time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    ax.set_title(
        f"{title}  OS={metrics['overshoot_percent']:.1f}%  "
        f"ζ={metrics['zeta']:.3f}  ωn={metrics['omega_n_rad_s']:.1f} rad/s"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def load_wn(label):
    path = DATA_DIR / f"section7_1_metrics_{label}.csv"
    if not path.exists():
        return None
    return float(pd.read_csv(path).iloc[0]["omega_n_rad_s"])


def report_summary(wn_bare=None, wn_loaded=None):
    j_cad = cad_disk_inertia()
    print("\n=== CAD mass-scaled reference ===")
    print(f"Disk mass:              {DISK_MASS_G:.1f} g")
    print(f"J_disk,CAD:             {j_cad:.6e} kg·m²")
    print(f"  (6.6185e-5 × {DISK_MASS_G}/{CAD_REF_MASS_G})")

    print("\n=== From closed-loop wn ratio ===")
    print(f"J_rotor (Q5.5):         {ROTOR_INERTIA_KG_M2:.6e} kg·m²")
    if wn_bare is None:
        wn_bare = load_wn("bare")
    if wn_loaded is None:
        wn_loaded = load_wn("loaded")

    metrics = {
        "disk_mass_g": DISK_MASS_G,
        "j_disk_cad_kg_m2": j_cad,
        "j_rotor_kg_m2": ROTOR_INERTIA_KG_M2,
        "omega_n_bare_rad_s": wn_bare if wn_bare is not None else np.nan,
        "omega_n_loaded_rad_s": wn_loaded if wn_loaded is not None else np.nan,
        "j_disk_identified_kg_m2": np.nan,
        "j_total_loaded_kg_m2": np.nan,
        "diff_vs_cad_percent": np.nan,
    }

    if wn_bare is not None and wn_loaded is not None and wn_loaded > 0:
        ratio = (wn_bare / wn_loaded) ** 2
        j_disk = ROTOR_INERTIA_KG_M2 * (ratio - 1.0)
        j_total = ROTOR_INERTIA_KG_M2 * ratio  # = J_rotor + J_disk
        diff = 100.0 * (j_disk - j_cad) / j_cad
        metrics.update({
            "omega_n_bare_rad_s": wn_bare,
            "omega_n_loaded_rad_s": wn_loaded,
            "j_disk_identified_kg_m2": j_disk,
            "j_total_loaded_kg_m2": j_total,
            "diff_vs_cad_percent": diff,
        })
        print(f"ωn bare:                {wn_bare:.2f} rad/s")
        print(f"ωn loaded:              {wn_loaded:.2f} rad/s")
        print(f"J_disk (from wn ratio): {j_disk:.6e} kg·m²")
        print(f"J_total - J_rotor:      {j_total - ROTOR_INERTIA_KG_M2:.6e} kg·m²")
        print(f"vs CAD:                 {diff:+.1f} %")
    else:
        print("Need both bare and loaded step metrics to identify J_disk.")
        print("  Run:  python section7_1_identify_inertia.py --bare")
        print("  then: python section7_1_identify_inertia.py --loaded")
        print("  (or --analyze after CSVs exist)")

    out = DATA_DIR / "section7_1_summary.csv"
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print("Saved:", out)
    return metrics


async def collect_labeled_step(controller, label: str):
    """One step capture + save + plot. Gains must already be applied."""
    print(f"\n--- {label.upper()} step (kp={KP}, kd={KD}) ---")
    data, y0, target = await run_home_and_step(controller)
    csv_path = DATA_DIR / f"section7_1_step_{label}.csv"
    data.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    metrics = analyze_step(data, y0, target)
    print(
        f"{label}: OS={metrics['overshoot_percent']:.1f}%  "
        f"zeta={metrics['zeta']:.3f}  wn={metrics['omega_n_rad_s']:.2f} rad/s"
    )
    mpath = DATA_DIR / f"section7_1_metrics_{label}.csv"
    pd.DataFrame([metrics]).to_csv(mpath, index=False)
    print("Saved:", mpath)
    plot_step(data, metrics, f"Q7.1 {label}")
    return metrics


async def run_guided():
    """Full Q7.1 live flow: bare -> mount disk -> loaded -> compare to CAD."""
    print("=" * 60)
    print("Q7.1 Disk inertia — guided bare then loaded steps")
    print(f"Using Section 5 gains: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    print(f"Disk mass for CAD: {DISK_MASS_G} g")
    print("Tip: re-run only one phase with  --bare  or  --loaded")
    print("=" * 60)

    input(
        "\nSTEP 1/2 — BARE SHAFT\n"
        "  • Remove / leave the inertia disk OFF the shaft\n"
        "  • Close tview\n"
        "  • Press Enter to apply gains and run the bare step..."
    )
    apply_gains(KP, KD, KI, ILIMIT)
    controller = moteus.Controller(id=MOTOR_ID)
    try:
        await controller.set_stop()
        bare = await collect_labeled_step(controller, "bare")
    finally:
        await controller.set_stop()
        del controller
        print("Motor stopped after bare step.\n")

    input(
        "STEP 2/2 — DISK MOUNTED\n"
        "  • Mount the inertia disk on the shaft\n"
        "  • Clamp the stand down\n"
        "  • Gains already set — do not open tview\n"
        "  • Press Enter to run the loaded step..."
    )
    # Skip second moteus_tool write — it often fails while USB is busy.
    print(f"Keeping existing gains (kp={KP}, kd={KD}).")
    await asyncio.sleep(0.5)
    controller = moteus.Controller(id=MOTOR_ID)
    try:
        await controller.set_stop()
        loaded = await collect_labeled_step(controller, "loaded")
    finally:
        await controller.set_stop()
        print("Motor stopped after loaded step.\n")

    report_summary(
        wn_bare=bare["omega_n_rad_s"],
        wn_loaded=loaded["omega_n_rad_s"],
    )


async def run_live(label: str, keep_gains: bool = False):
    """Single-phase live run (bare or loaded only)."""
    print("=" * 60)
    print(f"Q7.1 — {label.upper()} step only")
    print(f"Expected gains: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    print("=" * 60)
    if label == "bare":
        input("Shaft BARE (no disk). Close tview, then Enter...")
    else:
        bare_wn = load_wn("bare")
        if bare_wn is None:
            print(
                "Warning: no bare metrics yet "
                "(lab1_data/section7_1_metrics_bare.csv). "
                "Loaded step will still run; summary needs bare later."
            )
        else:
            print(f"Found bare wn={bare_wn:.2f} rad/s from earlier run.")
        input("Disk MOUNTED, stand clamped. Close tview, then Enter...")

    if keep_gains:
        print("Skipping moteus_tool (--keep-gains); using RAM gains as-is.")
    else:
        apply_gains(KP, KD, KI, ILIMIT)

    controller = moteus.Controller(id=MOTOR_ID)
    try:
        await controller.set_stop()
        metrics = await collect_labeled_step(controller, label)
    finally:
        await controller.set_stop()
        print("Motor stopped")
    report_summary(
        wn_bare=metrics["omega_n_rad_s"] if label == "bare" else load_wn("bare"),
        wn_loaded=metrics["omega_n_rad_s"] if label == "loaded" else load_wn("loaded"),
    )


def analyze_existing(label: str):
    csv_path = DATA_DIR / f"section7_1_step_{label}.csv"
    if label == "bare" and not csv_path.exists():
        alt = DATA_DIR / "section5_5_step.csv"
        if alt.exists():
            print(f"No {csv_path.name}; using {alt.name} as bare baseline.")
            csv_path = alt
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    print(f"Offline analysis of {csv_path}")
    data, y0, target = infer_step_endpoints(pd.read_csv(csv_path))
    metrics = analyze_step(data, y0, target)
    out_csv = DATA_DIR / f"section7_1_step_{label}.csv"
    data.to_csv(out_csv, index=False)
    mpath = DATA_DIR / f"section7_1_metrics_{label}.csv"
    pd.DataFrame([metrics]).to_csv(mpath, index=False)
    print(
        f"{label}: OS={metrics['overshoot_percent']:.1f}%  "
        f"zeta={metrics['zeta']:.3f}  wn={metrics['omega_n_rad_s']:.2f} rad/s"
    )
    print("Saved:", mpath)
    plot_step(data, metrics, f"Q7.1 {label}")


def main():
    parser = argparse.ArgumentParser(
        description="Q7.1 disk inertia identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python section7_1_identify_inertia.py\n"
            "  python section7_1_identify_inertia.py --bare\n"
            "  python section7_1_identify_inertia.py --loaded\n"
            "  python section7_1_identify_inertia.py --loaded --keep-gains\n"
            "  python section7_1_identify_inertia.py --analyze"
        ),
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--bare", action="store_true", help="Run only the bare-shaft step")
    phase.add_argument(
        "--loaded",
        action="store_true",
        help="Run only the disk-mounted step (reuses saved bare metrics)",
    )
    phase.add_argument("--both", action="store_true", help="Guided bare then loaded")
    phase.add_argument(
        "--analyze",
        choices=("bare", "loaded", "all"),
        nargs="?",
        const="all",
        help="Re-analyze step CSV(s) only (no motor)",
    )
    phase.add_argument("--summary", action="store_true", help="CAD / summary only")
    parser.add_argument(
        "--keep-gains",
        action="store_true",
        help="Do not call moteus_tool; assume gains already in RAM",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        plt.show = lambda *a, **k: None

    if args.summary:
        report_summary()
        return

    if args.analyze is not None:
        labels = ("bare", "loaded") if args.analyze == "all" else (args.analyze,)
        for label in labels:
            try:
                analyze_existing(label)
            except SystemExit as exc:
                print(exc)
        report_summary()
        return

    if args.bare:
        asyncio.run(run_live("bare", keep_gains=args.keep_gains))
        return
    if args.loaded:
        asyncio.run(run_live("loaded", keep_gains=args.keep_gains))
        return

    asyncio.run(run_guided())


if __name__ == "__main__":
    main()
