"""Section 5.5 / Q5.5 - Rotor inertia from step overshoot + caliper ring.

By default re-analyzes the earlier Q5.2 underdamped step CSV. Optionally run a
fresh motor step with --live.

  zeta from overshoot Mp,  omega_n from peak time t_p,  J = k_p^rad / omega_n^2

Caliper estimate: OD, radial ring thickness, axial width -> thin steel annulus.

  python section5_5_rotor_inertia.py --no-show
  python section5_5_rotor_inertia.py --live
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

# Gains used when the step was logged (section5_2: kp=1.6, kd=0).
KP = 1.6
KD = 0.0
KI = 0.0
ILIMIT = 0.0

STEP_REV = 0.05
HOME_POSITION_REV = 0.0
HOME_SETTLE_S = 3.0
PRE_STEP_S = 0.5
RUN_S = 3.0
SAMPLE_DT_S = 0.005

# Caliper dims of the outer steel ring (meters).
OUTER_DIAMETER_M = 0.063
RING_WIDTH_M = 0.004  # radial wall thickness
AXIAL_WIDTH_M = 0.0117
INNER_DIAMETER_M = OUTER_DIAMETER_M - 2.0 * RING_WIDTH_M
STEEL_DENSITY_KG_M3 = 7800.0

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section5_5_gains.cfg"
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


async def run_home_and_step(controller):
    """Hold command=0, then step by STEP_REV from the measured position."""
    hold_cmd = HOME_POSITION_REV
    rows = []
    start = time.perf_counter()
    total_pre = HOME_SETTLE_S + PRE_STEP_S
    print(
        f"Hold command={hold_cmd:.6f} for {total_pre:.1f} s, "
        f"then step +{STEP_REV} from measured"
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
    print(f"Step to {step_cmd:.6f}  (measured start={y0:.6f})")

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

    data = pd.DataFrame(rows)
    data = data[data["time_s"] >= HOME_SETTLE_S].copy().reset_index(drop=True)
    data["time_s"] = data["time_s"] - HOME_SETTLE_S
    return data, y0, step_cmd


def analyze_step(data, start_position, target):
    """Report formulas: zeta from Mp, omega_n from peak time t_p, J = kp_rad/wn^2."""
    step_data = data[data["time_s"] >= PRE_STEP_S].copy().reset_index(drop=True)
    t = step_data["time_s"].to_numpy() - PRE_STEP_S
    y = step_data["position_rev"].to_numpy()

    y0 = float(start_position)
    y_cmd = float(target)
    e0 = y_cmd - y0
    if abs(e0) < 1e-9:
        raise RuntimeError("Step size is zero.")

    if e0 > 0:
        i_peak = int(np.argmax(y))
        y_peak = float(y[i_peak])
    else:
        i_peak = int(np.argmin(y))
        y_peak = float(y[i_peak])

    mp = (y_peak - y_cmd) / e0
    if mp <= 1e-4:
        raise RuntimeError(
            f"No overshoot (Mp={mp:.4f}). Lower KD or raise KP and re-run — "
            "Q5.5 needs an underdamped step."
        )

    t_peak = float(t[i_peak])
    progress = (y - y0) * np.sign(e0)
    crossed = np.where(progress >= abs(e0) - 1e-6)[0]
    if crossed.size == 0:
        t_cross = np.nan
        rise_time = np.nan
    else:
        t_cross = float(step_data["time_s"].iloc[int(crossed[0])])
        rise_time = t_cross - PRE_STEP_S

    zeta = -np.log(mp) / np.sqrt(np.pi**2 + np.log(mp) ** 2)
    if zeta >= 1.0:
        raise RuntimeError(f"Computed zeta={zeta:.3f} >= 1; need more overshoot.")
    omega_n = np.pi / (t_peak * np.sqrt(1.0 - zeta**2))
    kp_rad = KP / (2.0 * np.pi)
    inertia = kp_rad / omega_n**2

    return {
        "kp": KP,
        "kd": KD,
        "e0_rev": e0,
        "y0_rev": y0,
        "y_ref_rev": y_cmd,
        "y_peak_rev": y_peak,
        "overshoot_frac": mp,
        "overshoot_percent": 100.0 * mp,
        "t_peak_s": t_peak + PRE_STEP_S,
        "t_peak_from_step_s": t_peak,
        "t_cross_s": t_cross,
        "rise_time_s": rise_time,
        "zeta": zeta,
        "omega_n_rad_s": omega_n,
        "kp_Nm_per_rev": KP,
        "kp_Nm_per_rad": kp_rad,
        "inertia_kg_m2": inertia,
    }


def geometric_ring_inertia():
    """Annular cylinder / thin steel ring about its axis: J = 1/2 m (Ri^2 + Ro^2)."""
    dims = (OUTER_DIAMETER_M, INNER_DIAMETER_M, AXIAL_WIDTH_M)
    if any(np.isnan(v) for v in dims):
        return None
    outer_r = OUTER_DIAMETER_M / 2.0
    inner_r = INNER_DIAMETER_M / 2.0
    if outer_r <= inner_r or AXIAL_WIDTH_M <= 0:
        raise ValueError("Need OUTER > INNER diameter and positive axial width.")
    volume = np.pi * (outer_r**2 - inner_r**2) * AXIAL_WIDTH_M
    mass = STEEL_DENSITY_KG_M3 * volume
    inertia = 0.5 * mass * (outer_r**2 + inner_r**2)
    return {
        "outer_diameter_m": OUTER_DIAMETER_M,
        "inner_diameter_m": INNER_DIAMETER_M,
        "ring_width_m": RING_WIDTH_M,
        "axial_width_m": AXIAL_WIDTH_M,
        "density_kg_m3": STEEL_DENSITY_KG_M3,
        "mass_kg": mass,
        "inertia_kg_m2": inertia,
    }


def infer_step_endpoints(data: pd.DataFrame):
    """Recover y0 / target from a logged step CSV (absolute multi-turn positions OK)."""
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


def plot_step(data, metrics):
    y0 = metrics["y0_rev"]
    y_ref = metrics["y_ref_rev"]
    pos_rel = data["position_rev"] - y0
    cmd_rel = np.where(
        data["time_s"].to_numpy() < PRE_STEP_S, 0.0, y_ref - y0
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["time_s"], cmd_rel, label="Command / target")
    ax.plot(data["time_s"], pos_rel, label="Measured")
    ax.axhline(metrics["y_peak_rev"] - y0, color="C3", linestyle=":", label="Peak")
    ax.axvline(metrics["t_peak_s"], color="C1", linestyle="--", label="Peak time")
    if not np.isnan(metrics["t_cross_s"]):
        ax.axvline(metrics["t_cross_s"], color="C2", linestyle="--", label="First crossing")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    ax.set_title(
        f"Q5.5 Step (kp={KP}, kd={KD})  "
        f"OS={metrics['overshoot_percent']:.1f}%  "
        f"ζ={metrics['zeta']:.3f}  "
        f"ωn={metrics['omega_n_rad_s']:.1f} rad/s  "
        f"J={metrics['inertia_kg_m2']:.3e} kg·m²"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def save_and_report(data, metrics):
    csv_path = DATA_DIR / "section5_5_step.csv"
    data.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    print("\n=== Q5.5 step-response inertia ===")
    print(f"Overshoot Mp:           {metrics['overshoot_percent']:.2f} %")
    print(f"Peak time t_p:          {metrics['t_peak_from_step_s']:.4f} s")
    print(f"First-crossing rise:    {metrics['rise_time_s']}")
    print(f"Damping ratio zeta:     {metrics['zeta']:.4f}")
    print(f"Natural freq omega_n:   {metrics['omega_n_rad_s']:.2f} rad/s")
    print(f"kp (rad):               {metrics['kp_Nm_per_rad']:.6f} N·m/rad")
    print(f"Rotor inertia J:        {metrics['inertia_kg_m2']:.6e} kg·m²")

    geo = geometric_ring_inertia()
    if geo is None:
        print(
            "\nCaliper ring skipped — set OUTER_DIAMETER_M, RING_WIDTH_M, "
            "AXIAL_WIDTH_M at top of script, then re-run."
        )
    else:
        print("\n=== Caliper thin-ring estimate ===")
        print(
            f"OD / ID / ring / axial: "
            f"{geo['outer_diameter_m']*1e3:.1f} / {geo['inner_diameter_m']*1e3:.1f} / "
            f"{geo['ring_width_m']*1e3:.1f} / {geo['axial_width_m']*1e3:.1f} mm"
        )
        print(f"Assumed density:        {geo['density_kg_m3']:.0f} kg/m³")
        print(f"Ring mass:              {geo['mass_kg']:.4f} kg")
        print(f"Ring inertia:           {geo['inertia_kg_m2']:.6e} kg·m²")
        print(
            f"Ratio step/caliper:     "
            f"{metrics['inertia_kg_m2'] / geo['inertia_kg_m2']:.3f}"
        )
        metrics.update({f"geo_{k}": v for k, v in geo.items()})

    metrics_path = DATA_DIR / "section5_5_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)
    plot_step(data, metrics)


def analyze_existing(csv_path: Path):
    print(f"Offline analysis of {csv_path}")
    raw = pd.read_csv(csv_path)
    data, y0, target = infer_step_endpoints(raw)
    metrics = analyze_step(data, y0, target)
    save_and_report(data, metrics)


async def run_experiment():
    print("Q5.5 needs visible overshoot — using underdamped gains.")
    print(f"kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    input("Close tview, then press Enter to apply gains and run the step...")

    apply_gains(KP, KD, KI, ILIMIT)
    controller = moteus.Controller(id=MOTOR_ID)
    try:
        await controller.set_stop()
        data, y0, target = await run_home_and_step(controller)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    metrics = analyze_step(data, y0, target)
    save_and_report(data, metrics)


def main():
    parser = argparse.ArgumentParser(description="Q5.5 rotor inertia from step + calipers")
    parser.add_argument(
        "--analyze",
        nargs="?",
        const="auto",
        metavar="CSV",
        help="Re-analyze a step CSV (default action; prefers section5_2_step.csv)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run a fresh motor step instead of reusing earlier CSV data",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show() (still writes CSVs)",
    )
    args = parser.parse_args()

    if args.no_show:
        plt.show = lambda *a, **k: None

    if args.live:
        asyncio.run(run_experiment())
        return

    csv_arg = "auto" if args.analyze is None else args.analyze
    if csv_arg == "auto":
        candidates = [
            DATA_DIR / "section5_2_step.csv",
            DATA_DIR / "section5_5_step.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise SystemExit("No step CSV found. Run with --live after closing tview.")
    else:
        csv_path = Path(csv_arg)
    analyze_existing(csv_path)


if __name__ == "__main__":
    main()
