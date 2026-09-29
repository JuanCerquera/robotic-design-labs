"""Regenerate all Lab 1 report plots from saved data (no motor needed).

Experiment scripts only show plots; this is the only place figures are saved.

Edit STYLE below, then run:
  python regenerate_plots.py

Outputs go to Plots/ (what report.tex includes). Copies are also written to
lab1_data/ so existing script paths stay consistent.

Plots that need motor logs you do not have yet are skipped with a message.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
PLOTS_DIR = HERE / "Plots"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Central style — change this once, re-run to restyle every figure.
# ---------------------------------------------------------------------------
STYLE = {
    "figure.figsize": (9.0, 4.0),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    # Optional: set a named style sheet, e.g. "seaborn-v0_8-whitegrid"
    "style_sheet": None,
}


def apply_style():
    if STYLE.get("style_sheet"):
        plt.style.use(STYLE["style_sheet"])
    rc = {k: v for k, v in STYLE.items() if k != "style_sheet" and v is not None}
    mpl.rcParams.update(rc)


def save_fig(fig: plt.Figure, *names: str) -> Path:
    """Save under Plots/ and mirror into lab1_data/. Accepts one or more filenames."""
    if not names:
        raise ValueError("save_fig requires at least one filename")
    for name in names:
        for folder in (PLOTS_DIR, DATA_DIR):
            out = folder / name
            fig.savefig(out)
            print(f"  saved {out.relative_to(HERE)}")
    plt.close(fig)
    return PLOTS_DIR / names[0]


def load_csv(*candidates: str) -> pd.DataFrame | None:
    for name in candidates:
        path = DATA_DIR / name
        if path.exists():
            print(f"  data  {path.name}")
            return pd.read_csv(path)
    print(f"  SKIP  missing data ({', '.join(candidates)})")
    return None


def skip(reason: str):
    print(f"  SKIP  {reason}")


# ---------------------------------------------------------------------------
# Individual plotters
# ---------------------------------------------------------------------------

def plot_section1_3_encoder_noise():
    data = load_csv("section1_3_encoder_noise.csv", "section1_encoder_noise.csv")
    if data is None:
        return
    pos_deg = data["position_rev"].to_numpy() * 360.0
    fig, ax = plt.subplots()
    ax.plot(data["time_s"], pos_deg)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [deg]")
    ax.set_title("Q1.3 Encoder noise floor")
    ax.legend(["Measured"])
    save_fig(fig, "section1_3_encoder_noise.png")


def plot_section2_3_direct_measurement():
    # Scale readings are hardcoded in section2_3_direct_measurement.py
    radius_m = 0.080
    current_rising = np.arange(1.0, 5.01, 0.5)
    current_falling = current_rising[::-1]
    scale_rising_g = np.array(
        [33.6, 49.4, 65.3, 81.6, 97.7, 113.8, 130.2, 146.5, 163.0]
    )
    scale_falling_g = np.array(
        [163.7, 149.4, 134.6, 119.9, 104.2, 88.6, 72.7, 56.7, 37.9]
    )

    def torque_from_scale(scale_g):
        return scale_g / 1000.0 * 9.81 * radius_m

    torque_rising = torque_from_scale(scale_rising_g)
    torque_falling = torque_from_scale(scale_falling_g)
    kt_rise, b_rise = np.polyfit(current_rising, torque_rising, 1)
    kt_fall, b_fall = np.polyfit(current_falling, torque_falling, 1)

    print("  data  hardcoded scale arrays (section2_3)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(current_rising, torque_rising, label="Rising")
    ax.scatter(current_falling, torque_falling, label="Falling")
    ax.plot(current_rising, kt_rise * current_rising + b_rise)
    ax.plot(current_falling, kt_fall * current_falling + b_fall)
    ax.set_xlabel("Commanded q-axis current magnitude [A]")
    ax.set_ylabel("Torque magnitude [N m]")
    ax.set_title("Q2.2 Direct torque-constant measurement")
    ax.legend()
    save_fig(fig, "section2_3_direct_measurement.png")


def plot_section3_1_phase_sweep():
    data = load_csv("section3_1_phase_sweep.csv", "section3_phase_sweep.csv")
    if data is None:
        return
    if "rotor_position_deg" in data.columns:
        y = data["rotor_position_deg"]
        ylabel = "Rotor position [deg]"
    else:
        y = data["rotor_position_rev"] * 360.0
        ylabel = "Rotor position [deg]"
    x = data["commanded_phase_rad"]
    fig, ax = plt.subplots()
    ax.plot(x, y, "o-", label="Measured")
    # Linear fit for pole-pair slope (mech deg vs elec rad → convert carefully)
    coef = np.polyfit(x, y if "rotor_position_deg" in data.columns else data.get("rotor_position_rev", y / 360.0), 1)
    # Prefer fitting in the native y of the CSV when possible
    if "rotor_position_deg" in data.columns:
        fit = np.poly1d(np.polyfit(x, data["rotor_position_deg"], 1))
        ax.plot(x, fit(x), "--", label=f"Fit (slope={fit.c[0]:.3f} deg/rad)")
    elif "rotor_position_rev" in data.columns:
        fit = np.poly1d(np.polyfit(x, data["rotor_position_rev"], 1))
        ax.plot(x, fit(x) * 360.0, "--", label=f"Fit (slope={fit.c[0]:.4f} rev/rad)")
    ax.set_xlabel("Commanded electrical phase [rad]")
    ax.set_ylabel(ylabel)
    ax.set_title("Q3.2 Mechanical angle vs electrical phase")
    ax.legend()
    save_fig(fig, "section3_1_phase_sweep.png")


def plot_section3_2_park_transform():
    timeseries = load_csv("section3_2_park_transform_timeseries.csv")
    averages = load_csv("section3_2_park_transform_averages.csv")
    if timeseries is None or averages is None:
        return

    commands = list(dict.fromkeys(timeseries["command"]))
    series = [
        ("position_deg", "Position [deg]"),
        ("velocity_rev_s", "Velocity [rev/s]"),
        ("measured_id_A", "Measured id [A]"),
        ("measured_iq_A", "Measured iq [A]"),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 9), sharex=True)
    smooth_n = max(3, int(0.05 / max(np.median(np.diff(timeseries["time_s"])), 1e-3)))
    for ax, (col, ylabel) in zip(axes, series):
        for command in commands:
            chunk = timeseries[timeseries["command"] == command].copy()
            ax.plot(chunk["time_s"], chunk[col], alpha=0.25, linewidth=1)
            smoothed = chunk[col].rolling(smooth_n, min_periods=1, center=True).mean()
            ax.plot(chunk["time_s"], smoothed, label=command, linewidth=2)
            settle = averages[
                (averages["command"] == command)
                & (averages["avg_window"].astype(str).str.startswith("last_"))
            ]
            if not settle.empty and col in settle.columns:
                ax.axhline(float(settle.iloc[0][col]), linestyle="--", linewidth=1.4, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time within command [s]")
    fig.suptitle("Q3.3 traces + settle averages (dashed)")
    fig.tight_layout()
    save_fig(fig, "section3_2_park_transform_timeseries.png")

    settle = averages[averages["avg_window"].astype(str).str.startswith("last_")].reset_index(drop=True)
    full = averages[averages["avg_window"] == "full_run"].reset_index(drop=True)
    if settle.empty or full.empty:
        skip("averages CSV missing settle/full windows for bar plot")
        return
    x = np.arange(len(settle))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - 1.5 * width, settle["measured_id_A"], width, label="id settle-avg")
    axes[0].bar(x - 0.5 * width, settle["measured_iq_A"], width, label="iq settle-avg")
    axes[0].bar(x + 0.5 * width, full["measured_id_A"], width, label="id full-avg")
    axes[0].bar(x + 1.5 * width, full["measured_iq_A"], width, label="iq full-avg")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settle["command"], rotation=15, ha="right")
    axes[0].set_ylabel("Current [A]")
    axes[0].set_title("Average id / iq")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 1.5 * width, settle["position_deg"], width, label="pos settle-avg")
    axes[1].bar(x - 0.5 * width, settle["velocity_rev_s"], width, label="vel settle-avg")
    axes[1].bar(x + 0.5 * width, full["position_deg"], width, label="pos full-avg")
    axes[1].bar(x + 1.5 * width, full["velocity_rev_s"], width, label="vel full-avg")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settle["command"], rotation=15, ha="right")
    axes[1].set_ylabel("Position [deg] / Velocity [rev/s]")
    axes[1].set_title("Average position / velocity")
    axes[1].legend(fontsize=8)
    fig.suptitle("Q3.3 Averaged Park-transform measurements")
    fig.tight_layout()
    save_fig(fig, "section3_2_park_transform_averages.png")


def _plot_hold_deg(data: pd.DataFrame, title: str, out_name: str):
    fig, ax = plt.subplots()
    if "command_deg" in data.columns:
        cmd = data["command_deg"]
        pos = data["position_deg"]
    else:
        cmd = data["command_rev"] * 360.0
        pos = data["position_rev"] * 360.0
    ax.plot(data["time_s"], cmd, label="Command")
    ax.plot(data["time_s"], pos, label="Measured")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [deg]")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_name)


def _plot_step_rev(data: pd.DataFrame, title: str, out_name: str, relative: bool = False):
    fig, ax = plt.subplots()
    t = data["time_s"].to_numpy()
    cmd = data["command_rev"].to_numpy()
    pos = data["position_rev"].to_numpy()
    if relative:
        # Relative to measured start; hold command drawn as 0 then step size.
        y0 = float(pos[0])
        # Detect step instant as first time command changes from initial.
        cmd0 = float(cmd[0])
        stepped = np.where(np.abs(cmd - cmd0) > 1e-9)[0]
        if stepped.size:
            y_cmd = float(cmd[stepped[0]])
            cmd_rel = np.where(t < t[stepped[0]], 0.0, y_cmd - y0)
        else:
            cmd_rel = cmd - cmd0
        pos_rel = pos - y0
        ax.plot(t, cmd_rel, label="Command / target")
        ax.plot(t, pos_rel, label="Measured")
        ax.set_ylabel("Position relative to start [rev]")
    else:
        ax.plot(t, cmd, label="Command")
        ax.plot(t, pos, label="Measured")
        ax.set_ylabel("Position [rev]")
    ax.set_xlabel("Time [s]")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_name)


def plot_section5_1():
    hold = load_csv("section5_1_hold_disturbance.csv")
    if hold is not None:
        kp = hold["kp"].iloc[0] if "kp" in hold.columns else "?"
        _plot_hold_deg(
            hold,
            f"Q5.1 Manual disturbance (kp={kp})",
            "section5_1_hold_disturbance.png",
        )
    # Q5.2 step: prefer the underdamped kd=0 capture that actually moved.
    # section5_1_step.csv commanded a step but measured never left the hold
    # (stuck / friction), so that plot looks "wrong".
    step = load_csv(
        "section5_5_step.csv",
        "section5_2_step.csv",
        "section5_1_step.csv",
    )
    if step is not None:
        _plot_step_rev(
            step,
            "Q5.2 Commanded step with $k_d = 0$",
            "section5_1_step.png",
            relative=True,
        )


def plot_section5_2():
    hold = load_csv("section5_2_hold_disturbance.csv")
    if hold is not None:
        _plot_hold_deg(
            hold,
            "Q5.2 Manual disturbance before step",
            "section5_2_hold_disturbance.png",
        )


def plot_section5_5():
    step = load_csv("section5_5_step.csv", "section5_2_step.csv")
    metrics = load_csv("section5_5_metrics.csv", "section5_2_inertia_metrics.csv")
    if step is None:
        return
    title = "Q5.5 Step response (inertia estimate)"
    if metrics is not None:
        m = metrics.iloc[0]
        os_pct = float(m.get("overshoot_percent", np.nan))
        j = float(m.get("inertia_kg_m2", np.nan))
        title = f"Q5.5 Step  OS={os_pct:.1f}%  J={j:.4e} kg m$^2$"
    fig, ax = plt.subplots()
    if metrics is not None and "y0_rev" in metrics.columns:
        y0 = float(metrics.iloc[0]["y0_rev"])
        y_ref = float(metrics.iloc[0]["y_ref_rev"])
        pre = 0.5
        pos_rel = step["position_rev"] - y0
        cmd_rel = np.where(step["time_s"].to_numpy() < pre, 0.0, y_ref - y0)
        ax.plot(step["time_s"], cmd_rel, label="Command / target")
        ax.plot(step["time_s"], pos_rel, label="Measured")
        ax.set_ylabel("Position relative to start [rev]")
        if pd.notna(metrics.iloc[0].get("t_peak_s", np.nan)):
            ax.axvline(
                float(metrics.iloc[0]["t_peak_s"]),
                color="C1",
                linestyle="--",
                label="Peak time",
            )
        if pd.notna(metrics.iloc[0].get("y_peak_rev", np.nan)):
            ax.axhline(
                float(metrics.iloc[0]["y_peak_rev"]) - y0,
                color="C3",
                linestyle=":",
                label="Peak",
            )
    else:
        ax.plot(step["time_s"], step["command_rev"], label="Command")
        ax.plot(step["time_s"], step["position_rev"], label="Measured")
        ax.set_ylabel("Position [rev]")
        if metrics is not None:
            m = metrics.iloc[0]
            if pd.notna(m.get("t_cross_s", np.nan)):
                ax.axvline(
                    float(m["t_cross_s"]), color="C2", linestyle="--", label="First crossing"
                )
            if pd.notna(m.get("y_peak_rev", np.nan)):
                ax.axhline(float(m["y_peak_rev"]), color="C3", linestyle=":", label="Peak")
    ax.set_xlabel("Time [s]")
    ax.set_title(title)
    ax.legend()
    # Primary + legacy report filename
    save_fig(fig, "section5_5_step.png", "section5_2_step_inertia.png")


def plot_section5_3():
    step = load_csv("section5_3_step.csv")
    metrics = load_csv("section5_3_metrics.csv")
    if step is None:
        return
    if metrics is not None:
        m = metrics.iloc[0]
        y0 = float(m["y0_rev"])
        y_ref = float(m["y_ref_rev"])
        title = (
            f"Q5.3 Step (kp={m['kp']}, kd={m['kd']})  "
            f"OS={float(m['overshoot_percent']):.2f}%"
        )
        t_cross = float(m["t_cross_s"]) if pd.notna(m.get("t_cross_s", np.nan)) else np.nan
        y_peak = float(m["y_peak_rev"])
    else:
        y0 = float(step["position_rev"].iloc[0])
        y_ref = float(step["command_rev"].iloc[-1])
        title = "Q5.3 Step response"
        t_cross = np.nan
        y_peak = float(step["position_rev"].max())

    # Match report relative view
    pre_step_s = 0.5
    pos_rel = step["position_rev"] - y0
    cmd_rel = np.where(step["time_s"].to_numpy() < pre_step_s, 0.0, y_ref - y0)
    fig, ax = plt.subplots()
    ax.plot(step["time_s"], cmd_rel, label="Command / target")
    ax.plot(step["time_s"], pos_rel, label="Measured")
    ax.axhline(y_peak - y0, color="C3", linestyle=":", label="Peak")
    if not np.isnan(t_cross):
        ax.axvline(t_cross, color="C2", linestyle="--", label="First crossing")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, "section5_3_step.png")

    # No high-kd CSV available offline
    skip("section5_3_kd_too_high.png — no high-kd CSV on disk")


def plot_section5_4():
    mapping = [
        ("section5_4_displace_release.csv", "section5_4_displace_release.png",
         "Q5.4 Displace-and-release with integral action", "hold"),
        ("section5_4_hold_release_ilimit.csv", "section5_4_hold_release_ilimit.png",
         "Q5.4 Hold-off then release (chosen ilimit)", "hold"),
        ("section5_4_hold_release_ilimit_max.csv", "section5_4_hold_release_ilimit_max.png",
         "Q5.4 Hold-off then release (ilimit = 0.1 N m)", "hold"),
        ("section5_4_step.csv", "section5_4_step.png",
         "Q5.4 Step with integral", "step"),
    ]
    any_done = False
    for csv_name, png_name, title, kind in mapping:
        data = load_csv(csv_name)
        if data is None:
            continue
        any_done = True
        if kind == "hold":
            _plot_hold_deg(data, title, png_name)
        else:
            _plot_step_rev(data, title, png_name, relative=True)
    if not any_done:
        skip("all section5_4 plots — run section5_4_integral.py on the motor first")


def plot_section6_1():
    data = load_csv("section6_1_chirp.csv")
    if data is None:
        skip("section6_1_bode.png — no chirp CSV")
        return

    # Prefer shared analyzer (annotations + metrics CSV).
    try:
        import section6_1_frequency_response as s61

        frequency, magnitude_db, phase_deg, sample_rate = s61.bode_from_chirp(data)
        metrics = s61.extract_metrics(frequency, magnitude_db, sample_rate, phase_deg)
        s61.print_metrics(metrics)
        pd.DataFrame([metrics]).to_csv(DATA_DIR / s61.METRICS_NAME, index=False)
        fig = s61.plot_bode(frequency, magnitude_db, phase_deg, metrics)
        save_fig(fig, "section6_1_bode.png")
        return
    except Exception as exc:
        print(f"  WARN shared analyzer failed ({exc}); falling back")

    from scipy import signal

    start_hz, end_hz = 0.5, 100.0
    times = data["time_s"].to_numpy()
    uniform_time = np.linspace(times[0], times[-1], len(times))
    command = np.interp(uniform_time, times, data["command_rev"])
    measured = np.interp(uniform_time, times, data["position_rev"])
    command -= np.mean(command)
    measured -= np.mean(measured)
    sample_rate = 1.0 / np.mean(np.diff(uniform_time))
    nperseg = min(4096, len(command) // 4)
    frequency, cross_spectrum = signal.csd(
        command, measured, fs=sample_rate, nperseg=nperseg
    )
    _, command_spectrum = signal.welch(command, fs=sample_rate, nperseg=nperseg)
    response = cross_spectrum / command_spectrum
    keep = (frequency >= start_hz) & (frequency <= end_hz)
    frequency = frequency[keep]
    response = response[keep]
    magnitude_db = 20.0 * np.log10(np.abs(response))
    phase_deg = np.unwrap(np.angle(response)) * 180.0 / np.pi
    reference_db = np.median(magnitude_db[frequency < 1.0])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].semilogx(frequency, magnitude_db)
    axes[0].axhline(reference_db - 3.0, color="k", linestyle="--")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].grid(True, which="both")
    axes[1].semilogx(frequency, phase_deg)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].grid(True, which="both")
    fig.suptitle("Q6.1 Closed-loop frequency response")
    fig.tight_layout()
    save_fig(fig, "section6_1_bode.png")


def plot_section7():
    def _q7_step(csv_name, metrics_name, title, out_name):
        step = load_csv(csv_name)
        if step is None:
            return
        metrics = load_csv(metrics_name)
        fig, ax = plt.subplots()
        if metrics is not None and "y0_rev" in metrics.columns:
            m = metrics.iloc[0]
            y0 = float(m["y0_rev"])
            y_ref = float(m["y_ref_rev"])
            pre = 0.5
            pos_rel = step["position_rev"] - y0
            cmd_rel = np.where(step["time_s"].to_numpy() < pre, 0.0, y_ref - y0)
            ax.plot(step["time_s"], cmd_rel, label="Command / target")
            ax.plot(step["time_s"], pos_rel, label="Measured")
            ax.set_ylabel("Position relative to start [rev]")
            os_pct = float(m.get("overshoot_percent", np.nan))
            wn = float(m.get("omega_n_rad_s", np.nan))
            title = f"{title}  OS={os_pct:.1f}%  ωn={wn:.1f} rad/s"
            if "kp" in m.index:
                title = (
                    f"{title.split('  OS')[0]} (kp={m['kp']}, kd={m['kd']})"
                    f"  OS={os_pct:.1f}%  ωn={wn:.1f} rad/s"
                )
        else:
            ax.plot(step["time_s"], step["command_rev"], label="Command")
            ax.plot(step["time_s"], step["position_rev"], label="Measured")
            ax.set_ylabel("Position [rev]")
        ax.set_xlabel("Time [s]")
        ax.set_title(title)
        ax.legend()
        save_fig(fig, out_name)

    _q7_step(
        "section7_1_step_bare.csv",
        "section7_1_metrics_bare.csv",
        "Q7.1 Bare step",
        "section7_1_step_bare.png",
    )
    _q7_step(
        "section7_1_step_loaded.csv",
        "section7_1_metrics_loaded.csv",
        "Q7.1 Loaded step",
        "section7_1_step_loaded.png",
    )
    _q7_step(
        "section7_2_step_retuned.csv",
        "section7_2_metrics.csv",
        "Q7.2 Retuned step",
        "section7_2_step_retuned.png",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLOTTERS = [
    ("Q1.3 encoder noise", plot_section1_3_encoder_noise),
    ("Q2.2 / Q2.3 Kt sweep", plot_section2_3_direct_measurement),
    ("Q3.2 phase sweep", plot_section3_1_phase_sweep),
    ("Q3.3 Park transform", plot_section3_2_park_transform),
    ("Q5.1 / Q5.2 hold+step", plot_section5_1),
    ("Q5.2 hold", plot_section5_2),
    ("Q5.5 inertia step", plot_section5_5),
    ("Q5.3 critical damping", plot_section5_3),
    ("Q5.4 integral", plot_section5_4),
    ("Q6.1 bode", plot_section6_1),
    ("Q7 inertia disk", plot_section7),
]


def main():
    apply_style()
    print(f"Data dir:  {DATA_DIR}")
    print(f"Plots dir: {PLOTS_DIR}\n")
    for label, fn in PLOTTERS:
        print(f"[{label}]")
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}")
        print()
    print("Done. Copy/update Plots/ as needed for report.tex.")


if __name__ == "__main__":
    main()
