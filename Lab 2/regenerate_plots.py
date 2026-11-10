"""Regenerate all Lab 2 report plots from saved data (no motor needed).

Experiment scripts only show plots; this saves figures to Plots/ and lab2_data/.

  python regenerate_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
PLOTS_DIR = HERE / "Plots"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

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
    "style_sheet": None,
}

PRE_STEP_S = 0.5


def apply_style():
    if STYLE.get("style_sheet"):
        plt.style.use(STYLE["style_sheet"])
    rc = {k: v for k, v in STYLE.items() if k != "style_sheet" and v is not None}
    mpl.rcParams.update(rc)


def save_fig(fig: plt.Figure, *names: str) -> Path:
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


def _plot_step_rel(data: pd.DataFrame, title: str, out_name: str, pre_step_s: float = PRE_STEP_S):
    fig, ax = plt.subplots()
    t = data["time_s"].to_numpy()
    cmd = data["command_rev"].to_numpy()
    pos = data["position_rev"].to_numpy()
    y0 = float(pos[0])
    cmd0 = float(cmd[0])
    stepped = np.where(np.abs(cmd - cmd0) > 1e-9)[0]
    if stepped.size:
        y_cmd = float(cmd[stepped[0]])
        t_step = float(t[stepped[0]])
        cmd_rel = np.where(t < t_step, 0.0, y_cmd - y0)
    else:
        cmd_rel = cmd - cmd0
    pos_rel = pos - y0
    ax.plot(t, cmd_rel, label="Command / target")
    ax.plot(t, pos_rel, label="Measured")
    ax.axvline(pre_step_s, color="0.5", linestyle=":", label="Step")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [rev]")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_name)


def plot_section6_1():
    data = load_csv("section6_1_step.csv")
    if data is None:
        return
    kp = data["kp"].iloc[0] if "kp" in data.columns else "?"
    kd = data["kd"].iloc[0] if "kd" in data.columns else "?"
    ki = data["ki"].iloc[0] if "ki" in data.columns else "?"
    # Prefer deg relative plot matching the live script.
    y0 = float(data["position_rev"].iloc[0])
    fig, ax = plt.subplots()
    ax.plot(data["time_s"], (data["command_rev"] - y0) * 360.0, label="Command")
    ax.plot(data["time_s"], (data["position_rev"] - y0) * 360.0, label="Measured")
    ax.axvline(PRE_STEP_S, color="0.5", linestyle=":", label="Step")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position relative to start [deg]")
    ax.set_title(f"§6.1 Step (kp={kp}, kd={kd}, ki={ki})")
    ax.legend()
    save_fig(fig, "section6_1_step.png")


def plot_section6_2():
    step = load_csv("section6_2_step.csv")
    metrics = load_csv("section6_2_metrics.csv")
    if step is None:
        return
    if metrics is not None:
        m = metrics.iloc[0]
        y0 = float(m["y0_rev"])
        y_ref = float(m["y_ref_rev"])
        title = (
            f"§6.2 Step (kp={m['kp']}, kd={m['kd']})  "
            f"OS={float(m['overshoot_percent']):.2f}%"
        )
        t_cross = float(m["t_cross_s"]) if pd.notna(m.get("t_cross_s", np.nan)) else np.nan
        y_peak = float(m["y_peak_rev"])
    else:
        y0 = float(step["position_rev"].iloc[0])
        y_ref = float(step["command_rev"].iloc[-1])
        title = "§6.2 Step response"
        t_cross = np.nan
        y_peak = float(step["position_rev"].max())

    pos_rel = step["position_rev"] - y0
    cmd_rel = np.where(step["time_s"].to_numpy() < PRE_STEP_S, 0.0, y_ref - y0)
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
    save_fig(fig, "section6_2_step.png")


def plot_section6_3():
    summary = load_csv("section6_3_summary.csv")
    if summary is not None:
        s = summary.iloc[0]
        kp, kd, ki = s.get("kp", "?"), s.get("kd", "?"), s.get("ki", "?")
        ilimit = s.get("ilimit_nm", "?")
        ilimit_max = s.get("ilimit_max_nm", 1.63)
    else:
        kp = kd = ki = ilimit = "?"
        ilimit_max = 1.63

    step1 = load_csv("section6_3_step_ilimit.csv")
    if step1 is not None:
        _plot_step_rel(
            step1,
            f"§6.3 Step (ki={ki}, ilimit={ilimit} N·m)",
            "section6_3_step_ilimit.png",
        )

    trial_a = load_csv("section6_3_hold_release_ilimit.csv")
    if trial_a is not None:
        _plot_hold_deg(
            trial_a,
            f"§6.3 Offset→return (ilimit={ilimit} N·m)",
            "section6_3_hold_release_ilimit.png",
        )

    trial_b = load_csv("section6_3_hold_release_ilimit_max.csv")
    if trial_b is not None:
        _plot_hold_deg(
            trial_b,
            f"§6.3 Offset→return (ilimit={ilimit_max} N·m)",
            "section6_3_hold_release_ilimit_max.png",
        )

    step = load_csv("section6_3_step.csv")
    if step is not None:
        _plot_step_rel(
            step,
            f"§6.3 Final step (kp={kp}, kd={kd}, ki={ki}, ilimit={ilimit})",
            "section6_3_step.png",
        )


def plot_section6_4():
    step = load_csv("section6_4_step.csv")
    metrics = load_csv("section6_4_inertia_metrics.csv")
    if step is None:
        return
    if metrics is not None:
        m = metrics.iloc[0]
        title = (
            f"Q6.2 Step response (kp={m.get('kp_Nm_per_rev', '?')})  "
            f"OS={float(m['overshoot_percent']):.1f}%  "
            f"tr={float(m['rise_time_s'])*1e3:.1f} ms  "
            f"J={float(m['inertia_kg_m2']):.4e} kg m$^2$"
        )
    else:
        title = "Q6.2 Step response"
        m = None

    fig, ax = plt.subplots()
    ax.plot(step["time_s"], step["command_rev"], label="Command")
    ax.plot(step["time_s"], step["position_rev"], label="Measured")
    if m is not None:
        if pd.notna(m.get("t_cross_s", np.nan)):
            ax.axvline(float(m["t_cross_s"]), color="C2", linestyle="--", label="First crossing")
        if pd.notna(m.get("y_peak_rev", np.nan)):
            ax.axhline(float(m["y_peak_rev"]), color="C3", linestyle=":", label="Peak")
        if pd.notna(m.get("y_ref_rev", np.nan)):
            ax.axhline(float(m["y_ref_rev"]), color="C4", linestyle="-.", label="Reference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [rev]")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, "section6_4_step.png")


def plot_section6_5():
    data = load_csv("section6_5_chirp.csv")
    if data is None:
        return
    try:
        import section6_5_frequency_response as s65

        frequency, magnitude_db, phase_deg, sample_rate = s65.bode_from_chirp(data)
        metrics = s65.extract_metrics(frequency, magnitude_db, sample_rate, phase_deg)
        s65.print_metrics(metrics)
        pd.DataFrame([metrics]).to_csv(DATA_DIR / s65.METRICS_NAME, index=False)
        fig = s65.plot_bode(frequency, magnitude_db, phase_deg, metrics)
        save_fig(fig, "section6_5_bode.png")
    except Exception as exc:
        print(f"  ERROR bode regenerate failed: {exc}")


PLOTTERS = [
    ("§6.1 kp step", plot_section6_1),
    ("§6.2 critical damping", plot_section6_2),
    ("§6.3 integral", plot_section6_3),
    ("Q6.2 inertia step", plot_section6_4),
    ("Q6.3 bode", plot_section6_5),
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
    print("Done.")


if __name__ == "__main__":
    main()
