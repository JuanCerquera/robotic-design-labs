"""Section 6 / Q6.3 - Chirp frequency response → −3 dB bandwidth.

Same Lab 1 §6.1 chirp. Report bandwidth and compare with the mj5208 from Lab 1.

  python section6_5_frequency_response.py              # live chirp
  python section6_5_frequency_response.py --analyze    # reuse CSV
  python section6_5_frequency_response.py --analyze --no-show
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
from scipy import signal


MOTOR_ID = 1
MAX_TORQUE_NM = 1.63
AMPLITUDE_REV = 0.005
START_FREQUENCY_HZ = 0.5
END_FREQUENCY_HZ = 100.0
DURATION_S = 30.0

# Settled gains from §6.1 tuning — edit after you finish section6_3.
KP = 1.0
KD = 0.05
KI = 1.5
ILIMIT = 0.01

FILENAME = "section6_5_chirp.csv"
METRICS_NAME = "section6_5_metrics.csv"

# Lab 1 mj5208 −3 dB bandwidth from lab1_data/section6_1_metrics.csv.
MJ5208_BANDWIDTH_HZ = 33.34

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
DATA_DIR.mkdir(exist_ok=True)


def apply_gains(kp, kd, ki, ilimit):
    cfg_path = DATA_DIR / "section6_5_gains.cfg"
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
            "moteus_tool --write-config failed. Close tview and retry."
        )
    print("Gains set (RAM).")


async def collect_chirp():
    controller = moteus.Controller(id=MOTOR_ID)
    first = await controller.query()
    center = first.values[moteus.Register.POSITION]
    rows = []
    start = time.perf_counter()
    sweep_constant = np.log(END_FREQUENCY_HZ / START_FREQUENCY_HZ) / DURATION_S

    print(
        f"Chirp {START_FREQUENCY_HZ:.1f} → {END_FREQUENCY_HZ:.1f} Hz "
        f"over {DURATION_S:.0f} s  (kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT})"
    )
    last_print_s = -1.0
    try:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= DURATION_S:
                break
            freq_hz = START_FREQUENCY_HZ * np.exp(sweep_constant * elapsed)
            if elapsed - last_print_s >= 0.5:
                print(f"  t={elapsed:5.1f} s   f={freq_hz:6.2f} Hz", flush=True)
                last_print_s = elapsed
            phase = (
                2.0 * np.pi * START_FREQUENCY_HZ
                * (np.exp(sweep_constant * elapsed) - 1.0)
                / sweep_constant
            )
            command = center + AMPLITUDE_REV * np.sin(phase)
            result = await controller.set_position(
                position=command,
                velocity=0.0,
                kp_scale=1.0,
                kd_scale=1.0,
                ilimit_scale=1.0,
                maximum_torque=MAX_TORQUE_NM,
                query=True,
            )
            rows.append({
                "time_s": elapsed,
                "command_rev": command,
                "position_rev": result.values[moteus.Register.POSITION],
                "frequency_hz": freq_hz,
            })
            await asyncio.sleep(0.002)
    finally:
        await controller.set_stop()
        print("Motor stopped")

    data = pd.DataFrame(rows)
    data.to_csv(DATA_DIR / FILENAME, index=False)
    return data


def bode_from_chirp(data: pd.DataFrame):
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
    keep = ((frequency >= START_FREQUENCY_HZ)
            & (frequency <= END_FREQUENCY_HZ))
    frequency = frequency[keep]
    response = response[keep]
    magnitude_db = 20.0 * np.log10(np.abs(response))
    phase_deg = np.unwrap(np.angle(response)) * 180.0 / np.pi
    return frequency, magnitude_db, phase_deg, sample_rate


def _smooth_db(magnitude_db, window=7):
    n = len(magnitude_db)
    w = max(3, int(window) | 1)
    kernel = np.ones(w) / w
    return np.convolve(magnitude_db, kernel, mode="same")


def extract_metrics(frequency, magnitude_db, sample_rate, phase_deg=None):
    del phase_deg
    mag_s = _smooth_db(magnitude_db)

    low = frequency < max(1.5, START_FREQUENCY_HZ * 2.0)
    if not np.any(low):
        low = slice(0, max(5, len(frequency) // 20))
    reference_db = float(np.median(magnitude_db[low]))

    f_lo, f_hi = 5.0, 40.0
    search = (frequency >= f_lo) & (frequency <= f_hi)
    if not np.any(search):
        search = (frequency >= frequency[0]) & (frequency <= min(f_hi, frequency[-1]))

    idx = np.flatnonzero(search)
    mag_band = mag_s[search]
    peaks_rel, _props = signal.find_peaks(
        mag_band,
        prominence=0.75,
        distance=max(3, len(mag_band) // 20),
    )
    if peaks_rel.size:
        i_rel = int(peaks_rel[np.argmax(mag_band[peaks_rel])])
    else:
        i_rel = int(np.argmax(mag_band))
    i_peak = int(idx[i_rel])

    half_win = max(2, int(0.05 * len(idx)))
    i0 = max(int(idx[0]), i_peak - half_win)
    i1 = min(int(idx[-1]), i_peak + half_win)
    i_raw = i0 + int(np.argmax(magnitude_db[i0:i1 + 1]))
    peak_db = float(magnitude_db[i_raw])
    f_peak_hz = float(frequency[i_raw])
    peak_rel_db = peak_db - reference_db

    thresh = reference_db - 3.0
    bandwidth_hz = np.nan
    after = np.flatnonzero(frequency >= f_peak_hz)
    for i in after:
        if magnitude_db[i] <= thresh and i + 2 < len(magnitude_db):
            if np.all(mag_s[i:i + 3] <= thresh):
                bandwidth_hz = float(frequency[i])
                break

    mr = float(10.0 ** (peak_rel_db / 20.0))
    vs_mj_pct = np.nan
    if np.isfinite(bandwidth_hz) and MJ5208_BANDWIDTH_HZ > 0:
        vs_mj_pct = 100.0 * (bandwidth_hz - MJ5208_BANDWIDTH_HZ) / MJ5208_BANDWIDTH_HZ

    return {
        "kp": KP,
        "kd": KD,
        "ki": KI,
        "ilimit": ILIMIT,
        "sample_rate_hz": sample_rate,
        "reference_db": reference_db,
        "bandwidth_hz": bandwidth_hz,
        "peak_search_lo_hz": f_lo,
        "peak_search_hi_hz": f_hi,
        "peak_db": peak_db,
        "peak_rel_db": peak_rel_db,
        "f_peak_hz": f_peak_hz,
        "Mr": mr,
        "mj5208_bandwidth_hz": MJ5208_BANDWIDTH_HZ,
        "diff_vs_mj5208_percent": vs_mj_pct,
    }


def print_metrics(m):
    print("\n=== Q6.3 bandwidth ===")
    print(f"Sample rate:            {m['sample_rate_hz']:.1f} Hz")
    print(f"DC / low-freq level:    {m['reference_db']:.2f} dB")
    print(f"-3 dB bandwidth:        {m['bandwidth_hz']:.2f} Hz")
    print(f"Peak magnitude:         {m['peak_db']:.2f} dB  "
          f"({m['peak_rel_db']:+.2f} dB vs DC)")
    print(f"mj5208 (Lab 1) BW:      {m['mj5208_bandwidth_hz']:.2f} Hz")
    if np.isfinite(m["diff_vs_mj5208_percent"]):
        print(f"RI60 vs mj5208:         {m['diff_vs_mj5208_percent']:+.1f} %")


def plot_bode(frequency, magnitude_db, phase_deg, metrics):
    ref = metrics["reference_db"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    axes[0].semilogx(frequency, magnitude_db, color="C0", label="Magnitude")
    axes[0].axvspan(
        metrics["peak_search_lo_hz"], metrics["peak_search_hi_hz"],
        color="C3", alpha=0.08, label="peak search band",
    )
    axes[0].axhline(ref, color="0.5", linestyle=":", label=f"DC ({ref:.1f} dB)")
    axes[0].axhline(ref - 3.0, color="k", linestyle="--", label="-3 dB vs DC")
    if np.isfinite(metrics["bandwidth_hz"]):
        axes[0].axvline(
            metrics["bandwidth_hz"], color="C2", linestyle="--",
            label=f"BW={metrics['bandwidth_hz']:.1f} Hz",
        )
    axes[0].axvline(
        metrics["f_peak_hz"], color="C3", linestyle="-.",
        label=f"fr={metrics['f_peak_hz']:.1f} Hz",
    )
    axes[0].axvline(
        MJ5208_BANDWIDTH_HZ, color="C4", linestyle=":",
        label=f"mj5208 BW={MJ5208_BANDWIDTH_HZ:.1f} Hz",
    )
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].grid(True, which="both")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].semilogx(frequency, phase_deg)
    if np.isfinite(metrics["bandwidth_hz"]):
        axes[1].axvline(metrics["bandwidth_hz"], color="C2", linestyle="--")
    axes[1].axvline(metrics["f_peak_hz"], color="C3", linestyle="-.")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].grid(True, which="both")
    clean = (frequency >= 0.5) & (frequency <= metrics["peak_search_hi_hz"])
    if np.any(clean):
        lo, hi = np.percentile(phase_deg[clean], [2, 98])
        pad = max(15.0, 0.15 * (hi - lo + 1e-9))
        axes[1].set_ylim(lo - pad, hi + pad)

    bw_s = f"{metrics['bandwidth_hz']:.1f}" if np.isfinite(metrics["bandwidth_hz"]) else "n/a"
    fig.suptitle(f"Q6.3 Bode  BW={bw_s} Hz  (mj5208 Lab1 = {MJ5208_BANDWIDTH_HZ:.1f} Hz)")
    fig.tight_layout()
    return fig


def analyze(data: pd.DataFrame):
    frequency, magnitude_db, phase_deg, sample_rate = bode_from_chirp(data)
    metrics = extract_metrics(frequency, magnitude_db, sample_rate, phase_deg)
    print_metrics(metrics)
    metrics_path = DATA_DIR / METRICS_NAME
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)
    plot_bode(frequency, magnitude_db, phase_deg, metrics)
    plt.show()
    return metrics


async def run_live(keep_gains: bool = False):
    print(f"Q6.3 using gains: kp={KP}, kd={KD}, ki={KI}, ilimit={ILIMIT}")
    input("Close tview, then press Enter to apply gains and start the chirp...")
    if keep_gains:
        print("Skipping moteus_tool (--keep-gains); using RAM gains as-is.")
    else:
        apply_gains(KP, KD, KI, ILIMIT)
    data = await collect_chirp()
    print("Saved:", DATA_DIR / FILENAME)
    analyze(data)


def main():
    parser = argparse.ArgumentParser(description="Q6.3 chirp frequency response")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Re-analyze existing section6_5_chirp.csv (no motor)",
    )
    parser.add_argument(
        "--keep-gains",
        action="store_true",
        help="Skip moteus_tool; assume gains already in RAM",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    args = parser.parse_args()

    if args.no_show:
        plt.show = lambda *a, **k: None

    if args.analyze:
        path = DATA_DIR / FILENAME
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run without --analyze first.")
        print(f"Offline analysis of {path}")
        analyze(pd.read_csv(path))
        return

    asyncio.run(run_live(keep_gains=args.keep_gains))


if __name__ == "__main__":
    main()
