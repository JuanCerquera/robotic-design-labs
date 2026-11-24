"""Section 4 - Burn-in: run the stage back and forth to work it in.

Velocity mode (position=NaN) at ±VELOCITY_REV_S output, 30 s each direction.
Live plot shows commanded vs measured velocity.

Requires the gear ratio already set (section4_1_set_ratio.py).
Torque capped at the RMS limit (3.4 N·m).

Close tview before running. Keep fingers out of the gears.

  python section4_3_burn_in.py
  python section4_3_burn_in.py --minutes 3
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
# Continuous motion: use RMS cap, not the 9.8 peak.
MAX_TORQUE_NM = 3.4
# Higher output speed for a while to work the stage in (lab aim is ~1 rev/s).
VELOCITY_REV_S = 3.0  # output → ~18 rev/s rotor at 6:1
HALF_CYCLE_S = 30.0   # hold each direction before reversing
ACCEL_LIMIT = 10.0
SAMPLE_DT_S = 0.01
DEFAULT_MINUTES = 3.0
HIGH_TORQUE_ABORT_S = 3.0
# Kit default is 0.1 s; any plot/UI stall longer than that stops the drive.
COMMAND_TIMEOUT_S = 1.0
# Kit max_velocity=10 is too low once ratio is set and we command >~1.5 rev/s out.
MAX_VELOCITY_REV_S = 50.0
PLOT_WINDOW_S = 60.0
PLOT_REFRESH_S = 0.25

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
PLOTS_DIR = HERE / "Plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOG_CSV = DATA_DIR / "section4_3_burn_in.csv"
PLOT_PNG = PLOTS_DIR / "section4_3_burn_in.png"


def make_controller():
    return moteus.Controller(id=MOTOR_ID)


def release_transport():
    """Drop the COM port so a moteus_tool subprocess can open it."""
    try:
        import moteus.transport_factory as tf
        if tf.GLOBAL_TRANSPORT is not None:
            tf.GLOBAL_TRANSPORT.close()
            tf.GLOBAL_TRANSPORT = None
    except Exception as exc:
        print(f"(transport release: {exc})")


def write_config(lines: list[str], label: str):
    release_transport()
    cfg_path = DATA_DIR / f"section4_3_{label}.cfg"
    cfg_path.write_text("\n".join(lines + [""]), encoding="utf-8")
    print(f"Writing {cfg_path.name}...")
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
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "moteus_tool --write-config failed. Close tview and retry.\n" + err
        )


def setup_burn_in():
    """Widen timeout + max_velocity so continuous velocity commands actually run."""
    write_config(
        [
            f"conf set servo.default_timeout_s {COMMAND_TIMEOUT_S}",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "conf set servo.max_position_slip nan",
            f"conf set servo.max_velocity {MAX_VELOCITY_REV_S}",
            "conf set servo.max_velocity_derate 2",
        ],
        "timeout_setup",
    )
    print(
        f"Burn-in setup (RAM): timeout={COMMAND_TIMEOUT_S:.1f}s, "
        f"max_velocity={MAX_VELOCITY_REV_S}."
    )


def restore_timeout():
    write_config(
        [
            "conf set servo.max_velocity 10",
            "conf set servo.max_velocity_derate 2",
            "conf set servo.default_timeout_s 0.1",
        ],
        "timeout_restore",
    )
    print("Restored max_velocity=10, default_timeout_s=0.1 (RAM).")


class LiveVelocityPlot:
    """Plot updates run in a separate asyncio task — never inside set_position."""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.line_cmd, = self.ax.plot([], [], label="Commanded", lw=1.5)
        self.line_meas, = self.ax.plot([], [], label="Measured", lw=1.2)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Velocity [rev/s] (output)")
        self.ax.set_title("§4.3 Burn-in — commanded vs measured speed")
        self.ax.set_ylim(-VELOCITY_REV_S * 1.5, VELOCITY_REV_S * 1.5)
        self.ax.axhline(0.0, color="0.7", lw=0.8)
        self.ax.grid(True)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.t: list[float] = []
        self.cmd: list[float] = []
        self.meas: list[float] = []
        self._running = True
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def append(self, time_s: float, cmd_v: float, meas_v: float):
        self.t.append(time_s)
        self.cmd.append(cmd_v)
        self.meas.append(meas_v)

    def redraw(self):
        if not self.t:
            return
        time_s = self.t[-1]
        t0 = max(0.0, time_s - PLOT_WINDOW_S)
        i0 = 0
        for i, ti in enumerate(self.t):
            if ti >= t0:
                i0 = i
                break
        tt = self.t[i0:]
        self.line_cmd.set_data(tt, self.cmd[i0:])
        self.line_meas.set_data(tt, self.meas[i0:])
        self.ax.set_xlim(t0, max(time_s, t0 + 1.0))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    async def run(self):
        while self._running:
            self.redraw()
            await asyncio.sleep(PLOT_REFRESH_S)

    def stop(self):
        self._running = False

    def finalize(self):
        self.redraw()
        self.fig.savefig(PLOT_PNG, dpi=150)
        self.fig.savefig(DATA_DIR / PLOT_PNG.name, dpi=150)
        print("Saved plot:", PLOT_PNG)
        plt.ioff()
        plt.show()


async def hold_velocity(controller, velocity_rev_s, duration_s, rows, cycle, t0, plot):
    """Command constant output velocity for duration_s; log only (no plotting)."""
    start = time.perf_counter()
    high_torque_since = None
    while time.perf_counter() - start < duration_s:
        result = await controller.set_position(
            position=math.nan,
            velocity=velocity_rev_s,
            velocity_limit=MAX_VELOCITY_REV_S,
            accel_limit=ACCEL_LIMIT,
            maximum_torque=MAX_TORQUE_NM,
            query=True,
        )
        fault = result.values.get(moteus.Register.FAULT, 0)
        if fault:
            raise RuntimeError(f"Fault {fault} during burn-in. d stop and inspect.")
        pos = float(result.values[moteus.Register.POSITION])
        vel = float(result.values[moteus.Register.VELOCITY])
        tau = float(result.values[moteus.Register.TORQUE])
        elapsed = time.perf_counter() - t0
        rows.append({
            "cycle": cycle,
            "time_s": elapsed,
            "cmd_velocity_rev_s": velocity_rev_s,
            "position_rev": pos,
            "velocity_rev_s": vel,
            "torque_nm": tau,
        })
        if plot is not None:
            plot.append(elapsed, velocity_rev_s, vel)

        if abs(tau) > 0.9 * MAX_TORQUE_NM:
            if high_torque_since is None:
                high_torque_since = time.perf_counter()
            elif time.perf_counter() - high_torque_since > HIGH_TORQUE_ABORT_S:
                raise RuntimeError(
                    f"Abort: |torque| near {MAX_TORQUE_NM} N·m for "
                    f"{HIGH_TORQUE_ABORT_S:.0f} s — possible jam."
                )
        else:
            high_torque_since = None

        await asyncio.sleep(SAMPLE_DT_S)


def save_log(rows):
    if not rows:
        print("No samples to save.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(LOG_CSV, index=False)
    print("Saved:", LOG_CSV)
    tau = df["torque_nm"].to_numpy(dtype=float)
    rms = float(np.sqrt(np.mean(tau ** 2)))
    print(
        f"Samples: {len(df)}  duration={df['time_s'].iloc[-1] / 60:.2f} min  "
        f"τ_rms={rms:.3f} N·m  τ_peak={float(np.max(np.abs(tau))):.3f} N·m"
    )
    if rms > MAX_TORQUE_NM:
        print(f"WARNING: τ_rms exceeded {MAX_TORQUE_NM} N·m RMS limit.")


async def run(minutes: float, no_show: bool):
    duration_s = minutes * 60.0
    setup_burn_in()
    controller = make_controller()
    await controller.set_stop()
    first = await controller.query()
    home = float(first.values[moteus.Register.POSITION])

    print(f"Burn-in for {minutes:.1f} min at ±{VELOCITY_REV_S} rev/s (output, velocity mode).")
    print(f"Home = {home:.4f} rev; reverse every {HALF_CYCLE_S:.0f} s.")
    print(f"MAX_TORQUE_NM = {MAX_TORQUE_NM} (RMS cap). Ctrl+C to stop early.")

    plot = None if no_show else LiveVelocityPlot()
    plot_task = None
    if plot is not None:
        plot_task = asyncio.create_task(plot.run())

    rows = []
    t0 = time.perf_counter()
    cycle = 0
    direction = +1.0
    try:
        while time.perf_counter() - t0 < duration_s:
            cycle += 1
            cmd_v = direction * VELOCITY_REV_S
            remaining = duration_s - (time.perf_counter() - t0)
            hold_s = min(HALF_CYCLE_S, remaining)
            if hold_s <= 0:
                break
            print(
                f"  cycle {cycle:4d}  cmd_v={cmd_v:+.1f} rev/s  "
                f"for {hold_s:.0f}s  remaining≈{remaining / 60:.1f} min"
            )
            await hold_velocity(controller, cmd_v, hold_s, rows, cycle, t0, plot)
            last = rows[-1]
            print(
                f"           t={last['time_s']:6.1f}s  "
                f"v={last['velocity_rev_s']:+.3f}  τ={last['torque_nm']:.3f} N·m"
            )
            direction *= -1.0
    except asyncio.CancelledError:
        print("\nCancelled — saving log.")
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if plot is not None:
            plot.stop()
        if plot_task is not None:
            try:
                await asyncio.wait_for(plot_task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                plot_task.cancel()
        try:
            await controller.set_stop()
            print("Motor stopped.")
        except Exception as exc:
            print(f"set_stop failed: {exc}")
        save_log(rows)
        try:
            restore_timeout()
        except Exception as exc:
            print(f"Timeout restore failed: {exc}")
        if plot is not None:
            try:
                plot.finalize()
            except Exception as exc:
                print(f"Plot finalize failed: {exc}")

    print("Next: Section 5 — python section5_1_drag_torque.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minutes", type=float, default=DEFAULT_MINUTES,
        help=f"Burn-in duration in minutes (default {DEFAULT_MINUTES}).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip live plot (still saves CSV).",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args.minutes, args.no_show))
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    main()
