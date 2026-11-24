"""Section 4 - Q4.1 hand-turn check, then set the planetary gear ratio.

Sequence:
  1. Clear rotor_to_output_ratio to 1.0
  2. Show the position register
  3. Wait while you turn the OUTPUT one full turn back to the mark
  4. Report position again and Δ (expect ≈ 6 rev with ratio=1)
  5. Set rotor_to_output_ratio = 1/6 and conf write

Close tview before running.

  python section4_1_set_ratio.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path

import moteus
import pandas as pd


MOTOR_ID = 1
ROTOR_TO_OUTPUT = 1.0 / 6.0  # sun in, ring fixed, carrier out → 6:1
DEFAULT_TIMEOUT_S = 0.1
EXPECTED_RATIO = 6.0
SAMPLE_DT_S = 0.05

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab3_data"
DATA_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "section4_1_hand_check.csv"
SUMMARY_PATH = DATA_DIR / "section4_1_hand_check_summary.csv"


def release_transport():
    """Drop COM port so moteus_tool can open it after Controller use."""
    try:
        import moteus.transport_factory as tf
        if tf.GLOBAL_TRANSPORT is not None:
            tf.GLOBAL_TRANSPORT.close()
            tf.GLOBAL_TRANSPORT = None
    except Exception as exc:
        print(f"(transport release: {exc})")


def write_config(lines: list[str], label: str):
    cfg_path = DATA_DIR / f"section4_1_{label}.cfg"
    cfg_path.write_text("\n".join(lines + [""]), encoding="utf-8")
    print(f"Writing {cfg_path.name}...")
    print("Close tview first if it is open.")
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
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("moteus_tool hung. Close tview and retry.") from exc
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError("moteus_tool --write-config failed. Close tview and retry.")


def clear_gear_ratio():
    write_config(
        [
            "conf set motor_position.rotor_to_output_ratio 1.0",
            f"conf set servo.default_timeout_s {DEFAULT_TIMEOUT_S}",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
        ],
        "clear_ratio",
    )
    print("rotor_to_output_ratio = 1.0 (motor-shaft units for Q4.1).")


def set_gear_ratio():
    write_config(
        [
            f"conf set motor_position.rotor_to_output_ratio {ROTOR_TO_OUTPUT:.8f}",
            f"conf set servo.default_timeout_s {DEFAULT_TIMEOUT_S}",
            "conf set servo.default_accel_limit nan",
            "conf set servo.default_velocity_limit nan",
            "conf set servopos.position_min nan",
            "conf set servopos.position_max nan",
            "conf write",
        ],
        "set_ratio",
    )
    print(f"rotor_to_output_ratio = {ROTOR_TO_OUTPUT:.8f} (written to flash).")


async def hand_turn_check() -> dict:
    controller = moteus.Controller(id=MOTOR_ID)
    await controller.set_stop()
    first = await controller.query()
    fault = first.values.get(moteus.Register.FAULT, None)
    voltage = first.values.get(moteus.Register.VOLTAGE, None)
    start = float(first.values[moteus.Register.POSITION])

    print()
    print("=== Q4.1 hand-turn check (ratio = 1.0) ===")
    print(f"Fault:       {fault}")
    print(f"Bus voltage: {voltage} V")
    print(f"Position register NOW: {start:.6f} rev  ({start * 360.0:.2f} deg)")
    print()
    print("Mark the output and the housing.")
    print("Press Enter, turn the OUTPUT one full turn back to the mark,")
    print("then press Enter again.")
    print(f"Expect |Δposition| ≈ {EXPECTED_RATIO:.0f} rev (motor turns 6× per output turn).")
    input("\nPress Enter to start watching...")

    stop_flag = {"done": False}

    def wait_enter():
        input()
        stop_flag["done"] = True

    thread = threading.Thread(target=wait_enter, daemon=True)
    thread.start()
    print("Watching... turn the output now. Press Enter when back on the mark.")

    rows = []
    t0 = time.perf_counter()
    last = start
    while not stop_flag["done"]:
        result = await controller.query()
        pos = float(result.values[moteus.Register.POSITION])
        elapsed = time.perf_counter() - t0
        rows.append({
            "time_s": elapsed,
            "position_rev": pos,
            "delta_from_start_rev": pos - start,
        })
        if abs(pos - last) > 1e-3:
            print(
                f"  t={elapsed:5.2f} s   pos={pos:.6f} rev  "
                f"Δ={pos - start:+.4f} rev  ({(pos - start) * 360:+.1f} deg)"
            )
            last = pos
        await asyncio.sleep(SAMPLE_DT_S)

    final = await controller.query()
    end = float(final.values[moteus.Register.POSITION])
    delta = end - start
    abs_delta = abs(delta)

    print()
    print(f"Position BEFORE turn: {start:.6f} rev  ({start * 360.0:.2f} deg)")
    print(f"Position AFTER turn:  {end:.6f} rev  ({end * 360.0:.2f} deg)")
    print(f"Δposition:            {delta:+.6f} rev  ({delta * 360.0:+.2f} deg)")
    print(f"|Δ| / 1 output turn ≈ {abs_delta:.3f}  (expect ≈ {EXPECTED_RATIO:.0f})")
    if abs_delta < 1.5:
        print("WARNING: |Δ| ≈ 1 — ratio may not have cleared to 1.0.")
    elif abs_delta > 4.5:
        print("Looks good: motor advanced ~6× for one output turn.")
    if delta > 0:
        print("Sign: register advanced the SAME way as a positive turn sense.")
    else:
        print("Sign: register advanced the OPPOSITE way to a positive turn sense.")
    print("Did the motor turn the same way as the output? (yes → matches P1.1)")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    summary = {
        "start_rev": start,
        "end_rev": end,
        "delta_rev": delta,
        "abs_delta_rev": abs_delta,
        "expected_ratio": EXPECTED_RATIO,
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)
    print("Saved:", CSV_PATH)
    print("Saved:", SUMMARY_PATH)

    await controller.set_stop()
    release_transport()
    return summary


async def async_main():
    print("Step 1/3 — clear gear ratio to 1.0")
    clear_gear_ratio()

    print("\nStep 2/3 — hand-turn check (Q4.1)")
    summary = await hand_turn_check()

    print("\nStep 3/3 — set gear ratio to 1/6 and conf write")
    input("Press Enter to write rotor_to_output_ratio = 1/6...")
    set_gear_ratio()

    print("\nDone.")
    print(f"Q4.1 |Δ| was {summary['abs_delta_rev']:.3f} rev (expect ≈ 6).")
    print("From here on: position/velocity = output revs; torque = N·m at output.")
    print("Q4.2: moteus reports τ_out = τ_motor / rotor_to_output_ratio (= τ_motor × 6).")
    print("Next: python section4_3_burn_in.py")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
