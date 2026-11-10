"""Section 4 - Configure MA600 as primary encoder on AUX2.

Writes the Lab 2 handout SPI / motor_position settings via moteus_tool, sets a
command timeout, and flashes with conf write.

Close tview before running:
  python section4_1_configure_ma600.py
"""

import subprocess
import sys
from pathlib import Path


MOTOR_ID = 1
DEFAULT_TIMEOUT_S = 0.1

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab2_data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    cfg_path = DATA_DIR / "section4_1_ma600.cfg"
    cfg_path.write_text(
        "\n".join([
            # AUX2 as SPI for the MA600 (handout Section 4).
            "conf set aux2.uart.mode 0",
            "conf set aux2.pins.0.mode 1",
            "conf set aux2.pins.1.mode 1",
            "conf set aux2.pins.2.mode 1",
            "conf set aux2.pins.3.mode 2",
            "conf set aux2.spi.mode 5",
            "conf set aux2.spi.rate_hz 6000000",
            "conf set motor_position.sources.0.aux_number 2",
            "conf set motor_position.sources.0.type 1",
            "conf set motor_position.sources.0.cpr 65536",
            # Kill output if commands stop (same idea as Lab 1).
            f"conf set servo.default_timeout_s {DEFAULT_TIMEOUT_S}",
            "conf write",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Writing MA600 config from {cfg_path.name}...")
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
        raise RuntimeError(
            "moteus_tool --write-config hung (timeout). Close tview and retry."
        ) from exc

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "moteus_tool --write-config failed. Close tview and retry."
        )

    print("MA600 configured as primary encoder (AUX2 SPI).")
    print(f"servo.default_timeout_s = {DEFAULT_TIMEOUT_S}")
    print("Next: python section4_2_connect.py  (or open tview and turn the shaft).")


if __name__ == "__main__":
    main()
