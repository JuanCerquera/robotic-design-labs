"""Section 4.1 - Byte worksheet for the command and reply frames."""

import struct
from pathlib import Path

import pandas as pd


COMMAND_HEX = "01000A0D200000C07F11001F01130D50"
REPLY_HEX = "2100002F0100282640FCA664BC00000000230D301F005050"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "lab1_data"
DATA_DIR.mkdir(exist_ok=True)


def byte_table(frame, hex_string):
    values = bytes.fromhex(hex_string)
    return pd.DataFrame({
        "frame": frame,
        "offset": range(len(values)),
        "raw_hex": [f"{value:02X}" for value in values],
        "meaning": [""] * len(values),
        "decoded_value": [""] * len(values),
        "units": [""] * len(values),
    })


def main():
    worksheet = pd.concat([
        byte_table("command", COMMAND_HEX),
        byte_table("reply", REPLY_HEX),
    ], ignore_index=True)
    output = DATA_DIR / "section4_1_decode_worksheet.csv"
    worksheet.to_csv(output, index=False)

    print(worksheet.to_string(index=False))
    print("Command bytes:", len(bytes.fromhex(COMMAND_HEX)))
    print("Reply bytes:", len(bytes.fromhex(REPLY_HEX)))
    print("Command position float:",
          struct.unpack("<f", bytes.fromhex("0000C07F"))[0])
    print("Saved:", output)


if __name__ == "__main__":
    main()

