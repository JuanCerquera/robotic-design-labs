"""Section 4.2 - Estimate CAN-FD command/reply rates."""

import numpy as np


LONG_COMMAND_HEX = (
    "01000A0C0620002826400000000000000000CDCC4C3D"
    "CDCC4C3DFBAE6A3D0E2855558542ABAA264411001F01130D5050"
)
LONG_REPLY_HEX = "2100032F014028264069B203BA00000000230D301F005050"

DATA_RATE_BITS_S = 1_000_000
OVERHEAD_BITS_PER_FRAME = 30

# Replace after constructing/counting equivalent int16 frames.
INT16_COMMAND_BYTES = np.nan
INT16_REPLY_BYTES = np.nan


def round_trip_rate(command_bytes, reply_bytes):
    total_bits = 8 * (command_bytes + reply_bytes)
    total_bits += 2 * OVERHEAD_BITS_PER_FRAME
    return total_bits, DATA_RATE_BITS_S / total_bits


def main():
    command_bytes = len(bytes.fromhex(LONG_COMMAND_HEX))
    reply_bytes = len(bytes.fromhex(LONG_REPLY_HEX))
    bits, rate = round_trip_rate(command_bytes, reply_bytes)
    print("Float command payload bytes:", command_bytes)
    print("Float reply payload bytes:", reply_bytes)
    print("Float round-trip bits:", bits)
    print("Float estimated rate [Hz]:", rate)

    if np.isfinite(INT16_COMMAND_BYTES) and np.isfinite(INT16_REPLY_BYTES):
        bits, rate = round_trip_rate(
            int(INT16_COMMAND_BYTES), int(INT16_REPLY_BYTES))
        print("Int16 round-trip bits:", bits)
        print("Int16 estimated rate [Hz]:", rate)
    else:
        print("Set INT16_COMMAND_BYTES and INT16_REPLY_BYTES to finish Q4.4.")


if __name__ == "__main__":
    main()

