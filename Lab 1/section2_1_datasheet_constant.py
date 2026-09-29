"""Section 2.1 - Convert published Kv to ideal Kt."""

import numpy as np


KV_RPM_PER_V = 330.0


def main():
    kt = 60.0 / (2.0 * np.pi * KV_RPM_PER_V)
    print("Published Kv [rpm/V]:", KV_RPM_PER_V)
    print("Ideal Kt [N m/A]:", kt)


if __name__ == "__main__":
    main()

