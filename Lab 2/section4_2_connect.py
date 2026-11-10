"""Section 4 - Verify communication and that the MA600 position follows.

Close tview before running (port is exclusive), or use tview instead:
  python -m moteus_gui.tview --target 1

  python section4_2_connect.py
"""

import asyncio
import time

import moteus


MOTOR_ID = 1
SAMPLE_DT_S = 0.05
DURATION_S = 8.0


async def main():
    controller = moteus.Controller(id=MOTOR_ID)
    first = await controller.query()

    fault = first.values.get(moteus.Register.FAULT, None)
    voltage = first.values.get(moteus.Register.VOLTAGE, None)
    position = first.values[moteus.Register.POSITION]

    print("Controller response:", first)
    print(f"Fault:        {fault}")
    print(f"Bus voltage:  {voltage} V")
    print(f"Position:     {position:.6f} rev  ({position * 360.0:.2f} deg)")
    print(f"\nTurn the shaft by hand for {DURATION_S:.0f} s — watching position...")

    start = time.perf_counter()
    last = position
    while time.perf_counter() - start < DURATION_S:
        result = await controller.query()
        pos = result.values[moteus.Register.POSITION]
        if abs(pos - last) > 1e-4:
            print(f"  t={time.perf_counter() - start:5.2f} s   "
                  f"pos={pos:.6f} rev  ({pos * 360.0:.2f} deg)")
            last = pos
        await asyncio.sleep(SAMPLE_DT_S)

    print("Done. Expect fault=0, sensible bus voltage (~24 V), and position that moves.")


if __name__ == "__main__":
    asyncio.run(main())
