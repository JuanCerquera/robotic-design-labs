# Lab 3 — Planetary Gear Stage (RI60 + mjbots M0.5 6×)

Code for baseline friction, gear-ratio bring-up, and stage friction / torque /
efficiency. Same pattern as Labs 1–2: close tview, then run each script.

## Setup

```bash
pip install moteus moteus_gui matplotlib pandas numpy scipy
```

Bolt the actuator to the test stand and clamp the stand before powering.
`d stop` in tview kills the output stage.

**Before the gear ratio is set** (Section 3): keep `maximum_torque ≤ 1.63 N·m`.

**After** `motor_position.rotor_to_output_ratio = 1/6` (Section 4+): torque and
position are in **output** units. Cap peak at **9.8 N·m**; keep RMS ≤ **3.4 N·m**.
Keep fingers out of the gears. Wear safety glasses with the lever arm on.

## Run order

### 3. Baseline (shaft free — no gear stage yet)

Do this **before** mounting the stage. Ratio must stay at 1.0.

```bash
python section3_1_drag_torque.py      # Q3.1 velocity sweep → Coulomb + viscous
python section3_2_breakaway.py       # Q3.2 torque ramp until shaft moves
```

### 4. Assembly and bring-up

Assemble the stage. Turn the output by hand — it should feel smooth.

```bash
python section4_1_set_ratio.py       # clear ratio → Q4.1 hand turn → set 1/6
python section4_3_burn_in.py         # ~10 min back/forth at 1 rev/s output
```

### 5. Friction, torque, efficiency (ratio already set)

```bash
python section5_1_drag_torque.py     # Q5.1 output drag vs §3.1 referred to output
python section5_2_output_torque.py   # Q5.2 lever arm + scale → efficiency
python section5_3_backdrive.py       # Q5.3 record backdrive scale pushes
```

Offline re-analysis (no motor):

```bash
python section3_1_drag_torque.py --analyze --no-show
python section5_1_drag_torque.py --analyze --no-show
python section5_2_output_torque.py --analyze --no-show
python section5_3_backdrive.py --analyze --no-show
```

## Gearset (mjbots M0.5 6×)

| Quantity | Value |
|---|---|
| Sun / planet / ring teeth | 20 / 40 / 100 |
| Ratio (sun in, ring fixed, carrier out) | 6 : 1 |
| `rotor_to_output_ratio` | `0.16666667` |
| Peak / RMS torque at output | 9.8 / 3.4 N·m |
| Lever arm radius (Section 5) | 100 mm |

## Safety constants

| Phase | `MAX_TORQUE_NM` |
|---|---|
| Section 3 (motor shaft) | 1.63 |
| Sections 4–5 (output) | 9.8 |
