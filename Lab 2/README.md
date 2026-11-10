# Lab 2 — Frameless Motor Actuator (RI60 + moteus-c1)

Code for bring-up, calibration, tuning, and characterization. Same pattern as
Lab 1: edit gains at the top of each script, close tview, then run.

## Setup

```bash
pip install moteus matplotlib pandas numpy scipy
```

Bolt the actuator to the test stand and clamp the stand before powering.
`d stop` in tview kills the output stage. Peak torque is limited to **1.63 N·m**
in every command; keep RMS torque at or below **0.57 N·m**.

## Run order

### 4. Bring-up

```bash
# Open tview (optional, for live watching):
python -m moteus_gui.tview --target 1

# Or apply MA600 + safety config from this repo:
python section4_1_configure_ma600.py

# Check bus voltage / encoder follows hand turns:
python section4_2_connect.py
```

### 5. Calibration

```bash
# Shaft free to turn. Capture console to a file if you like.
python -m moteus.moteus_tool --target 1 --calibrate

# Place the moteus-cal-*.log beside these scripts, then:
python section5_1_calibration.py   # Q5.1 vs RI60 datasheet
python section5_2_kt_comparison.py # Q5.2 Kt + vs mj5208 from Lab 1
```

### 6. Tuning (Lab 1 §5.1 method)

Edit `KP` / `KD` / `KI` / `ILIMIT` at the top of each file.

```bash
python section6_1_tune_kp.py            # kp only, disturb by hand
python section6_2_critical_damping.py   # raise kd until critically damped
python section6_3_integral.py           # set ki / ilimit, check windup
```

### 6. Characterization

```bash
python section6_4_rotor_inertia.py      # Q6.2 step-response J vs CAD
python section6_5_frequency_response.py # Q6.3 chirp → −3 dB bandwidth
```

Offline re-analysis (no motor):

```bash
python section6_5_frequency_response.py --analyze --no-show
```

## Datasheet (RI60 KV120)

| Quantity | Value |
|---|---|
| Kv | 120 rpm/V |
| Kt | 0.100 N·m/A |
| Pole pairs | 14 |
| Rated / peak torque | 0.57 / 1.63 N·m |
| R / L (phase–phase) | 900 mΩ / 877.5 μH |

## Safety constants in every motion script

- `MAX_TORQUE_NM = 1.63`
- `servo.default_timeout_s` set during MA600 config
