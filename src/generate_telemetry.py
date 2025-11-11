from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def generate_time_series(run_duration: int = 300, fs: float = 1.0) -> np.ndarray:
    """
    Generate a simple time axis, e.g. 0..300 seconds with 1 Hz sampling.
    """
    t = np.arange(0.0, run_duration, 1.0 / fs)
    return t


def normal_behavior(t: np.ndarray) -> pd.DataFrame:
    """
    Nominal operation of a liquid-propellant rocket engine.

    Channels:
    - Pc:          chamber pressure
    - N_pump:      turbopump speed (RPM)
    - T_in:        injector inlet temperature
    - Vib:         overall vibration level
    - fuel_flow:   fuel mass flow
    - bearing_temp: bearing temperature
    - thrust:      simplified thrust proxy
    """
    n = len(t)

    # Chamber pressure: ramp up, steady, ramp down
    Pc = 50 + 30 * (np.tanh((t - 30) / 10)) - 30 * (np.tanh((t - 270) / 10))
    Pc += np.random.normal(0, 0.5, size=n)

    # Turbopump speed
    N_pump = 3000 + 1500 * (np.tanh((t - 30) / 15)) - 1500 * (np.tanh((t - 270) / 15))
    N_pump += np.random.normal(0, 20, size=n)

    # Inlet temperature with slow oscillation
    T_in = 300 + 20 * np.sin(t / 50) + np.random.normal(0, 1, size=n)

    # Base vibrations
    Vib = 0.3 + 0.05 * np.random.normal(0, 1, size=n)

    # Fuel flow (kg/s), slightly higher during main phase
    fuel_flow = 100 + 5 * np.tanh((t - 40) / 20)
    fuel_flow += np.random.normal(0, 0.5, size=n)

    # Bearing temperature, slow drift upwards
    bearing_temp = 80 + 5 * np.tanh((t - 60) / 40)
    bearing_temp += np.random.normal(0, 0.5, size=n)

    # Simplified thrust proxy: mix of pressure and fuel flow
    thrust = 0.8 * Pc + 0.2 * fuel_flow
    thrust += np.random.normal(0, 1.0, size=n)

    return pd.DataFrame(
        {
            "time": t,
            "Pc": Pc,
            "N_pump": N_pump,
            "T_in": T_in,
            "Vib": Vib,
            "fuel_flow": fuel_flow,
            "bearing_temp": bearing_temp,
            "thrust": thrust,
            "label": "normal",
        }
    )


def faulty_behavior(t: np.ndarray, fault_type: str) -> pd.DataFrame:
    """
    Start from normal behavior and inject specific fault patterns.
    """
    df = normal_behavior(t)

    if fault_type == "pressure_decay":
        # Slow chamber pressure decay over the run
        decay = -0.1 * (t / t.max())
        df["Pc"] += decay * 30

    elif fault_type == "turbopump_overspeed":
        # Turbopump overspeed in the second half of the run
        overspeed = df["N_pump"] + 2000 * (np.tanh((t - 150) / 10))
        df["N_pump"] = np.where(t > 150, overspeed, df["N_pump"])

    elif fault_type == "temp_rise":
        # Strong injector temperature rise later in the run
        rise = 100 * (np.tanh((t - 200) / 10))
        df["T_in"] += rise

    elif fault_type == "vibration_increase":
        # Global vibration increase (e.g. imbalance, cavitation)
        increase = 1.5 * (np.tanh((t - 180) / 10))
        df["Vib"] += increase + 0.1 * np.random.normal(0, 1, size=len(t))

    elif fault_type == "fuel_leak":
        # Fuel flow increase while Pc and thrust degrade
        leak = 10 * (np.tanh((t - 160) / 15))
        df["fuel_flow"] += leak
        df["Pc"] -= 0.4 * leak
        df["thrust"] -= 0.6 * leak

    elif fault_type == "bearing_overheat":
        # Bearing overheating and increased vibration
        over = 40 * (np.tanh((t - 170) / 10))
        df["bearing_temp"] += over
        df["Vib"] += 0.5 * (np.tanh((t - 170) / 10))

    df["label"] = fault_type
    return df


def generate_and_save_runs(
    out_dir: str = "data",
    fs: float = 1.0,
    run_duration: int = 300,
) -> None:
    """
    Generate several normal and faulty runs and save as CSV files in `out_dir`.
    """
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    t = generate_time_series(run_duration=run_duration, fs=fs)

    # Several normal runs
    for i in range(3):
        df = normal_behavior(t)
        df.to_csv(out_path / f"normal_run_{i}.csv", index=False)

    faults = [
        "pressure_decay",
        "turbopump_overspeed",
        "temp_rise",
        "vibration_increase",
        "fuel_leak",
        "bearing_overheat",
    ]
    for fault in faults:
        for i in range(2):
            df = faulty_behavior(t, fault)
            df.to_csv(out_path / f"{fault}_run_{i}.csv", index=False)

    print(f"Saved synthetic runs to {out_path.resolve()}")


if __name__ == "__main__":
    generate_and_save_runs()
