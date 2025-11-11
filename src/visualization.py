from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def plot_time_series_with_labels(
    df: pd.DataFrame,
    channel: str = "Pc",
    title: Optional[str] = None,
    filename: str = "time_series_Pc.png",
) -> None:
    """
    Plot a single telemetry channel over time, color-coded by label.
    Assumes df has columns: 'time', channel, 'label'.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    labels = df["label"].unique()
    for lbl in labels:
        mask = df["label"] == lbl
        ax.plot(df.loc[mask, "time"], df.loc[mask, channel], label=str(lbl))

    ax.set_xlabel("Time, s")
    ax.set_ylabel(channel)
    ax.set_title(title or f"{channel} time series by regime")
    ax.grid(True)
    ax.legend()

    out_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_health_index(
    df: pd.DataFrame,
    filename: str = "health_index.png",
    threshold: float = 0.7,
) -> None:
    """
    Plot health index (P(normal)) over time with a simple threshold.
    Assumes df has columns: 'time', 'health_index', 'label'.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["time"], df["health_index"], label="Health index (P(normal))")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")

    ax.set_xlabel("Time, s")
    ax.set_ylabel("Health index")
    ax.set_title("Engine health index over time")
    ax.grid(True)
    ax.legend()

    out_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
