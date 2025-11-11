from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


CHANNELS: List[str] = [
    "Pc",
    "N_pump",
    "T_in",
    "Vib",
    "fuel_flow",
    "bearing_temp",
    "thrust",
]


def load_data(filenames: Iterable[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files with telemetry.
    """
    dfs = [pd.read_csv(f) for f in filenames]
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def feature_engineering(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Simple rolling-window features for each telemetry channel:
    - mean and std over the given window size.

    Also carries over:
    - time: so we can plot health index vs time;
    - label: fault / normal class for supervised learning.
    """
    df_feat = pd.DataFrame()

    for col in CHANNELS:
        df_feat[f"{col}_mean"] = df[col].rolling(window).mean()
        df_feat[f"{col}_std"] = df[col].rolling(window).std()

    df_feat["time"] = df["time"]
    df_feat["label"] = df["label"]

    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat
