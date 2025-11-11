from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .preprocessing import feature_engineering, load_data
from .visualization import plot_health_index


def main() -> None:
    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError(
            "data/ directory not found. Run src/generate_telemetry.py first."
        )

    # Collect all CSV files with runs
    csv_files: List[str] = [str(p) for p in data_dir.glob("*.csv")]
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/. Run src/generate_telemetry.py first."
        )

    print(f"Loading {len(csv_files)} telemetry files...")
    df_raw = load_data(csv_files)

    # Feature engineering
    df_feat = feature_engineering(df_raw, window=10)

    # Prepare features and labels
    feature_cols = [c for c in df_feat.columns if c not in ("label", "time")]
    X = df_feat[feature_cols]
    y = df_feat["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Classification report (fault vs normal regimes):")
    print(classification_report(y_val, y_pred))

    # Compute health index on the full feature set
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    if "normal" not in classes:
        raise ValueError(
            f"'normal' class not found in classifier classes: {classes}"
        )
    normal_idx = classes.index("normal")
    health = proba[:, normal_idx]

    health_df = pd.DataFrame(
        {
            "time": df_feat["time"].values,
            "health_index": health,
            "label": df_feat["label"].values,
        }
    )

    out_path = Path("data") / "health_index.csv"
    health_df.to_csv(out_path, index=False)
    print(f"Saved health index time series to {out_path.resolve()}")

    # Plot health index
    plot_health_index(health_df)


if __name__ == "__main__":
    main()
