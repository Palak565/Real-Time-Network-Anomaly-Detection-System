"""
Read-side helpers for the Streamlit app. Kept separate from Spark code
so Streamlit doesn't need to import pyspark just to render a table.
"""
import json
import os

import pandas as pd

import config


def load_predictions(limit=500):
    """Latest live predictions written by stream_processor.py."""
    if not os.path.exists(config.PREDICTIONS_FILE):
        return pd.DataFrame()

    rows = []
    with open(config.PREDICTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df["verdict"] = df["prediction"].map({0: "normal", 1: "attack"})
    return df.sort_values("captured_at", ascending=False).head(limit)


def load_metrics(model_dir):
    """Offline metrics.json written by train_basic.py / train_full.py."""
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def attacks_over_time(df, bucket="10s"):
    """Time-bucketed counts of normal vs attack flows, for a time series chart."""
    if df.empty:
        return pd.DataFrame(columns=["captured_at", "verdict", "count"])
    grouped = (
        df.set_index("captured_at")
        .groupby([pd.Grouper(freq=bucket), "verdict"])
        .size()
        .reset_index(name="count")
    )
    return grouped


def breakdown_by(df, column):
    """Counts of flagged attacks grouped by a categorical column (protocol_type, service, ...)."""
    if df.empty:
        return pd.DataFrame(columns=[column, "count"])
    attacks = df[df["verdict"] == "attack"]
    if attacks.empty:
        return pd.DataFrame(columns=[column, "count"])
    return attacks.groupby(column).size().reset_index(name="count").sort_values("count", ascending=False)
