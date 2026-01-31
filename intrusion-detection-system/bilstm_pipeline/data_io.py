import os
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected CSV not found: {path}. Place the dataset at the configured path."
        )
    return pd.read_csv(path, low_memory=False, dtype=str, on_bad_lines="skip")


def drop_common_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)
    return df
