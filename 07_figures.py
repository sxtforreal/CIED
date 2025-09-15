import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plan 5: key figures (minimal placeholders)
IN_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
IN_CSV = "/workspace/clean.csv"
OUT_DIR = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/figs"


def load_clean() -> pd.DataFrame:
    if os.path.exists(IN_PARQUET):
        return pd.read_parquet(IN_PARQUET)
    return pd.read_csv(IN_CSV)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def dose_response(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)
    df = df.copy()
    df["artifact_q"] = pd.qcut(df["artifact_burden"], 5, duplicates="drop")
    p = df.groupby(["device_cat", "artifact_q"])['dx_change'].mean().reset_index()
    plt.figure(figsize=(6, 4))
    for dev, sub in p.groupby("device_cat"):
        x = sub["artifact_q"].astype(str)
        y = sub["dx_change"]
        plt.plot(x, y, marker='o', label=dev)
    plt.ylabel("Pr(ΔDx=1)")
    plt.xlabel("artifact_burden quintile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dose_response_dx_change.png"), dpi=300)
    plt.close()


def ridgeline_placeholder(df: pd.DataFrame) -> None:
    ensure_dir(OUT_DIR)
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="ratio_TFE", hue="device_cat", fill=True, common_norm=False, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ridge_ratio_TFE.png"), dpi=300)
    plt.close()


def main() -> None:
    df = load_clean()
    if "artifact_burden" in df.columns and "dx_change" in df.columns:
        dose_response(df)
    if "ratio_TFE" in df.columns:
        ridgeline_placeholder(df)


if __name__ == "__main__":
    main()

