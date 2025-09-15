import os
import pandas as pd

# plan 3.1: descriptive tables by device groups (Table 1/2)
IN_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
IN_CSV = "/workspace/clean.csv"
OUT1 = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/table1_by_device.xlsx"
OUT2 = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/table2_outcomes_by_device.xlsx"


def load_clean() -> pd.DataFrame:
    if os.path.exists(IN_PARQUET):
        return pd.read_parquet(IN_PARQUET)
    return pd.read_csv(IN_CSV)


def make_table1(df: pd.DataFrame) -> pd.DataFrame:
    # minimal: descriptive stats by device_cat
    cols = [
        "age",
        "sex_male",
        "bmi",
        "hf",
        "htn",
        "cad",
        "mi",
        "afib",
        "ckd",
        "mr_conditional",
        "n_leads",
        "left_vs_other",
        "manufacturer_other",
        "norm_dist_card_sil",
        "norm_dist_LV_apex",
        "rotation",
        "has_TFE",
        "has_CINE",
        "has_VIAB",
        "artifact_burden",
        "lv_visibility_score",
    ]
    available = [c for c in cols if c in df.columns]
    g = df.groupby("device_cat")[available]
    desc = g.agg(["count", "mean", "std", "min", "max"])
    desc.columns = ["_".join([c for c in col if c]) for col in desc.columns.to_flat_index()]
    return desc.reset_index()


def make_table2(df: pd.DataFrame) -> pd.DataFrame:
    # outcomes incidence by device
    outcomes = ["dx_change", "MgmtChange", "AddInfo", "NonDiagnostic", "Confirmed"]
    outcomes = [c for c in outcomes if c in df.columns]
    counts = df.groupby("device_cat")[outcomes].sum(min_count=1)
    denoms = df.groupby("device_cat")[outcomes].count()
    rates = (counts / denoms * 100).round(1)
    out = pd.DataFrame(index=counts.index)
    for c in outcomes:
        out[c] = counts[c].astype("Int64").astype(str) + " (" + rates[c].astype(str) + "%)"
    return out.reset_index()


def main() -> None:
    df = load_clean()
    t1 = make_table1(df)
    t2 = make_table2(df)
    with pd.ExcelWriter(OUT1, engine="openpyxl") as w:
        t1.to_excel(w, index=False, sheet_name="Table1")
    with pd.ExcelWriter(OUT2, engine="openpyxl") as w:
        t2.to_excel(w, index=False, sheet_name="Table2")


if __name__ == "__main__":
    main()

