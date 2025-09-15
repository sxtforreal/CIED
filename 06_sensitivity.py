import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# plan 3.5 and 4: sensitivity analyses
IN_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
IN_CSV = "/workspace/clean.csv"
OUT_DIR = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/"


def load_clean() -> pd.DataFrame:
    if os.path.exists(IN_PARQUET):
        return pd.read_parquet(IN_PARQUET)
    return pd.read_csv(IN_CSV)


def run_sensitivity(df: pd.DataFrame) -> None:
    # alt artifact metrics
    outcomes = ["dx_change", "MgmtChange", "AddInfo", "NonDiagnostic"]
    metrics = ["max_ratio", "severe_art_TFE", "severe_art_CINE", "severe_art_VIAB"]
    for y in outcomes:
        for m in metrics:
            if m not in df.columns:
                continue
            formula = f"{y} ~ C(device_cat) + mr_conditional + {m} + lv_visibility_score + age + sex_male + hf + htn + cad + mi + afib + ckd + C(pre_dx_cat) + has_TFE + has_CINE + has_VIAB + C(device_cat):{m} + C(device_cat):mr_conditional"
            try:
                res = smf.glm(formula, data=df.dropna(), family=sm.families.Binomial()).fit(cov_type="HC3")
                res.summary2().tables[1].to_csv(os.path.join(OUT_DIR, f"sens_{y}_{m}.csv"))
            except Exception as e:
                print(f"Sensitivity failed for {y} with {m}: {e}")

    # exclude non-diagnostic vs penalty in UtilityScore
    try:
        df_cc = df[df["NonDiagnostic"] == 0]
        res_cc = smf.glm("dx_change ~ C(device_cat) + mr_conditional + artifact_burden + age + sex_male", data=df_cc.dropna(), family=sm.families.Binomial()).fit(cov_type="HC3")
        res_cc.summary2().tables[1].to_csv(os.path.join(OUT_DIR, "sens_dx_change_complete_case.csv"))
    except Exception:
        pass


def main() -> None:
    df = load_clean()
    run_sensitivity(df)


if __name__ == "__main__":
    main()

