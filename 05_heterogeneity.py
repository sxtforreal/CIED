import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# plan 3.4: heterogeneity (stratified by device)
IN_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
IN_CSV = "/workspace/clean.csv"
OUT = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/heterogeneity_models.csv"


def load_clean() -> pd.DataFrame:
    if os.path.exists(IN_PARQUET):
        return pd.read_parquet(IN_PARQUET)
    return pd.read_csv(IN_CSV)


def stratified_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dev in ["PPM", "ICD", "CRT"]:
        sdf = df[df["device_cat"] == dev].copy()
        if len(sdf) < 20:
            continue
        for outcome in ["dx_change", "MgmtChange", "AddInfo", "NonDiagnostic"]:
            formula = f"{outcome} ~ artifact_burden + max_ratio + lv_visibility_score + age + sex_male + hf + htn + cad + mi + afib + ckd + C(pre_dx_cat) + has_TFE + has_CINE + has_VIAB"
            try:
                res = smf.glm(formula, data=sdf.dropna(), family=sm.families.Binomial()).fit(cov_type="HC3")
                for idx, r in res.summary2().tables[1].iterrows():
                    rows.append({"device_cat": dev, "outcome": outcome, "term": idx, "coef": r["Coef."], "se": r["Std.Err."], "p": r["P>|z|"]})
            except Exception as e:
                rows.append({"device_cat": dev, "outcome": outcome, "term": "ERROR", "coef": None, "se": None, "p": None})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_clean()
    out = stratified_models(df)
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()

