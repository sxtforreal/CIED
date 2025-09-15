import os
import itertools
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# plan 3.2: main effects models (logit/probit)
IN_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
IN_CSV = "/workspace/clean.csv"
OUT_DIR = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/"


def load_clean() -> pd.DataFrame:
    if os.path.exists(IN_PARQUET):
        return pd.read_parquet(IN_PARQUET)
    return pd.read_csv(IN_CSV)


def build_formula(outcome: str, artifact_metric: str = "artifact_burden") -> str:
    # covariates
    base = [
        "C(device_cat)",
        "mr_conditional",
        artifact_metric,
        "lv_visibility_score",
        "age",
        "sex_male",
        "hf + htn + cad + mi + afib + ckd",
        "C(pre_dx_cat)",
        "has_TFE + has_CINE + has_VIAB",
    ]
    # interactions
    inter = [
        f"C(device_cat):{artifact_metric}",
        "C(device_cat):mr_conditional",
        "C(pre_dx_cat):has_TFE",
        "C(pre_dx_cat):has_CINE",
        "C(pre_dx_cat):has_VIAB",
    ]
    rhs = " + ".join(base + inter)
    return f"{outcome} ~ {rhs}"


def fit_glm(df: pd.DataFrame, formula: str):
    # drop rows with any missing vars in formula
    model_df = df.copy()
    # naive parse of variables used in formula
    tokens = (
        formula.replace("~", "+")
        .replace("*", "+")
        .replace(":", "+")
        .replace("C(", "C_")
        .replace(")", "")
        .replace(" ", "")
        .split("+")
    )
    vars_used = [t for t in tokens if t and t != "1" and not t.startswith("C_")]
    model_df = model_df.dropna(subset=vars_used)
    model = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial())
    res = model.fit(cov_type="HC3")
    return res, model_df


def run_main_models(df: pd.DataFrame) -> None:
    outcomes = ["dx_change", "MgmtChange", "AddInfo", "NonDiagnostic"]
    for y in outcomes:
        for metric in ["artifact_burden", "max_ratio"]:
            formula = build_formula(y, artifact_metric=metric)
            try:
                res, used = fit_glm(df, formula)
            except Exception as e:
                print(f"Model failed for {y} with {metric}: {e}")
                continue
            coefs = res.summary2().tables[1]
            coefs.to_csv(os.path.join(OUT_DIR, f"model_{y}_{metric}_coefs.csv"))
            # marginal effects (robust)
            try:
                me = res.get_margeff(at="overall").summary_frame()
                me.to_csv(os.path.join(OUT_DIR, f"model_{y}_{metric}_margins.csv"))
            except Exception as e:
                print(f"Margins failed for {y} with {metric}: {e}")
            # device x artifact marginal effects grid for plotting
            try:
                q = used[metric].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).rename("q").reset_index()
                grid = []
                for dev in ["PPM", "ICD", "CRT"]:
                    for _, row in q.iterrows():
                        val = float(row[metric])
                        pred_df = used.copy()
                        pred_df[metric] = val
                        pred_df["device_cat"] = dev
                        p = res.predict(pred_df).mean()
                        grid.append({"device_cat": dev, "metric": metric, "q": row["index"], "value": val, "mean_prob": p, "outcome": y})
                pd.DataFrame(grid).to_csv(os.path.join(OUT_DIR, f"model_{y}_{metric}_dose_response.csv"), index=False)
            except Exception as e:
                print(f"Dose-response grid failed for {y} with {metric}: {e}")


def main() -> None:
    df = load_clean()
    run_main_models(df)


if __name__ == "__main__":
    main()

