import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# plan 3.3: simple mediation (device -> artifact_burden -> dx_change)
IN_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
IN_CSV = "/workspace/clean.csv"
OUT = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/mediation_results.csv"


def load_clean() -> pd.DataFrame:
    if os.path.exists(IN_PARQUET):
        return pd.read_parquet(IN_PARQUET)
    return pd.read_csv(IN_CSV)


def mediation_acme_ade(df: pd.DataFrame, mediator: str = "artifact_burden", outcome: str = "dx_change") -> pd.DataFrame:
    # mediator model
    m_formula = f"{mediator} ~ C(device_cat) + left_vs_other + norm_dist_card_sil + norm_dist_LV_apex + n_leads + mr_conditional + age + sex_male + hf + htn + cad + mi + afib + ckd"
    m_res = smf.ols(m_formula, data=df.dropna(subset=[mediator, outcome])).fit(cov_type="HC3")

    # outcome model
    y_formula = f"{outcome} ~ C(device_cat) + {mediator} + age + sex_male + hf + htn + cad + mi + afib + ckd + C(pre_dx_cat)"
    y_model = smf.glm(y_formula, data=df.dropna(subset=[mediator, outcome]), family=sm.families.Binomial())
    y_res = y_model.fit(cov_type="HC3")

    # compute ACME/ADE via parametric g-formula at contrasting devices (ICD vs PPM, CRT vs PPM)
    results = []
    for comp in [("ICD", "PPM"), ("CRT", "PPM")]:
        a1, a0 = comp
        df_a0 = df.copy()
        df_a1 = df.copy()
        df_a0["device_cat"] = a0
        df_a1["device_cat"] = a1
        # predicted mediator
        m_a0 = m_res.predict(df_a0)
        m_a1 = m_res.predict(df_a1)
        # natural indirect: E[Y(a1, M(a1)) - Y(a1, M(a0))]
        y_a1_m_a1 = y_res.predict(df_a1.assign(**{mediator: m_a1}))
        y_a1_m_a0 = y_res.predict(df_a1.assign(**{mediator: m_a0}))
        acme = float((y_a1_m_a1 - y_a1_m_a0).mean())
        # natural direct: E[Y(a1, M(a0)) - Y(a0, M(a0))]
        y_a0_m_a0 = y_res.predict(df_a0.assign(**{mediator: m_a0}))
        ade = float((y_a1_m_a0 - y_a0_m_a0).mean())
        total = float((y_a1_m_a1 - y_a0_m_a0).mean())
        prop = acme / total if total != 0 else np.nan
        results.append({"contrast": f"{a1} vs {a0}", "ACME": acme, "ADE": ade, "Total": total, "PropMediated": prop})
    return pd.DataFrame(results)


def main() -> None:
    df = load_clean()
    out = mediation_acme_ade(df, mediator="artifact_burden", outcome="dx_change")
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()

