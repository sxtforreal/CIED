import pandas as pd
import numpy as np

# plan 0: load data
SRC_XLSX = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/Database 6.12 clean LB.xlsx"
OUT_PARQUET = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
OUT_CSV = "/workspace/clean.csv"


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_bin(series: pd.Series) -> pd.Series:
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    if s.dtype == object:
        s = s.str.strip().str.lower()
        s = s.replace({
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
        })
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num = to_num(num)
    den = to_num(den)
    return num / den.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # plan 1.1: basic & disease phenotype
    df["age"] = to_num(df.get("Age"))
    df["sex_male"] = to_bin(df.get("Sex (1-M)", 0))
    df["height_cm"] = to_num(df.get("Height cm"))
    df["weight_kg"] = to_num(df.get("Weight Kg"))
    df["bmi"] = to_num(df.get("BMI"))

    comorb_cols = {
        "hf": "HF (1=Y)",
        "htn": "HTN (1=Y)",
        "cad": "CAD",
        "mi": "MI (Y=1)",
        "vt_vf": "VT/VF",
        "vfib": "Vfib (1=Y)",
        "vt": "VT (1=Y)",
        "afib": "Afib (1=Y)",
        "ckd": "CKD (1=Y)",
    }
    for new, old in comorb_cols.items():
        df[new] = to_bin(df.get(old, 0))

    pre_col = "Pre-MR diagnosis/suspected_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)"
    post_col = "Post-MR diagnosis/suspected_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)"
    df["pre_dx_cat"] = to_num(df.get(pre_col))
    df["post_dx_cat"] = to_num(df.get(post_col))
    dx_map = {
        1: "infiltrative",
        2: "valvular",
        3: "hcm",
        4: "myopericarditis",
        5: "ischemia",
        6: "other_cmp",
        7: "other",
        8: "vt",
    }
    df["pre_dx_name"] = df["pre_dx_cat"].map(dx_map)
    df["post_dx_name"] = df["post_dx_cat"].map(dx_map)

    # plan 1.2: device & leads
    df["device_type_code"] = to_num(df.get("Type of Device (ICD =1 PPM = 2 CRT = 3)"))
    df["device_cat"] = df["device_type_code"].map({1: "ICD", 2: "PPM", 3: "CRT"}).astype("object")
    df["is_CRT"] = (df["device_type_code"] == 3).astype(int)

    df["icd_indication_code"] = to_num(df.get("ICD indication (Primary prevention = 1, secodary prevention =2)"))
    df["ppm_indication_code"] = to_num(df.get("PPM indications (CHB = 1, SND = 2, Other = 3)"))
    df["indication_missing"] = (
        df["icd_indication_code"].isna() & df["ppm_indication_code"].isna()
    ).astype(int)

    df["mr_conditional"] = to_bin(df.get("MR Conditional ", 0))

    pos_body_col = "Position in body (left chest = 1, right chest = 2, leadless = 3, 4=subC)"
    df["position_body_code"] = to_num(df.get(pos_body_col))
    df["left_chest"] = (df["position_body_code"] == 1).astype(int)
    df["right_chest"] = (df["position_body_code"] == 2).astype(int)
    df["leadless"] = (df["position_body_code"] == 3).astype(int)
    df["subQ"] = (df["position_body_code"] == 4).astype(int)
    df["left_vs_other"] = (df["position_body_code"] == 1).astype(int)

    df["left_vs_other_coarse"] = (
        to_num(df.get("Position (L chest=1, other=2)")) == 1
    ).astype(int)

    df["manufacturer_code"] = to_num(df.get("Manufacturer (1=Boston,2=MDT,3=Bio,4=StJude,5=other)"))
    manu_map = {1: "Boston", 2: "MDT", 3: "Bio", 4: "StJude", 5: "Other"}
    df["manufacturer_name"] = df["manufacturer_code"].map(manu_map).astype("object")
    df["manufacturer_other"] = (df["manufacturer_name"] == "Other").astype(int)

    df["atrial_lead"] = to_bin(df.get("Atrial Lead Yes/No", 0))
    df["ventricular_lead"] = to_bin(df.get("Ventricular lead Yes/No", 0))
    df["lv_lead"] = to_bin(df.get("LV lead Yes/No", 0))
    subq_raw = df.get("SubQ lead - (0=no,1=yes,3=other (leadless)")
    df["subq_lead"] = (to_num(subq_raw) == 1).fillna(0).astype(int) if subq_raw is not None else 0

    df["n_leads"] = (
        df[["atrial_lead", "ventricular_lead", "lv_lead", "subq_lead"]].sum(axis=1)
    )
    df["has_LV"] = df["lv_lead"]
    df["has_SQ"] = df["subq_lead"]

    # plan 1.3: position/distances & normalization
    df["rotation"] = to_bin(df.get("Rotation. 0 = normal, 1 rotated.", 0))
    df["dist_card_sil_mm"] = to_num(df.get("CXR- PPM to cardiac silhouette - shortest (mm)"))
    df["dist_lv_apex_mm"] = to_num(df.get("PPM to LV Apex"))
    df["widest_chest_mm"] = to_num(df.get("Widest Chest Transverse Diameter"))
    df["norm_dist_card_sil"] = safe_divide(df["dist_card_sil_mm"], df["widest_chest_mm"])
    df["norm_dist_LV_apex"] = safe_divide(df["dist_lv_apex_mm"], df["widest_chest_mm"])

    # plan 1.4: sequences & artefact
    seqs = [
        {"name": "TFE", "sfx": "", "any": "Any artefact (grade 3 or above)", "ratio": "Artefact ratio (biventricular)"},
        {"name": "CINE", "sfx": ".1", "any": "Any artefact (grade 3 or above).1", "ratio": "Artefact ratio (biventricular).1"},
        {"name": "VIAB", "sfx": ".2", "any": "Any artefact (grade 3 or above).2", "ratio": "Artefact ratio (biventricular).2"},
    ]

    def region_cols(sfx: str) -> dict:
        return {
            "lateral": [f"B_Anterolateral{sfx}", f"B_Inferolateral{sfx}", f"M_Anterolateral{sfx}", f"M_Inferolateral{sfx}", f"A_Lateral{sfx}"],
            "septal": [f"B_Inferoseptal{sfx}", f"B_Anteroseptal{sfx}", f"M_Inferoseptal{sfx}", f"M_Anteroseptal{sfx}", f"A_Septal{sfx}"],
            "anterior": [f"B_Anterior{sfx}", f"M_Anterior{sfx}", f"A_Anterior{sfx}"],
            "inferior": [f"B_Inferior{sfx}", f"M_Inferior{sfx}", f"A_Inferior{sfx}"],
        }

    ratio_cols = []
    wall_any_severe_cols = {"lateral": [], "septal": [], "anterior": [], "inferior": []}
    for info in seqs:
        name, sfx = info["name"], info["sfx"]
        any_col, ratio_col = info["any"], info["ratio"]
        df[f"severe_art_{name}"] = (to_num(df.get(any_col)) >= 1).fillna(0).astype(int)
        df[f"ratio_{name}"] = to_num(df.get(ratio_col))
        ratio_cols.append(f"ratio_{name}")

        walls = region_cols(sfx)
        for wall, cols in walls.items():
            existing = [c for c in cols if c in df.columns]
            if existing:
                sev_any = (pd.concat([to_num(df[c]) for c in existing], axis=1) >= 3).any(axis=1).astype(int)
            else:
                sev_any = pd.Series(0, index=df.index)
            df[f"{wall}_severe_{name}"] = sev_any
            df[f"lv_{wall}_clean_{name}"] = (1 - sev_any).astype(int)
            wall_any_severe_cols[wall].append(f"{wall}_severe_{name}")

    df["artifact_burden"] = df[ratio_cols].apply(lambda r: r.dropna().mean(), axis=1)
    df["max_ratio"] = df[ratio_cols].max(axis=1, skipna=True)

    for wall, cols in wall_any_severe_cols.items():
        if cols:
            any_severe = (df[cols].max(axis=1) >= 1).astype(int)
            df[f"lv_{wall}_clean"] = (1 - any_severe).astype(int)
        else:
            df[f"lv_{wall}_clean"] = np.nan

    df["lv_visibility_score"] = df[[
        "lv_lateral_clean",
        "lv_septal_clean",
        "lv_anterior_clean",
        "lv_inferior_clean",
    ]].sum(axis=1, min_count=1)

    # cause of artefact (binaryization)
    cause_cols = [
        ("Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None)", "RV Artefact cause (Lead alone=1, lead and device=2, neither=3,IPG=4)"),
        ("Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None).1", "RV Artefact cause (Lead alone=1, lead and device=2, neither=3, IPG=4)"),
        ("Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None).2", "RV Artefact cause (Lead alone=1, lead and device=2, neither=3, IPG=4).1"),
    ]
    ipg_any, lead_any, both_any = [], [], []
    for cause_col, rv_col in cause_cols:
        c = to_num(df[cause_col]) if cause_col in df.columns else pd.Series(np.nan, index=df.index)
        rvc = to_num(df[rv_col]) if rv_col in df.columns else pd.Series(np.nan, index=df.index)
        ipg_any.append(((c == 1) | (rvc == 4)).astype(int))
        lead_any.append(((c == 2) | (rvc.isin([1, 2]))).astype(int))
        both_any.append((c == 3).astype(int))
    df["cause_IPG"] = (pd.concat(ipg_any, axis=1).max(axis=1) >= 1).astype(int)
    df["cause_lead"] = (pd.concat(lead_any, axis=1).max(axis=1) >= 1).astype(int)
    df["cause_both"] = (pd.concat(both_any, axis=1).max(axis=1) >= 1).astype(int)

    # sequence availability
    df["has_TFE"] = 0
    df["has_CINE"] = 0
    df["has_VIAB"] = 0
    if "TFE (exact sequence listed)" in df.columns:
        df["has_TFE"] = df["TFE (exact sequence listed)"].notna().astype(int)
    if "CINE_SSFP (exact sequence listed)" in df.columns:
        df["has_CINE"] = df["CINE_SSFP (exact sequence listed)"].notna().astype(int)
    if "VIAB (exact sequence listed)" in df.columns:
        df["has_VIAB"] = df["VIAB (exact sequence listed)"].notna().astype(int)

    # plan 2: outcomes
    df["NonDiagnostic"] = to_bin(df.get("Non-diagnostic", 0))
    df["AddInfo"] = to_bin(df.get("Did MRI provide additional information to the existing diagnosis? (ie quantity of iron, location of scar, etc) ", 0))
    df["Confirmed"] = to_bin(df.get(" Was the pre-MRI (tentative) diagnosis confirmed?", 0))
    df["MgmtChange"] = to_bin(df.get("Was patient management altered as a result of the scan data?", 0))
    df["dx_change"] = (
        df["pre_dx_cat"].notna() & df["post_dx_cat"].notna() & (df["pre_dx_cat"] != df["post_dx_cat"])  # noqa: E712
    ).astype(int)
    df["UtilityScore"] = (
        df["dx_change"].fillna(0)
        + df["MgmtChange"].fillna(0)
        + df["AddInfo"].fillna(0)
        - df["NonDiagnostic"].fillna(0)
    )

    return df


def main() -> None:
    df = pd.read_excel(SRC_XLSX, engine="openpyxl")
    df = build_features(df)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception:
        pass
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    # plan 6: export clean dataset
    main()

