import pandas as pd
import numpy as np

# plan 0: load data
df = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/CIED/Database 6.12 clean LB.xlsx",
    engine="openpyxl",
)

# helpers
def to_num(series):
    return pd.to_numeric(series, errors="coerce")

def to_bin(series):
    s = series if isinstance(series, pd.Series) else pd.Series(series, index=df.index)
    if s.dtype == object:
        s = s.str.strip().str.lower()
        s = s.replace({
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
        })
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def safe_divide(num, den):
    num = to_num(num)
    den = to_num(den)
    out = num / den.replace(0, np.nan)
    return out

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
    if old in df.columns:
        df[new] = to_bin(df[old])
    else:
        df[new] = 0

# unify pre/post dx
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
type_col = "Type of Device (ICD =1 PPM = 2 CRT = 3)"
df["device_type_code"] = to_num(df.get(type_col))
df["device_cat"] = df["device_type_code"].map({1: "ICD", 2: "PPM", 3: "CRT"}).astype("object")
df["is_CRT"] = (df["device_type_code"] == 3).astype(int)

df["icd_indication_code"] = to_num(df.get("ICD indication (Primary prevention = 1, secodary prevention =2)"))
df["ppm_indication_code"] = to_num(df.get("PPM indications (CHB = 1, SND = 2, Other = 3)"))
df["indication_missing"] = (
    df["icd_indication_code"].isna() & df["ppm_indication_code"].isna()
).astype(int)

mr_cond_col = "MR Conditional "
df["mr_conditional"] = to_bin(df.get(mr_cond_col, 0))

pos_body_col = "Position in body (left chest = 1, right chest = 2, leadless = 3, 4=subC)"
df["position_body_code"] = to_num(df.get(pos_body_col))
df["left_chest"] = (df["position_body_code"] == 1).astype(int)
df["right_chest"] = (df["position_body_code"] == 2).astype(int)
df["leadless"] = (df["position_body_code"] == 3).astype(int)
df["subQ"] = (df["position_body_code"] == 4).astype(int)
df["left_vs_other"] = (df["position_body_code"] == 1).astype(int)

left_other_col = "Position (L chest=1, other=2)"
df["left_vs_other_coarse"] = (to_num(df.get(left_other_col)) == 1).astype(int)

manu_col = "Manufacturer (1=Boston,2=MDT,3=Bio,4=StJude,5=other)"
df["manufacturer_code"] = to_num(df.get(manu_col))
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
rot_col = "Rotation. 0 = normal, 1 rotated."
df["rotation"] = to_bin(df.get(rot_col, 0))
dist_card_col = "CXR- PPM to cardiac silhouette - shortest (mm)"
dist_lv_apex_col = "PPM to LV Apex"
chest_w_col = "Widest Chest Transverse Diameter"
df["dist_card_sil_mm"] = to_num(df.get(dist_card_col))
df["dist_lv_apex_mm"] = to_num(df.get(dist_lv_apex_col))
df["widest_chest_mm"] = to_num(df.get(chest_w_col))
df["norm_dist_card_sil"] = safe_divide(df["dist_card_sil_mm"], df["widest_chest_mm"])
df["norm_dist_LV_apex"] = safe_divide(df["dist_lv_apex_mm"], df["widest_chest_mm"])

# plan 1.4: sequences & artefact
seqs = [
    {"name": "TFE", "sfx": "", "any": "Any artefact (grade 3 or above)", "ratio": "Artefact ratio (biventricular)"},
    {"name": "CINE", "sfx": ".1", "any": "Any artefact (grade 3 or above).1", "ratio": "Artefact ratio (biventricular).1"},
    {"name": "VIAB", "sfx": ".2", "any": "Any artefact (grade 3 or above).2", "ratio": "Artefact ratio (biventricular).2"},
]

def region_cols(sfx):
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

# global artefact burden
df["artifact_burden"] = df[ratio_cols].apply(lambda r: r.dropna().mean(), axis=1)
df["max_ratio"] = df[ratio_cols].max(axis=1, skipna=True)

# global LV visibility
for wall, cols in wall_any_severe_cols.items():
    if cols:
        any_severe = (df[cols].max(axis=1) >= 1).astype(int)
        df[f"lv_{wall}_clean"] = (1 - any_severe).astype(int)
    else:
        df[f"lv_{wall}_clean"] = np.nan

df["lv_visibility_score"] = df[[
    "lv_lateral_clean", "lv_septal_clean", "lv_anterior_clean", "lv_inferior_clean"
]].sum(axis=1, min_count=1)

# cause of artefact (binaryization)
# explicit mapping due to dataset suffix inconsistencies for RV cause
cause_cols = [
    ("Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None)", "RV Artefact cause (Lead alone=1, lead and device=2, neither=3,IPG=4)"),
    ("Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None).1", "RV Artefact cause (Lead alone=1, lead and device=2, neither=3, IPG=4)"),
    ("Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None).2", "RV Artefact cause (Lead alone=1, lead and device=2, neither=3, IPG=4).1"),
]
ipg_any, lead_any, both_any = [], [], []
for cause_col, rv_col in cause_cols:
    if cause_col in df.columns:
        c = to_num(df[cause_col])
    else:
        c = pd.Series(np.nan, index=df.index)
    if rv_col in df.columns:
        rvc = to_num(df[rv_col])
    else:
        rvc = pd.Series(np.nan, index=df.index)
    ipg_any.append(((c == 1) | (rvc == 4)).astype(int))
    lead_any.append(((c == 2) | (rvc.isin([1, 2]))).astype(int))
    both_any.append((c == 3).astype(int))

df["cause_IPG"] = (pd.concat(ipg_any, axis=1).max(axis=1) >= 1).astype(int)
df["cause_lead"] = (pd.concat(lead_any, axis=1).max(axis=1) >= 1).astype(int)
df["cause_both"] = (pd.concat(both_any, axis=1).max(axis=1) >= 1).astype(int)

# sequence availability flags
df["has_TFE"] = df.get("TFE (exact sequence listed)").notna() if "TFE (exact sequence listed)" in df.columns else False
df["has_CINE"] = df.get("CINE_SSFP (exact sequence listed)").notna() if "CINE_SSFP (exact sequence listed)" in df.columns else False
df["has_VIAB"] = df.get("VIAB (exact sequence listed)").notna() if "VIAB (exact sequence listed)" in df.columns else False
df[["has_TFE", "has_CINE", "has_VIAB"]] = df[["has_TFE", "has_CINE", "has_VIAB"]].astype(int)

# plan 2: outcomes
non_diag_col = "Non-diagnostic"
df["NonDiagnostic"] = to_bin(df.get(non_diag_col, 0))
add_info_col = "Did MRI provide additional information to the existing diagnosis? (ie quantity of iron, location of scar, etc) "
df["AddInfo"] = to_bin(df.get(add_info_col, 0))
confirm_col = " Was the pre-MRI (tentative) diagnosis confirmed?"
df["Confirmed"] = to_bin(df.get(confirm_col, 0))
mgmt_col = "Was patient management altered as a result of the scan data?"
df["MgmtChange"] = to_bin(df.get(mgmt_col, 0))

df["dx_change"] = (
    (df["pre_dx_cat"].notna() & df["post_dx_cat"].notna() & (df["pre_dx_cat"] != df["post_dx_cat"]))
).astype(int)

df["UtilityScore"] = (
    df["dx_change"].fillna(0)
    + df["MgmtChange"].fillna(0)
    + df["AddInfo"].fillna(0)
    - df["NonDiagnostic"].fillna(0)
)

# plan 6: save clean dataset
out_parquet = "/home/sunx/data/aiiih/projects/sunx/projects/CIED/clean.parquet"
try:
    df.to_parquet(out_parquet, index=False)
except Exception:
    pass
df.to_csv("/workspace/clean.csv", index=False)
