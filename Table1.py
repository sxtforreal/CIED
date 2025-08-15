import pandas as pd
from tableone import TableOne
from scipy.stats import shapiro

# Load the Excel file
df = pd.read_excel(
    "/home/sunx/data/aiiih/projects/sunx/projects/CIED/Database 6.12 clean LB.xlsx",
    engine="openpyxl",
)

# Rename columns
column_rename_map = {
    "MR_indication_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)\n\n*changed to mached pre/post)": "MR_indication_simplified"
}
df = df.rename(columns=column_rename_map)

# Debug: Inspect the column
print(
    "Unique values in MR_indication_simplified:",
    df["MR_indication_simplified"].unique(),
)
print(
    "Data types in MR_indication_simplified:",
    df["MR_indication_simplified"].apply(type).value_counts(),
)

# Convert MR_indication_simplified to integer, handling non-numeric values
df["MR_indication_simplified"] = pd.to_numeric(
    df["MR_indication_simplified"], errors="coerce"
)
print("After conversion, unique values:", df["MR_indication_simplified"].unique())
print("NaN count:", df["MR_indication_simplified"].isna().sum())

# Drop rows with NaN in MR_indication_simplified if any
if df["MR_indication_simplified"].isna().sum() > 0:
    print(
        f"Dropping {df['MR_indication_simplified'].isna().sum()} rows with NaN in MR_indication_simplified"
    )
    df = df.dropna(subset=["MR_indication_simplified"])

# Define your column lists
demo = ["Age", "Sex (1-M)", "Height cm", "Weight Kg", "BMI"]
disease = [
    "HF (1=Y)",
    "HTN (1=Y)",
    "CAD",
    "MI (Y=1)",
    "VT/VF",
    "Vfib (1=Y)",
    "VT (1=Y)",
    "Afib (1=Y)",
    "CKD (1=Y)",
]
device_info = [
    "Type of Device (ICD =1 PPM = 2 CRT = 3)",
    "ICD indication (Primary prevention = 1, secodary prevention =2)",
    "PPM indications (CHB = 1, SND = 2, Other = 3)",
    "MR Conditional ",
    "Position in body (left chest = 1, right chest = 2, leadless = 3, 4=subC)",
    "Position (L chest=1, other=2)",
    "Manufacturer (1=Boston,2=MDT,3=Bio,4=StJude,5=other)",
    "Atrial Lead Yes/No",
    "Ventricular lead Yes/No",
    "SubQ lead - (0=no,1=yes,3=other (leadless)",
    "LV lead Yes/No",
]
position = [
    "Rotation. 0 = normal, 1 rotated.",
    "PPM to RV Lead",
    "PPM to LV Lead",
    "PPM to subQ (can to subQ lead)",
    "CXR- PPM to cardiac silhouette - shortest (mm)",
    "PPM to LV Apex",
    "Widest Chest Transverse Diameter",
    "Lat view- Inf edge of PPM to RV lead tip",
    "Lat view- Inf edge of PPM to LV lead tip",
    "Lat view- Inf edge of PPM to SubQ lead tip",
]
label = [
    "Pre-MR diagnosis/suspected_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)",
    "Post-MR diagnosis/suspected_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)",
    "Non-diagnostic",
    "Did the patients' suspected diagnosis change post MR",
    "Did MRI provide additional information to the existing diagnosis? (ie quantity of iron, location of scar, etc) ",
    " Was the pre-MRI (tentative) diagnosis confirmed?",
    "Was patient management altered as a result of the scan data?",
]


tfe = [
    "TFE (exact sequence listed)",
    "Any artefact (grade 3 or above)",
    "Artefact ratio (biventricular)",
    "B_Anterior",
    "B_Anterolateral",
    "B_Inferolateral",
    "B_Inferior",
    "B_Inferoseptal",
    "B_Anteroseptal",
    "M_Anterior",
    "M_Anterolateral",
    "M_Inferolateral",
    "M_Inferior",
    "M_Inferoseptal",
    "M_Anteroseptal",
    "A_Anterior",
    "A_Lateral",
    "A_Inferior",
    "A_Septal",
    "RV_Base",
    "RV_Mid",
    "RV_Apex",
    "Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None)",
    "RV Artefact cause (Lead alone=1, lead and device=2, neither=3,IPG=4)",
]
cine_ssfp = [
    "CINE_SSFP (exact sequence listed)",
    "Any artefact (grade 3 or above).1",
    "Artefact ratio (biventricular).1",
    "B_Anterior.1",
    "B_Anterolateral.1",
    "B_Inferolateral.1",
    "B_Inferior.1",
    "B_Inferoseptal.1",
    "B_Anteroseptal.1",
    "M_Anterior.1",
    "M_Anterolateral.1",
    "M_Inferolateral.1",
    "M_Inferior.1",
    "M_Inferoseptal.1",
    "M_Anteroseptal.1",
    "A_Anterior.1",
    "A_Lateral.1",
    "A_Inferior.1",
    "A_Septal.1",
    "RV_Base.1",
    "RV_Mid.1",
    "RV_Apex.1",
    "Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None).1",
    "RV Artefact cause (Lead alone=1, lead and device=2, neither=3, IPG=4)",
]
viab = [
    "VIAB (exact sequence listed)",
    "Any artefact (grade 3 or above).2",
    "Artefact ratio (biventricular).2",
    "B_Anterior.2",
    "B_Anterolateral.2",
    "B_Inferolateral.2",
    "B_Inferior.2",
    "B_Inferoseptal.2",
    "B_Anteroseptal.2",
    "M_Anterior.2",
    "M_Anterolateral.2",
    "M_Inferolateral.2",
    "M_Inferior.2",
    "M_Inferoseptal.2",
    "M_Anteroseptal.2",
    "A_Anterior.2",
    "A_Lateral.2",
    "A_Inferior.2",
    "A_Septal.2",
    "RV_Base.2",
    "RV_Mid.2",
    "RV_Apex.2",
    "Breathing artifact (No=0, Yes=1)",
    "Cause of Artifact (1=IPG, 2=Lead, 3=Both, 0=None).2",
    "RV Artefact cause (Lead alone=1, lead and device=2, neither=3, IPG=4).1",
]


def create_table1(group_name, columns, discrete):
    # Convert non-numeric values to NaN for all columns
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")

    # Determine continuous columns (all columns minus discrete)
    continuous = [col for col in columns if col not in discrete]

    # Check normality for continuous variables
    nonnormal = []
    for col in continuous:
        if (
            col in df.columns and len(df[col].dropna()) > 3
        ):  # Need at least 4 values for Shapiro test
            stat, p = shapiro(df[col].dropna())
            if p < 0.05:  # Non-normal if p < 0.05
                nonnormal.append(col)
    print(f"Non-normal continuous variables: {nonnormal}")

    # Create TableOne instance with nonnormal variables and ignore NaN in p-values
    table1 = TableOne(
        df,
        columns=columns,
        groupby="MR_indication_simplified",
        pval=True,
        nonnormal=nonnormal,  # Use nonnormal list based on normality test
        categorical=discrete,
        rename={
            1: "Infiltrative",
            2: "Valvulopathy",
            3: "HOCM",
            4: "Mypericarditis",
            5: "Ischemia",
            6: "Other Unexplained CMP",
            7: "Other",
            8: "VT",
        },
        missing=True,
        pval_adjust="bonferroni",  # Adjusts p-values, handles NaN implicitly by default
    )

    # Convert TableOne object to DataFrame
    try:
        table1_df = table1.to_dataframe()
    except AttributeError:
        print("to_dataframe() not available. Trying tableone attribute...")
        if hasattr(table1, "tableone"):
            table1_df = table1.tableone
        elif hasattr(table1, "data"):
            table1_df = table1.data
        else:
            raise AttributeError(
                "Neither to_dataframe(), tableone, nor data attribute is available. Please check tableone version or update the package."
            )

    # Save to Excel
    output_file = (
        f"/home/sunx/data/aiiih/projects/sunx/projects/CIED/table1_{group_name}.xlsx"
    )
    table1_df.to_excel(output_file)
    print(f"Table 1 for {group_name} has been saved to {output_file}")


# Example usage with specified discrete and continuous variables
# Define the categories for each group
discrete_demo = ["Sex (1-M)"]
discrete_disease = [
    "HF (1=Y)",
    "HTN (1=Y)",
    "MI (Y=1)",
    "Vfib (1=Y)",
    "VT (1=Y)",
    "Afib (1=Y)",
    "CKD (1=Y)",
    "CAD",
    "VT/VF",
]
discrete_device_info = [
    "Type of Device (ICD =1 PPM = 2 CRT = 3)",
    "ICD indication (Primary prevention = 1, secodary prevention =2)",
    "PPM indications (CHB = 1, SND = 2, Other = 3)",
    "MR Conditional ",
    "Position in body (left chest = 1, right chest = 2, leadless = 3, 4=subC)",
    "Position (L chest=1, other=2)",
    "Manufacturer (1=Boston,2=MDT,3=Bio,4=StJude,5=other)",
    "Atrial Lead Yes/No",
    "Ventricular lead Yes/No",
    "SubQ lead - (0=no,1=yes,3=other (leadless)",
    "LV lead Yes/No",
]
discrete_position = ["Rotation. 0 = normal, 1 rotated."]
discrete_label = [
    "Pre-MR diagnosis/suspected_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)",
    "Post-MR diagnosis/suspected_simplified (1=infiltrative,2=valvulopathy,3=HOCM,4=myopericarditis,5=ischemia,6=other unexplained CMP,7=Othe, 8= VT)",
    "Non-diagnostic",
    "Did the patients' suspected diagnosis change post MR",
    "Did MRI provide additional information to the existing diagnosis? (ie quantity of iron, location of scar, etc) ",
    " Was the pre-MRI (tentative) diagnosis confirmed?",
    "Was patient management altered as a result of the scan data?",
]

# Create Table 1 for each group
create_table1("demo", demo, discrete_demo)
create_table1("disease", disease, discrete_disease)
create_table1("device_info", device_info, discrete_device_info)
create_table1("position", position, discrete_position)
create_table1("label", label, discrete_label)
