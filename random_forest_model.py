#%%
# Joining example code using SEQN
import pandas as pd
import requests

# Function to download and load an XPT file into a DataFrame
def download_xpt(url, file_name):
    response = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(response.content)
    return pd.read_sas(file_name, format='xport')

## DEMOGRAPHIC DATA
demographics_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt'
demographics = download_xpt(demographics_url, 'demographics.xpt')
demographics.rename(columns={'RIDAGEYR': 'Age in years at screening'}, inplace=True)

## BLOOD TEST DATA
blood_test_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.XPT'
blood_test = download_xpt(blood_test_url, 'blood_test.xpt')
blood_test.rename(columns={'LBXRBCSI': 'Red blood cell count (million cells/uL)'}, inplace=True)
blood_test.rename(columns={'LBXRDW': 'Red cell distribution width (%)'}, inplace=True)

## BLOOD PRESSURE DATA
BP_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.XPT'
BP = download_xpt(BP_url, 'BP.xpt')

## CHOLESTROL TOTAL DATA
cholestrol_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.XPT'
cholestrol = download_xpt(cholestrol_url, 'cholestrol.xpt')
cholestrol.rename(columns={'LBXTC': 'Total Cholesterol (mg/dL)'}, inplace=True)

## GLUCOSE DATA
glucose_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.XPT'
glucose = download_xpt(glucose_url, 'glucose.xpt')
glucose.rename(columns={'LBXGLU': 'Fasting Glucose (mg/dL)'}, inplace=True)

#LEAD
lead_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PBCD_L.XPT'
lead = download_xpt(lead_url, 'lead.xpt')
lead_columns = lead.columns

# Steroid Hormones
steroidhormones_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TST_L.XPT'
steroidhormones = download_xpt(steroidhormones_url, 'steroidhormones.xpt')
steroidhormones.rename(columns={'WTPH2YR_x': 'dont use'}, inplace=True)

#Acid Glycoprotein
acidGlycoprotein_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/AGP_L.XPT'
acidGlycoprotein = download_xpt(acidGlycoprotein_url, 'acidGlycoprotein.xpt')

#Cholesterol – High-Density Lipoprotein
cholesterolHighDensity_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HDL_L.XPT'
cholesterolHighDensity = download_xpt(cholesterolHighDensity_url, 'cholesterolHighDensity.xpt')


#Ferritin
ferritin_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FERTIN_L.XPT'
ferritin = download_xpt(ferritin_url, 'ferritin.xpt')

#Folate - RBC
folate_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FOLATE_L.XPT'
folate = download_xpt(folate_url, 'folate.xpt')

#Glycohemoglobin
glycohemoglobin_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.XPT'
glycohemoglobin = download_xpt(glycohemoglobin_url, 'glycohemoglobin.xpt')


#Hepatitis A
hepatitisA_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HEPA_L.XPT'
hepatitisA = download_xpt(hepatitisA_url, 'hepatitisA.xpt')


#Hepatitis B Surface Antibody
hepatitisB_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HEPB_S_L.XPT'
hepatitisB = download_xpt(hepatitisB_url, 'hepatitisB.xpt')

#High-Sensitivity C-Reactive Protein
creactiveProtein_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HSCRP_L.XPT'
creactiveProtein = download_xpt(creactiveProtein_url, 'creactiveProtein.xpt')

#Insulin
insulin_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/INS_L.XPT'
insulin = download_xpt(insulin_url, 'insulin.xpt')

#Mercury: Inorganic, Ethyl, and Methyl - Blood
mercury_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/IHGEM_L.XPT'
mercury = download_xpt(mercury_url, 'mercury.xpt')

#Serum
serum_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FOLFMS_L.XPT'
serum = download_xpt(serum_url, 'serum.xpt')

#Transferrin Receptor
transferrinReceptor_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TFR_L.XPT'
transferrinReceptor = download_xpt(transferrinReceptor_url, 'transferrinReceptor.xpt')

#Vitamin D
vitaminD_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/VID_L.XPT'
vitaminD = download_xpt(vitaminD_url, 'vitaminD.xpt')


#merged_dfFinal = pd.merge(merged_df7, steroidhormones, on='SEQN', how='inner', suffixes=('_left', '_right'))
merged_dfFinal= demographics
datasets = [
    blood_test, BP, cholestrol, glucose, lead, steroidhormones, acidGlycoprotein,
    cholesterolHighDensity, ferritin, folate, glycohemoglobin, hepatitisA, hepatitisB,
    creactiveProtein, insulin, mercury, serum, transferrinReceptor, vitaminD
]

# Merge datasets on SEQN with suffixes to avoid conflicts
for dataset in datasets:
    merged_dfFinal = merged_dfFinal.merge(dataset, on='SEQN', how='outer', suffixes=('', '_dup'))

# Remove duplicated columns if they are unnecessary
duplicated_columns = [col for col in merged_dfFinal.columns if col.endswith('_dup')]
merged_dfFinal.drop(columns=duplicated_columns, inplace=True)

rename_dict = {
    "LBDDHESI": "DHEAS (µmol/L)",
    "LBDAMHLC": "Anti-Mullerian hormone Comment Code",
    "LBXFSH": "Follicle Stimulating Hormone (mIU/mL)",
    "LBDANDSI": "Androstenedione (nmol/L)",
    "LBDAMHSI": "Anti-Mullerian hormone (pmol/L)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBXLUH": "Luteinizing Hormone (mIU/mL)",
    "LBXHA": "Hepatitis A antibody",
    "LBDLUHLC": "Luteinizing Hormone Comment Code",
    "LBDTHGLC": "Blood mercury, total comment code",
    "WTSAF2YR": "Fasting Subsample 2 Year MEC Weight",
    "LBDBCDLC": "Blood cadmium comment code",
    "LBDBCDSI": "Blood cadmium (nmol/L)",
    "LBXFER": "Ferritin (ng/mL)",
    "LBD17HSI": "17α-hydroxyprogesterone (nmol/L)",
    "LBXSF1SI": "5-Methyl-tetrahydrofolate (nmol/L)",
    "LBDGLUSI": "Fasting Glucose (mmol/L)",
    "LBDFOTSI": "Serum total folate (nmol/L)",
    "LBXGH": "Glycohemoglobin (%)",
    "LBDBMNSI": "Blood manganese (nmol/L)",
    "LBDIHGLC": "Mercury, inorganic comment code",
    "LBDHRPLC": "HS C-Reactive Protein Comment Code",
    "LBDBGMLC": "Mercury, methyl comment code",
    "LBDPG4SI": "Progesterone (nmol/L)",
    "LBXAGP": "alpha-1-acid glycoprotein (g/L)",
    "LBDHDD": "Direct HDL-Cholesterol (mg/dL)",
    "LBDHDDSI": "Direct HDL-Cholesterol (mmol/L)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBDTCSI": "Total Cholesterol (mmol/L)",
    "LBXLYPCT": "Lymphocyte percent (%)",
    "LBXMOPCT": "Monocyte percent (%)",
    "LBXNEPCT": "Segmented neutrophils percent (%)",
    "LBXBAPCT": "Basophils percent (%)",
    "LBDLYMNO": "Lymphocyte number (1000 cells/uL)",
    "LBDMONO": "Monocyte number (1000 cells/uL)",
    "LBDNENO": "Segmented neutrophils number (1000 cells/uL)",
    "LBDBANO": "Basophils number (1000 cells/uL)",
    "LBXRBCSI": "Red blood cell count (million cells/uL)",
    "LBXHGB": "Hemoglobin (g/dL)",
    "LBXHCT": "Hematocrit (%)",
    "LBXMCVSI": "Mean cell volume (fL)",
    "LBXMC": "Mean Cell Hgb Concentration (g/dL)",
    "LBXMCHSI": "Mean cell hemoglobin (pg)",
    "LBXRDW": "Red cell distribution width (%)",
    "LBXMPSI": "Mean platelet volume (fL)",
    "LBXFER": "Ferritin (ng/mL)",
    "LBDFERSI": "Ferritin (µg/L)",
    "LBDRFO": "RBC folate (ng/mL)",
    "LBDRFOSI": "RBC folate (nmol/L)",
    "LBXGH": "Glycohemoglobin (%)",
    "LBXHA": "Hepatitis A antibody",
    "LBXHBS": "Hepatitis B Surface Antibody",
    "LBXHSCRP": "HS C-Reactive Protein (mg/L)",
    "LBDHRPLC": "HS C-Reactive Protein Comment Code",
    "LBXIN": "Insulin (uU/mL)",
    "LBDINSI": "Insulin (pmol/L)",
    "LBXBPB": "Blood lead (ug/dL)",
    "LBDBPBSI": "Blood lead (umol/L)",
    "LBXBCD": "Blood cadmium (ug/L)",
    "LBDBCDSI": "Blood cadmium (nmol/L)",
    "LBDBCDLC": "Blood cadmium comment code",
    "LBXTHG": "Blood mercury, total (ug/L)",
    "LBDTHGSI": "Blood mercury, total (nmol/L)",
    "LBDTHGLC": "Blood mercury, total comment code",
    "LBXBSE": "Blood selenium (ug/L)",
    "LBDBSESI": "Blood selenium (umol/L)",
    "LBXBMN": "Blood manganese (ug/L)",
    "LBDBMNSI": "Blood manganese (nmol/L)",
    "LBXIHG": "Mercury, inorganic (ug/L)",
    "LBDIHGSI": "Mercury, inorganic (nmol/L)",
    "LBDIHGLC": "Mercury, inorganic comment code",
    "LBXBGM": "Mercury, methyl (ug/L)",
    "LBDBGMSI": "Mercury, methyl (nmol/L)",
    "LBDBGMLC": "Mercury, methyl comment code",
    "LBXGLU": "Fasting Glucose (mg/dL)",
    "LBDGLUSI": "Fasting Glucose (mmol/L)",
    "LBDFOTSI": "Serum total folate (nmol/L)",
    "LBDFOT": "Serum total folate (ng/mL)",
    "LBXSF1SI": "5-Methyl-tetrahydrofolate (nmol/L)",
    "WTSAF2YR" : "Fasting Subsample 2 Year MEC Weight",
    "LBDANDSI": "Androstenedione (nmol/L)",
    "LBDAMHLC": "Anti-Mullerian hormone Comment Code",
    "LBXFSH": "Follicle Stimulating Hormone (mIU/mL)",
    "LBDAMHSI": "Anti-Mullerian hormone (pmol/L)",
    "LBXLUH": "Luteinizing Hormone (mIU/mL)",
    "LBXMCVSI": "Mean cell volume (fL)",
    "LBDPG4SI": "Progesterone (nmol/L)",
    "LBDES1SI": "Estrone Sulfate (pmol/L)",
    "LBDESTSI": "Estradiol (pmol/L)",
    "LBDFSHLC": "FSH Comment Code",
    "LBDESTLC": "Estradiol Comment Code",
    "LBDLUHLC": "Luteinizing Hormone Comment Code",
    "LBDTSTSI": "Testosterone, total (nmol/L)",
    "RIDAGEYR": "Age in years at screening"

 }


merged_dfFinal.rename(columns=rename_dict, inplace=True)


selected_vars = ['DHEAS (µmol/L)',
'Follicle Stimulating Hormone (mIU/mL)',
'Androstenedione (nmol/L)',
'Anti-Mullerian hormone (pmol/L)',
'Total Cholesterol (mg/dL)',
'Hepatitis A antibody',
'Red blood cell count (million cells/uL)',
'Segmented neutrophils percent (%)',
'Lymphocyte percent (%)',
'Testosterone, total (nmol/L)',
'HS C-Reactive Protein (mg/L)',
'RBC folate (nmol/L)',
'RBC folate (ng/mL)',
'Total Cholesterol (mmol/L)',
'Hemoglobin (g/dL)',
'Fasting Glucose (mg/dL)',
'Fasting Glucose (mmol/L)',
'Glycohemoglobin (%)',
'Progesterone (nmol/L)',
'Insulin (pmol/L)',
'Insulin (uU/mL)',
'Red cell distribution width (%)',
'Age in years at screening',

'Blood manganese (nmol/L)',
'Serum total folate (nmol/L)',
'Blood cadmium (nmol/L)',
'Ferritin (ng/mL)',
'17α-hydroxyprogesterone (nmol/L)',
'5-Methyl-tetrahydrofolate (nmol/L)',
'Fasting Subsample 2 Year MEC Weight',
'Luteinizing Hormone (mIU/mL)',


'Mean platelet volume (fL)',
'Mean cell volume (fL)',
 'Mean cell hemoglobin (pg)',

]


df_for_model = merged_dfFinal[selected_vars]


# Assuming df_for_model is your DataFrame
df = df_for_model.copy()

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Identify columns with missing values
columns_with_nan = df.columns[df.isna().any()].tolist()

#df.dropna(inplace=True)

# Iterate through columns with missing values
#for column in columns_with_nan:
#  print(column)
column = 'Ferritin (ng/mL)'
#  try:
#    # Attempt to convert column to numeric if it has mixed types
if not pd.api.types.is_numeric_dtype(df[column]):
  df[column] = pd.to_numeric(df[column], errors='coerce')

if pd.api.types.is_numeric_dtype(df[column]):
  # For numerical columns, fill missing values with the median
  median_value = df[column].median()
  df[column] = df[column].fillna(median_value)
else:
  # For non-numerical columns, fill missing values with the mode
  mode_value = df[column].mode()[0]
  df[column] = df[column].fillna(mode_value)
#  except Exception as e:
#    print(f"An error occurred with column '{column}': {e}")

#filt = df['Age in years at screening'] <= 10
#df.drop(index=df[filt].index)

df.dropna(inplace=True)
#df = df.iloc[2000:3000]
df
df.to_excel('built_nhanes_dataset.xlsx')
# Build a framework using scikit-learn to run ml models of different types with age_in_years_at_screening as the target variable.
# Start with linear regression, then do random forest models, then a simple one-layer neural network.

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create a copy of the original data
starting_data = df.copy()

# Split the data into features (X) and the target variable (y)
X = starting_data.drop(columns=['Age in years at screening'])
y = starting_data['Age in years at screening']

# Split the training and test data, use a random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# After fitting the scaler and the model
from sklearn.preprocessing import StandardScaler
import joblib

# Assuming 'scaler' is your StandardScaler instance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(rf_model, 'rf_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

y_pred_rf = rf_model.predict(X_test_scaled)

# Print RF evaluation outputs
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Root Mean Squared Error: {rmse_rf}")
print(f"Random Forest R-squared: {r2_rf}")
# %%
