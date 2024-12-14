import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set the title of the app
st.title("True Age: Unlocking the Secrets of Your Biological Clock")

# Add a description
st.markdown("""
This application predicts your age based on various health parameters. Please enter the required information below and click on **Calculate my age** to see the prediction.
""")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.stop()

# Define the list of input features
selected_vars = [
    'DHEAS (µmol/L)',
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

# Have user input their age:
# **1. Add a new text input for Age**  
age_input = st.text_input('Age', placeholder='Enter your age') 

# Create input fields
st.header("Enter Your Health Parameters")
input_data = {}
for var in selected_vars:
    # You can customize input types based on variable names or other logic
    input_data[var] = st.text_input(var, placeholder=f"Enter {var}")

# Button to trigger prediction
if st.button("Calculate my age"):
    # Validate inputs
    missing_fields = [var for var in selected_vars if input_data[var].strip() == '']
    if missing_fields:
        st.error(f"Please fill in all fields. Missing: {', '.join(missing_fields)}")
    else:
        try:
            # Convert inputs to float
            input_values = []
            for var in selected_vars:
                value = float(input_data[var])
                input_values.append(value)
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_values], columns=selected_vars)
            
            # Scale the input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            predicted_age = model.predict(input_scaled)[0]
            
            # Display the result
            st.success(f"**Predicted Age:** {predicted_age:.2f} years")
        except ValueError:
            st.error("Please ensure all inputs are numerical values.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
