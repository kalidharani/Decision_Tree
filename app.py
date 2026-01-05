import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
model = joblib.load('pet_adoption_model.pkl')

# Get the exact feature names the model expects
expected_features = model.feature_names_in_

st.set_page_config(page_title="Pet Adoption Predictor", page_icon="🐾")
st.title("🐾 Pet Adoption Likelihood Predictor")

col1, col2 = st.columns(2)

with col1:
    age_months = st.number_input("Age (Months)", min_value=1, max_value=240, value=12)
    weight_kg = st.number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=5.0)
    timein_shelter = st.number_input("Time in Shelter (Days)", min_value=1, value=30)
    adoption_fee = st.number_input("Adoption Fee ($)", min_value=0, value=100)
    pet_type = st.selectbox("Pet Type", ["Dog", "Cat", "Bird", "Rabbit"])

with col2:
    # Added Breed selection to fix the error
    breed = st.selectbox("Breed", ["Labrador", "Parakeet", "Persian", "Poodle", "Rabbit", "Golden Retriever", "Siamese", "Hamster"])
    size = st.selectbox("Size", ["Small", "Medium", "Large"])
    health = st.selectbox("Health Condition", ["Healthy", "Minor Issues", "Critical"])
    vaccinated = st.selectbox("Vaccinated?", ["Yes", "No"])
    prev_owner = st.selectbox("Has Previous Owner?", ["Yes", "No"])
    color = st.selectbox("Color", ["Brown", "Gray", "Orange", "White", "Black"])

# --- DATA PREPROCESSING ---
# Mapping numerical values
size_map = {"Small": 0, "Medium": 1, "Large": 2}
health_map = {"Healthy": 0, "Minor Issues": 1, "Critical": 2}

# Create a dictionary with ALL features set to 0 initially
input_data = {feat: 0 for feat in expected_features}

# Fill in the numerical/binary features
input_data['age_months'] = age_months
input_data['weight_kg'] = weight_kg
input_data['timein_shelter_days'] = timein_shelter
input_data['adoption_fee'] = adoption_fee
input_data['size'] = size_map[size]
input_data['health_condition'] = health_map[health]
input_data['vaccinated'] = 1 if vaccinated == "Yes" else 0
input_data['previous_owner'] = 1 if prev_owner == "Yes" else 0

# Fill in One-Hot Encoded features (set the selected ones to 1)
if f"pet_type_{pet_type}" in input_data:
    input_data[f"pet_type_{pet_type}"] = 1

if f"breed_{breed}" in input_data:
    input_data[f"breed_{breed}"] = 1

if f"color_{color}" in input_data:
    input_data[f"color_{color}"] = 1

# Convert to DataFrame and ensure correct column order
input_df = pd.DataFrame([input_data])[expected_features]

if st.button("Predict Adoption Likelihood"):
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.success("🌟 This pet is **LIKELY** to be adopted!")
    else:
        st.warning("⏳ This pet is **UNLIKELY** to be adopted soon.")