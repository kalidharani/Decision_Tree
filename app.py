import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
# Ensure 'pet_adoption_model.pkl' is uploaded to your GitHub repo
model = joblib.load('pet_adoption_model.pkl')

# Get the exact features the model was trained on
expected_features = model.feature_names_in_

st.set_page_config(page_title="Pet Adoption Predictor", page_icon="🐾")
st.title("🐾 Pet Adoption Predictor")

# 2. User Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Months)", 1, 240, 12)
    weight = st.number_input("Weight (kg)", 0.1, 100.0, 5.0)
    shelter_time = st.number_input("Days in Shelter", 1, 1000, 30)
    fee = st.number_input("Adoption Fee ($)", 0, 1000, 100)
    pet_type = st.selectbox("Pet Type", ["Dog", "Cat", "Bird", "Rabbit"])

with col2:
    # Breeds must match the ones in your training CSV exactly
    breed = st.selectbox("Breed", ["Labrador", "Poodle", "Golden Retriever", "Persian", "Siamese", "Parakeet", "Rabbit", "Hamster"])
    size = st.selectbox("Size", ["Small", "Medium", "Large"])
    health = st.selectbox("Health", ["Healthy", "Minor Issues", "Critical"])
    vac = st.radio("Vaccinated?", ["Yes", "No"])
    prev = st.radio("Previous Owner?", ["Yes", "No"])
    color = st.selectbox("Color", ["Black", "White", "Brown", "Gray", "Orange"])

# 3. Preprocessing logic
size_map = {"Small": 0, "Medium": 1, "Large": 2}
health_map = {"Healthy": 0, "Minor Issues": 1, "Critical": 2}

# Start with a dictionary of all zeros for every expected feature
input_dict = {feat: 0 for feat in expected_features}

# Fill numerical/binary fields
input_dict['age_months'] = age
input_dict['weight_kg'] = weight
input_dict['timein_shelter_days'] = shelter_time
input_dict['adoption_fee'] = fee
input_dict['size'] = size_map[size]
input_dict['health_condition'] = health_map[health]
input_dict['vaccinated'] = 1 if vac == "Yes" else 0
input_dict['previous_owner'] = 1 if prev == "Yes" else 0

# Handle One-Hot Encoding: Set the specific category column to 1
if f"pet_type_{pet_type}" in input_dict: input_dict[f"pet_type_{pet_type}"] = 1
if f"breed_{breed}" in input_dict: input_dict[f"breed_{breed}"] = 1
if f"color_{color}" in input_dict: input_dict[f"color_{color}"] = 1

# Convert to DataFrame and reorder columns to match the model's training
input_df = pd.DataFrame([input_dict])[expected_features]

# 4. Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("✨ High Likelihood of Adoption!")
    else:
        st.info("⌛ Lower Likelihood of Adoption.")