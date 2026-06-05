import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model/crop_model.pkl", "rb"))

st.title("🌾 Smart Crop Advisory System")

st.write("Enter soil and weather details")

N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
K = st.number_input("Potassium (K)")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Recommend Crop"):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(data)

    st.success(f"Recommended Crop: {prediction[0]}")