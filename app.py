from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "final_model.joblib"
st.set_page_config(page_title="CarMetric | Price Estimator", page_icon="🚗", layout="centered")
st.title("🚗 CarMetric")
st.caption("Used-car resale price estimation")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
st.write("Enter the vehicle details to estimate its resale price.")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        insurance = st.selectbox("Insurance validity", ["Comprehensive", "Third Party insurance", "Third Party", "Zero Dep", "Not Available"])
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
        ownership = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth Owner", "Fifth Owner"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        kms = st.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=45_000, step=1_000)
    with c2:
        seats = st.selectbox("Seats", [4, 5, 6, 7, 8], index=1)
        year = st.number_input("Manufacturing year", min_value=1990, max_value=2026, value=2018, step=1)
        mileage = st.number_input("Mileage (km/l)", min_value=1.0, max_value=100.0, value=18.0, step=0.5)
        engine = st.number_input("Engine capacity (cc)", min_value=50, max_value=10_000, value=1200, step=50)
        power = st.number_input("Max power (bhp)", min_value=1.0, max_value=2_000.0, value=80.0, step=1.0)
        torque = st.number_input("Torque (Nm)", min_value=1.0, max_value=10_000.0, value=110.0, step=5.0)
    submitted = st.form_submit_button("Estimate resale price", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "insurance_validity": insurance, "fuel_type": fuel, "ownsership": ownership,
        "transmission": transmission, "kms_driven": kms, "seats": seats,
        "manufacturing_year": year, "mileage(kmpl)": mileage, "engine(cc)": engine,
        "max_power(bhp)": power, "torque(Nm)": torque,
    }])
    prediction = float(pipeline.predict(row)[0])
    st.success(f"Estimated resale price: ₹{prediction:.2f} lakh")
    st.caption("This is a model estimate, not a dealer quote or guaranteed market value.")
