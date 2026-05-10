# streamlit_app.py - COMPLETE CODE
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="Railway Seat Demand Predictor",
    page_icon="🚂",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("decision_tree_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train and save the model first.")
        st.stop()

classifier, scaler = load_model()
st.success("✅ Model loaded successfully!")

# Title
st.title("🚂 Indian Railway Seat Demand Prediction System")
st.markdown("---")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Train Details")
    train_type = st.selectbox(
        "Train Type",
        options=[("Express", 0), ("Passenger", 1), ("Rajdhani", 2), ("Shatabdi", 3), ("Superfast", 4)],
        format_func=lambda x: x[0]
    )[1]
    
    seat_type = st.selectbox(
        "Seat Type",
        options=[("AC First Class", 0), ("AC 2 Tier", 1), ("AC 3 Tier", 2), ("General", 3), ("Sleeper", 4)],
        format_func=lambda x: x[0]
    )[1]
    
    season = st.selectbox(
        "Season",
        options=[("Normal", 0), ("Off Season", 1), ("Peak Season", 2)],
        format_func=lambda x: x[0]
    )[1]

with col2:
    st.subheader("📊 Journey Details")
    distance_km = st.number_input("Distance (KM)", min_value=50, max_value=2500, value=300)
    base_price = st.number_input("Base Ticket Price (₹)", min_value=100, max_value=5000, value=500)
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        is_holiday = st.radio("Is Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    with col2_2:
        is_weekend = st.radio("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

# Predict button
if st.button("🔮 Predict Seat Demand", type="primary", use_container_width=True):
    # Prepare input
    input_data = np.array([[train_type, seat_type, season, distance_km, is_holiday, is_weekend, base_price]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = classifier.predict(input_scaled)[0]
    probabilities = classifier.predict_proba(input_scaled)[0]
    
    # Maps
    label_map = {0: "🔴 HIGH DEMAND", 1: "🟢 LOW DEMAND", 2: "🟡 MEDIUM DEMAND"}
    recommendations = {
        0: "⚠️ Book immediately! Seats will sell out fast.",
        1: "✅ Good availability. Flexible booking.",
        2: "📅 Book 1-2 weeks in advance."
    }
    
    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Demand Level", label_map[prediction].split()[1])
    with col_res2:
        st.metric("Confidence", f"{np.max(probabilities)*100:.1f}%")
    with col_res3:
        st.metric("Recommendation", "Act Now" if prediction == 0 else "Plan Ahead")
    
    # Progress bars for probabilities
    st.markdown("### 📊 Confidence Breakdown")
    prob_df = pd.DataFrame({
        "Demand Type": ["High Demand", "Low Demand", "Medium Demand"],
        "Confidence (%)": [probabilities[0]*100, probabilities[1]*100, probabilities[2]*100]
    })
    st.bar_chart(prob_df.set_index("Demand Type"))
    
    # Recommendation
    st.info(f"💡 **Recommendation:** {recommendations[prediction]}")