import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Background Image
# -----------------------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1531297484001-80022131f5a1");
            background-size: cover;
            background-position: center;
        }

        /* Optional dark overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: -1;
        }

        h1, h2, h3, h4, p {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# -----------------------------
# Page Title
# -----------------------------
st.title("🎓 Students' AI Usage and Academic Performance Prediction")

st.write(
"This application predicts students' academic performance based on "
"study habits and AI usage."
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("students_ai_usage.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("best_model.pkl")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Student Details")

grades_before_ai = st.sidebar.slider("Grades Before Using AI", 0, 100, 70)

study_hours_per_day = st.sidebar.slider("Study Hours Per Day", 0, 12, 4)

daily_screen_time_hours = st.sidebar.slider("Daily Screen Time (Hours)", 0, 12, 5)

# Education Level
education_map = {
    "School": 0,
    "College": 1,
    "University": 2
}
education_selected = st.sidebar.selectbox("Education Level", list(education_map.keys()))
education_level_encoded = education_map[education_selected]

# AI Tool
ai_tools_map = {
    "Gemini": 0,
    "Copilot": 1,
    "ChatGPT": 2,
    "None": 3
}
ai_tool_selected = st.sidebar.selectbox("AI Tool Used", list(ai_tools_map.keys()))
ai_tools_used_encoded = ai_tools_map[ai_tool_selected]

# Purpose
purpose_map = {
    "Homework": 0,
    "Learning": 1,
    "Projects": 2
}
purpose_selected = st.sidebar.selectbox("Purpose of AI", list(purpose_map.keys()))
purpose_of_ai_encoded = purpose_map[purpose_selected]

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict"):

    input_data = np.array([[
        grades_before_ai,
        study_hours_per_day,
        daily_screen_time_hours,
        education_level_encoded,
        ai_tools_used_encoded,
        purpose_of_ai_encoded
    ]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("🎯 High Academic Performance")
    else:
        st.error("⚠️ Low Academic Performance")