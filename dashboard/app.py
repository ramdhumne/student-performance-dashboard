import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "data", "students.csv"))

# Calculate average
df["average"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3

st.title("🎓 Student Performance Dashboard")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

lunch_filter = st.sidebar.multiselect(
    "Select Lunch Type",
    options=df["lunch"].unique(),
    default=df["lunch"].unique()
)

course_filter = st.sidebar.multiselect(
    "Test Preparation Course",
    options=df["test preparation course"].unique(),
    default=df["test preparation course"].unique()
)

# Apply filters
filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["lunch"].isin(lunch_filter)) &
    (df["test preparation course"].isin(course_filter))
]

# ---------------- OVERVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

st.metric("Average Score", round(filtered_df["average"].mean(),2))

# ---------------- VISUALS ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance by Gender")
    fig = px.box(filtered_df, x="gender", y="average", color="gender")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Lunch Impact")
    fig2 = px.bar(
        filtered_df.groupby("lunch")["average"].mean().reset_index(),
        x="lunch", y="average", color="lunch"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- CORRELATION ----------------
st.subheader("Subject Correlation")
corr = filtered_df[["math score","reading score","writing score"]].corr()
fig3 = px.imshow(corr, text_auto=True)
st.plotly_chart(fig3, use_container_width=True)

# ---------------- PREDICTION PANEL ----------------
st.header("🎯 Predict Student Score")

model = joblib.load(os.path.join(BASE_DIR, "models", "student_model.pkl"))

with st.form("prediction_form"):
    gender = st.selectbox("Gender", df["gender"].unique())
    race = st.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
    parent = st.selectbox("Parental Education", df["parental level of education"].unique())
    lunch = st.selectbox("Lunch", df["lunch"].unique())
    course = st.selectbox("Test Preparation", df["test preparation course"].unique())

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent,
        "lunch": lunch,
        "test preparation course": course
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Average Score: {round(prediction,2)}")