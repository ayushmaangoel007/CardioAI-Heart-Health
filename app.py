from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="CardioAI Prototype", page_icon="❤️", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
HEALTH_FILE = DATA_DIR / "health_data.csv"
REC_FILE = DATA_DIR / "recommendations.csv"

@st.cache_data
def load_data():
    health = pd.read_csv(HEALTH_FILE)
    recs = pd.read_csv(REC_FILE)
    return health, recs

@st.cache_resource
def train_model(health):
    features = ["age", "cholesterol", "systolic_bp", "glucose", "bmi", "heart_rate"]
    X = health[features]
    y = health["risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return model, acc, features, X_test, y_test, pred

def risk_order(risk):
    order = {"Low": 0, "Moderate": 1, "High": 2}
    return order.get(risk, 0)

health, recs = load_data()
model, accuracy, features, X_test, y_test, pred = train_model(health)

st.title("❤️ CardioAI: Heart-Health Risk & Prevention Prototype")
st.caption(
    "School-project prototype • Local CSV data • Educational use only"
)

st.warning(
    "This prototype is not a medical diagnosis or treatment tool. "
    "The numerical dataset is synthetic/demo data. Real cardiovascular assessment "
    "should use clinically validated tools and a qualified healthcare professional."
)

with st.sidebar:
    st.header("User Input")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    cholesterol = st.number_input("Total cholesterol (mg/dL)", 100.0, 350.0, 180.0, step=1.0)
    systolic_bp = st.number_input("Systolic blood pressure (mmHg)", 80.0, 220.0, 120.0, step=1.0)
    glucose = st.number_input("Blood glucose (mg/dL)", 60.0, 400.0, 95.0, step=1.0)
    bmi = st.number_input("BMI (kg/m²)", 15.0, 60.0, 22.0, step=0.1)
    heart_rate = st.number_input("Resting heart rate (bpm)", 40.0, 160.0, 70.0, step=1.0)

    submitted = st.button("Analyze Heart-Health Profile", type="primary")

if submitted:
    user_df = pd.DataFrame([{
        "age": age,
        "cholesterol": cholesterol,
        "systolic_bp": systolic_bp,
        "glucose": glucose,
        "bmi": bmi,
        "heart_rate": heart_rate
    }])

    prediction = model.predict(user_df)[0]
    probabilities = model.predict_proba(user_df)[0]
    classes = model.classes_

    st.subheader("1. Model Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted risk class", prediction)
    c2.metric("Training/test demo accuracy", f"{accuracy * 100:.1f}%")
    c3.metric("Dataset size", f"{len(health)} records")

    st.info(
        f"The model classified this example as **{prediction}** risk using the six "
        "input variables. This is a machine-learning demonstration, not a clinical risk score."
    )

    st.subheader("2. Model Confidence (demo)")
    prob_df = pd.DataFrame({
        "Risk class": classes,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)
    st.dataframe(
        prob_df.style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("3. Input Profile")
    chart_df = pd.DataFrame({
        "Indicator": [
            "Cholesterol", "Systolic BP", "Glucose", "BMI", "Heart Rate"
        ],
        "User value": [cholesterol, systolic_bp, glucose, bmi, heart_rate],
        "Reference midpoint": [200, 120, 100, 22, 70]
    })

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(chart_df))
    width = 0.35
    ax.bar(x - width/2, chart_df["User value"], width, label="User")
    ax.bar(x + width/2, chart_df["Reference midpoint"], width, label="Reference")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["Indicator"], rotation=20)
    ax.set_ylabel("Value")
    ax.set_title("User values vs simple reference midpoints")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig, clear_figure=True)

    st.subheader("4. Personalized Prevention Plan")
    selected = recs[recs["risk_level"] == prediction].copy()

    # Sort by educational priority.
    selected = selected.sort_values(
        by=["priority", "recommendation"],
        ascending=[True, True]
    ).head(8)

    for _, row in selected.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['recommendation']}")
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Why it appears:** {row['reason']}")
            st.write(f"**How to use the idea:** {row['action']}")
            if row["plant_or_environment"] == "yes":
                st.caption(
                    "Plant/environment note: this is an environmental or educational "
                    "suggestion, not a claim that the plant treats heart disease."
                )

    st.subheader("5. Simple Risk-Factor Flags")
    flags = []
    if systolic_bp >= 140:
        flags.append("Systolic BP is in a high range.")
    elif systolic_bp >= 130:
        flags.append("Systolic BP is above the preferred range used in this demo.")

    if cholesterol >= 240:
        flags.append("Cholesterol is high in this demo.")
    elif cholesterol >= 200:
        flags.append("Cholesterol is above the reference value used in this demo.")

    if glucose >= 126:
        flags.append("Glucose is in a high range in this demo.")
    elif glucose >= 100:
        flags.append("Glucose is above the reference value used in this demo.")

    if bmi >= 30:
        flags.append("BMI is in the obesity category by conventional adult BMI classification.")
    elif bmi >= 25:
        flags.append("BMI is above 25 in this demo.")

    if not flags:
        st.success("No major demo thresholds were crossed.")
    else:
        for f in flags:
            st.write("• " + f)

st.divider()
st.subheader("How the prototype works")
st.code(
    """CSV dataset
    ↓
Pandas loads the data
    ↓
Random Forest learns risk_class
    ↓
User enters six measurements
    ↓
Model predicts Low / Moderate / High
    ↓
CSV recommendation table filters matching advice
    ↓
Streamlit displays result + chart""",
    language="text"
)

with st.expander("Model details"):
    st.write(
        "The model is a Random Forest classifier. The project deliberately keeps the "
        "model simple so it can run entirely on a normal laptop without a GPU or server."
    )
    st.write(f"Demo hold-out accuracy: {accuracy * 100:.1f}%")

st.caption("CardioAI Prototype — replace the synthetic dataset with validated research data before making scientific claims.")
