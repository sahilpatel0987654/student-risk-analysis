import os
os.system('pip install matplotlib seaborn scikit-learn')
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Academic Risk Predictor",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# Dataset Creation
# --------------------------------------------------

@st.cache_data
def create_dataset():
    np.random.seed(42)
    n_students = 500

    data = {
        "Attendance_Percentage": np.random.randint(55, 100, n_students),
        "Previous_Exam_Score": np.random.randint(40, 100, n_students),
        "Study_Hours_Per_Week": np.random.randint(5, 35, n_students),
        "Assignment_Completion_Rate": np.random.randint(50, 100, n_students),
        "Participation_Score": np.random.randint(1, 10, n_students)
    }

    df = pd.DataFrame(data)

    def determine_risk(row):
        if (
            row["Attendance_Percentage"] < 75
            or row["Previous_Exam_Score"] < 50
            or row["Study_Hours_Per_Week"] < 10
        ):
            return 1
        return 0

    df["Academic_Risk"] = df.apply(determine_risk, axis=1)

    return df


# --------------------------------------------------
# Train Machine Learning Model
# --------------------------------------------------

@st.cache_resource
def train_model(df):
    features = [
        "Attendance_Percentage",
        "Previous_Exam_Score",
        "Study_Hours_Per_Week",
        "Assignment_Completion_Rate",
        "Participation_Score"
    ]

    X = df[features]
    y = df["Academic_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    return model, accuracy, report, features


# --------------------------------------------------
# Load Data and Model
# --------------------------------------------------

df = create_dataset()
model, accuracy, report, feature_names = train_model(df)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Risk Prediction",
        "Dataset Overview",
        "Model Performance"
    ]
)


# --------------------------------------------------
# Main Header
# --------------------------------------------------

st.title("🎓 Academic Risk Prediction System")
st.markdown(
    """
    This application predicts whether a student is at **academic risk**
    using attendance, examination scores, study hours, assignment completion,
    and participation.
    """
)


# --------------------------------------------------
# Risk Prediction Page
# --------------------------------------------------

if page == "Risk Prediction":
    st.header("Predict Student Academic Risk")

    st.info(
        "A student is classified as being at risk if attendance is below 75%, "
        "previous exam score is below 50%, or weekly study hours are below 10."
    )

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider(
            "Attendance Percentage",
            min_value=55,
            max_value=99,
            value=75,
            step=1
        )

        previous_score = st.slider(
            "Previous Exam Score",
            min_value=40,
            max_value=99,
            value=65,
            step=1
        )

        study_hours = st.slider(
            "Study Hours Per Week",
            min_value=5,
            max_value=34,
            value=15,
            step=1
        )

    with col2:
        assignment_rate = st.slider(
            "Assignment Completion Rate",
            min_value=50,
            max_value=99,
            value=80,
            step=1
        )

        participation = st.slider(
            "Participation Score",
            min_value=1,
            max_value=9,
            value=6,
            step=1
        )

    input_data = pd.DataFrame(
        {
            "Attendance_Percentage": [attendance],
            "Previous_Exam_Score": [previous_score],
            "Study_Hours_Per_Week": [study_hours],
            "Assignment_Completion_Rate": [assignment_rate],
            "Participation_Score": [participation]
        }
    )

    st.subheader("Student Information")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Predict Academic Risk", type="primary"):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        risk_probability = probabilities[1] * 100
        safe_probability = probabilities[0] * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Academic Risk Detected")
            st.write(
                f"The estimated academic risk probability is "
                f"**{risk_probability:.2f}%**."
            )

            st.warning(
                "Recommended actions: improve attendance, increase study hours, "
                "and seek academic support."
            )
        else:
            st.success("✅ Student Is Academically Safe")
            st.write(
                f"The estimated safe probability is "
                f"**{safe_probability:.2f}%**."
            )

        probability_df = pd.DataFrame(
            {
                "Category": ["Safe", "Academic Risk"],
                "Probability": [
                    safe_probability,
                    risk_probability
                ]
            }
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=probability_df,
            x="Category",
            y="Probability",
            palette=["#2ca02c", "#d62728"],
            ax=ax
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Prediction Probability")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%")

        st.pyplot(fig)


# --------------------------------------------------
# Dataset Overview Page
# --------------------------------------------------

elif page == "Dataset Overview":
    st.header("Dataset Overview")

    total_students = len(df)
    students_at_risk = int(df["Academic_Risk"].sum())
    safe_students = total_students - students_at_risk

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Students", total_students)

    with col2:
        st.metric("Students at Risk", students_at_risk)

    with col3:
        st.metric("Safe Students", safe_students)

    st.subheader("Dataset Sample")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Academic Risk Distribution")

    risk_counts = df["Academic_Risk"].value_counts().rename(
        index={
            0: "Safe",
            1: "Academic Risk"
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=risk_counts.index,
        y=risk_counts.values,
        palette=["#2ca02c", "#d62728"],
        ax=ax
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Students")
    ax.set_title("Academic Risk Distribution")

    for container in ax.containers:
        ax.bar_label(container)

    st.pyplot(fig)

    st.subheader("Feature Distributions")

    selected_feature = st.selectbox(
        "Select a feature",
        feature_names
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=selected_feature,
        hue="Academic_Risk",
        bins=20,
        kde=True,
        palette={
            0: "#2ca02c",
            1: "#d62728"
        },
        ax=ax
    )

    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)


# --------------------------------------------------
# Model Performance Page
# --------------------------------------------------

elif page == "Model Performance":
    st.header("Model Performance")

    st.metric(
        "Random Forest Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    performance_df = pd.DataFrame(report).transpose()

    st.subheader("Classification Report")
    st.dataframe(
        performance_df.round(3),
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        by="Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        palette="Blues_r",
        ax=ax
    )

    ax.set_title("Feature Importance from Random Forest")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("Academic Risk Prediction System")
