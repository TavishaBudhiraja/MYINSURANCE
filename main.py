from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "insurance.csv"


# ---------------------------------------------------------
# Load model and dataset
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def convert_df(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


try:
    model = load_model()
    df = load_data()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error.filename}")
    st.info(
        "Keep main.py, model.pkl and insurance.csv inside the same folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Application loading error: {error}")
    st.stop()


# ---------------------------------------------------------
# Page heading and navigation
# ---------------------------------------------------------
st.header(
    "Insurance Prediction using Machine Learning",
    divider="rainbow"
)

st.sidebar.title("Navigation")

options = st.sidebar.radio(
    "Select an option:",
    [
        "Exploratory Data Analysis",
        "Insurance Prediction"
    ]
)


# ---------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------
def stats():
    st.header(":blue[Exploratory Data Analysis]")

    st.subheader("Insurance Dataset")
    st.dataframe(df, use_container_width=True)

    csv = convert_df(df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="insurance_data.csv",
        mime="text/csv"
    )

    st.subheader("Sex and Smoking Status")

    fig1 = px.histogram(
        df,
        x="sex",
        color="smoker",
        barmode="group"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Children and Smoking Status")

    fig2 = px.histogram(
        df,
        x="children",
        color="smoker",
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Regional Distribution")

    fig3 = px.histogram(
        df,
        x="region",
        color="smoker",
        barmode="group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Smoking Status and Insurance Charges")

    fig4 = px.box(
        df,
        x="smoker",
        y="charges",
        color="sex"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Age Distribution")

    age_data = [df["age"].dropna().tolist()]

    fig5 = ff.create_distplot(
        age_data,
        ["Age"]
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("BMI Distribution")

    bmi_data = [df["bmi"].dropna().tolist()]

    fig6 = ff.create_distplot(
        bmi_data,
        ["BMI"]
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Age and Insurance Charges")

    fig7 = px.scatter(
        df,
        x="age",
        y="charges",
        opacity=0.65,
        color="smoker",
        trendline="ols",
        trendline_color_override="darkblue"
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("BMI and Insurance Charges")

    fig8 = px.scatter(
        df,
        x="bmi",
        y="charges",
        color="smoker"
    )
    st.plotly_chart(fig8, use_container_width=True)


# ---------------------------------------------------------
# Insurance Prediction
# ---------------------------------------------------------
def enterdata():
    st.header(":blue[Insurance Charges Prediction]")

    def predict(age, sex, bmi, children, smoke, region):
        sex_mapping = {
            "Female": 0,
            "Male": 1
        }

        smoker_mapping = {
            "No": 0,
            "Yes": 1
        }

        region_mapping = {
            "northeast": 0,
            "southwest": 1,
            "southeast": 2,
            "northwest": 3
        }

        input_data = pd.DataFrame(
            [
                {
                    "age": float(age),
                    "sex": sex_mapping[sex],
                    "bmi": float(bmi),
                    "children": int(children),
                    "smoker": smoker_mapping[smoke],
                    "region": region_mapping[region]
                }
            ]
        )

        prediction = model.predict(input_data)

        return float(np.asarray(prediction).flatten()[0])

    with st.form("insurance_prediction_form"):
        age = st.number_input(
            "Enter your age",
            min_value=1,
            max_value=120,
            value=25,
            step=1
        )

        sex = st.selectbox(
            "Select your gender",
            ["Male", "Female"]
        )

        bmi = st.number_input(
            "Enter your BMI",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        children = st.selectbox(
            "Select number of children",
            [0, 1, 2, 3, 4, 5]
        )

        smoke = st.selectbox(
            "Do you smoke?",
            ["No", "Yes"]
        )

        region = st.selectbox(
            "Select your region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            output = predict(
                age,
                sex,
                bmi,
                children,
                smoke,
                region
            )

            st.success(
                f"Predicted insurance charges: {output:,.2f}"
            )

        except ValueError as error:
            st.error(f"Invalid input or model feature error: {error}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ---------------------------------------------------------
# Display selected page
# ---------------------------------------------------------
if options == "Exploratory Data Analysis":
    stats()

elif options == "Insurance Prediction":
    enterdata()