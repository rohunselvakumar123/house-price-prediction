import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------------------
# Background Styling
# -----------------------------------------
st.markdown(
    """
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1564013799919-ab600027ffc6");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.block-container {
    background-color: rgba(0,0,0,0.65);
    padding: 25px;
    border-radius: 12px;
}

.main-title {
    font-size: 45px;
    font-weight: bold;
    color: #ffffff;
    text-align: center;
}

.result-box {
    background: linear-gradient(90deg,#1f4037,#99f2c8);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------
# Title
# -----------------------------------------
st.markdown(
    '<p class="main-title">🏠 House Price Prediction</p>',
    unsafe_allow_html=True,
)

# -----------------------------------------
# Load Model
# -----------------------------------------
with open("house_model.pkl", "rb") as f:
    model, columns = pickle.load(f)

# -----------------------------------------
# Input Section
# -----------------------------------------
st.subheader("📋 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    OverallCond = st.slider("Overall Condition", 1, 10, 5)
    LotArea = st.number_input("Lot Area", value=8000)

with col2:
    GarageCars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])
    TotalBsmtSF = st.number_input("Total Basement SF", value=800)

# -----------------------------------------
# Create DataFrame for Prediction
# -----------------------------------------
input_data = pd.DataFrame(columns=columns)
input_data.loc[0] = 0

if "OverallCond" in input_data.columns:
    input_data["OverallCond"] = OverallCond

if "LotArea" in input_data.columns:
    input_data["LotArea"] = LotArea

if "GarageCars" in input_data.columns:
    input_data["GarageCars"] = GarageCars

if "TotalBsmtSF" in input_data.columns:
    input_data["TotalBsmtSF"] = TotalBsmtSF

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("🔍 Predict Price"):

    prediction = model.predict(input_data)

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f'<div class="result-box">💰 Estimated Price: ${prediction[0]:,.0f}</div>',
            unsafe_allow_html=True,
        )

        st.progress(90)

    with col2:
        st.image(
            "https://images.unsplash.com/photo-1564013799919-ab600027ffc6",
            caption="Sample House",
            use_container_width=True,
        )

    # -----------------------------------------
    # Chart Visualization
    # -----------------------------------------
    fig, ax = plt.subplots()

    ax.bar(["Predicted Price"], [prediction[0]])
    ax.set_ylabel("Price ($)")
    ax.set_title("House Price Prediction")

    st.pyplot(fig)
