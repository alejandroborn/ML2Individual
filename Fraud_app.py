import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load pre-trained model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("logistic_model.pkl")

# Streamlit app title
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon=":credit_card:", layout="wide")
st.title("🔍 Credit Card Fraud Detection")

# Sidebar for navigation and info
st.sidebar.title("Navigation")
st.sidebar.info("Upload a CSV file with transaction data to detect potential fraud.")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load and display dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Dataset")
    st.dataframe(data.head(10))

    # Drop the 'Class' column if present (not part of the features)
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    # Check for required features
    required_columns = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"❌ The uploaded dataset is missing the following required columns: {missing_columns}")
    else:
        # Scale the 'Amount' column
        data["Amount"] = scaler.transform(data[["Amount"]])

        # Predict fraud
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        # Add results to the dataset
        data["Fraud Prediction"] = predictions
        data["Fraud Probability"] = probabilities

        # Highlight fraud predictions
        fraud_data = data[data["Fraud Prediction"] == 1]
        non_fraud_data = data[data["Fraud Prediction"] == 0]

        st.subheader("🚩 Fraudulent Transactions")
        if len(fraud_data) > 0:
            st.write("🌟 Highlighted Transactions Detected as Fraudulent:")
            # Limit displayed rows to avoid Streamlit rendering limits
            fraud_display_limit = 100
            st.dataframe(
                fraud_data.head(fraud_display_limit).style.highlight_max(
                    subset=["Fraud Probability"], color="red"
                )
            )
            if len(fraud_data) > fraud_display_limit:
                st.warning(f"Displaying the first {fraud_display_limit} fraudulent transactions.")
        else:
            st.success("No fraudulent transactions detected!")

        st.subheader("📊 Full Dataset with Predictions")
        # Limit full dataset display
        data_display_limit = 100  # Limit to 100 rows, because streamlit couldnt load the full dataset 
        st.dataframe(
            data.head(data_display_limit).style.highlight_max(
                subset=["Fraud Probability"], color="yellow"
            )
        )
        if len(data) > data_display_limit:
            st.warning(f"Displaying the first {data_display_limit} rows of the dataset.")

        # Visualize fraud probability, bar chart for each row.
        st.subheader("📈 Fraud Probability Distribution")
        st.bar_chart(data["Fraud Probability"].head(100))  # Display chart for first 100 rows
else:
    st.info("👆 Please upload a CSV file to proceed.")
