import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import load_data, preprocess_data
from utils.eda import show_basic_info, plot_correlation, plot_distribution
from utils.model import load_model, predict
from utils.download import download_button

def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: linear-gradient(to bottom right, rgba(255,255,255,0.85), rgba(240,240,255,0.9)),
                        url('https://images.unsplash.com/photo-1526948128573-703ee1aeb6fa?fit=crop&w=1400&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
            margin: 2rem auto;
        }

        .stButton>button {
            background-color: #2962FF;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0039cb;
        }

        .stFileUploader, .stTextInput, .stNumberInput, .stSelectbox {
            border-radius: 10px !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #1e3c72, #2a5298);
            color: white;
            padding: 2rem 1rem;
            height: 100%;
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
            font-weight: 500;
        }

        section[data-testid="stSidebar"] .stFileUploader span:first-child {
            color: black !important;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] .stFileUploader span:nth-child(2) {
            color: black !important;
            font-weight: 500;
        }

        section[data-testid="stSidebar"] .stFileUploader button {
            color: black !important;
            border-radius: 8px;
            font-weight: 600;
        }

        section[data-testid="stSidebar"] .stFileUploader {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
        }

        .stRadio > div {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
        }

        input[type="radio"]:checked + div > label {
            color: #ffe600 !important;
        }
        
    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# Page Config
st.set_page_config(page_title="AdVision", layout="wide")
st.title("📊 AdVision: Advertising Sales Predictor")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Clean", "EDA", "Predict", "Visualize", "Download"])

# Upload File
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load & preprocess
    df_raw = load_data(uploaded_file)
    df = preprocess_data(df_raw)
    st.session_state['df'] = df  # Cache for use across pages

    if page == "Upload & Clean":
        st.subheader("📥 Uploaded Data")
        st.write(df.head())

    elif page == "EDA":
        st.subheader("🔎 Exploratory Data Analysis")
        show_basic_info(df)
        plot_correlation(df)
        plot_distribution(df)

    elif page == "Predict":
        st.subheader("📈 Predicting Sales with MLR")

        # Load model
        model = load_model("models/mlr_model.joblib")

        # Features used for prediction
        feature_cols = ['tv', 'radio', 'social_media', 'influencer']
        if not all(col in df.columns for col in feature_cols):
            st.error(f"Required columns not found in data: {feature_cols}")
        else:
            X = df[feature_cols]
            predictions = predict(model, X)
            result_df = df.copy()
            result_df['Predicted_Sales'] = predictions

            st.write("📊 Prediction Results")
            st.dataframe(result_df.head())
            st.session_state['predictions'] = result_df

            st.markdown("---")
            st.subheader("🎯 Predict Sales from Your Own Inputs")

            influencer_mapping = {
                "Micro": 0,
                "Macro": 1,
                "Nano": 2,
                "Celebrity": 3
            }

            with st.form("prediction_form"):
                tv_input = st.number_input("TV Advertisement Budget", min_value=0.0, step=100.0)
                radio_input = st.number_input("Radio Advertisement Budget", min_value=0.0, step=100.0)
                social_input = st.number_input("Social Media Budget", min_value=0.0, step=100.0)
                influencer_input = st.selectbox("Influencer Type", options=influencer_mapping.keys())

                submit = st.form_submit_button("Predict")

            if submit:
                influencer_encoded = influencer_mapping[influencer_input]

                input_data = pd.DataFrame(
                    [[tv_input, radio_input, social_input, influencer_encoded]],
                    columns=['tv', 'radio', 'social_media', 'influencer']
                )

                prediction = model.predict(input_data)[0]
                st.success(f"💰 Predicted Sales: **{prediction:,.2f}**")

    elif page == "Visualize":
        st.subheader("🎯Feature-wise Sales Visualization")

        plot_type = st.selectbox("Choose Plot Type", ["📦Boxplot", "🎻Violinplot", "📊Barplot", "📉Histogram"])
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if plot_type in ["📦Boxplot", "🎻Violinplot", "📉Histogram"]:
            selected_feature = st.selectbox("Select Numeric Feature", numeric_cols)

            fig, ax = plt.subplots()

            if plot_type == "📦Boxplot":
                sns.boxplot(y=df[selected_feature], ax=ax)
                ax.set_title(f'Boxplot of {selected_feature}')
                st.pyplot(fig)
                st.info(f"ℹ️ Insight: {selected_feature.capitalize()} shows potential outliers and spread. Check for right/left skew.")

            elif plot_type == "🎻Violinplot":
                sns.violinplot(y=df[selected_feature], ax=ax)
                ax.set_title(f'Violin Plot of {selected_feature}')
                st.pyplot(fig)
                st.info(f"ℹ️ Insight: {selected_feature.capitalize()} shows distribution with thickness indicating density.")

            elif plot_type == "📉Histogram":
                sns.histplot(df[selected_feature], kde=True, ax=ax, color="orange")
                ax.set_title(f'Histogram of {selected_feature}')
                st.pyplot(fig)
                st.info(f"ℹ️ Insight: {selected_feature.capitalize()} has {'a long right tail' if df[selected_feature].skew() > 1 else 'a fairly symmetric distribution'}.")

        elif plot_type == "📊Barplot":
            category_col = st.selectbox("Select Categorical Column to Group By", df.columns)
            if category_col in df.columns:
                fig, ax = plt.subplots()
                sns.barplot(x=category_col, y="sales", data=df, ax=ax)
                ax.set_title(f"Bar Plot: Sales vs {category_col}")
                st.pyplot(fig)
                st.info(f"ℹ️ Insight: Compares average sales across {category_col} values.")
        

    elif page == "Download":
        st.subheader("📥 Download Outputs")

        if 'predictions' in st.session_state:
            download_button(st.session_state['predictions'], "predictions.csv", "📥 Download Predictions")
        else:
            st.warning("No predictions available yet.")

        if 'df' in st.session_state:
            download_button(st.session_state['df'], "cleaned_data.csv", "📥 Download Cleaned Data")
else:
    st.info("👈 Upload a CSV file to begin.")
