import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_basic_info(df):
    """
    Display basic dataset info and statistics.
    """
    st.subheader("🔍 Data Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Descriptive Statistics:**")
    st.write(df.describe())


def plot_correlation(df):
    """
    Display a correlation heatmap.
    """
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)


def plot_distribution(df):
    """
    Plot distributions of all numeric columns.
    """
    st.subheader("📈 Distribution Plots")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        
def plot_pairwise(df):
    """
    Pairplot of all features (use only if dataset is small).
    """
    st.subheader("🔗 Pairplot (Relationships)")
    fig = sns.pairplot(df)
    st.pyplot(fig)
