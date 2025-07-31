import base64
import pandas as pd
import streamlit as st

def get_csv_download_link(df, filename="data.csv", link_text="📥 Download CSV"):
    """
    Generates a download link for a CSV file.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def download_button(df, filename, label):
    """
    Renders a styled download button inside Streamlit.
    """
    if isinstance(df, pd.DataFrame):
        st.markdown(get_csv_download_link(df, filename, label), unsafe_allow_html=True)
    else:
        st.warning("Provided object is not a DataFrame")
