import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Upload CSV", "Visualization"])

# Home Page
if page == "Home":
    st.title("Enhanced Streamlit App with Docker 🚀")
    name = st.text_input("Enter your name:", "Guest")
    st.write(f"Hello, {name}! Welcome to the upgraded Streamlit app.")

    # Counter with Session State
    if "count" not in st.session_state:
        st.session_state.count = 0

    if st.button("Increase Count"):
        st.session_state.count += 1

    st.write(f"Count: {st.session_state.count}")

# CSV Upload Page
elif page == "Upload CSV":
    st.title("Upload and Display CSV 📂")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded file:")
        st.dataframe(df)

# Data Visualization Page
elif page == "Visualization":
    st.title("Simple Data Visualization 📊")

    # Generate Sample Data
    data = {"Category": ["A", "B", "C", "D"], "Values": [10, 30, 50, 20]}
    df = pd.DataFrame(data)

    # Plot Bar Chart
    fig, ax = plt.subplots()
    ax.bar(df["Category"], df["Values"], color=["blue", "green", "red", "purple"])
    st.pyplot(fig)
