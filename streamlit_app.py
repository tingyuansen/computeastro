# main_app.py
import streamlit as st
import home
import lecture1_linear_regression

def main():
    st.sidebar.title('Course Content')
    page = st.sidebar.radio("Go to", ["Home", "Chaper 1: Linear Regression"])

    if page == "Home":
        home.show_page()
    elif page == "Chaper 1: Linear Regression":
        lecture1_linear_regression.show_page()

if __name__ == "__main__":
    main()
