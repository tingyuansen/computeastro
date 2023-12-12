# main_app.py
import streamlit as st
import home
import lecture1_linear_regression
import lecture2_logistic_regression

def main():
    st.sidebar.title('Course Content')
    page = st.sidebar.radio("Go to", ["Home", "Chaper 1: Linear Regression", "Chaper 1: Logistic Regression"])

    if page == "Home":
        home.show_page()
    elif page == "Chaper 1: Linear Regression":
        lecture1_linear_regression.show_page()
    elif page == "Chaper 2: Logistic Regression":
        lecture2_logistic_regression.show_page()

if __name__ == "__main__":
    main()
