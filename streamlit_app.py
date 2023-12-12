# main_app.py
import streamlit as st
import home
import lecture1_linear_regression
import lecture2_logistic_regression
import lecture3_clustering_kmeans
import lecture3_clustering_gmm

def main():
    st.sidebar.title('Course Content')
    page = st.sidebar.radio("Go to", ["Home", "Chaper 1: Linear Regression", "Chaper 2: Logistic Regression", "Chaper 3: Clustering - K-Means", "Chaper 3: Clustering - Gaussian Mixture Models"])

    if page == "Home":
        home.show_page()
    elif page == "Chaper 1: Linear Regression":
        lecture1_linear_regression.show_page()
    elif page == "Chaper 2: Logistic Regression":
        lecture2_logistic_regression.show_page()
    elif page == "Chaper 3: Clustering - K-Means":
        lecture3_clustering_kmeans.show_page()
    elif page == "Chaper 3: Clustering - Gaussian Mixture Models":
        lecture3_clustering_gmm.show_page()

if __name__ == "__main__":
    main()
