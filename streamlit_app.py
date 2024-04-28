import streamlit as st
import home

import tutorial1_linear_regression
import tutorial2_logistic_regression
import tutorial3_clustering_kmeans
import tutorial3_clustering_gmm
import tutorial4_pca
import tutorial5_pytorch
import tutorial5_multilayer_perceptron
import tutorial6_autoencoder
import tutorial6_mixed_density_network
import tutorial7_gaussian_process_regression
import tutorial8_gaussian_process_classification
import tutorial9_sampling
import tutorial10_markov_chain_monte_carlo


def main():
    st.sidebar.title('Course Content')

    # Initialize the current page to 'Home'
    page = 'Home'

    # Home Page
    if st.sidebar.button("Home"):
        current_page = 'Home'

    if st.sidebar.button("Tutorial 1: Linear Regression"):
        page = 'Tutorial 1: Linear Regression'

    if st.sidebar.button("Tutorial 2: Logistic Regression"):
        page = 'Tutorial 2: Logistic Regression'

    if st.sidebar.button("Tutorial 3 - K-Means"):
        page = 'Tutorial 3 - K-Means'

    if st.sidebar.button("Tutorial 3 - Gaussian Mixture Models"):
        page = 'Tutorial 3 - Gaussian Mixture Models'

    if st.sidebar.button("Tutorial 4 - Principal Component Analysis"):
        page = 'Tutorial 4 - Principal Component Analysis'

    if st.sidebar.button("Tutorial 5: Pytorch Basics"):
        page = 'Tutorial 5: Pytorch Basics'

    if st.sidebar.button("Tutorial 5: Multilayer Perceptron"):
        page = 'Tutorial 5: Multilayer Perceptron'

    if st.sidebar.button("Tutorial 6: Autoencoder"):
        page = 'Tutorial 6: Autoencoder'

    if st.sidebar.button("Tutorial 6: Mixed Density Network"):
        page = 'Tutorial 6: Mixed Density Network'

    if st.sidebar.button("Tutorial 7: Gaussian Process Regression"):
        page = 'Tutorial 7: Gaussian Process Regression'

    if st.sidebar.button("Tutorial 8: Gaussian Process Classification"):
        page = 'Tutorial 8: Gaussian Process Classification'

    if st.sidebar.button("Tutorial 9: Sampling - Basic Techniques"):
        page = 'Tutorial 9: Sampling - Basic Techniques'

    if st.sidebar.button("Tutorial 10: Sampling - Markov Chain Monte Carlo"):
        page = 'Tutorial 10: Sampling - Markov Chain Monte Carlo'

    
    
    if page == "Home":
        home.show_page()
        
    elif page == "Tutorial 1: Linear Regression":
        tutorial1_linear_regression.show_page()

    elif page == "Tutorial 2: Logistic Regression":
        tutorial2_logistic_regression.show_page()
        
    elif page == "Tutorial 3 - K-Means":
        tutorial3_clustering_kmeans.show_page()
        
    elif page == "Tutorial 3 - Gaussian Mixture Models":
        tutorial3_clustering_gmm.show_page()

    elif page == "Tutorial 4 - Principal Component Analysis":
        tutorial4_pca.show_page()
        
    elif page == "Tutorial 5: Pytorch Basics":
        tutorial5_pytorch.show_page()
        
    elif page == "Tutorial 5: Multilayer Perceptron":
        tutorial5_multilayer_perceptron.show_page()

    elif page == "Tutorial 6: Autoencoder":
        tutorial6_autoencoder.show_page()
        
    elif page == "Tutorial 6: Mixed Density Network":
        tutorial6_mixed_density_network.show_page()

    elif page == "Tutorial 7: Gaussian Process Regression":
        tutorial7_gaussian_process_regression.show_page()

    elif page == "Tutorial 8: Gaussian Process Classification":
        tutorial8_gaussian_process_classification.show_page()

    elif page == "Tutorial 9: Sampling - Basic Techniques":
        tutorial9_sampling.show_page()

    elif page == "Tutorial 10: Sampling - Markov Chain Monte Carlo":
        tutorial10_markov_chain_monte_carlo.show_page()


if __name__ == "__main__":
    main()
