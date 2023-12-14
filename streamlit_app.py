import streamlit as st
import home

import lecture1_linear_regression
import tutorial1_linear_regression

import lecture2_logistic_regression
import tutorial2_logistic_regression

import lecture3_clustering
import tutorial3_clustering_kmeans
import tutorial3_clustering_gmm

import lecture4_pca
import tutorial4_pca

import lecture5_neural_network_supervised
import tutorial5_pytorch
import tutorial5_multilayer_perceptron

import lecture6_neural_network_unsupervised
import tutorial6_autoencoder
import tutorial6_mixed_density_network

import lecture7_gaussian_process_regression
import tutorial7_gaussian_process_regression

import lecture8_gaussian_process_classification
import tutorial8_gaussian_process_classification

import lecture9_sampling
import tutorial9_sampling

import lecture10_markov_chain_monte_carlo
import tutorial10_markov_chain_monte_carlo


def main():
    st.sidebar.title('Course Content')

    # Initialize the current page to 'Home'
    page = 'Home'

    # Home Page
    if st.sidebar.button("Home"):
        current_page = 'Home'

    if st.sidebar.button("Chapter 1: Supervised Learning - Regression"):
        page = 'Chapter 1: Supervised Learning - Regression'

    if st.sidebar.button("Tutorial 1: Linear Regression"):
        page = 'Tutorial 1: Linear Regression'

    
    if st.sidebar.button("Chapter 2: Supervised Learning - Classification"):
        page = 'Chapter 2: Supervised Learning - Classification'

    if st.sidebar.button("Tutorial 2: Logistic Regression"):
        page = 'Tutorial 2: Logistic Regression'


    if st.sidebar.button("Chapter 3: Unsupervised Learning - Clustering"):
        page = 'Chapter 3: Unsupervised Learning - Clustering'

    if st.sidebar.button("Tutorial 3 - K-Means"):
        page = 'Tutorial 3 - K-Means'

    if st.sidebar.button("Tutorial 3 - Gaussian Mixture Models"):
        page = 'Tutorial 3 - Gaussian Mixture Models'

    
    if st.sidebar.button("Chapter 4: Unsupervised Learning - Dimension Reduction"):
        page = 'Chapter 4: Unsupervised Learning - Dimension Reduction'

    if st.sidebar.button("Tutorial 4 - Principal Component Analysis"):
        page = 'Tutorial 4 - Principal Component Analysis'
        

    if st.sidebar.button("Chapter 5: Neural Networks - Supervised Learning"):
        page = 'Chapter 5: Neural Networks - Supervised Learning'

    if st.sidebar.button("Tutorial 5: Pytorch Basics"):
        page = 'Tutorial 5: Pytorch Basics'

    if st.sidebar.button("Tutorial 5: Multilayer Perceptron"):
        page = 'Tutorial 5: Multilayer Perceptron'

    
    if st.sidebar.button("Chapter 6: Neural Networks - Unsupervised Learning"):
        page = 'Chapter 6: Neural Networks - Unsupervised Learning'

    if st.sidebar.button("Tutorial 6: Autoencoder"):
        page = 'Tutorial 6: Autoencoder'

    if st.sidebar.button("Tutorial 6: Mixed Density Network"):
        page = 'Tutorial 6: Mixed Density Network'


    if st.sidebar.button("Chapter 7: Gaussian Process Regression"):
        page = 'Chapter 7: Gaussian Process Regression'

    if st.sidebar.button("Tutorial 7: Gaussian Process Regression"):
        page = 'Tutorial 7: Gaussian Process Regression'


    if st.sidebar.button("Chapter 8: Gaussian Process Classification"):
        page = 'Chapter 8: Gaussian Process Classification'

    if st.sidebar.button("Tutorial 8: Gaussian Process Classification"):
        page = 'Tutorial 8: Gaussian Process Classification'


    if st.sidebar.button("Chapter 9: Sampling - Basic Techniques"):
        page = 'Chapter 9: Sampling - Basic Techniques'

    if st.sidebar.button("Tutorial 9: Sampling - Basic Techniques"):
        page = 'Tutorial 9: Sampling - Basic Techniques'


    if st.sidebar.button("Chapter 10: Sampling - Markov Chain Monte Carlo"):
        page = 'Chapter 10: Sampling - Markov Chain Monte Carlo'

    if st.sidebar.button("Tutorial 10: Sampling - Markov Chain Monte Carlo"):
        page = 'Tutorial 10: Sampling - Markov Chain Monte Carlo'

    
    
    if page == "Home":
        home.show_page()

    
    elif page == "Chapter 1: Supervised Learning - Regression":
        lecture1_linear_regression.show_page()
        
    elif page == "Tutorial 1: Linear Regression":
        tutorial1_linear_regression.show_page()

    
    elif page == "Chapter 2: Supervised Learning - Classification":
        lecture2_logistic_regression.show_page()

    elif page == "Tutorial 2: Logistic Regression":
        tutorial2_logistic_regression.show_page()

    
    elif page == "Chapter 3: Unsupervised Learning - Clustering":
        lecture3_clustering.show_page()
        
    elif page == "Tutorial 3 - K-Means":
        tutorial3_clustering_kmeans.show_page()
        
    elif page == "Tutorial 3 - Gaussian Mixture Models":
        tutorial3_clustering_gmm.show_page()


    elif page == "Chapter 4: Unsupervised Learning - Dimension Reduction":
        lecture4_pca.show_page()

    elif page == "Tutorial 4 - Principal Component Analysis":
        tutorial4_pca.show_page()
        

    elif page == "Chapter 5: Neural Networks - Supervised Learning":
        lecture5_neural_network_supervised.show_page()
        
    elif page == "Tutorial 5: Pytorch Basics":
        tutorial5_pytorch.show_page()
        
    elif page == "Tutorial 5: Multilayer Perceptron":
        tutorial5_multilayer_perceptron.show_page()


    elif page == "Chapter 6: Neural Networks - Unsupervised Learning":
        lecture6_neural_network_unsupervised.show_page()

    elif page == "Tutorial 6: Autoencoder":
        tutorial6_autoencoder.show_page()
        
    elif page == "Tutorial 6: Mixed Density Network":
        tutorial6_mixed_density_network.show_page()

    
    elif page == "Chapter 7: Gaussian Process Regression":
        lecture7_gaussian_process_regression.show_page()

    elif page == "Tutorial 7: Gaussian Process Regression":
        tutorial7_gaussian_process_regression.show_page()

    
    elif page == "Chapter 8: Gaussian Process Classification":
        lecture8_gaussian_process_classification.show_page()

    elif page == "Tutorial 8: Gaussian Process Classification":
        tutorial8_gaussian_process_classification.show_page()
        

    elif page == "Chapter 9: Sampling - Basic Techniques":
        lecture9_sampling.show_page()

    elif page == "Tutorial 9: Sampling - Basic Techniques":
        tutorial9_sampling.show_page()

    
    elif page == "Chapter 10: Sampling - Markov Chain Monte Carlo":
        lecture10_markov_chain_monte_carlo.show_page()

    elif page == "Tutorial 10: Sampling - Markov Chain Monte Carlo":
        tutorial10_markov_chain_monte_carlo.show_page()


if __name__ == "__main__":
    main()
