


import streamlit as st
import home
import tutorial1_linear_regression
import lecture1_linear_regression
import lecture2_logistic_regression
import lecture3_clustering_kmeans
import lecture3_clustering_gmm
import lecture4_pca
import lecture5_pytorch
import lecture5_multilayer_perceptron
import lecture6_autoencoder
import lecture6_mixed_density_network
import lecture7_gaussian_process_regression
import lecture8_gaussian_process_classification
import lecture9_sampling
import lecture10_markov_chain_monte_carlo


def main():
    st.sidebar.title('Course Content')

    # Initialize the current page to 'Home'
    page = 'Home'

    # Home Page
    if st.sidebar.button("Home"):
        current_page = 'Home'

    if st.sidebar.button("Chapter 1: Supervised Learning - Linear Regression"):
        page = 'Chapter 1: Supervised Learning - Linear Regression'

    if st.sidebar.button("Tutorial 1"):
        page = 'Tutorial 1'
        
    if page == "Home":
        home.show_page()
    elif page == "Chapter 1: Supervised Learning - Linear Regression":
        lecture1_linear_regression.show_page()
    elif page == "Tutorial 1":
        tutorial1_linear_regression.show_page()
    elif page == "Chaper 2: Logistic Regression":
        lecture2_logistic_regression.show_page()
    elif page == "Chaper 3: Clustering - K-Means":
        lecture3_clustering_kmeans.show_page()
    elif page == "Chaper 3: Clustering - Gaussian Mixture Models":
        lecture3_clustering_gmm.show_page()
    elif page == "Chaper 4: Dimension Reduction - Principal Component Analysis":
        lecture4_pca.show_page()
    elif page == "Chaper 5: Neural Networks - Pytorch Basics":
        lecture5_pytorch.show_page()
    elif page == "Chaper 5: Neural Networks - Multilayer Perceptron":
        lecture5_multilayer_perceptron.show_page()
    elif page == "Chaper 6: Neural Networks - Autoencoder":
        lecture6_autoencoder.show_page()
    elif page == "Chaper 6: Neural Networks - Mixed Density Network":
        lecture6_mixed_density_network.show_page()
    elif page == "Chaper 7: Gaussian Process Regression":
        lecture7_gaussian_process_regression.show_page()
    elif page == "Chaper 8: Gaussian Process Classification":
        lecture8_gaussian_process_classification.show_page()
    elif page == "Chaper 9: Sampling":
        lecture9_sampling.show_page()
    elif page == "Chaper 10: Markov Chain Monte Carlo":
        lecture10_markov_chain_monte_carlo.show_page()

if __name__ == "__main__":
    main()

#     # Chapter 1: Linear Regression
#     with st.sidebar.expander("):
#         if st.sidebar.button("Lecture 1"):
#             current_page = 'Lecture 1'
#         if st.sidebar.button("Tutorial 1"):
#             current_page = 'Tutorial 1'

#     # Chapter 1: Linear Regression
#     with st.sidebar.expander("Chapter 2: Supervised Learning - Logistic Regression"):
#         if st.sidebar.button("Lecture 2"):
#             current_page = 'Lecture 2'
            
#     # Chapter 2: Logistic Regression
#     # ... similarly for other chapters ...

#     # Render the selected page
#     if current_page == 'Home':
#         home.show_page()
#     elif current_page == 'Lecture 1':
#         lecture1_linear_regression.show_page()
#     elif current_page == 'Tutorial 1':
#         tutorial1_linear_regression.show_page()
#     elif current_page == 'Lecture 2':
#         lecture2_logistic_regression.show_page()
#     # ... similarly for other pages ...




# def main():
#     st.sidebar.title('Course Content')

#     # Home Page
#     if st.sidebar.button("Home"):
#         home.show_page()

#     # Chapter 1: Linear Regression
#     with st.sidebar.expander("Chapter 1: Supervised Learning - Linear Regression"):
#         if st.button("Lecture 1"):
#             lecture1_linear_regression.show_page()
#         if st.button("Tutorial 1"):
#             tutorial1_linear_regression.show_page()

#     # Chapter 2: Logistic Regression
#     with st.sidebar.expander("Chapter 2: - Logistic Regression"):
#         if st.button("Tutorial 2"):
#             lecture2_logistic_regression.show_page()

#     # Continue with more chapters following the same pattern

#     # Chapter 10: Markov Chain Monte Carlo
#     with st.sidebar.expander("Chapter 10: Markov Chain Monte Carlo"):
#         if st.button("Lecture 10"):
#             lecture10_markov_chain_monte_carlo.show_page()

# if __name__ == "__main__":
#     main()

# # main_app.py
# import streamlit as st
# import home


# def main():
#     st.sidebar.title('Course Content')

#     # Home Page
#     if st.sidebar.button("Home"):
#         home.show_page()

#     # Chapter 1: Linear Regression
#     with st.sidebar.beta_expander("Chapter 1: Linear Regression"):
#         if st.button("Lecture"):
#             lecture1_linear_regression.show_page()
#         if st.button("Tutorial"):
#             # Add your tutorial page function here

#     # Chapter 2: Logistic Regression
#     with st.sidebar.beta_expander("Chapter 2: Logistic Regression"):
#         if st.button("Lecture"):
#             lecture2_logistic_regression.show_page()
#         if st.button("Tutorial"):
#             # Add your tutorial page function here

#     # Add more chapters following the same pattern

#     # Chapter 10: Markov Chain Monte Carlo
#     with st.sidebar.beta_expander("Chapter 10: Markov Chain Monte Carlo"):
#         if st.button("Lecture"):
#             lecture10_markov_chain_monte_carlo.show_page()
#         if st.button("Tutorial"):
#             # Add your tutorial page function here
            
#     # page = st.sidebar.radio("Go to", ["Home", "Chaper 1: Linear Regression", "Chaper 2: Logistic Regression", "Chaper 3: Clustering - K-Means", "Chaper 3: Clustering - Gaussian Mixture Models", "Chaper 4: Dimension Reduction - Principal Component Analysis", "Chaper 5: Neural Networks - Pytorch Basics", "Chaper 5: Neural Networks - Multilayer Perceptron", "Chaper 6: Neural Networks - Autoencoder", "Chaper 6: Neural Networks - Mixed Density Network", "Chaper 7: Gaussian Process Regression", "Chaper 8: Gaussian Process Classification", "Chaper 9: Sampling", "Chaper 10: Markov Chain Monte Carlo"])

#     # if page == "Home":
#     #     home.show_page()
#     # elif page == "Chaper 1: Linear Regression":
#     #     lecture1_linear_regression.show_page()
#     # elif page == "Chaper 2: Logistic Regression":
#     #     lecture2_logistic_regression.show_page()
#     # elif page == "Chaper 3: Clustering - K-Means":
#     #     lecture3_clustering_kmeans.show_page()
#     # elif page == "Chaper 3: Clustering - Gaussian Mixture Models":
#     #     lecture3_clustering_gmm.show_page()
#     # elif page == "Chaper 4: Dimension Reduction - Principal Component Analysis":
#     #     lecture4_pca.show_page()
#     # elif page == "Chaper 5: Neural Networks - Pytorch Basics":
#     #     lecture5_pytorch.show_page()
#     # elif page == "Chaper 5: Neural Networks - Multilayer Perceptron":
#     #     lecture5_multilayer_perceptron.show_page()
#     # elif page == "Chaper 6: Neural Networks - Autoencoder":
#     #     lecture6_autoencoder.show_page()
#     # elif page == "Chaper 6: Neural Networks - Mixed Density Network":
#     #     lecture6_mixed_density_network.show_page()
#     # elif page == "Chaper 7: Gaussian Process Regression":
#     #     lecture7_gaussian_process_regression.show_page()
#     # elif page == "Chaper 8: Gaussian Process Classification":
#     #     lecture8_gaussian_process_classification.show_page()
#     # elif page == "Chaper 9: Sampling":
#     #     lecture9_sampling.show_page()
#     # elif page == "Chaper 10: Markov Chain Monte Carlo":
#     #     lecture10_markov_chain_monte_carlo.show_page()
        
# if __name__ == "__main__":
#     main()
