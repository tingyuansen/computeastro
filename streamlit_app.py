# home.py
import streamlit as st

def navigation_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.title("Course Content")
    st.sidebar.page_link("home.py", label="Home")
    st.sidebar.markdown("Ch. 1: Linear Regression")
    st.sidebar.page_link("pages/lecture1_linear_regression.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial1_linear_regression.py", label="Tutorial")
    st.sidebar.markdown("Ch. 2: Logistic Regression")
    st.sidebar.page_link("pages/lecture2_logistic_regression.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial2_logistic_regression.py", label="Tutorial")
    st.sidebar.markdown("Ch. 3: Clustering")  # Changed from "Neural Networks"
    st.sidebar.page_link("pages/lecture3_clustering.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial3_clustering_kmeans.py", label="Tutorial: K-Means")
    st.sidebar.page_link("pages/tutorial3_clustering_gmm.py", label="Tutorial: GMM")
    st.sidebar.markdown("Ch. 4: PCA")
    st.sidebar.page_link("pages/lecture4_pca.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial4_pca.py", label="Tutorial")
    st.sidebar.markdown("Ch. 5: Neural Network Supervised")
    st.sidebar.page_link("pages/lecture5_neural_network_supervised.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial5_multilayer_perceptron.py", label="Tutorial: MLP")
    st.sidebar.page_link("pages/tutorial5_pytorch.py", label="Tutorial: PyTorch")
    st.sidebar.markdown("Ch. 6: Neural Network Unsupervised")
    st.sidebar.page_link("pages/lecture6_neural_network_unsupervised.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial6_autoencoder.py", label="Tutorial: Autoencoder")
    st.sidebar.page_link("pages/tutorial6_mixed_density_network.py", label="Tutorial: MDN")
    st.sidebar.markdown("Ch. 7: Gaussian Process Regression")
    st.sidebar.page_link("pages/lecture7_gaussian_process_regression.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial7_gaussian_process_regression.py", label="Tutorial")
    st.sidebar.markdown("Ch. 8: Gaussian Process Classification")
    st.sidebar.page_link("pages/lecture8_gaussian_process_classification.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial8_gaussian_process_classification.py", label="Tutorial")
    st.sidebar.markdown("Ch. 9: Sampling")
    st.sidebar.page_link("pages/lecture9_sampling.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial9_sampling.py", label="Tutorial")
    st.sidebar.markdown("Ch. 10: Markov Chain Monte Carlo")
    st.sidebar.page_link("pages/lecture10_markov_chain_monte_carlo.py", label="Lecture")
    st.sidebar.page_link("pages/tutorial10_markov_chain_monte_carlo.py", label="Tutorial")

# Page Title
st.title('A Beginner Guide to Astro-Statistics and Astro-Machine-Learning')

navigation_menu()

# Author Information with Larger Font
st.markdown('### **Author: Yuan-Sen Ting**')

# Course Description
st.write('''
A comprehensive textbook will explore essential computational, statistical and data analysis techniques for daily astronomical research.

In the vast canvas of the universe, deciphering astronomical data is intricately tied to advanced statistical machine learning techniques. Gaining profound insights and enhancing our understanding of the cosmos necessitates a deep dive into methodologies that seamlessly weave together the disciplines of statistics and astronomy.

In this textbook, we prioritize the mathematical, statistical, and computational foundations of these intertwined fields. The textbook encompasses a range of topics: from Bayesian inference, linear and logistic regression, and dimension reduction techniques to neural networks, sampling, Gaussian Process, and Markov Chain Monte Carlo methods. 

Each chapter also comes with a companion tutorial, where we will utilize Python 3 and the Jupyter notebook, ensuring a hands-on, applied understanding of the concepts discussed.
''')

# Additional Course Information
st.info('''
All copyright reserved.
''')

# Citation and Bibliography Section
st.header('How to Cite This Work')
st.write('''
This course represents a significant effort on my part to bring advanced concepts in Astro-Statistics and Astro-Machine-Learning to a wider audience. If you find this course useful for your research or teaching, please consider cite this work. Your citation not only recognizes my effort but also helps to disseminate this knowledge to a broader community.

**Citation Format**:
**Yuan-Sen Ting**. (2023). *A Beginner Guide to Astro-Statistics and Astro-Machine-Learning* Available at [computeastro.streamlit.app](https://computeastro.streamlit.app/)
''')
