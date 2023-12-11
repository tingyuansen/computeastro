# home.py
import streamlit as st

def app():
    # Page Title
    st.title('A Beginner Guide to Astro-Statistics and Astro-Machine-Learning')

    # Author Information
    st.markdown('**Author: Yuan-Sen Ting**')

    # Course Description
    st.write('''
    A comprehensive course delves into essential computational, statistical and data analysis techniques pivotal for daily astronomical research.

    In the vast canvas of the universe, deciphering astronomical data is intricately tied to advanced statistical machine learning techniques. Gaining profound insights and enhancing our understanding of the cosmos necessitates a deep dive into methodologies that seamlessly weave together the disciplines of statistics and astronomy.

    As we transition to the latter half of the course, our focus sharpens on the synergy of statistics, machine learning, and astronomy. We prioritize the mathematical, statistical, and computational foundations of these intertwined fields. The curriculum encompasses a range of topics: from Bayesian inference, linear and logistic regression, and dimension reduction techniques to neural networks, sampling, Gaussian Process, and Markov Chain Monte Carlo methods. Time permitting, we will also explore the potential of contemporary large-language models in the realm of astronomical research.

    For all programming-related assignments, participants will utilize Python 3 and the Jupyter notebook, ensuring a hands-on, applied understanding of the concepts discussed.
    ''')

    # Additional Course Information
    st.info('''
    This course is developed as my teaching for honour year and Master level course ASTR 4004 / 8004 at the Australia National University.
    All copyright reserved.
    ''')

    # Citation and Bibliography Section
    st.header('Citation and Bibliography')
    st.write('Details for citing this course and relevant bibliography will be provided here.')

# Call the app function
app()
