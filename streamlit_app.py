import streamlit as st
from menu import navigation_menu

st.set_page_config(page_title="Comp Astro",
                       page_icon="./image/tutor_favicon.png", layout="wide")

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
