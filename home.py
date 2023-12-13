# home.py
import streamlit as st
import streamlit.components.v1 as components

def show_page():

    # Embed a YouTube Video
    video_url = 'https://www.youtube.com/watch?v=YG0C0r21VAw'

    # Embed a PDF Viewer using Google Docs Viewer
    st.subheader('Course Slides Testing 1')
    pdf_file = 'https://storage.googleapis.com/compute_astro/Week7b_Linear_Regression.pdf'  # Replace with your PDF URL
    google_docs_viewer_url = f"https://docs.google.com/gview?url={pdf_file}&embedded=true"
    components.html(
        f'<iframe src="{google_docs_viewer_url}" style="width:100%; height:480px; border:none;" frameborder="0" allowfullscreen></iframe>',
        height=480,
    )
    
    # Embed a PDF Viewer for Slides
    st.subheader('Course Slides')
      # Replace with your PDF URL
    components.html(
        f'<iframe src="{pdf_file}" style="width:100%" frameborder="0"></iframe>'
    )

    # Page Title
    st.title('A Beginner Guide to Astro-Statistics and Astro-Machine-Learning')

    # Author Information with Larger Font
    st.markdown('### **Author: Yuan-Sen Ting**')

    # Course Description
    st.write('''
    A comprehensive textbook delves into essential computational, statistical and data analysis techniques pivotal for daily astronomical research.

    In the vast canvas of the universe, deciphering astronomical data is intricately tied to advanced statistical machine learning techniques. Gaining profound insights and enhancing our understanding of the cosmos necessitates a deep dive into methodologies that seamlessly weave together the disciplines of statistics and astronomy.

    In this textbook, we prioritize the mathematical, statistical, and computational foundations of these intertwined fields. The textbook encompasses a range of topics: from Bayesian inference, linear and logistic regression, and dimension reduction techniques to neural networks, sampling, Gaussian Process, and Markov Chain Monte Carlo methods. 

    Each chapter also comes with a companion tutorial, where we will utilize Python 3 and the Jupyter notebook, ensuring a hands-on, applied understanding of the concepts discussed.
    ''')

    # Additional Course Information
    st.info('''
    This course is developed as my teaching for honour year and Master level course ASTR 4004 / 8004 at the Australia National University.
    All copyright reserved.
    ''')

    # Citation and Bibliography Section
    st.header('How to Cite This Work')
    st.write('''
    This course represents a significant effort on my part to bring advanced concepts in Astro-Statistics and Astro-Machine-Learning to a wider audience. If you find this course useful for your research or teaching, please consider cite this work. Your citation not only recognizes my effort but also helps to disseminate this knowledge to a broader community.

    **Citation Format**:
    **Yuan-Sen Ting**. (2023). *A Beginner Guide to Astro-Statistics and Astro-Machine-Learning* Available at [computeastro.streamlit.app](https://computeastro.streamlit.app/)
    ''')

# This ensures that if someone runs this script directly, it shows the page content.
if __name__ == "__main__":
    show_page()
