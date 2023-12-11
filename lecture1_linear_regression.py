# course1.py
import streamlit as st

def app():
    # Course Page Title
    st.title('Course 1: Introduction to Astro-Statistics')

    # Introduction Section
    st.header('Introduction')
    st.write('''
    Welcome to the first part of our journey into Astro-Statistics and Astro-Machine-Learning. In this section, we will cover the basics of statistical methods as they apply to astronomical data. This will lay the foundation for more advanced topics in later courses.
    ''')

    # Learning Objectives
    st.header('Learning Objectives')
    st.write('''
    By the end of this course, you should be able to:
    - Understand the basic principles of statistical analysis in astronomy.
    - Apply fundamental statistical techniques to analyze astronomical data.
    - Interpret the results of statistical analyses in the context of astronomical research.
    ''')

    # Course Content
    st.header('Course Content')
    st.write('''
    - **Module 1**: Statistical Foundations
      - Descriptive Statistics
      - Probability Distributions
      - Hypothesis Testing
    - **Module 2**: Applying Statistics to Astronomical Data
      - Data Collection and Preparation
      - Statistical Modeling in Astronomy
      - Case Studies and Practical Examples
    - **Module 3**: Tools and Techniques
      - Introduction to Python for Data Analysis
      - Working with Astronomical Data Sets
      - Hands-on Exercises and Assignments
    ''')

    # Additional Resources
    st.header('Additional Resources')
    st.write('''
    Here are some additional resources to help you with this course:
    - [Link to external resource 1]
    - [Link to external resource 2]
    - [Link to external resource 3]
    ''')

    # Feedback and Questions
    st.header('Feedback and Questions')
    st.write('''
    If you have any questions or feedback about this course, please feel free to reach out. Your input is invaluable in improving this learning experience.
    ''')

# Call the app function
app()
