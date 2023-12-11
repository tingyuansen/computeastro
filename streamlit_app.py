import streamlit as st
import home, lecture1_linear_regression

PAGES = {
    "Home": home,
    "Linear Regression Tutorial": lecture1_linear_regression
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.app()

