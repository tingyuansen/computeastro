# app.py
import streamlit as st
import home, course1, course2

PAGES = {
    "Home": home,
    "Course 1": course1,
    "Course 2": course2
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.app()
