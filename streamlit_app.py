# main_app.py
import streamlit as st
import home
import course1

def main():
    st.sidebar.title('Course Content')
    page = st.sidebar.radio("Go to", ["Home", "Course 1"])
    # page = st.sidebar.selectbox("Choose a page", ["Home", "Course 1"])

    if page == "Home":
        home.show_page()
    elif page == "Course 1":
        course1.show_page()
    elif page == "Course 2":
        course2.show_page()

if __name__ == "__main__":
    main()
