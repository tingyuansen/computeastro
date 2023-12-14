# home.py
import streamlit as st
import streamlit.components.v1 as components

def show_page():

    # Page Title
    st.title('Sampling - Markov Chain Monte Carlo')

    # Embed the external HTML page
    st.info('''
    Course Slides
    ''')
    slideshare_embed_code = """
        <iframe src="https://www.slideshare.net/slideshow/embed_code/key/l59x3CC7iaHQ2t" 
                width="700" height="394" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" 
                style="border:0px solid #000; border-width:0px; margin-bottom:0px; max-width: 100%;" allowfullscreen> 
        </iframe> 
    """
    components.html(slideshare_embed_code, height=412)

    st.markdown('---')

    # Embed a YouTube Video
    st.info('''
        Course Video Recording
    ''')
    video_embed_code = """
        <iframe width="700" height="394" src="https://www.youtube.com/embed/5KeqlTEya58?rel=0&modestbranding=1&autohide=1&showinfo=0&controls=0" frameborder="0" allowfullscreen style="border:none; max-width: 100%;"></iframe>
    """
    components.html(video_embed_code, height=412)


# This ensures that if someone runs this script directly, it shows the page content.
if __name__ == "__main__":
    show_page()
