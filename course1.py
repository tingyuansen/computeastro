import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def app():
    st.markdown(r'''# Linear Regression

- By Yuan-Sen Ting, August 2023, for ASTR 4004/8004.''')

    st.code('''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd''', language='python')

    # Sample Data
    x = np.linspace(0, 10, 30)
    y = np.sin(x)

    # Creating a plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-', label='sin(x)')
    ax.legend()

    # Displaying the plot
    st.pyplot(fig)

# Call the app function
app()
