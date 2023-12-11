import streamlit as st

st.markdown('''# Python - Hello World! 

by Yuan-Sen Ting (ANU, RSAA) and Tomasz Rozanski (ANU, RSAA)

---

Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. Google Colab uses Jupyter Notebooks to provide an interactive environment for coding and data analysis.

A Jupyter notebook is composed of a set of cells. Each cell can contain either code or text, and they can be executed independently. This allows you to organize your code and data analysis into logical chunks, making it easy to understand and share with others.

When working with Jupyter Notebooks on Google Colab, you can create new cells by clicking the "+ Code" or "+ Text" button below the menu. This will add a new cell to the notebook that can be used to write code or text.

To execute a cell, you can press Shift + Enter simultaneously or click the play button on the toolbar. This will execute the code or text in the cell and display the output below it.

Jupyter Notebooks can also be saved to your Google Drive, which allows you to access them from anywhere, share them with others, and collaborate on them in real-time. Additionally, Colab also allows you to use the power of the cloud, by providing the ability to use GPUs and TPUs for computation-intensive tasks.

Jupyter Notebook is a powerful tool for data analysis, machine learning, and scientific computing. It allows you to combine code, text, and visualizations in one document and make it easy to share and collaborate with others. Google Colab's integration with Jupyter Notebooks provides an easy-to-use platform for working with these tools, even on the cloud.

---
''')

st.markdown('''# "What can be written in a text cell?

Jupyter notebook uses Markdown. Markdown is a simple markup language that allows you to format text in a plain-text document. It is often used to create documents, such as README files, documentation, and blog posts, that can be easily converted to HTML or other formats. Markdown provides a simple syntax for creating headings, lists, links, images, and other formatting elements.

When creating a document in Markdown, the [Markdown Cheat Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Here-Cheatsheet) is a useful reference for the different formatting options available. The cheat sheet provides examples of the different syntax used to create headings, lists, links, images, and other formatting elements.

In the example provided, the # character is used to create headings of different levels. The number of # characters used before the text determines the level of the heading. For example,
 ## Headers 

of the next level creates a heading of level 2, while 
### Further... 

creates a heading of level 3.

You can also use **bold text** and *italic text* by enclosing the text in double asterisks for bold and single asterisks for italic. 
You can create numbered lists by starting each line with a number followed by a period and a space, as well as unnumbered lists by starting each line with an asterisk and a space.

Markdown is a simple, easy-to-use markup language that allows you to format text in a plain-text document


A key advantage of using markdown is the creation of formulas: both in text $E=mc^2$ and on a separate line:

$$ E=mc^2$$

that uses the syntax known from [LaTeX](https://en.wikipedia.org/wiki/LaTeX). The ability to enter formulas is very useful in practice in astronomy.



---





''')

st.markdown('''# Python

Python is a high-level programming language, it is easy to learn, easy to read and write, offers an interactive environment, has a vast library which makes it a popular choice for beginners and experts.

Python is often considered as the "Swiss Army Knife" of programming languages because it can be used for a wide variety of tasks like web development, data analysis, machine learning, artificial intelligence, and much more. It's easy to learn and has a vast community of users and developers, which makes it a popular choice for beginners and experts alike.

In a cell containing code, you can place code in the Python language, which will be executed line by line after pressing the **Shift + Enter** combination.

The following cell contains one line of code that will print the string `'Hello world!'`''')

st.code('''print("Hello world!")''', language='python')

st.write("Hello world!")

st.markdown('''But it can also perform much more complex operations, e.g.,''')

st.code('''import matplotlib.pyplot as plt # Load the library for creating drawings and create an alias for it: plt
import numpy as np # Numerical Python Library
# This is a comment, a piece for the programmer, in which he explains his or her intentions

x= np.arange(0,2*np.pi,0.01) # Creates an array of x's
y= np.sin(x) # Creates an array of y, filled with sin(x)

# Below I create a plot:
plt.figure(figsize=(12,6))
plt.title("Example plot")
plt.plot(x,y,label='y=sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
''', language='python')

import matplotlib.pyplot as plt # Load the library for creating drawings and create an alias for it: plt
import numpy as np # Numerical Python Library
# This is a comment, a piece for the programmer, in which he explains his or her intentions

x= np.arange(0,2*np.pi,0.01) # Creates an array of x's
y= np.sin(x) # Creates an array of y, filled with sin(x)

# Below I create a plot:
plt.figure(figsize=(12,6))
plt.title("Example plot")
plt.plot(x,y,label='y=sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

st.pyplot()




st.markdown('''In the next notebook, you can find information about the basic syntax of Python.

It is important to have a good understanding of the basics of the language before moving on to more advanced topics. Some of the basic concepts of Python include data types, variables, control structures, functions, and modules. These concepts are fundamental to any programming language and will be used throughout your journey in learning Python. It is suggested to practice by writing simple codes and experimenting with different examples to better grasp the language.''')

st.code('''''', language='python')



