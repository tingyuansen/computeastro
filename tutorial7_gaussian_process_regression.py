import streamlit as st

def show_page():

    st.markdown(r'''# Gaussian Processes Regression

- By Yuan-Sen Ting, October 2023, for ASTR 4004/8004. Built upon the COMP 4670/8600 tutorial prepared by Josh Nguyen.''')

    st.markdown(r'''Welcome to our lab on Gaussian process (GP) regression and classification techniques! The universe of Gaussian processes is fascinating, and understanding them is key to a myriad of applications in machine learning and beyond. In this tutorial, we will embark on a journey to explore the depths of Gaussian processes, starting from the foundational understanding of multivariate Gaussians and moving to real-world application scenarios.

The tutorial consists of three parts

1. **Properties of Multivariate Gaussian**: Delving deep into the marginal and conditional distributions.
2. **A Simple Toy Example**: A hands-on approach to understanding GP regression.
3. **Case Study**: Analyzing light curves from quasars and discovering the mysteries they hold.

### Assumed Knowledge

Before we begin, it would be beneficial if you are already familiar with the following concepts:

- **Multivariate Gaussians**: We will frequently encounter joint, marginal, and conditional distributions. If you need a refresher or more detailed mathematical insights, Chapter 2, Section 2.3 of Bishop's book is an excellent reference.
  
- **Kernels and the Kernel Trick**: The magic behind the scene in many machine learning algorithms, including Gaussian processes. Understanding the kernel trick will enable us to project our data into high-dimensional spaces effortlessly.

- **Two Views of Supervised Learning**: In our journey, we will often oscillate between two perspectives – the parameter space, which concerns the weights and biases in traditional machine learning models, and the function space, where we view things from the standpoint of functions.

- **Application of Kernels in Bayesian Regression and Classification**: To enhance our GP models, we will harness the power of kernels in the Bayesian context, bringing more flexibility and robustness.

### Learning Objectives

By the end of this lab, you should be able to:

- **Understand the Essence of Gaussian Processes**: Grasp the underlying principles and appreciate the beauty and power of Gaussian processes in regression and classification.

- **Implement GP Regression**: Put theory into practice by coding GP regression from scratch, understanding each step, and being aware of potential pitfalls.

- **Analyze and Interpret Results**: Not just stop at implementing, but also draw meaningful conclusions from the results and visualize them effectively.

---
''')

    st.code('''import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import scipy

%matplotlib inline''', language='python')

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import gridspec
    import scipy
    
    

    st.markdown(r'''## Getting Used to Multivariate Gaussians

### Gaussian Distribution: A Foundation in Machine Learning

Gaussian distributions, often referred to as Normal distributions, stand as one of the central pillars in the realms of statistics and machine learning. Their mathematical properties make them indispensable tools, especially when dealing with uncertainty or modeling real-world data. Most learners start with the univariate Gaussian — representing distributions in one dimension. However, as we venture into more advanced topics like Gaussian Process regression, the multivariate version becomes crucial.

A multivariate Gaussian distribution captures the behavior of random vectors, considering not only individual element variances but also the relationships (covariances) between them.

Formally, the probability density function (pdf) of a random vector $ x \in \mathbb{R}^d $ that follows a multivariate Gaussian distribution with mean $ \mu \in \mathbb{R}^d $ and covariance matrix $ \Sigma \in \mathbb{R}^{d \times d} $ is:

$$
\mathcal{N}(x \mid \mu, \Sigma) = \dfrac{1}{(2 \pi)^{d/2}} \dfrac{1}{\lvert \Sigma \rvert^{1/2}} \exp \bigg\{ -\dfrac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \bigg\}.
$$

To break this down:
- The term $ (2 \pi)^{d/2} $ is there to ensure the distribution is normalized.
- $ \lvert \Sigma \rvert $ represents the determinant of the covariance matrix. This factor adjusts for the "volume" or "spread" of the distribution in the multi-dimensional space.

### Connecting to Gaussian Processes

Now, why is understanding multivariate Gaussians so pivotal when delving into Gaussian Process regression?

A Gaussian Process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. Essentially, when we talk about a GP, we're discussing an infinite-dimensional generalization of multivariate Gaussian distributions. Instead of a finite vector of outputs (like in multivariate Gaussians), a GP gives us a distribution over functions. 

Understanding the nuances of multivariate Gaussians serves as the perfect stepping stone, as many properties and intuitions carry over to Gaussian Processes, especially when it comes to defining kernels, which can be seen as covariance functions.

Now, as a starting point, let's see how we can represent the aforementioned multivariate Gaussian distribution in Python:
''')

    st.code('''def multivariate_normal(x, mean, cov):
    """
    PDF of the multivariate normal distribution.
    Args:
        x: example, of shape (d, )
        mean: mu, of shape (d,)
        cov: Sigma, of shape (d, d)
    Returns:
        A scalar representing the density
    """
    # Ensure dimensions match
    d = x.shape[0]
    assert x.shape == (d,), f"x is not of shape ({{}d{}},)"
    assert mean.shape == (d,), f"Mean vector is not of shape ({{}d{}},)"
    assert cov.shape == (d,d), f"Cov matrix vector is not of shape ({{}d{}}, {{}d{}})"
    
    x_m = x - mean
    return (1. / ((2 * np.pi) ** (d / 2))) * \
           (1. / ((np.linalg.det(cov)) ** (1/2))) * \
           np.exp(- (1/2) * (np.linalg.solve(cov, x_m).T.dot(x_m)))''', language='python')

    def multivariate_normal(x, mean, cov):
        """
        PDF of the multivariate normal distribution.
        Args:
            x: example, of shape (d, )
            mean: mu, of shape (d,)
            cov: Sigma, of shape (d, d)
        Returns:
            A scalar representing the density
        """
        # Ensure dimensions match
        d = x.shape[0]
        assert x.shape == (d,), f"x is not of shape ({d},)"
        assert mean.shape == (d,), f"Mean vector is not of shape ({d},)"
        assert cov.shape == (d,d), f"Cov matrix vector is not of shape ({d}, {d})"
        
        x_m = x - mean
        return (1. / ((2 * np.pi) ** (d / 2))) * \
               (1. / ((np.linalg.det(cov)) ** (1/2))) * \
               np.exp(- (1/2) * (np.linalg.solve(cov, x_m).T.dot(x_m)))
    
    

    st.markdown(r'''Testing is crucial. To ensure our implementation is correct, we'll compare our function's output against that of a well-established library: SciPy.''')

    st.code('''# Setting the random seed for reproducibility
np.random.seed(100)

# Define the multivariate Gaussian parameters
d = 2  # Number of dimensions
mean = np.array([1, 0])
cov = np.array([[1, -0.7],
                [-0.7, 1]])

# Test point
x = np.array([1, 2])
if len(x) != d:
    raise ValueError(f"Dimension mismatch: x should be of dimension {{}d{}}.")

# Calculate the PDF using the implemented method
pdf_x_implemented = multivariate_normal(x, mean, cov)
print(f"[YOUR    METHOD] p(x) = {{}pdf_x_implemented:.10f{}}")

# Calculate the PDF using SciPy's method
pdf_x_scipy = scipy.stats.multivariate_normal(mean=mean, cov=cov).pdf(x)
print(f"[SCIPY's METHOD] p(x) = {{}pdf_x_scipy:.10f{}}")

# Check and print if both methods match
print("Do the methods produce the same result?", np.allclose(pdf_x_implemented, pdf_x_scipy))
''', language='python')

    # Setting the random seed for reproducibility
    np.random.seed(100)
    
    # Define the multivariate Gaussian parameters
    d = 2  # Number of dimensions
    mean = np.array([1, 0])
    cov = np.array([[1, -0.7],
                    [-0.7, 1]])
    
    # Test point
    x = np.array([1, 2])
    if len(x) != d:
        raise ValueError(f"Dimension mismatch: x should be of dimension {d}.")
    
    # Calculate the PDF using the implemented method
    pdf_x_implemented = multivariate_normal(x, mean, cov)
    st.write(f"[YOUR    METHOD] p(x) = {pdf_x_implemented:.10f}")
    
    # Calculate the PDF using SciPy's method
    pdf_x_scipy = scipy.stats.multivariate_normal(mean=mean, cov=cov).pdf(x)
    st.write(f"[SCIPY's METHOD] p(x) = {pdf_x_scipy:.10f}")
    
    # Check and print if both methods match
    st.write("Do the methods produce the same result?", np.allclose(pdf_x_implemented, pdf_x_scipy))
    
    

    st.markdown(r'''Now, let's bring our distribution to life by visualizing the PDF. Visualization can offer deeper insights and confirm if our distribution looks as expected:
''')

    st.code('''def generate_pdf_data(mean, cov):
    """
    Compute the multivariate Gaussian PDF values over a grid.
    
    Args:
    - mean (array-like): Array of shape (d,).
    - cov (array-like): Covariance matrix of shape (d, d).
    
    Returns:
    - grid (list): List of meshgrid arrays.
    - pdf (array-like): PDF values over the grid.
    """
    d = mean.shape[0]
    
    N = 50
    x1s = np.linspace(mean[0]-3, mean[0]+3, num=N)
    x2s = np.linspace(mean[1]-3, mean[1]+3, num=N)
    x1, x2 = np.meshgrid(x1s, x2s)

    # Convert meshgrid arrays to 2D array of shape (N*N, d)
    points = np.column_stack((x1.ravel(), x2.ravel()))

    # Compute the PDF values for all points at once
    pdf_values = np.array([multivariate_normal(pt, mean, cov) for pt in points])

    # Reshape the result back to 2D grid
    pdf = pdf_values.reshape(N, N)
            
    return [x1, x2], pdf


def plot_gaussian_pdf(mean, cov, ax):
    """
    Plot the 2D Gaussian distribution's PDF.
    
    Args:
    - mean (array-like): Array of shape (2,).
    - cov (array-like): Covariance matrix of shape (2, 2).
    - ax (AxesSubplot): Axes to plot on.
    """
    grid, pdf_values = generate_pdf_data(mean, cov)
    cb = ax.contourf(*grid, pdf_values, 100, cmap=cm.YlGnBu)
    
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.set_aspect('equal')

    # Show the mean
    ax.scatter(*mean, marker="x", color="red", s=60, label="Mean")
    ax.legend()
''', language='python')

    def generate_pdf_data(mean, cov):
        """
        Compute the multivariate Gaussian PDF values over a grid.
        
        Args:
        - mean (array-like): Array of shape (d,).
        - cov (array-like): Covariance matrix of shape (d, d).
        
        Returns:
        - grid (list): List of meshgrid arrays.
        - pdf (array-like): PDF values over the grid.
        """
        d = mean.shape[0]
        
        N = 50
        x1s = np.linspace(mean[0]-3, mean[0]+3, num=N)
        x2s = np.linspace(mean[1]-3, mean[1]+3, num=N)
        x1, x2 = np.meshgrid(x1s, x2s)
    
        # Convert meshgrid arrays to 2D array of shape (N*N, d)
        points = np.column_stack((x1.ravel(), x2.ravel()))
    
        # Compute the PDF values for all points at once
        pdf_values = np.array([multivariate_normal(pt, mean, cov) for pt in points])
    
        # Reshape the result back to 2D grid
        pdf = pdf_values.reshape(N, N)
                
        return [x1, x2], pdf
    
    def plot_gaussian_pdf(mean, cov, ax):
        """
        Plot the 2D Gaussian distribution's PDF.
        
        Args:
        - mean (array-like): Array of shape (2,).
        - cov (array-like): Covariance matrix of shape (2, 2).
        - ax (AxesSubplot): Axes to plot on.
        """
        grid, pdf_values = generate_pdf_data(mean, cov)
        cb = ax.contourf(*grid, pdf_values, 100, cmap=cm.YlGnBu)
        
        ax.set_xlabel('$x_1$', fontsize=13)
        ax.set_ylabel('$x_2$', fontsize=13)
        ax.set_aspect('equal')
    
        # Show the mean
        ax.scatter(*mean, marker="x", color="red", s=60, label="Mean")
        ax.legend()
    
    

    st.code('''# Setting up the distribution
np.random.seed(100)
d = 2
mean = np.array([1, 0])
cov = np.array([[1, -0.7],
                [-0.7, 1]])

# Plotting the PDF
fig, ax = plt.subplots(1, 1)
plot_gaussian_pdf(mean, cov, ax)
ax.set_title('2D Gaussian Distribution', fontsize=14)
plt.show()''', language='python')

    # Setting up the distribution
    np.random.seed(100)
    d = 2
    mean = np.array([1, 0])
    cov = np.array([[1, -0.7],
                    [-0.7, 1]])
    
    # Plotting the PDF
    fig, ax = plt.subplots(1, 1)
    plot_gaussian_pdf(mean, cov, ax)
    ax.set_title('2D Gaussian Distribution', fontsize=14)
    st.pyplot(fig)
    

    st.markdown(r'''Take a moment to observe the plot. The color gradient indicates the density, with the red `x` marking the mean of the distribution. Adjust the mean and covariance values to see how they affect the shape and orientation of the distribution!

---''')

    st.markdown(r'''### The Essence of Marginal Distributions

When dealing with multivariate Gaussian distributions, the concept of marginal distributions is fundamental. Marginal distributions let us focus solely on one or a subset of variables by integrating out the others.

Given our two-component system, $x_a$ and $x_b$:

$$
    \mathcal{N}(x \mid \mu, \Sigma) = 
    \mathcal{N}
        \bigg(
        \begin{matrix}
        x_a \\
        x_b
        \end{matrix}
        \bigg|
        \begin{matrix}
        \mu_a \\
        \mu_b
        \end{matrix},
        \begin{matrix}
        \Sigma_{aa} & \Sigma_{ab} \\
        \Sigma_{ba} & \Sigma_{bb}
        \end{matrix}
    \bigg),
$$

To obtain the marginal distributions, we integrate:

$$
p(x_a) = \int p(x_a, x_b) \, dx_b
$$

$$
p(x_b) = \int p(x_a, x_b) \, dx_a
$$

Now, it turns out that if we perform the integrals, the marginal distributions of a multivariate Gaussian are also Gaussian. The mean and covariance of the marginal distribution of $x_a$ are directly taken from the original distribution: $ \mu_a $ and $ \Sigma_{aa} $, respectively. Similarly, for $x_b$, the mean is $ \mu_b $ and the covariance is $ \Sigma_{bb} $.

This is one of the many elegant properties of Gaussian distributions: even after marginalization, the resulting distributions remain Gaussian. This quality makes Gaussian distributions an attractive choice in a wide range of applications including the Gaussian Process regression that we will explore in the following.''')

    st.markdown(r'''### Connecting Marginal Distributions to Gaussian Processes

The relationship between marginal distributions and Gaussian Processes might not be immediately evident. While predictions in GPs hinge on conditional distributions, but as we will see below, marginal likelihoods—or evidence—become particularly vital when optimizing hyperparameters in GPs.

The marginal likelihood measures the probability of the observed data for different choices of hyperparameters. By optimizing this likelihood, we can effectively choose hyperparameters that make our observed data most probable. 

This notion of optimizing hyperparameters using marginal likelihood stems from the properties of Gaussian distributions. The fact that marginal distributions of a multivariate Gaussian retain their Gaussian nature ensures that we can compute the marginal likelihood in a closed form, making hyperparameter optimization computationally tractable.

In the grander context of Gaussian Processes, understanding the characteristics and implications of marginal distributions is crucial. It provides a foundation for model selection and hyperparameter tuning, ensuring our GP model is both accurate and robust.
''')

    st.markdown(r'''### Visualizing the Marginal Distributions

For a practical grasp, let's visualize these marginal distributions. In the upcoming example, let's assume:
- $x \in \mathbb{R}^{2}$
- $x_a = x_1 \in \mathbb{R}$
- $x_b = x_2 \in \mathbb{R}$
''')

    st.code('''# Function Definitions
def univariate_normal(x, mean, var):
    """
    PDF of the univariate normal distribution.
    Args:
        x: Input value, scalar.
        mean: Mean of the distribution, scalar.
        var: Variance of the distribution, scalar.
    Returns:
        A scalar representing the density at x.
    """
    return (1. / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))
''', language='python')

    # Function Definitions
    def univariate_normal(x, mean, var):
        """
        PDF of the univariate normal distribution.
        Args:
            x: Input value, scalar.
            mean: Mean of the distribution, scalar.
            var: Variance of the distribution, scalar.
        Returns:
            A scalar representing the density at x.
        """
        return (1. / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))
    
    

    st.code('''# Main Execution
# Set random seed for reproducibility
np.random.seed(100)

# Define Gaussian distribution parameters
d = 2
mean_values = np.array([1, 0])
cov_matrix = np.array([[1, -0.7], [-0.7, 1]])

# Extract individual means and variances
mean_xa, mean_xb = mean_values
var_xa = cov_matrix[0, 0]
var_xb = cov_matrix[1, 1]

# Setup figure
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

# Plot joint 2D Gaussian distribution
ax_joint = plt.subplot(gs[0])
plot_gaussian_pdf(mean_values, cov_matrix, ax_joint)
ax_joint.set_title('2D Gaussian Distribution', fontsize=14)

# Plot marginal distribution for x1
ax_x1 = plt.subplot(gs[2])
x_range = np.linspace(-5, 5, num=50)
prob_x1 = univariate_normal(x_range, mean_xa, var_xa)
ax_x1.plot(x_range, prob_x1, 'r--', label="$p(x_1)$")
ax_x1.legend()
ax_x1.set_ylabel("$p(x_1)$", fontsize=13)
ax_x1.yaxis.set_label_position('right')
ax_x1.set_xlim(-2, 4)

# Plot marginal distribution for x2
ax_x2 = plt.subplot(gs[1])
x_range = np.linspace(-3, 3, num=50)
prob_x2 = univariate_normal(x_range, mean_xb, var_xb)
ax_x2.plot(prob_x2, x_range, 'b--', label="$p(x_2)$")
ax_x2.legend()
ax_x2.set_xlabel('$p(x_2)$', fontsize=13)
ax_x2.set_ylim(-3, 3)

# Show the plots
plt.show()
''', language='python')

    # Main Execution
    # Set random seed for reproducibility
    np.random.seed(100)
    
    # Define Gaussian distribution parameters
    d = 2
    mean_values = np.array([1, 0])
    cov_matrix = np.array([[1, -0.7], [-0.7, 1]])
    
    # Extract individual means and variances
    mean_xa, mean_xb = mean_values
    var_xa = cov_matrix[0, 0]
    var_xb = cov_matrix[1, 1]
    
    # Setup figure
    fig = fig, ax = plt.subplots(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # Plot joint 2D Gaussian distribution
    ax_joint = plt.subplot(gs[0])
    plot_gaussian_pdf(mean_values, cov_matrix, ax_joint)
    ax_joint.set_title('2D Gaussian Distribution', fontsize=14)
    
    # Plot marginal distribution for x1
    ax_x1 = plt.subplot(gs[2])
    x_range = np.linspace(-5, 5, num=50)
    prob_x1 = univariate_normal(x_range, mean_xa, var_xa)
    ax_x1.plot(x_range, prob_x1, 'r--', label="$p(x_1)$")
    ax_x1.legend()
    ax_x1.set_ylabel("$p(x_1)$", fontsize=13)
    ax_x1.yaxis.set_label_position('right')
    ax_x1.set_xlim(-2, 4)
    
    # Plot marginal distribution for x2
    ax_x2 = plt.subplot(gs[1])
    x_range = np.linspace(-3, 3, num=50)
    prob_x2 = univariate_normal(x_range, mean_xb, var_xb)
    ax_x2.plot(prob_x2, x_range, 'b--', label="$p(x_2)$")
    ax_x2.legend()
    ax_x2.set_xlabel('$p(x_2)$', fontsize=13)
    ax_x2.set_ylim(-3, 3)
    
    # Show the plots
    st.pyplot(fig)
    
    

    st.markdown(r'''Observe the plots. The first displays the joint distribution of $x_a$ and $x_b$. The other two provide a visualization for the marginal distributions of $x_a$ and $x_b$ individually. By examining these plots, you can gain an understanding of the distributions' shapes, spread, and central tendencies.

---
''')

    st.markdown(r'''### Conditional Distributions: A Key Aspect of Gaussian Processes

#### Introduction to Conditional Distributions

In many real-world scenarios, understanding how the distribution of a variable changes given the knowledge of another becomes crucial. This concept, represented as conditional distributions, is a cornerstone in Gaussian Processes (GPs) and multivariate Gaussian distributions.

Given a Gaussian random variable $x$ split into two components, $x_a$ and $x_b$, as:

$$
    \mathcal{N}(x \mid \mu, \Sigma) = 
    \mathcal{N}
    \bigg(
        \begin{matrix}
            x_a \\
            x_b
        \end{matrix}
        \bigg|
        \begin{matrix}
            \mu_a \\
            \mu_b
        \end{matrix},
        \begin{matrix}
            \Sigma_{aa} & \Sigma_{ab} \\
            \Sigma_{ba} & \Sigma_{bb}
        \end{matrix}
    \bigg),
$$
where $x_a \in \mathbb{R}^{m}$ and $x_b \in \mathbb{R}^{d-m}$.

The task at hand is to decipher the conditional distributions $p(x_a \mid x_b)$ and $p(x_b \mid x_a)$.

#### Deriving the Conditional Distributions

Both $x_a \mid x_b$ and $x_b \mid x_a$ are Gaussian random variables. The parameters for these conditional distributions can be derived from the properties of multivariate Gaussians. For $p(x_a \mid x_b)$, the parameters are:

$$
\begin{align*}
\mu_{a \mid b} &= \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1}(x_b - \mu_b), \\
\Sigma_{a \mid b} &= \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}.
\end{align*}
$$

Similarly, for $p(x_b \mid x_a)$:

$$
\begin{align*}
\mu_{b \mid a} &= \mu_b + \Sigma_{ba} \Sigma_{aa}^{-1}(x_a - \mu_a), \\
\Sigma_{b \mid a} &= \Sigma_{bb} - \Sigma_{ba} \Sigma_{aa}^{-1} \Sigma_{ab}.
\end{align*}
$$


#### Tying Conditional Distributions to Gaussian Processes

While the idea of conditional distributions in multivariate Gaussians is powerful on its own, its real prowess emerges in the context of Gaussian Processes. In GPs, making predictions for new points given observed data relies on conditional distributions. Given a set of training points and their outputs, the GP framework lets us deduce the conditional distribution of outputs at new points. This not only provides an estimate for the function value at these new points but also a measure of uncertainty associated with these estimates.

In essence, the entire predictive capability of GPs is underpinned by conditional distributions. Understanding them in the context of multivariate Gaussians paves the way for effective utilization of Gaussian Processes in various machine learning applications.

#### Visualization of Conditional Distributions

Visualization aids in developing an intuition about conditional distributions. Let's consider an example where $x \in \mathbb{R}^{2}$, $x_a = x_1 \in \mathbb{R}$, and $x_b = x_2 \in \mathbb{R}$. In this 2D space, understanding the influence of one variable on another becomes more tangible.
''')

    st.code('''# Seed for reproducibility
np.random.seed(100)

# Define the distribution
d = 2  # Number of dimensions
mean_values = np.array([1, 0])
cov_matrix = np.array([[1, -0.7],
                       [-0.7, 1]])
mean_xa, mean_xb = mean_values
s_aa, s_ab, s_ba, s_bb = cov_matrix.flatten()

# Initialize the plot space
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

# 1. Plot the 2D Gaussian distribution 
ax_joint = plt.subplot(gs[0])
plot_gaussian_pdf(mean_values, cov_matrix, ax_joint)
ax_joint.set_title('2D Gaussian Distribution', fontsize=14)
ax_joint.plot([-2, 4], [1, 1], "r--")
ax_joint.plot([2, 2], [-3, 3], "b--")

# 2. Plot the conditional distribution p(x1 | x2=1)
x_b_given = 1
cond_mean_xa = mean_xa + s_ab * (1/s_bb) * (x_b_given - mean_xb)
cond_var_xa = s_aa - s_ab * (1/s_bb) * s_ba
ax_xa_given_xb = plt.subplot(gs[2])
x_range = np.linspace(-5, 5, num=50)
prob_xa_given_xb = univariate_normal(x_range, cond_mean_xa, cond_var_xa)
ax_xa_given_xb.plot(x_range, prob_xa_given_xb, 'r--', label="$p(x_1 | x_2 = 1)$")
ax_xa_given_xb.legend(loc=0)
ax_xa_given_xb.set_ylabel("$p(x_1 | x_2 = 1)$", fontsize=13)
ax_xa_given_xb.yaxis.set_label_position('right')
ax_xa_given_xb.set_xlim(-2, 4)

# 3. Plot the conditional distribution p(x2 | x1=2)
x_a_given = 2
cond_mean_xb = mean_xb + s_ba * (1/s_aa) * (x_a_given - mean_xa)
cond_var_xb = s_bb - s_ba * (1/s_aa) * s_ab
ax_xb_given_xa = plt.subplot(gs[1])
x_range = np.linspace(-3, 3, num=50)
prob_xb_given_xa = univariate_normal(x_range, cond_mean_xb, cond_var_xb)
ax_xb_given_xa.plot(prob_xb_given_xa, x_range, 'b--', label="$p(x_2 | x_1 = 2)$")
ax_xb_given_xa.legend(loc=0)
ax_xb_given_xa.set_xlabel('$p(x_2 | x_1 = 2)$', fontsize=13)
ax_xb_given_xa.yaxis.set_label_position('right')
ax_xb_given_xa.set_ylim(-3, 3)

# Render the plot
plt.show()
''', language='python')

    # Seed for reproducibility
    np.random.seed(100)
    
    # Define the distribution
    d = 2  # Number of dimensions
    mean_values = np.array([1, 0])
    cov_matrix = np.array([[1, -0.7],
                           [-0.7, 1]])
    mean_xa, mean_xb = mean_values
    s_aa, s_ab, s_ba, s_bb = cov_matrix.flatten()
    
    # Initialize the plot space
    fig = fig, ax = plt.subplots(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # 1. Plot the 2D Gaussian distribution 
    ax_joint = plt.subplot(gs[0])
    plot_gaussian_pdf(mean_values, cov_matrix, ax_joint)
    ax_joint.set_title('2D Gaussian Distribution', fontsize=14)
    ax_joint.plot([-2, 4], [1, 1], "r--")
    ax_joint.plot([2, 2], [-3, 3], "b--")
    
    # 2. Plot the conditional distribution p(x1 | x2=1)
    x_b_given = 1
    cond_mean_xa = mean_xa + s_ab * (1/s_bb) * (x_b_given - mean_xb)
    cond_var_xa = s_aa - s_ab * (1/s_bb) * s_ba
    ax_xa_given_xb = plt.subplot(gs[2])
    x_range = np.linspace(-5, 5, num=50)
    prob_xa_given_xb = univariate_normal(x_range, cond_mean_xa, cond_var_xa)
    ax_xa_given_xb.plot(x_range, prob_xa_given_xb, 'r--', label="$p(x_1 | x_2 = 1)$")
    ax_xa_given_xb.legend(loc=0)
    ax_xa_given_xb.set_ylabel("$p(x_1 | x_2 = 1)$", fontsize=13)
    ax_xa_given_xb.yaxis.set_label_position('right')
    ax_xa_given_xb.set_xlim(-2, 4)
    
    # 3. Plot the conditional distribution p(x2 | x1=2)
    x_a_given = 2
    cond_mean_xb = mean_xb + s_ba * (1/s_aa) * (x_a_given - mean_xa)
    cond_var_xb = s_bb - s_ba * (1/s_aa) * s_ab
    ax_xb_given_xa = plt.subplot(gs[1])
    x_range = np.linspace(-3, 3, num=50)
    prob_xb_given_xa = univariate_normal(x_range, cond_mean_xb, cond_var_xb)
    ax_xb_given_xa.plot(prob_xb_given_xa, x_range, 'b--', label="$p(x_2 | x_1 = 2)$")
    ax_xb_given_xa.legend(loc=0)
    ax_xb_given_xa.set_xlabel('$p(x_2 | x_1 = 2)$', fontsize=13)
    ax_xb_given_xa.yaxis.set_label_position('right')
    ax_xb_given_xa.set_ylim(-3, 3)
    
    # Render the plot
    st.pyplot(fig)
    
    

    st.markdown(r'''---

## Gaussian Process Regression

Gaussian Process Regression (GPR) is a non-parametric Bayesian approach to regression that uses the concept of a Gaussian process to predict continuous values for new input points. It's fundamentally based on setting a prior over functions and then updating this prior with observed data to get a posterior over functions.

### Introducing a Toy Dataset

For clarity, we'll begin with a one-dimensional dataset. This will allow us to visualize and better understand the underlying function we're trying to estimate.

Our synthetic dataset is generated from the function:

$$
y(x) = 0.08 x^{2} + 2 \sin(x) + 0.1 \tanh(x^{3}).
$$

This function combines polynomial, sinusoidal, and hyperbolic tangent components. Let's create data based on this function and add some noise to simulate real-world scenarios:
''')

    st.code('''def generate_regression_data(N=20, beta=3):
    """
    Generate synthetic regression data.
    Args:
        N (int): Number of data points.
    Returns:
        X (ndarray): Input values.
        t (ndarray): Target values.
    """
    np.random.seed(42)
    X = np.linspace(-10, 10, N)
    t = 0.08 * X ** 2 + 2 * np.sin(X) + 0.1 * np.tanh(X ** 3)
    
    # Introduce Gaussian noise to t for realistic simulation
    t += (1/np.sqrt(beta)) * np.random.randn(N)
    
    return np.expand_dims(X, 1), t''', language='python')

    def generate_regression_data(N=20, beta=3):
        """
        Generate synthetic regression data.
        Args:
            N (int): Number of data points.
        Returns:
            X (ndarray): Input values.
            t (ndarray): Target values.
        """
        np.random.seed(42)
        X = np.linspace(-10, 10, N)
        t = 0.08 * X ** 2 + 2 * np.sin(X) + 0.1 * np.tanh(X ** 3)
        
        # Introduce Gaussian noise to t for realistic simulation
        t += (1/np.sqrt(beta)) * np.random.randn(N)
        
        return np.expand_dims(X, 1), t
    
    

    st.markdown(r'''We can visualize the true function and the generated noisy data:
''')

    st.code('''fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Default setting for noise (inverse variance)
beta_assumed = 10

# Plot the underlying true function
x_coords = np.linspace(-10, 10, 100)
true_function = 0.08 * x_coords ** 2 + 2 * np.sin(x_coords) \
                + 0.1 * np.tanh(x_coords ** 3)
ax.plot(x_coords, true_function, 'r--', linewidth=2, label="True function")

# Plot the generated noisy data
X, t = generate_regression_data(N=10, beta=beta_assumed)
ax.scatter(X, t, c="k", marker="o", label="Noisy data")
ax.set_title("Synthetic Data and Underlying Function")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()
plt.show()''', language='python')

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    # Default setting for noise (inverse variance)
    beta_assumed = 10
    
    # Plot the underlying true function
    x_coords = np.linspace(-10, 10, 100)
    true_function = 0.08 * x_coords ** 2 + 2 * np.sin(x_coords) \
                    + 0.1 * np.tanh(x_coords ** 3)
    ax.plot(x_coords, true_function, 'r--', linewidth=2, label="True function")
    
    # Plot the generated noisy data
    X, t = generate_regression_data(N=10, beta=beta_assumed)
    ax.scatter(X, t, c="k", marker="o", label="Noisy data")
    ax.set_title("Synthetic Data and Underlying Function")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    st.pyplot(fig)
    

    st.markdown(r'''### Understanding the RBF Kernel

The Radial Basis Function (RBF) kernel, often called the Gaussian kernel, is a popular choice in Gaussian process regression. It is defined as:

$$
k(x_n, x_m) = \exp \bigg\{ - \frac{\left\| x_n - x_m \right\|_2^2} {2 \sigma^2} \bigg\}
$$

Here, $\left\| x_n - x_m \right\|_2^2$ represents the squared Euclidean distance between data points $ x_n $ and $ x_m $, and $ \sigma $ is a hyperparameter determining the width of the kernel.

This kernel measures the similarity between points. When the points are close, the kernel value is high, and as the points move apart, the kernel value decays.

We can compute the RBF kernel using:

''')

    st.code('''def rbf(xa, xb, sigma=1):
    """
    Compute the RBF kernel.
    Args:
        xa (ndarray): First array of shape (N1, d).
        xb (ndarray): Second array of shape (N2, d).
        sigma (float): Kernel hyperparameter.
    Returns:
        Kernel matrix of shape (N1, N2).
    """
    # Calculate squared Euclidean distance
    sqdist = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    
    # Return the computed RBF kernel
    return np.exp(-sqdist / (2 * sigma ** 2))''', language='python')

    def rbf(xa, xb, sigma=1):
        """
        Compute the RBF kernel.
        Args:
            xa (ndarray): First array of shape (N1, d).
            xb (ndarray): Second array of shape (N2, d).
            sigma (float): Kernel hyperparameter.
        Returns:
            Kernel matrix of shape (N1, N2).
        """
        # Calculate squared Euclidean distance
        sqdist = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
        
        # Return the computed RBF kernel
        return np.exp(-sqdist / (2 * sigma ** 2))
    
    

    st.markdown(r'''The kernel essentially provides us a measure of similarity between any two points in our input space. The kernel values inform the Gaussian process about how to make predictions for new, unseen data points based on the observed data.
''')

    st.markdown(r'''### Introductory Concepts of Gaussian Process Regression

In Gaussian Process Regression (GPR), we assume that function values are jointly Gaussian-distributed. For a training set of inputs $ X $ and their outputs $ t $, and a new input $ x^* $, the joint distribution of the training outputs and the new output is:

$$
\begin{bmatrix}
t \\
y^*
\end{bmatrix}
\sim \mathcal{N}
\bigg(
\begin{bmatrix}
0 \\
0
\end{bmatrix},
\begin{bmatrix}
K(X, X) & K(X, x^*) \\
K(x^*, X) & K(x^*, x^*)
\end{bmatrix}
\bigg)
$$

Here, $ K(X, X) $ is the kernel matrix of the training data.

### Incorporating Observation Noise

In real-world scenarios, the observations are often noisy. To account for this noise, we incorporate a noise term into our Gaussian Process, which leads us to:

$$
\begin{bmatrix}
t \\
y^*
\end{bmatrix}
\sim \mathcal{N}
\bigg(
\begin{bmatrix}
0 \\
0
\end{bmatrix},
\begin{bmatrix}
K(X, X) + \beta^{-1}I & K(X, x^*) \\
K(x^*, X) & K(x^*, x^*) + \beta^{-1}
\end{bmatrix}
\bigg)
$$

where $ \beta^{-1} $ represents the noise variance. By adding this term, we're saying that there's a small amount of Gaussian noise associated with each observation in our training set.

The logic behind this is straightforward: in many real-world scenarios, measurements come with some inherent uncertainty or noise. By adding a noise variance to our covariance matrix, we account for this uncertainty in our model. This is especially important when our new input $ x^* $ is a training point, as we would expect the variance to be non-zero due to the observation noise.

Given this joint distribution, our predictive distribution for $ y^* $ given $ t $ becomes:

$$
p(y^* | t) = \mathcal{N}(y^*; m(x^*), \sigma^2(x^*))
$$

with:

1. $ m(x^*) = K(x^*, X)[K(X, X) + \beta^{-1}I]^{-1}t $ - the mean of the predictive distribution.
2. $ \sigma^2(x^*) = K(x^*, x^*) + \beta^{-1} - K(x^*, X)[K(X, X) + \beta^{-1}I]^{-1}K(X, x^*) $

For computational efficiency and stability, especially when inverting the kernel matrix for large datasets, we often use the Cholesky decomposition. This decomposes the positive definite kernel matrix as $ K(X, X) + \beta^{-1}I = L L^T $, where $ L $ is a lower triangular matrix. Using this decomposition, we can more efficiently solve for the weights vector and compute the predictive mean and variance.
''')

    st.code('''def gp_reg(x_star, X, t, kernel, beta, **kernel_kwargs):
    """
    Gaussian Process Regression to compute the mean and variance of y_star.
    
    Args:
        x_star (ndarray): New example.
        X (ndarray): Training examples.
        t (ndarray): Training targets.
        kernel (function): Kernel function.
        beta (float): Precision of the noise in the targets.
        kernel_kwargs (dict): Additional arguments for the kernel function.

    Returns:
        float: Mean of y_star.
        float: Variance of y_star.
    """
    N, d = X.shape
    
    # Compute kernel matrices
    K_X_X = kernel(X, X, **kernel_kwargs) + (1/beta) * np.eye(N) 
    K_x_star_X = kernel(x_star, X, **kernel_kwargs)
    K_x_star_x_star = kernel(x_star, x_star, **kernel_kwargs)
    
    # Cholesky decomposition for stability and efficiency
    L = np.linalg.cholesky(K_X_X)
    
    # Compute the weights vector alpha
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))
    
    # Predicted mean for y_star
    m_x_star = K_x_star_X @ alpha
    
    # Predicted variance for y_star
    v = np.linalg.solve(L, K_x_star_X.T)
    variance_x_star = K_x_star_x_star + (1/beta) - v.T @ v

    return m_x_star.flatten()[0], variance_x_star.flatten()[0]
''', language='python')

    def gp_reg(x_star, X, t, kernel, beta, **kernel_kwargs):
        """
        Gaussian Process Regression to compute the mean and variance of y_star.
        
        Args:
            x_star (ndarray): New example.
            X (ndarray): Training examples.
            t (ndarray): Training targets.
            kernel (function): Kernel function.
            beta (float): Precision of the noise in the targets.
            kernel_kwargs (dict): Additional arguments for the kernel function.
    
        Returns:
            float: Mean of y_star.
            float: Variance of y_star.
        """
        N, d = X.shape
        
        # Compute kernel matrices
        K_X_X = kernel(X, X, **kernel_kwargs) + (1/beta) * np.eye(N) 
        K_x_star_X = kernel(x_star, X, **kernel_kwargs)
        K_x_star_x_star = kernel(x_star, x_star, **kernel_kwargs)
        
        # Cholesky decomposition for stability and efficiency
        L = np.linalg.cholesky(K_X_X)
        
        # Compute the weights vector alpha
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))
        
        # Predicted mean for y_star
        m_x_star = K_x_star_X @ alpha
        
        # Predicted variance for y_star
        v = np.linalg.solve(L, K_x_star_X.T)
        variance_x_star = K_x_star_x_star + (1/beta) - v.T @ v
    
        return m_x_star.flatten()[0], variance_x_star.flatten()[0]
    
    

    st.markdown(r'''### Visualizing Predictions

Let's visualize the true function, the noisy data we've generated, and the predictions from our Gaussian Process Regression.
''')

    st.code('''fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Plot the true underlying function
x_coords = np.linspace(-10, 10, 100)
true_function = 0.08 * x_coords ** 2 + 2 * np.sin(x_coords) + 0.1 * np.tanh(x_coords ** 3)
ax.plot(x_coords, true_function, 'r--', linewidth=2, label="True function")

# Display the training data
ax.scatter(X.flatten(), t, c="k", marker="o", label="Noisy data")

sigma_assumed = 1 # the assumed hyperparameter for the RBF kernel

# Predict using Gaussian Process and visualize
x_coords = np.expand_dims(np.linspace(-10, 10, 100), 1)

predicted = [gp_reg(np.expand_dims(x, 1), X, t, beta=beta_assumed, kernel=rbf, sigma=sigma_assumed) for x in x_coords]
predicted_m = np.array([m for m, v in predicted])
predicted_stddev = np.array([np.sqrt(v) for m, v in predicted])

ax.plot(x_coords.flatten(), predicted_m, 'b-', lw=2, label='Predicted $m(x^*)$')
ax.fill_between(x_coords.flatten(), predicted_m-2*predicted_stddev, predicted_m+2*predicted_stddev, color='blue', alpha=0.15, label='Predicted $2 \sigma(x^*)$ interval')

ax.set_title("Gaussian Process Regression using RBF Kernel")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()

plt.show()''', language='python')

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    # Plot the true underlying function
    x_coords = np.linspace(-10, 10, 100)
    true_function = 0.08 * x_coords ** 2 + 2 * np.sin(x_coords) + 0.1 * np.tanh(x_coords ** 3)
    ax.plot(x_coords, true_function, 'r--', linewidth=2, label="True function")
    
    # Display the training data
    ax.scatter(X.flatten(), t, c="k", marker="o", label="Noisy data")
    
    sigma_assumed = 1 # the assumed hyperparameter for the RBF kernel
    
    # Predict using Gaussian Process and visualize
    x_coords = np.expand_dims(np.linspace(-10, 10, 100), 1)
    
    predicted = [gp_reg(np.expand_dims(x, 1), X, t, beta=beta_assumed, kernel=rbf, sigma=sigma_assumed) for x in x_coords]
    predicted_m = np.array([m for m, v in predicted])
    predicted_stddev = np.array([np.sqrt(v) for m, v in predicted])
    
    ax.plot(x_coords.flatten(), predicted_m, 'b-', lw=2, label='Predicted $m(x^*)$')
    ax.fill_between(x_coords.flatten(), predicted_m-2*predicted_stddev, predicted_m+2*predicted_stddev, color='blue', alpha=0.15, label='Predicted $2 \sigma(x^*)$ interval')
    
    ax.set_title("Gaussian Process Regression using RBF Kernel")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    st.pyplot(fig)
    

    st.markdown(r'''The shaded blue region in the plot represents the confidence interval (±2 standard deviations) around the predictions. This provides a measure of uncertainty in our predictions. As expected, the uncertainty is usually higher in regions where there's less data.

The above code showcases the essence of Gaussian Process Regression. It defines a distribution over possible functions that could fit the data. Our predictions are made by considering all these possible functions. As you might observe, the prediction seems to be more jagged and does not necessary follow the true function, this is because the choice of kernel (in this case, RBF) and its hyperparameters can significantly influence the predictions, something that we will optimise next.

---''')

    st.markdown(r'''## Tuning Hyperparameters in the Gaussian Process Model

In the Gaussian Process Regression, the choice of kernel and its hyperparameters play a significant role in the model's predictive capability. Specifically, for the Radial Basis Function (RBF) kernel, also known as the Gaussian kernel, there's a hyperparameter $ \sigma $ (often called the "width" or "length scale") that defines the shape of the function. A smaller $ \sigma $ can make the function more wiggly, whereas a larger $ \sigma $ can smoothen it out.

While we previously used an arbitrary value for $ \sigma $, that choice might not be optimal for our dataset. The right way to determine the best value of $ \sigma $ is by maximizing the log likelihood of the observed data.

#### Understanding the Log Likelihood with Respect to $ \sigma $

The log likelihood is a measure that indicates how probable our observed data is, given our model and the choice of hyperparameters. The mathematical formulation for the log likelihood in the context of Gaussian Processes involves the kernel matrix and its Cholesky decomposition. Specifically, the equation is:

$$
\ln p(t \mid X) = - \dfrac{1}{2} \ln \lvert K \rvert - \dfrac{1}{2} t^T K^{-1} t - \dfrac{N}{2} \ln(2\pi)
$$

With the Cholesky decomposition, this becomes:

$$
\ln p(t \mid X) = - \sum_{i=1}^{N} \ln(L_{ii}) - \dfrac{1}{2} t^T \alpha - \dfrac{N}{2} \ln(2\pi)
$$

Where:
- $ K $ is the covariance matrix generated using our kernel function.
- $ L $ is the result of the Cholesky decomposition of $ K $.
- $ \alpha $ is a vector that is solved using the equation $ K \alpha = t $.

The hyperparameter $ \sigma $ in the kernel affects the elements of $ K $, and thus, indirectly affects our log likelihood.

Let's now implement the computation of the log likelihood for a range of $ \sigma $ values and visualize which $ \sigma $ gives the highest log likelihood:
''')

    st.code('''def log_likelihood(X, t, kernel, beta, **kernel_kwargs):
    """
    Calculate the data log likelihood.
    
    Args:
    - X: Training examples, shape (N, d).
    - t: Training targets, shape (N,).
    - kernel: Kernel function.
    - beta: Precision of the noise in the targets (scalar).
    - kernel_kwargs: Any additional arguments for the kernel function.

    Returns:
    - Log likelihood (scalar).
    """
    N, d = X.shape
    K_X_X = kernel(X, X, **kernel_kwargs) + (1/beta) * np.eye(N)  # Compute the kernel matrix with noise term
    L = np.linalg.cholesky(K_X_X)  # Cholesky decomposition for stable and efficient inversion
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))  # Solve for alpha
    
    # Compute the log likelihood
    loglik = - np.sum(np.log(np.diag(L))) - 0.5 * np.dot(t, alpha) - (N / 2) * np.log(2 * np.pi)
    
    return loglik
''', language='python')

    def log_likelihood(X, t, kernel, beta, **kernel_kwargs):
        """
        Calculate the data log likelihood.
        
        Args:
        - X: Training examples, shape (N, d).
        - t: Training targets, shape (N,).
        - kernel: Kernel function.
        - beta: Precision of the noise in the targets (scalar).
        - kernel_kwargs: Any additional arguments for the kernel function.
    
        Returns:
        - Log likelihood (scalar).
        """
        N, d = X.shape
        K_X_X = kernel(X, X, **kernel_kwargs) + (1/beta) * np.eye(N)  # Compute the kernel matrix with noise term
        L = np.linalg.cholesky(K_X_X)  # Cholesky decomposition for stable and efficient inversion
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))  # Solve for alpha
        
        # Compute the log likelihood
        loglik = - np.sum(np.log(np.diag(L))) - 0.5 * np.dot(t, alpha) - (N / 2) * np.log(2 * np.pi)
        
        return loglik
    
    

    st.code('''# Define a set of values for sigma
sigma_range = np.linspace(0.0001, 2.5, 100)

# Fix beta for the noise variance
beta = beta_assumed

# Calculate log likelihood for each sigma
logliks = []
for sigma in sigma_range:
    logliks.append(log_likelihood(X, t, rbf, beta, **{{}"sigma": sigma{}}))

# Plot the log likelihood against sigma
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(sigma_range, logliks, color="black")
ax.set_xlabel("$\sigma$ in the RBF kernel", size=15)
ax.set_ylabel("Log likelihood", size=15)

# Highlight the best sigma
best_idx = np.argmax(logliks)
best_sigma = sigma_range[best_idx]
best_loglik = logliks[best_idx]
ax.scatter([best_sigma], [best_loglik], color="red", s=80)
ax.grid(alpha=0.3)
plt.show()

# Print the best hyperparameter
print(f"The optimal value for sigma (with beta fixed at {{}beta{}}) is", best_sigma)
''', language='python')

    # Define a set of values for sigma
    sigma_range = np.linspace(0.0001, 2.5, 100)
    
    # Fix beta for the noise variance
    beta = beta_assumed
    
    # Calculate log likelihood for each sigma
    logliks = []
    for sigma in sigma_range:
        logliks.append(log_likelihood(X, t, rbf, beta, **{"sigma": sigma}))
    
    # Plot the log likelihood against sigma
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(sigma_range, logliks, color="black")
    ax.set_xlabel("$\sigma$ in the RBF kernel", size=15)
    ax.set_ylabel("Log likelihood", size=15)
    
    # Highlight the best sigma
    best_idx = np.argmax(logliks)
    best_sigma = sigma_range[best_idx]
    best_loglik = logliks[best_idx]
    ax.scatter([best_sigma], [best_loglik], color="red", s=80)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # Print the best hyperparameter
    st.write(f"The optimal value for sigma (with beta fixed at {beta}) is", best_sigma)
    

    st.markdown(r'''### Using the Best $ \sigma $ for Prediction

Having optimized $ \sigma $ using the log likelihood of our data, we'll now see its impact on the Gaussian Process Regression prediction. 

The optimal $ \sigma $ gives us a kernel that better captures the underlying data generation process. This means the GP regression should provide a smoother, more accurate prediction. 

Let's visualize this below:''')

    st.code('''# Set sigma to the optimized value
sigma_optimized = best_sigma 

# Initialize the figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot the true function for comparison
x_coords = np.linspace(-10, 10, 100)
true_function_vals = 0.08 * x_coords ** 2 + 2 * np.sin(x_coords) + 0.1 * np.tanh(x_coords ** 3)
ax.plot(x_coords, true_function_vals, 'r--', linewidth=2, label="True function")

# Plot the training data
ax.scatter(X.flatten(), t, c="k", s=30, label="Noisy observations")

# Predict using the Gaussian Process with the optimized sigma
predictions = [gp_reg(x.reshape(-1, 1), X, t, beta=beta_assumed, kernel=rbf, sigma=sigma_optimized) for x in x_coords]
predicted_means = [mean for mean, var in predictions]
predicted_stds = [np.sqrt(var) for mean, var in predictions]

# Plot the GP regression results
ax.plot(x_coords, predicted_means, 'b-', lw=2, label='Predicted Mean $m(x^*)$')
ax.fill_between(x_coords.flatten(), 
                np.array(predicted_means) - 2 * np.array(predicted_stds), 
                np.array(predicted_means) + 2 * np.array(predicted_stds), 
                color='blue', alpha=0.15, 
                label='Predicted $2 \sigma(x^*)$ interval')

# Set plot details
ax.set_title("Optimized Gaussian Process Regression with RBF Kernel")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc='upper left')

plt.show()
''', language='python')

    # Set sigma to the optimized value
    sigma_optimized = best_sigma 
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot the true function for comparison
    x_coords = np.linspace(-10, 10, 100)
    true_function_vals = 0.08 * x_coords ** 2 + 2 * np.sin(x_coords) + 0.1 * np.tanh(x_coords ** 3)
    ax.plot(x_coords, true_function_vals, 'r--', linewidth=2, label="True function")
    
    # Plot the training data
    ax.scatter(X.flatten(), t, c="k", s=30, label="Noisy observations")
    
    # Predict using the Gaussian Process with the optimized sigma
    predictions = [gp_reg(x.reshape(-1, 1), X, t, beta=beta_assumed, kernel=rbf, sigma=sigma_optimized) for x in x_coords]
    predicted_means = [mean for mean, var in predictions]
    predicted_stds = [np.sqrt(var) for mean, var in predictions]
    
    # Plot the GP regression results
    ax.plot(x_coords, predicted_means, 'b-', lw=2, label='Predicted Mean $m(x^*)$')
    ax.fill_between(x_coords.flatten(), 
                    np.array(predicted_means) - 2 * np.array(predicted_stds), 
                    np.array(predicted_means) + 2 * np.array(predicted_stds), 
                    color='blue', alpha=0.15, 
                    label='Predicted $2 \sigma(x^*)$ interval')
    
    # Set plot details
    ax.set_title("Optimized Gaussian Process Regression with RBF Kernel")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    
    

    st.markdown(r'''
By optimizing the kernel hyperparameter $ \sigma $ and then utilizing it in our Gaussian Process Regression, we should witness a more accurate and smoother representation of the underlying function.

---''')

    st.markdown(r'''### Gaussian Processes in Astrophysical Research: Quasar Light Curves

Astrophysics, in its quest to unravel the intricacies of the universe, routinely grapples with vast and complex datasets. Among the array of tools at its disposal, Gaussian Processes have emerged as particularly instrumental, especially when dealing with time-series observations known as "light curves." These curves plot the luminosity of celestial objects against time, providing insights into their underlying physical processes.

Quasars, distinguished by their immense luminosity and energy output, serve as prime subjects for such investigations. Their luminous signatures emanate from material accreting onto supermassive black holes at galactic centers. The variability in this emitted light is a direct consequence of the dynamic nature of the accretion process, which is modulated by an array of factors inherent to the accretion disk. Hence, an in-depth analysis of quasar light curves can yield profound insights into the complex physics underpinning these astronomical phenomena.

The variable nature of quasar luminosity presents both a formidable challenge and a rewarding research avenue for astrophysicists. GPs, with their capacity to discern intricate correlations in time-series data, offer a powerful solution. Their non-parametric nature permits GPs to capture nuanced patterns without confinement to a specific functional form, rendering them highly effective for quasar light curve analyses. Notably, while specialized kernels, exemplified by the autoregressive model delineated in seminal studies such as [Kelley et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...698..895K/abstract), have been tailored for such tasks, this tutorial will employ the universally-applicable RBF kernel for simplicity and broader understanding.

In the following, we will study mock light curves, generated with the autoregressive paradigms of Kelley et al. (2009), our endeavor is to ascertain the characteristic timescales over which quasar luminosities vary, harnessing the capabilities of Gaussian Processes. To embark on this exploration, let's start by visually inspecting one of these captivating light curves.
''')

    st.code('''# Loading the dataset
temp = np.load("./quasar_light_curve_tutorial_week10b.npz")
time = temp["time"]
light_curve = temp["light_curve"]

# For our initial exploration, we'll focus on one light curve
selected_light_curve = light_curve[0]
corresponding_time = time[0]

# Plotting this selected light curve to visualize its structure
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(corresponding_time, selected_light_curve, c="k", s=3, label="Observations from Quasar")
ax.set_title("Light Curve of a Quasar")
ax.set_xlabel("Time [day]")
ax.set_ylabel("Brightness [mag]")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()''', language='python')

    # Loading the dataset
    import requests
    from io import BytesIO
    
    # Load the dataset using np.load
    response = requests.get('https://storage.googleapis.com/compute_astro/quasar_light_curve_tutorial_week10b.npz')
    f = BytesIO(response.content)
    temp = np.load(f, allow_pickle=True)

    time = temp["time"]
    light_curve = temp["light_curve"]
    
    # For our initial exploration, we'll focus on one light curve
    selected_light_curve = light_curve[0]
    corresponding_time = time[0]
    
    # Plotting this selected light curve to visualize its structure
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(corresponding_time, selected_light_curve, c="k", s=3, label="Observations from Quasar")
    ax.set_title("Light Curve of a Quasar")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Brightness [mag]")
    ax.legend(loc='upper right')
    st.pyplot(fig)
    

    st.markdown(r'''### Hyperparameter Optimization for Quasar Light Curve Analysis

In machine learning applications, the efficacy of the model often hinges on the judicious selection of hyperparameters. For quasar light curve analysis, these hyperparameters are of paramount importance, impacting the model's capability to accurately discern and represent brightness fluctuations.

To ensure computational efficiency while preserving the integrity of our dataset, we have employed a strategy of subsampling. By selecting every 10th timestamp, we strike a balance between computational feasibility and data representation. An intrinsic characteristic of quasars is that their significant brightness variations tend to span periods longer a day. Our original dataset, with its granularity of 0.1 day per sample, could introduce computational challenges without proportionate benefits in model accuracy.

The rationale behind subsampling is twofold:
1. **Computational Efficiency**: By reducing the data size, computational resources are optimized.
2. **Model Evaluation**: The subset provides a practical benchmark to assess the performance of the Gaussian Process.

Our mock dataset, like most real-world data, is subject to noise. The data set assumes a noise level of 0.01 mag. This gives rise to a precision, $ \beta $, set at 10,000. Accounting for this noise is crucial; overlooking it risks conflating random noise with genuine quasar brightness variations.

The RBF kernel, integral to our Gaussian Process, operates with a pivotal hyperparameter, $ \sigma $. This parameter delineates the function's "time scale". A smaller $ \sigma $ focuses on capturing localized variations, while a larger value offers a more generalized view. The objective is to identify an optimal $ \sigma $ value. For this, the log likelihood is employed as a guiding metric. By optimizing $ \sigma $ to maximize the log likelihood, the Gaussian Process is fine-tuned to best represent the quasar data.

Let's now proceed to the detailed implementation of the search of hyperparameters.

''')

    st.code('''# Subsample the data for efficient hyperparameter search
t_sub = corresponding_time[::10]
lc_sub = selected_light_curve[::10]

# Define a range of values for sigma to search over
log_sigma_range = np.linspace(1, 3, 100)
sigma_range = 10.**log_sigma_range

# Fix beta for the noise variance, assuming uncertainty of 0.01 mag
beta = (1/0.01)**2

# Calculate the log likelihood for each sigma
logliks = []
for sigma in sigma_range:
    logliks.append(log_likelihood(t_sub[:, None], lc_sub, rbf, beta, **{{}"sigma": sigma{}}))

# Plotting the log likelihood against the sigma values
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(sigma_range, logliks, color="black")
ax.set_xlabel("$\sigma$ in the RBF kernel", size=15)
ax.set_ylabel("Log likelihood", size=15)

# Identify and highlight the best sigma value
best_idx = np.argmax(logliks)
best_sigma = sigma_range[best_idx]
ax.scatter([best_sigma], [logliks[best_idx]], color="red", s=80)

ax.grid(alpha=0.3)
plt.show()

print(f"The optimal value for sigma is approximately: {{}best_sigma:.2f{}} days")
''', language='python')

    # Subsample the data for efficient hyperparameter search
    t_sub = corresponding_time[::10]
    lc_sub = selected_light_curve[::10]
    
    # Define a range of values for sigma to search over
    log_sigma_range = np.linspace(1, 3, 100)
    sigma_range = 10.**log_sigma_range
    
    # Fix beta for the noise variance, assuming uncertainty of 0.01 mag
    beta = (1/0.01)**2
    
    # Calculate the log likelihood for each sigma
    logliks = []
    for sigma in sigma_range:
        logliks.append(log_likelihood(t_sub[:, None], lc_sub, rbf, beta, **{"sigma": sigma}))
    
    # Plotting the log likelihood against the sigma values
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(sigma_range, logliks, color="black")
    ax.set_xlabel("$\sigma$ in the RBF kernel", size=15)
    ax.set_ylabel("Log likelihood", size=15)
    
    # Identify and highlight the best sigma value
    best_idx = np.argmax(logliks)
    best_sigma = sigma_range[best_idx]
    ax.scatter([best_sigma], [logliks[best_idx]], color="red", s=80)
    
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    st.write(f"The optimal value for sigma is approximately: {best_sigma:.2f} days")
    

    st.markdown(r'''Our Gaussian Process regression, utilizing the RBF kernel, indicates an optimal timescale of approximately 20 days. This result aligns with anticipated timescales based on previous studies and underscores the validity of the model within its constraints. 

It is crucial to highlight, however, that quasars are complex astronomical objects, influenced by a multitude of physical processes operating on varying timescales. The multiple peaks observed in our log likelihood curve suggest the presence of other significant timescales, indicating the multifaceted nature of quasar variability.

While the RBF kernel captures the dominant timescale effectively, a more comprehensive model might better represent the complexity of quasars. For the purposes of this tutorial, we will proceed with the identified dominant timescale from the RBF kernel.

With the optimal timescale identified, we will now employ our model to predict the general trend of the quasar light curve. The following section will visualize and evaluate the model's performance against the observed data.
''')

    st.code('''# Extracting the optimal sigma value with single decimal precision
print(f"The optimal value for sigma (with beta fixed at %d) is %.1f days" % (beta, best_sigma))

# Setting sigma to the deduced optimal value
sigma_optimized = best_sigma

# Initializing the plot canvas
fig, ax = plt.subplots(figsize=(7, 5))

# Displaying the observed data points
ax.scatter(corresponding_time, selected_light_curve, c="k", s=3, label="Observations from Quasar")

# Generating predictions using the Gaussian Process with the optimal sigma
predictions = [gp_reg(t.reshape(-1, 1), t_sub[:,None], lc_sub, beta=beta, kernel=rbf, sigma=sigma_optimized) for t in t_sub]

# Extracting means and standard deviations from predictions
predicted_means = [mean for mean, var in predictions]
predicted_stds = [np.sqrt(var) for mean, var in predictions]

# Rendering the Gaussian Process regression outcome on the plot
ax.plot(t_sub, predicted_means, 'b-', lw=2, label='Predicted Mean $m(x^*)$')
ax.fill_between(t_sub.flatten(),
                np.array(predicted_means) - 2 * np.array(predicted_stds),
                np.array(predicted_means) + 2 * np.array(predicted_stds),
                color='blue', alpha=0.15,
                label='Predicted $2 \sigma(x^*)$ interval')

# Configuring plot attributes
ax.set_title("Optimized Gaussian Process Regression with RBF Kernel")
ax.set_xlabel("Time [day]")
ax.set_ylabel("Brightness [mag]")
ax.legend(loc='upper left')

# Displaying the plot
plt.tight_layout()
''', language='python')

    # Extracting the optimal sigma value with single decimal precision
    st.write(f"The optimal value for sigma (with beta fixed at %d) is %.1f days" % (beta, best_sigma))
    
    # Setting sigma to the deduced optimal value
    sigma_optimized = best_sigma
    
    # Initializing the plot canvas
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Displaying the observed data points
    ax.scatter(corresponding_time, selected_light_curve, c="k", s=3, label="Observations from Quasar")
    
    # Generating predictions using the Gaussian Process with the optimal sigma
    predictions = [gp_reg(t.reshape(-1, 1), t_sub[:,None], lc_sub, beta=beta, kernel=rbf, sigma=sigma_optimized) for t in t_sub]
    
    # Extracting means and standard deviations from predictions
    predicted_means = [mean for mean, var in predictions]
    predicted_stds = [np.sqrt(var) for mean, var in predictions]
    
    # Rendering the Gaussian Process regression outcome on the plot
    ax.plot(t_sub, predicted_means, 'b-', lw=2, label='Predicted Mean $m(x^*)$')
    ax.fill_between(t_sub.flatten(),
                    np.array(predicted_means) - 2 * np.array(predicted_stds),
                    np.array(predicted_means) + 2 * np.array(predicted_stds),
                    color='blue', alpha=0.15,
                    label='Predicted $2 \sigma(x^*)$ interval')
    
    # Configuring plot attributes
    ax.set_title("Optimized Gaussian Process Regression with RBF Kernel")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Brightness [mag]")
    ax.legend(loc='upper left')
    
    # Displaying the plot
    st.pyplot(fig)
    
    

    st.markdown(r'''### Systematic Exploration of Optimal Timescales for Quasar Light Curves

Having studied and identified the optimal timescale for one particular quasar light curve, we now expand our scope. The cosmos is replete with quasars, and understanding the typical variability timescales for a representative sample can provide invaluable insights into the general processes governing these astronomical objects. Each quasar, although governed by common underlying principles, might exhibit unique variabilities due to its specific environment and historical events.

To systematically study these timescales, we'll compute the optimal $ \sigma $ for each quasar light curve in our dataset. This endeavor will not only provide a distribution of optimal timescales but also enable us to discern any common patterns or outliers among them. The following function encapsulates this process, iterating over all the light curves and identifying the optimal $ \sigma $ value for each.

By executing below, we can extract the optimal $ \sigma $ value for each quasar in our dataset, enabling a comprehensive analysis of the inherent variability timescales. As we will see the quasar variability can span from days to years, which is what we expect.''')

    st.code('''from tqdm import tqdm

def find_optimal_sigma_for_all_light_curves(time_data, light_curve_data, sigma_range, beta):
    """
    Calculate the optimal sigma value for each light curve using log likelihood.
    
    Args:
        time_data: A 2D array where each row corresponds to time points of a light curve.
        light_curve_data: A 2D array where each row corresponds to brightness values of a light curve.
        sigma_range: A list or array of sigma values to search over.
        beta: Precision of the noise in the targets.
        
    Returns:
        A list of optimal sigma values for each light curve.
    """
    optimal_sigmas = []
    
    for t, lc in tqdm(list(zip(time_data, light_curve_data))):
        t_sub = t[::10]
        lc_sub = lc[::10]

        logliks = [log_likelihood(t_sub[:, None], lc_sub, rbf, beta, **{{}"sigma": sigma{}}) for sigma in sigma_range]
        best_sigma = sigma_range[np.argmax(logliks)]
        optimal_sigmas.append(best_sigma)
    
    return optimal_sigmas

# Calculate the optimal sigma values for all light curves
all_optimal_sigmas = find_optimal_sigma_for_all_light_curves(time, light_curve, sigma_range, beta)
''', language='python')

    from tqdm import tqdm
    
    def find_optimal_sigma_for_all_light_curves(time_data, light_curve_data, sigma_range, beta):
        """
        Calculate the optimal sigma value for each light curve using log likelihood.
        
        Args:
            time_data: A 2D array where each row corresponds to time points of a light curve.
            light_curve_data: A 2D array where each row corresponds to brightness values of a light curve.
            sigma_range: A list or array of sigma values to search over.
            beta: Precision of the noise in the targets.
            
        Returns:
            A list of optimal sigma values for each light curve.
        """
        optimal_sigmas = []
        
        for t, lc in tqdm(list(zip(time_data, light_curve_data))):
            t_sub = t[::10]
            lc_sub = lc[::10]
    
            logliks = [log_likelihood(t_sub[:, None], lc_sub, rbf, beta, **{"sigma": sigma}) for sigma in sigma_range]
            best_sigma = sigma_range[np.argmax(logliks)]
            optimal_sigmas.append(best_sigma)
        
        return optimal_sigmas
    
    # Calculate the optimal sigma values for all light curves
    all_optimal_sigmas = find_optimal_sigma_for_all_light_curves(time, light_curve, sigma_range, beta)
    
    

    st.code('''# Plotting the histogram of optimal sigma values
plt.hist(np.log10(all_optimal_sigmas), bins=20)

# Configuring plot attributes
plt.xlabel("Log Optimal Sigma [day]")
plt.ylabel("Number of Light Curves")
''', language='python')

    fig, ax = plt.subplots()
    
    # Plotting the histogram of optimal sigma values
    ax.hist(np.log10(all_optimal_sigmas), bins=20)
    
    # Configuring plot attributes
    ax.set_xlabel("Log Optimal Sigma [day]")
    ax.set_ylabel("Number of Light Curves")
    st.pyplot(fig)
    
    

    st.markdown(r'''---

### Conclusion

In this tutorial, we delved deeply into the intricacies of Gaussian Processes, a methodology that elegantly combines the precision of statistical modeling with the adaptability inherent in non-parametric techniques.

- **Understanding Gaussian Processes**: We rigorously explored the foundations and principles of GPs, transitioning from high-level mathematical constructs to tangible, applicable concepts. GPs, anchored in the philosophy of distributions over functions, tap into infinite-dimensional spaces, establishing themselves as an intricate yet lucid solution for regression tasks. Their strength is exemplified in their ability to intuitively model and forecast intricate patterns, unbounded by predetermined functional constraints.

- **The Transition from Theory to Implementation**: Our journey encompassed the hands-on implementation of GP regression. We dissected the kernel method, learned to forecast outputs at previously unobserved points using the GP framework through conditional distributions, and honed our skills in hyperparameter optimization by maximizing the marginal likelihood.

- **Astronomical Applications**: Our application to quasar light curves transcended mere academic exercise, vividly showcasing the practical potency of GPs. Our foray into astrophysics underscored the versatility of GPs, illustrating their capability to decode complex celestial phenomena. The study not only allowed us to model but also to extract pivotal insights, such as identifying the dominant timescales influencing quasar luminosity variations.

To encapsulate, Gaussian Processes emerge as an indispensable asset for data analysts and scientists. They seamlessly weave theoretical foundations with tangible applications, presenting a formidable toolset to address multifaceted challenges. As we conclude this tutorial, our aspiration is for you to not only internalize the acquired knowledge and techniques but also to recognize and respect the sophistication and immense potential housed within Gaussian Processes.

''')

if __name__ == '__main__':
    show_page()
