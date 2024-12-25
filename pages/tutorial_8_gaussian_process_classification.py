import streamlit as st
from streamlit_app import navigation_menu

def show_page():
    st.set_page_config(page_title="Comp Astro",
                       page_icon="./image/tutor_favicon.png", layout="wide")

    navigation_menu()
    st.markdown(r'''# Gaussian Processes Classification''')

    st.markdown(r'''In the previous lab, we explored the depths of Gaussian Process Regression (GPR), equipping ourselves with the tools to model continuous data. Now, as we set our sights on the discrete universe of data, we'll uncover the elegance and utility of Gaussian Process Classification (GPC).

### Prerequisite Knowledge:
- **Multivariate Gaussians**: We will frequently encounter joint, marginal, and conditional distributions. If you need a refresher or more detailed mathematical insights, Chapter 2, Section 2.3 of Bishop's book is an excellent reference.
  
- **Kernels and the Kernel Trick**: The magic behind the scene in many machine learning algorithms, including Gaussian processes. Understanding the kernel trick will enable us to project our data into high-dimensional spaces effortlessly.

- **Two Views of Supervised Learning**: In our journey, we will often oscillate between two perspectives – the parameter space, which concerns the weights and biases in traditional machine learning models, and the function space, where we view things from the standpoint of functions.

- **Application of Kernels in Bayesian Regression and Classification**: To enhance our GP models, we will harness the power of kernels in the Bayesian context, bringing more flexibility and robustness.

### The Strengths of Gaussian Process Classification:

- **Non-parametric Nature**: One of the crowning glories of Gaussian Processes is their non-parametric nature. This means GPs aren't bound by a predefined model structure, allowing them to flexibly adapt to the data they're modeling.

- **Soft Predictions**: Instead of giving hard classifications, Gaussian Processes provide a probability distribution over possible classes. This offers a nuanced view of the model's confidence in its predictions.

- **Connection to GPR**: Gaussian Process Classification can be thought of as a natural extension of GPR. At its core, GPC uses GPR and then passes the predictions through a "squashing function" to ensure they lie between 0 and 1, making them valid probabilities.

### Goals for this Lab:
By the culmination of this tutorial:
- **Master Gaussian Process Classification**: Understand its foundational principles, discern its differences and similarities with GPR, and recognize the contexts where it shines.

- **Hands-on Implementation**: Transition from theoretical knowledge to practical application by implementing Gaussian Process Classification.

- **Result Interpretation**: Build the acumen to meticulously analyze, visualize, and derive meaningful inferences from the outcomes of Gaussian Process Classification.

With the groundwork set in Gaussian Process Regression, let's traverse the enlightening pathway of Gaussian Process Classification, exploring its intricacies and power."
''')

    st.markdown(r'''Setting up the environment''')

    st.code('''import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

%matplotlib inline''', language='python')

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import scipy
    
    

    st.markdown(r'''--- 

### Gaussian Process Classification: A Toy Example

To deeply understand the mechanics of Gaussian Process Classification, it's invaluable to start with a toy example. This controlled exploration not only allows us to grasp the foundational concepts but also prepares us for intricate real-world datasets.

In a binary classification setting, we're dealing with data points that can belong to one of two distinct classes. The task's crux lies in identifying or drawing a decision boundary, which acts as a separator between these classes.

Our proposed decision boundary is given by:

$$
3x^{2}+3y^{3}-2y^{2}-4xy-0.5x^{2}y-2=0.
$$

This equation describes a curve in the feature space that theoretically divides our two classes. While the boundary provides a hard division—points on one side belong to one class and points on the other side to another class—in real-world scenarios, there's often ambiguity. Some points could be on the fence, so to speak, and may not strongly belong to one class or the other.

This is where our sigmoid function comes into play. Instead of providing a hard assignment of class labels, the sigmoid allows for a soft assignment. It takes values from the decision boundary and squashes them between 0 and 1. This "squashed" value can be interpreted as the probability of a data point belonging to a particular class. For instance, if the sigmoid output is 0.7 for a data point, it suggests that the point has a 70% chance of belonging to class 1 and a 30% chance of being in class 0. This probabilistic interpretation is a hallmark of Gaussian Process Classification and offers a nuanced view of classification tasks.

With this foundation, our objective now becomes twofold:

1. Visualize the true decision boundary to understand its shape and complexity.
2. Generate a synthetic dataset using the given boundary and apply soft assignments using the sigmoid function to simulate real-world uncertainty in data labeling.

Let's embark on this exploration.
''')

    st.code('''import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Sigmoid function to squash values between 0 and 1
def sigmoid(a):
    return scipy.special.expit(a)

# True decision function based on the provided equation
def true_fn(x, y, binary=True):
    a = 3*x**2 + 3*y**3 - 2*y**2 - 4*x*y - 0.5*x**2*y - 2 
    if not binary:
        return a
    else:
        # Convert continuous value into binary class (0 or 1)
        return (a > 0).astype(int)

# Generate synthetic dataset with points belonging to two classes
def generate_classification_data(N=20):
    np.random.seed(42)
    x1 = np.random.uniform(low=-3, high=3, size=N)
    x2 = np.random.uniform(low=-3, high=3, size=N)
    
    # Determine true labels and introduce some noise for realism
    labels = true_fn(x1, x2, binary=False)
    labels = sigmoid(labels)
    labels_binary = np.array([np.random.binomial(1, p) for p in labels])
    
    return np.vstack([x1, x2]).T, labels_binary
''', language='python')

    import scipy.special
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Sigmoid function to squash values between 0 and 1
    def sigmoid(a):
        return scipy.special.expit(a)
    
    # True decision function based on the provided equation
    def true_fn(x, y, binary=True):
        a = 3*x**2 + 3*y**3 - 2*y**2 - 4*x*y - 0.5*x**2*y - 2 
        if not binary:
            return a
        else:
            # Convert continuous value into binary class (0 or 1)
            return (a > 0).astype(int)
    
    # Generate synthetic dataset with points belonging to two classes
    def generate_classification_data(N=20):
        np.random.seed(42)
        x1 = np.random.uniform(low=-3, high=3, size=N)
        x2 = np.random.uniform(low=-3, high=3, size=N)
        
        # Determine true labels and introduce some noise for realism
        labels = true_fn(x1, x2, binary=False)
        labels = sigmoid(labels)
        labels_binary = np.array([np.random.binomial(1, p) for p in labels])
        
        return np.vstack([x1, x2]).T, labels_binary
    
    

    st.code('''# Plot the decision boundary and the synthetic data points
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# Custom colormap
my_gradient = LinearSegmentedColormap.from_list('my_gradient', (
                 (0.000, (255, 255, 255)),
                 (0.100, (0, 0, 0)),
                 (0.500, (0, 0, 0)),
                 (0.900, (0, 0, 0)),
                 (1.000, (255, 255, 255))))

# Visualize the decision boundary
N = 100
x1s = np.linspace(-3, 3, num=N)
x2s = np.linspace(-3, 3, num=N)
x1, x2 = np.meshgrid(x1s, x2s)
f = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        f[i,j] = true_fn(x1[i, j], x2[i, j])
cb = ax.contourf(x1, x2, f, cmap=my_gradient, alpha=1)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13, rotation=0)
ax.set_aspect('equal')
ax.set_ylim([-3, 3])
ax.set_xlim([-3, 3])

# Plot the generated synthetic data points
X, t = generate_classification_data(N=200)
ax.scatter(X[t==0, 0], X[t==0, 1], s=30, facecolors='none', edgecolors='b', marker="o", alpha=0.8)
ax.scatter(X[t==1, 0], X[t==1, 1], s=30, marker="x", color="red", alpha=0.8)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_aspect('equal')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_title("True Decision Boundary and Sampled Data Points")
plt.show()''', language='python')

    # Plot the decision boundary and the synthetic data points
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    # Custom colormap
    my_gradient = LinearSegmentedColormap.from_list('my_gradient', (
                     (0.000, (255, 255, 255)),
                     (0.100, (0, 0, 0)),
                     (0.500, (0, 0, 0)),
                     (0.900, (0, 0, 0)),
                     (1.000, (255, 255, 255))))
    
    # Visualize the decision boundary
    N = 100
    x1s = np.linspace(-3, 3, num=N)
    x2s = np.linspace(-3, 3, num=N)
    x1, x2 = np.meshgrid(x1s, x2s)
    f = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            f[i,j] = true_fn(x1[i, j], x2[i, j])
    cb = ax.contourf(x1, x2, f, cmap=my_gradient, alpha=1)
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13, rotation=0)
    ax.set_aspect('equal')
    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])
    
    # Plot the generated synthetic data points
    X, t = generate_classification_data(N=200)
    ax.scatter(X[t==0, 0], X[t==0, 1], s=30, facecolors='none', edgecolors='b', marker="o", alpha=0.8)
    ax.scatter(X[t==1, 0], X[t==1, 1], s=30, marker="x", color="red", alpha=0.8)
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.set_aspect('equal')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_title("True Decision Boundary and Sampled Data Points")
    st.pyplot(fig)
    

    st.markdown(r'''
Through this exercise, we have visualized the inherent complexity of the decision boundary and seen how data points can be distributed on both sides. The next steps involve employing Gaussian Process Classification to discern this boundary based on the provided data points.''')

    st.markdown(r'''### Delving into Gaussian Process Classification Algorithm

We now pivot to the mainstay of this exercise: the Gaussian Process Classification (GPC) algorithm. We'll employ the Laplace approximation scheme—a technique for approximating the mode of a function. This is the foundation of our classification approach, and understanding its workings will be pivotal to mastering GPC.

#### MAP Estimation: $a_{\mathrm{MAP}}$

At the core of our approach is the estimation of $a_{\mathrm{MAP}}$. Given a kernel matrix $K \in \mathbb{R}^{N \times N}$ and a target vector $t \in \mathbb{R}^{N}$, the mode of the distribution $p(a \mid X, t)$ can be represented as:

$$
a_{\mathrm{MAP}} = K (t - \sigma(a_{\mathrm{MAP}})).
$$

What's intriguing here is that $a_{\mathrm{MAP}}$ is implicitly defined—it appears on both sides of the equation. This self-referential nature means there's typically no direct formula for its solution. Instead, we approach it using the iterative method known as fixed-point iteration.

Let's delve deeper into the concept of the fixed-point iteration method.

---

### The Fixed-Point Iteration Method: A Closer Look

The fixed-point iteration method is a powerful and classic numerical technique to approximate solutions to equations of the form $ x = f(x) $. In this format, the value of $ x $ that makes the equation hold true is termed the "fixed point" of the function $ f $.

The crux of the method is rooted in iterative refinement: starting with an initial guess $ x_0 $, we apply the function $ f $ repeatedly to generate a sequence of approximations:

$$
 x_1 = f(x_0) 
$$
$$
 x_2 = f(x_1) 
$$
$$
 x_3 = f(x_2) 
$$
$$
 \vdots 
$$

The hope is that, under certain conditions on $ f $, this sequence will converge to the true fixed point $ x^* $ as the number of iterations increases.

The method's convergence hinges on a few factors:

1. **The function $ f $:** For a region around the true fixed point $ x^* $, $ f $ must be continuous and possess a derivative whose absolute value is less than 1. This ensures that the iterations don't spiral out of control.
2. **The initial guess $ x_0 $:** Even if $ f $ satisfies the above criteria, an inappropriate initial guess can lead to divergent behavior. Hence, often, some prior insight or repeated trials with different starting points might be necessary.

#### Connection to our Problem

In our context, the fixed-point iteration is leveraged to solve the implicit equation:

$$
 a_{\mathrm{MAP}} = K (t - \sigma(a_{\mathrm{MAP}})) 
$$

The challenge arises from the fact that $ a_{\mathrm{MAP}} $ appears on both sides, making it impossible to isolate and directly compute. The fixed-point iteration offers a path to approximate $ a_{\mathrm{MAP}} $ without needing an explicit solution.

Starting with an initial guess for $ a_{\mathrm{MAP}} $, we iterate using the function on the right-hand side. Over successive iterations, the method refines its estimate of $ a_{\mathrm{MAP}} $ until the values stabilize or the change between iterations falls below a predefined threshold. We will adopt the `scipy.optimize` package to perform such a search.
''')

    st.code('''def f_a_map(a, K, t):
    """
    Calculates the right-hand side of the aforementioned equation.
    """
    return K @ (t - sigmoid(a))

def get_a_map(K, t): 
    """
    Uses the fixed-point iteration to approximate a_MAP.
    """
    return scipy.optimize.fixed_point(
        f_a_map, 
        x0=np.random.rand(t.shape[0]),  # Initiating with a random point
        args=(K, t),  
        xtol=1e-03,
        maxiter=10000,
    )''', language='python')

    def f_a_map(a, K, t):
        """
        Calculates the right-hand side of the aforementioned equation.
        """
        return K @ (t - sigmoid(a))
    
    def get_a_map(K, t): 
        """
        Uses the fixed-point iteration to approximate a_MAP.
        """
        return scipy.optimize.fixed_point(
            f_a_map, 
            x0=np.random.rand(t.shape[0]),  # Initiating with a random point
            args=(K, t),  
            xtol=1e-03,
            maxiter=10000,
        )
    
    

    st.markdown(r'''--- 

### Constructing Gaussian Process Classification

Our groundwork with $a_{\mathrm{MAP}}$ enables us to derive the predictive distribution for GPC. 
In the context of our GPC, recall that the predictive distribution for a new data point $ x^* $ is given by:

$$
 p(t^* = 1 | x^*, {\bf X}, {\bf t}) = \sigma \big( ( 1 + {\pi d^2}/{8} )^{-1/2} \, c \big) 
$$

Where $ c $ and $ d^2 $ are parameters defined by:

$$
 c = k({\bf x}^*, {\bf X}) \big({\bf t} - \sigma({\bf a}_{\rm MAP})\big) 
$$
$$
 d^2 = k({\bf x}^*, {\bf x}^*) - k({\bf x}^*,{\bf X}) \, \Big({\rm diag}\big\{ \sigma({\bf a}_{\rm MAP})(1-\sigma({\bf a}_{\rm MAP})) \big\}^{-1}  + k({\bf X},{\bf X}) \Big)^{-1} \, k({\bf X},{\bf x}^*) 
$$

These equations might appear intricate initially. However, they lay the foundation for understanding how we determine the probability of a new data point belonging to a particular class.

Let's unpack the accompanying code, translating these equations into actionable Python functions.
 
''')

    st.code('''def gp_classif(x_star, X, t, kernel, **kernel_kwargs):
    """
    Given a test point x_star, this function computes the mean prediction using Gaussian Process Classification.
    
    Args:
    - x_star (ndarray): The test point, of shape (1, d).
    - X (ndarray): The training examples, of shape (N, d).
    - t (ndarray): The training targets, of shape (N,).
    - kernel (function): The kernel function to compute similarity.
    - **kernel_kwargs (dict): Additional arguments for the kernel function.
    
    Returns:
    - float: The predicted probability of the test point x_star belonging to class 1.
    """
    N, d = X.shape
    
    # Constructing the kernels
    K_X_X = kernel(X, X, **kernel_kwargs) 
    K_x_star_X = kernel(x_star, X, **kernel_kwargs)
    K_x_star_x_star = kernel(x_star, x_star, **kernel_kwargs)
    
    # Obtaining the mode of p(a | X, t)
    a_map = get_a_map(K_X_X, t)
    
    # Calculating c and d^2 parameters
    c = K_x_star_X @ (t - sigmoid(a_map))
    d_sq = K_x_star_x_star.flatten()[0] - \
           K_x_star_X @ np.linalg.inv(np.diag(1./(sigmoid(a_map) * (1 - sigmoid(a_map)))) + K_X_X) @ K_x_star_X.T
    
    return sigmoid(np.sqrt(1./(1 + np.pi * d_sq / 8)) * c).flatten()[0]
''', language='python')

    def gp_classif(x_star, X, t, kernel, **kernel_kwargs):
        """
        Given a test point x_star, this function computes the mean prediction using Gaussian Process Classification.
        
        Args:
        - x_star (ndarray): The test point, of shape (1, d).
        - X (ndarray): The training examples, of shape (N, d).
        - t (ndarray): The training targets, of shape (N,).
        - kernel (function): The kernel function to compute similarity.
        - **kernel_kwargs (dict): Additional arguments for the kernel function.
        
        Returns:
        - float: The predicted probability of the test point x_star belonging to class 1.
        """
        N, d = X.shape
        
        # Constructing the kernels
        K_X_X = kernel(X, X, **kernel_kwargs) 
        K_x_star_X = kernel(x_star, X, **kernel_kwargs)
        K_x_star_x_star = kernel(x_star, x_star, **kernel_kwargs)
        
        # Obtaining the mode of p(a | X, t)
        a_map = get_a_map(K_X_X, t)
        
        # Calculating c and d^2 parameters
        c = K_x_star_X @ (t - sigmoid(a_map))
        d_sq = K_x_star_x_star.flatten()[0] - \
               K_x_star_X @ np.linalg.inv(np.diag(1./(sigmoid(a_map) * (1 - sigmoid(a_map)))) + K_X_X) @ K_x_star_X.T
        
        return sigmoid(np.sqrt(1./(1 + np.pi * d_sq / 8)) * c).flatten()[0]
    
    

    st.markdown(r'''Once the Gaussian Process Classification machinery is in place, the true test lies in its application. By evaluating sample points, we can observe how the GPC assigns probabilities, effectively classifying new, unseen data based on the patterns learned from the training dataset.''')

    st.code('''# define the RBF kernel
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
    return np.exp(-sqdist / (2 * sigma ** 2))''', language='python')

    # define the RBF kernel
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
    
    

    st.code('''x_star = np.array([[0, -2]])
print("For the data point: [0, -2]")
print("Predicted probability of belonging to class 1:", gp_classif(x_star, X, t, rbf))

x_star = np.array([[2.5, 2]])
print("For the data point: [2.5, 2]")
print("Predicted probability of belonging to class 1:", gp_classif(x_star, X, t, rbf))''', language='python')

    x_star = np.array([[0, -2]])
    st.write("For the data point: [0, -2]")
    st.write("Predicted probability of belonging to class 1:", gp_classif(x_star, X, t, rbf))
    
    x_star = np.array([[2.5, 2]])
    st.write("For the data point: [2.5, 2]")
    st.write("Predicted probability of belonging to class 1:", gp_classif(x_star, X, t, rbf))
    
    

    st.markdown(r'''The outputs provide probabilistic assignments to each class, showcasing the strength of Gaussian Process Classification in modeling uncertainty. As we continue, it's pivotal to appreciate this nuance—a blend of deterministic patterns and probabilistic insights, which makes GPC so versatile and powerful in real-world applications.

While we computed $ a_{\mathrm{MAP}} $ for a single data point, this mode of the posterior distribution need only be calculated once. With this in hand, we can then compute the predictive probabilities for any number of points.

In the code below, we will vectorise the calculation over each point in `X_star` and compute the predicted probabilities, we will also use the Cholesky decomposition, which is often a numerically stable way to invert the positive definite matrix like the one in the GPC setting. It's used to derive the `d_sq` values.

''')

    st.code('''def gp_classif_batch(X_star, X, t, kernel, **kernel_kwargs):
    """
    Given a test point x_star, this function computes the mean prediction using Gaussian Process Classification.
    
    Args:
    - X_star: new examples, of shape (M, d), where M is number of test examples
    - X (ndarray): The training examples, of shape (N, d).
    - t (ndarray): The training targets, of shape (N,).
    - kernel (function): The kernel function to compute similarity.
    - **kernel_kwargs (dict): Additional arguments for the kernel function.
    
    Returns:
    - ndarray: The predicted probability of the test points X_star belonging to class 1.
    """

    N, _ = X.shape
    
    # Kernels
    K_X_X = kernel(X, X, **kernel_kwargs) 
    
    # Get the mode of p(a | X, t)
    a_map = get_a_map(K_X_X, t)
    sigmoid_a_map = sigmoid(a_map)
    
    # Compute kernels for all points in X_star with respect to X
    K_x_star_X = kernel(X_star, X, **kernel_kwargs)
    K_x_star_x_star = np.diagonal(kernel(X_star, X_star, **kernel_kwargs)).reshape(-1, 1)
    
    # Compute intermediate matrices for d_sq calculation
    L = np.linalg.cholesky(np.diag(sigmoid_a_map * (1 - sigmoid_a_map)) + K_X_X)
    L_inv_K_x_star_X_T = np.linalg.solve(L, K_x_star_X.T)
    
    # Compute c and d_sq for all x_star points
    c = K_x_star_X @ (t - sigmoid_a_map)
    d_sq = K_x_star_x_star.flatten() - np.sum(L_inv_K_x_star_X_T**2, axis=0)
    
    # Return predicted probabilities for all x_star points
    return sigmoid(np.sqrt(1./(1 + np.pi * d_sq / 8)) * c).flatten()
''', language='python')

    def gp_classif_batch(X_star, X, t, kernel, **kernel_kwargs):
        """
        Given a test point x_star, this function computes the mean prediction using Gaussian Process Classification.
        
        Args:
        - X_star: new examples, of shape (M, d), where M is number of test examples
        - X (ndarray): The training examples, of shape (N, d).
        - t (ndarray): The training targets, of shape (N,).
        - kernel (function): The kernel function to compute similarity.
        - **kernel_kwargs (dict): Additional arguments for the kernel function.
        
        Returns:
        - ndarray: The predicted probability of the test points X_star belonging to class 1.
        """
    
        N, _ = X.shape
        
        # Kernels
        K_X_X = kernel(X, X, **kernel_kwargs) 
        
        # Get the mode of p(a | X, t)
        a_map = get_a_map(K_X_X, t)
        sigmoid_a_map = sigmoid(a_map)
        
        # Compute kernels for all points in X_star with respect to X
        K_x_star_X = kernel(X_star, X, **kernel_kwargs)
        K_x_star_x_star = np.diagonal(kernel(X_star, X_star, **kernel_kwargs)).reshape(-1, 1)
        
        # Compute intermediate matrices for d_sq calculation
        L = np.linalg.cholesky(np.diag(sigmoid_a_map * (1 - sigmoid_a_map)) + K_X_X)
        L_inv_K_x_star_X_T = np.linalg.solve(L, K_x_star_X.T)
        
        # Compute c and d_sq for all x_star points
        c = K_x_star_X @ (t - sigmoid_a_map)
        d_sq = K_x_star_x_star.flatten() - np.sum(L_inv_K_x_star_X_T**2, axis=0)
        
        # Return predicted probabilities for all x_star points
        return sigmoid(np.sqrt(1./(1 + np.pi * d_sq / 8)) * c).flatten()
    
    

    st.markdown(r'''Visualization offers clarity. To truly understand and validate our Gaussian Process Classification, it's crucial to visualize our results. Let's first plot the true decision boundary alongside our dataset. Then, in contrast, we'll plot the predicted probabilities for each grid point in our feature space.

The left plot provides context, showing the true decision boundary and how our generated data points relate to it. The right plot, on the other hand, showcases the power of our GPC. The color intensity represents the predicted probability, with deeper shades of blue indicating higher probabilities of belonging to class 1.

This visualization serves as a testament to the power and adaptability of Gaussian Processes. It allows us to visualize not only our predictions but also the confidence (probability) associated with each.
''')

    st.code('''# Set up a canvas with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# -- PLOT DECISION BOUNDARY --

# Prepare the axis for the first plot
ax = axes[0]

# Create a grid over our feature space
x1s = np.linspace(-3, 3, num=N)
x2s = np.linspace(-3, 3, num=N)
x1, x2 = np.meshgrid(x1s, x2s)

# Evaluate the true function over our feature grid
f = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        f[i,j] = true_fn(x1[i, j], x2[i, j])

# Visualize the decision boundary
cb = ax.contourf(x1, x2, f, cmap=my_gradient, alpha=1)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13, rotation=0)
ax.set_aspect('equal')
ax.set_ylim([-3, 3])
ax.set_xlim([-3, 3])

# Overlay the training data on the decision boundary
X, t = generate_classification_data(N=200)
ax.scatter(X[t==0, 0], X[t==0, 1], s=30, facecolors='none', edgecolors='b', marker="o", alpha=0.8)
ax.scatter(X[t==1, 0], X[t==1, 1], s=30, marker="x", color="red", alpha=0.8)

#---------------------------------------------
# -- PLOT GP CLASSIFICATION PREDICTIONS --

# Prepare the axis for the second plot
ax = axes[1]

# Create a grid over our feature space
X_star = np.zeros((N * N, 2))
for i in range(N):
    for j in range(N):
        X_star[i * N + j, 0] = x1[i, j]
        X_star[i * N + j, 1] = x2[i, j]

# Get predictions over our feature grid using the GP classifier
pred = gp_classif_batch(X_star, X, t, rbf)
pred = pred.reshape(N, N)

# Visualize the GP classifier's predictions
cb = ax.contourf(x1, x2, pred, 100, cmap=cm.coolwarm, alpha=1)
cbar = plt.colorbar(cb, fraction=0.05)

ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13, rotation=0)
ax.set_aspect('equal')
ax.set_ylim([-3, 3])
ax.set_xlim([-3, 3])

# Setting titles for the subplots
axes[0].set_title("Dataset", size=15)
axes[1].set_title("$p(t^* = 1 \mid x^*, X, t)$", size=15)

# Render the plots
plt.show()
''', language='python')

    # Set up a canvas with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # -- PLOT DECISION BOUNDARY --
    
    # Prepare the axis for the first plot
    ax = axes[0]
    
    # Create a grid over our feature space
    x1s = np.linspace(-3, 3, num=N)
    x2s = np.linspace(-3, 3, num=N)
    x1, x2 = np.meshgrid(x1s, x2s)
    
    # Evaluate the true function over our feature grid
    f = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            f[i,j] = true_fn(x1[i, j], x2[i, j])
    
    # Visualize the decision boundary
    cb = ax.contourf(x1, x2, f, cmap=my_gradient, alpha=1)
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13, rotation=0)
    ax.set_aspect('equal')
    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])
    
    # Overlay the training data on the decision boundary
    X, t = generate_classification_data(N=200)
    ax.scatter(X[t==0, 0], X[t==0, 1], s=30, facecolors='none', edgecolors='b', marker="o", alpha=0.8)
    ax.scatter(X[t==1, 0], X[t==1, 1], s=30, marker="x", color="red", alpha=0.8)
    
    #---------------------------------------------
    # -- PLOT GP CLASSIFICATION PREDICTIONS --
    
    # Prepare the axis for the second plot
    ax = axes[1]
    
    # Create a grid over our feature space
    X_star = np.zeros((N * N, 2))
    for i in range(N):
        for j in range(N):
            X_star[i * N + j, 0] = x1[i, j]
            X_star[i * N + j, 1] = x2[i, j]
    
    # Get predictions over our feature grid using the GP classifier
    pred = gp_classif_batch(X_star, X, t, rbf)
    pred = pred.reshape(N, N)
    
    # Visualize the GP classifier's predictions
    cb = ax.contourf(x1, x2, pred, 100, cmap=cm.coolwarm, alpha=1)
    cbar = fig.colorbar(cb, fraction=0.05)
    
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13, rotation=0)
    ax.set_aspect('equal')
    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])
    
    # Setting titles for the subplots
    axes[0].set_title("Dataset", size=15)
    axes[1].set_title("$p(t^* = 1 \mid x^*, X, t)$", size=15)
    
    # Render the plots
    st.pyplot(fig)
    
    

    st.markdown(r'''As we conclude this part, take a moment to appreciate the contours and patterns in the visualization. Each point, each shade, and each contour is a testament to the sophisticated machinery of Gaussian Process Classification at work, bridging theory and practice.

---''')

    st.markdown(r'''## Star-Galaxy Separation

When we observe the vast expanse of the sky, we are limited to a 2D projected view. In this realm, distinguishing stars from galaxies becomes a quintessential task. This distinction aids us in making informed decisions, like optimally allocating resources for spectroscopic follow-ups, ensuring that we maximize the utility of spectroscopic time.

However, the star-galaxy separation task is not without its challenges:

1. **Non-Linear Boundaries:** The boundary that separates stars from galaxies in the feature space, especially when considering photometric colors, isn't always linear. This non-linearity diminishes the effectiveness of simpler models like logistic regression.

2. **Ambiguities in Classification:** Often, the difference between a star and a galaxy isn't stark. There are instances where an object can almost equally qualify as both. Hence, it's crucial to provide a probabilistic assignment rather than a definitive classification.

For this exploration, we'll analyze a dataset containing 537 objects (stars and galaxies). This dataset provides us with the Gaia `BP-RP` and `G-RP` colors of these objects. We'll be leveraging star/galaxy labels from the study by [Hughes et al. (2022)](https://arxiv.org/abs/2210.05505). 

As we will see below, the GPC codes that we have developped can be easily adopted for this scenario with no modification.''')

    st.code('''# Load the dataset
data = np.load("star_galaxy_tutorial_week11a.npz")
X = np.vstack([data['bp_rp'], data['g_rp']]).T
t = data['star_label'] # 1=star, 0=galaxy

# Setting parameters for our visualization grid
N = 100

# Compute the aspect ratio based on data ranges for accurate visualization
x_range = 2.5 - 0.5
y_range = 5 - 0.5
aspect_ratio = x_range / y_range

#---------------------------------------------
# Create a canvas with two subplots for visualization
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# plot properties
ax = axes[0]  
x1s = np.linspace(0.5, 2.5, num=N)
x2s = np.linspace(0.5, 5, num=N)
x1, x2 = np.meshgrid(x1s, x2s)
ax.scatter(X[t==0, 0], X[t==0, 1], s=30, facecolors='none', edgecolors='b', marker="o", alpha=0.8)
ax.scatter(X[t==1, 0], X[t==1, 1], s=30, marker="x", color="red", alpha=0.8)
ax.set(title="Dataset", xlabel='BP-RP', ylabel='G-RP', aspect=aspect_ratio, xlim=[0.5, 2.5], ylim=[0.5, 5])

# -- PLOT GP CLASSIFICATION PREDICTIONS --
ax = axes[1]
X_star = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
pred = gp_classif_batch(X_star, X, t, rbf).reshape(N, N)
cb = ax.contourf(x1, x2, pred, 100, cmap=cm.coolwarm, alpha=1)
cbar = plt.colorbar(cb, ax=ax, fraction=0.05)
ax.set(title="$p(t^* = 1 \mid x^*, X, t)$", xlabel='BP-RP', ylabel='G-RP', aspect=aspect_ratio, xlim=[0.5, 2.5], ylim=[0.5, 5])

# Adjust layout for clear visualization
plt.subplots_adjust(wspace=0.2, right=0.85)
plt.tight_layout() 
plt.show()
''', language='python')

    # Loading the dataset
    import requests
    from io import BytesIO
    
    # Load the dataset using np.load
    response = requests.get('https://storage.googleapis.com/compute_astro/star_galaxy_tutorial_week11a.npz')
    f = BytesIO(response.content)
    data = np.load(f, allow_pickle=True)

    X = np.vstack([data['bp_rp'], data['g_rp']]).T
    t = data['star_label'] # 1=star, 0=galaxy
    
    # Setting parameters for our visualization grid
    N = 100
    
    # Compute the aspect ratio based on data ranges for accurate visualization
    x_range = 2.5 - 0.5
    y_range = 5 - 0.5
    aspect_ratio = x_range / y_range
    
    #---------------------------------------------
    # Create a canvas with two subplots for visualization
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # plot properties
    ax = axes[0]  
    x1s = np.linspace(0.5, 2.5, num=N)
    x2s = np.linspace(0.5, 5, num=N)
    x1, x2 = np.meshgrid(x1s, x2s)
    ax.scatter(X[t==0, 0], X[t==0, 1], s=30, facecolors='none', edgecolors='b', marker="o", alpha=0.8)
    ax.scatter(X[t==1, 0], X[t==1, 1], s=30, marker="x", color="red", alpha=0.8)
    ax.set(title="Dataset", xlabel='BP-RP', ylabel='G-RP', aspect=aspect_ratio, xlim=[0.5, 2.5], ylim=[0.5, 5])
    
    # -- PLOT GP CLASSIFICATION PREDICTIONS --
    ax = axes[1]
    X_star = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
    pred = gp_classif_batch(X_star, X, t, rbf).reshape(N, N)
    cb = ax.contourf(x1, x2, pred, 100, cmap=cm.coolwarm, alpha=1)
    cbar = fig.colorbar(cb, ax=ax, fraction=0.05)
    ax.set(title="$p(t^* = 1 \mid x^*, X, t)$", xlabel='BP-RP', ylabel='G-RP', aspect=aspect_ratio, xlim=[0.5, 2.5], ylim=[0.5, 5])
    
    # Adjust layout for clear visualization
    plt.subplots_adjust(wspace=0.2, right=0.85)
     
    st.pyplot(fig)
    
    

    st.markdown(r'''We note from the results that

1. **Efficiency in Simplicity:** A salient feature of our approach is its compactness. In just a few lines of code, we've implemented a robust system capable of probabilistically classifying stars and galaxies. And, rather than a straightforward, linear classification, it establishes a nuanced, non-linear boundary.

2. **Scalability:** While our current analysis harnesses features in a two-dimensional space, our method's beauty lies in its scalability. The same code, without substantial modifications, can handle feature spaces of much higher dimensions. This flexibility ensures that as our data grows or becomes more complex, our methodology can adapt with ease.

3. **Hyperparameter Tuning:** In this exercise, we used a fixed hyperparameter for the RBF kernel, a decision that simplifies our initial exploration. However, in a more exhaustive analysis, especially when optimizing for performance, it's advisable (and entirely feasible) to tune this hyperparameter. Recall our discussions on Gaussian Process Regression—much like in that context, hyperparameters in our current scenario can be optimized to enhance model precision.

---

''')

    st.markdown(r'''### Conclusion

In this tutorial, we embarked on a journey through Gaussian Process Classification, delving deep into its intricacies and capabilities. Our exploration consisted of the following phases. We began with a toy example, establishing a foundational understanding of GPC. This served as a gentle introduction, ensuring that even those new to GPC could grasp its core concepts.

With the basic principles in place, we transitioned to a real-world application - the star-galaxy separation. This not only showcased the practicality of GPC but also its versatility in handling more complex, real-world datasets. Our analyses, especially in the star-galaxy context, showcased the power of GPC in establishing non-linear boundaries and offering probabilistic classifications.

In essence, this tutorial aimed to provide a balanced mix of theory and application. We hope that as you close this tutorial, you carry with you not just an understanding of GPC, but an appreciation for its potential in diverse scenarios for probabilistic classification.

''')

#if __name__ == '__main__':
show_page()
