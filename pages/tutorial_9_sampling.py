import streamlit as st
from menu import navigation_menu

def show_page():
    st.set_page_config(page_title="Comp Astro",
                       page_icon="./image/tutor_favicon.png", layout="wide")

    navigation_menu()
    st.markdown(r'''# Sampling''')

    st.markdown(r'''

### Assumed knowledge

- **Sampling (lectures)**: You should have attended and understood the main concepts from the lectures on sampling. This includes understanding what sampling is, why we sample, and familiarizing yourself with basic methods of sampling. If you're not confident about these concepts, consider revisiting the lecture materials.

### Objectives

By the end of this lab, you'll have hands-on experience with several pivotal sampling techniques. Specifically:

1. **Sampling distributions using inverse CDF**:
   - *What it means*: Given any integrable distribution with a known and invertible CDF, we can employ a simple uniform distribution (where all values between 0 and 1 are equally likely) to sample from it using the Inverse CDF method.
   - *Why it's important*: This foundational method enables generating random samples from non-uniform distributions with the help of a uniform random number generator, a prevalent feature in most programming languages.

###

2. **Rejection Sampling**:
   - *What it means*: When direct sampling from a distribution proves challenging, we can resort to a proposal distribution (often simpler) to generate samples and subsequently "reject" those that don't align with our target distribution.
   - *Why it's important*: Rejection sampling emerges as a handy tool when an explicit form for the integration of the distribution remains elusive.

###

3. **Importance Sampling**:
   - *What it means*: Instead of sampling directly from our desired distribution, we sample from an alternative distribution and adjust or weigh the samples to account for the disparities between the two distributions. This approach facilitates efficient estimations of specific properties (like moments) of the target distribution.
   - *Why it's important*: Importance sampling becomes invaluable when direct sampling poses computational challenges, especially when pinpointing particular properties of a distribution. Its strength lies in variance reduction and potentially faster convergence for some estimates.

###
---
''')

    st.code('''import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import uniform, multivariate_normal

%matplotlib inline''', language='python')

    import math
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    from scipy.stats import uniform, multivariate_normal
    
    

    st.markdown(r'''## Sampling Distributions Using Inverse CDF

When it comes to generating random numbers, most programming languages offer easy ways to pull values from a uniform distribution $\mathcal{U}(0,1)$. However, the real art and science arise when we want to generate samples from non-uniform distributions.

### The Two-Step Process

For a given distribution $ p(y) $:

1. **Compute the CDF**:

   The CDF, $ h(y) $, of your target distribution $ p $ is:
   
   $$
   h(y) = \int_{-\infty}^y p(\hat{y}) d\hat{y}.
   $$

2. **Transform Using the Uniform Distribution**:

   After obtaining the CDF, you can then generate samples from a uniform distribution $ \mathcal{U}(z \mid 0, 1) $ and transform these samples using the inverse of the CDF:

   $$
   y = h^{-1}(z).
   $$

### The Exponential Distribution: Derivation

Let's start by showing how to draw a simple exponential distribution. Given the probability density function of the exponential distribution:

$$
    p(y) = 
    \begin{cases}
        \lambda e^{-\lambda y}, & y \geq 0 \\
        0, & y < 0
    \end{cases}
$$

Let's compute the CDF and then find the transformation using the uniform samples.

1. **CDF of $ p $**:

   We want to compute:
   
   $$
   h(y) = \int_{0}^y \lambda e^{-\lambda \hat{y}} d\hat{y}.
   $$
   
   Integrating the above, we get:

   $$
   h(y) = 1 - e^{-\lambda y}.
   $$

2. **Transformation with Uniform Samples $ z $**:

   Setting $ h(y) $ equal to $ z $ (from our uniform distribution), we have:

   $$
   z = 1 - e^{-\lambda y}.
   $$

   Solving for $ y $, we get:

   $$
   y = -\dfrac{1}{\lambda} \ln (1-z).
   $$

With these derivations in hand, you now have a firm grasp of how to sample from the exponential distribution using a uniform random number generator. Now, let's implement it.

''')

    st.code('''def sample_exponential(l, size=1):
    """
    Vectorized function to sample from the exponential distribution.
    
    Parameters:
    - l (float): The rate parameter of the exponential distribution.
    - size (int): The number of samples to draw.
    
    Returns:
    - y (numpy array): Samples from the exponential distribution.
    """
    
    # Generate an array of uniform random numbers
    z = np.random.uniform(size=size)
    
    # Transform the uniform samples using the derived formula
    y = -(1 / l) * np.log(1 - z)
    
    return y''', language='python')

    def sample_exponential(l, size=1):
        """
        Vectorized function to sample from the exponential distribution.
        
        Parameters:
        - l (float): The rate parameter of the exponential distribution.
        - size (int): The number of samples to draw.
        
        Returns:
        - y (numpy array): Samples from the exponential distribution.
        """
        
        # Generate an array of uniform random numbers
        z = np.random.uniform(size=size)
        
        # Transform the uniform samples using the derived formula
        y = -(1 / l) * np.log(1 - z)
        
        return y
    
    

    st.markdown(r'''With our vectorized sampling function ready, let's draw samples and visualize them against the true exponential distribution.''')

    st.code('''l = 1  # Rate parameter

# Draw 10,000 samples
samples = sample_exponential(l, size=10000)''', language='python')

    l = 1  # Rate parameter
    
    # Draw 10,000 samples
    samples = sample_exponential(l, size=10000)
    
    

    st.markdown(r'''Now, let's visualize the results.''')

    st.code('''# Define the true exponential distribution curve
xs = np.arange(0, 10.5, 0.5)
ys = [l * np.exp(-l * x) for x in xs]

# Plot the histogram of samples and the true distribution curve
plt.figure(figsize=(7,5))
plt.hist(samples, 50, density=True, alpha=0.7, label='Samples')
plt.plot(xs, ys, 'r-', label='True distribution')
plt.xlabel('Sample Value')
plt.ylabel('Probability Density')
plt.title('Density Histogram vs. True Exponential Distribution')
plt.legend()
plt.show()''', language='python')

    # Define the true exponential distribution curve
    xs = np.arange(0, 10.5, 0.5)
    ys = [l * np.exp(-l * x) for x in xs]
    
    # Plot the histogram of samples and the true distribution curve
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(samples, 50, density=True, alpha=0.7, label='Samples')
    ax.plot(xs, ys, 'r-', label='True distribution')
    ax.set_xlabel('Sample Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Density Histogram vs. True Exponential Distribution')
    ax.legend()
    st.pyplot(fig)
    

    st.markdown(r'''This approach leverages the power of numpy to draw multiple samples efficiently. When you run the cells, you should see the histogram of your samples overlayed with the exponential distribution curve. The histogram should closely follow the curve, validating our sampling approach.

---''')

    st.markdown(r'''### The Kroupa Initial Mass Function

The Initial Mass Function (IMF) is a pivotal concept in astrophysics, describing the distribution of stellar masses when they're initially formed in star-forming regions. It fundamentally influences many aspects of galactic evolution and dynamics, playing a role in determining the luminosity of a galaxy, its chemical evolution, and the rate at which it forms stars.

The Kroupa IMF, in particular, has become one of the standard IMFs used in astrophysical simulations and calculations. Understanding and being able to sample from the Kroupa IMF is crucial for generating synthetic stellar populations, modeling light from distant galaxies, and predicting the occurrence rates of different types of supernovae, among other applications.

#### Mathematical Formulation

The Kroupa IMF is defined as a piecewise power-law:

$$ p(m) = 
    \begin{cases} 
      K_1 m^{-0.3} & 0.01 \leq m < 0.08 \\
      K_2 m^{-1.3} & 0.08 \leq m < 0.5 \\
      K_3 m^{-2.3} & 0.5 \leq m \leq 1 
   \end{cases}
$$

Where the $ K $ values are normalization constants ensuring that the IMF is continuous across the entire mass range and properly normalized, i.e., integrates to 1 over the range of stellar masses.

--- 

#### Derivation of Normalization Constants

To determine the $ K $ values, we need to ensure two things:

1. Continuity at the boundaries of the mass ranges.

    a. At $ m = 0.08 $:

    $$ 
    K_1 \times 0.08^{-0.3} = K_2 \times 0.08^{-1.3} 
    $$
    
    This gives:
    
    $$ 
    K_2 = K_1 \times 0.08 
    $$

    b. At $ m = 0.5 $:

    $$ 
    K_2 \times 0.5^{-1.3} = K_3 \times 0.5^{-2.3} 
    $$
    
    This yields:
    $$ 
    K_3 = K_2 \times 0.5 
    $$

2. The entire function integrates to $1$ over the mass range $[0.01, 1]$.

    The integral of $ p(m) $ over $[0.01, 1]$ should be $1$:

    $$ 
    \int_{0.01}^{1} p(m) \, dm = 1 
    $$

    Breaking down this integral over the piecewise segments:

    $$ 
    K_1 \int_{0.01}^{0.08} m^{-0.3} \, dm + K_2 \int_{0.08}^{0.5}  m^{-1.3} \, dm + K_3 \int_{0.5}^{1} m^{-2.3} \, dm = 1 
    $$

To evaluate the individual integral is quite straightforward even analytically, but here we will benefit from the numerical integration using scipy.''')

    st.code('''# integrate with scipy
print("integrate 0.01 to 0.08 with m^-0.3 = ", scipy.integrate.quad(lambda x: x**(-0.3), 0.01, 0.08)[0])
print("integrate 0.08 to 0.5 with m^-1.3 = ", scipy.integrate.quad(lambda x: x**(-1.3), 0.08, 0.5)[0])
print("integrate 0.5 to 1.0 with m^-2.3 = ", scipy.integrate.quad(lambda x: x**(-2.3), 0.5, 1.0)[0])
''', language='python')

    # integrate with scipy
    st.write("integrate 0.01 to 0.08 with m^-0.3 = ", scipy.integrate.quad(lambda x: x**(-0.3), 0.01, 0.08)[0])
    st.write("integrate 0.08 to 0.5 with m^-1.3 = ", scipy.integrate.quad(lambda x: x**(-1.3), 0.08, 0.5)[0])
    st.write("integrate 0.5 to 1.0 with m^-2.3 = ", scipy.integrate.quad(lambda x: x**(-2.3), 0.5, 1.0)[0])
    
    

    st.markdown(r'''And this leads to 
$$ 
0.187 K_1 + 3.008 K_2 + 1.125 K_3 = 1 
$$

Plugging in the continuity relationships we found earlier,

$$ 
K_2 = K_1 \times 0.08, 
$$

$$ 
K_3 = K_2 \times 0.5, 
$$

this yields 
$$ 
0.187 K_1 + 3.008 \times 0.08 K_1 + 1.125 \times 0.5 \times 0.08 K_1 = 1
$$

$$ 
\Rightarrow K_1 = 2.116, K_2 = 0.169, K_3 = 0.085 
$$''')

    st.markdown(r'''We'll start by incorporating the derived values for $K_1$, $K_2$, and $K_3$:

--- 

### Cumulative Distribution Function (CDF)

The CDF for each segment is computed by integrating the respective segment of the IMF.

**1. For $ 0.01 \leq m < 0.08 $:**

$$ 
h(m) = 2.116 \int_{0.01}^{m} \hat{m}^{-0.3} d\hat{m} 
$$

$$ 
h(m) = 2.116 \bigg[ \hat{m}^{0.7} \times \dfrac{1}{0.7} \bigg]_{0.01}^{m} 
$$

**2. For $ 0.08 \leq m < 0.5 $:**

$$ 
h(m) = 0.169 \int_{0.08}^{m} \hat{m}^{-1.3} d\hat{m} 
$$

$$ 
h(m) = 0.169 \bigg[ \hat{m}^{-0.3} \times \dfrac{1}{-0.3} \bigg]_{0.08}^{m} 
$$

**3. For $ 0.5 \leq m \leq 1 $:**

$$ 
h(m) = 0.085 \int_{0.5}^{m} \hat{m}^{-2.3} d\hat{m} 
$$

$$ 
h(m) = 0.085 \bigg[ \hat{m}^{-1.3} \times \dfrac{1}{-1.3} \bigg]_{0.5}^{m} 
$$

Using the inverse transform sampling method, we can convert uniform random numbers to match our desired distribution. To effectively do this, we must ascertain the proportion of the probability contained within each segment.

Let's breakdown the cumulative distribution at each segment boundary:

1. For $ 0.01 \leq m < 0.08 $:

$$ 
\Delta h_1 = h(0.08) - h(0.01) = 2.116 \bigg[ \hat{m}^{0.7} \times \dfrac{1}{0.7} \bigg]_{0.01}^{0.08} 
$$

2. For $ 0.08 \leq m < 0.5 $:

$$ 
\Delta h_2 = h(0.5) - h(0.08) = 0.169 \bigg[ \hat{m}^{-0.3} \times \dfrac{1}{-0.3} \bigg]_{0.08}^{0.5} 
$$

3. For $ 0.5 \leq m \leq 1 $:

$$ 
\Delta h_3 = h(1) - h(0.5) = 0.085 \bigg[ \hat{m}^{-1.3} \times \dfrac{1}{-1.3} \bigg]_{0.5}^{1} 
$$

From these, let:
- $A = \Delta h_1$
- $B = \Delta h_2$
- $C = \Delta h_3$

This categorizes our uniform random variable, $z$, as:
- $0 \leq z < A$ for the first segment,
- $A \leq z < A+B$ for the second segment, and
- $A+B \leq z \leq 1$ (or $A+B \leq z < A+B+C$ if it doesn't sum to 1) for the third segment.

Finally we will determine the inverse of the CDF for each segment. The inverse CDF will map from our uniform random numbers in the range [0, 1] to stellar masses in the range defined by the IMF.

Let's start with the inverse CDF for each segment:

1. **For $0.01 \leq m < 0.08$:**

Given:

$$ 
h(m) = 2.116 \bigg[ \hat{m}^{0.7} \times \dfrac{1}{0.7} \bigg]_{0.01}^{m} 
$$

The inverse of this CDF is:

$$ 
m(z) = \bigg( 0.7z/2.116 + 0.01^{0.7} \bigg)^{1/0.7} 
$$

2. **For $0.08 \leq m < 0.5$:**

Given:

$$ 
h(m) = 0.169 \bigg[ \hat{m}^{-0.3} \times \dfrac{1}{-0.3} \bigg]_{0.08}^{m} 
$$

The inverse of this CDF is:

$$ 
m(z) = \bigg( -0.3(z-A)/0.169 + 0.08^{-0.3} \bigg)^{-1/0.3} 
$$

3. **For $0.5 \leq m \leq 1$:**

Given:

$$ 
h(m) = 0.085 \bigg[ \hat{m}^{-1.3} \times \dfrac{1}{-1.3} \bigg]_{0.5}^{m} 
$$

The inverse of this CDF is:

$$ 
m(z) = \bigg( -1.3(z-A-B)/0.085 + 0.5^{-1.3} \bigg)^{-1/1.3} 
$$

Now, we can incorporate these details into our sampling code:''')

    st.code('''import numpy as np

def sample_kroupa_imf(size=1):
    """Sample stellar masses from the Kroupa IMF.
    
    Parameters:
    - size (int): The number of samples to draw.
    
    Returns:
    - masses (numpy array): Stellar masses sampled from the Kroupa IMF.
    """
    
    K1 = 2.116
    K2 = 0.169
    K3 = 0.085
    
    # Generate an array of uniform random numbers
    z = np.random.uniform(size=size)
    
    # Initialize an array for the resulting samples
    masses = np.zeros(size)
    
    # Calculate the integrals for each segment:
    A = 2.116 * (0.08 ** 0.7 - 0.01 ** 0.7) / 0.7
    B = 0.169 * (0.5 ** (-0.3) - 0.08 ** (-0.3)) / (-0.3)
    C = 0.085 * (1 ** (-1.3) - 0.5 ** (-1.3)) / (-1.3)

    # Define the conditions for each segment:
    mask1 = (z < A)
    mask2 = (z >= A) & (z < A+B)
    mask3 = (z >= A+B) # For the third segment
    
    # Apply the inverse CDF transformations for each mass range
    masses[mask1] = (0.7 * z[mask1] / K1 + 0.01**0.7)**(1/0.7)
    masses[mask2] = (-0.3 * (z[mask2] - A) / K2 + 0.08**(-0.3))**(-1/0.3)
    masses[mask3] = (-1.3 * (z[mask3] - A - B) / K3 + 0.5**(-1.3))**(-1/1.3)
    
    return masses
''', language='python')

    import numpy as np
    
    def sample_kroupa_imf(size=1):
        """Sample stellar masses from the Kroupa IMF.
        
        Parameters:
        - size (int): The number of samples to draw.
        
        Returns:
        - masses (numpy array): Stellar masses sampled from the Kroupa IMF.
        """
        
        K1 = 2.116
        K2 = 0.169
        K3 = 0.085
        
        # Generate an array of uniform random numbers
        z = np.random.uniform(size=size)
        
        # Initialize an array for the resulting samples
        masses = np.zeros(size)
        
        # Calculate the integrals for each segment:
        A = 2.116 * (0.08 ** 0.7 - 0.01 ** 0.7) / 0.7
        B = 0.169 * (0.5 ** (-0.3) - 0.08 ** (-0.3)) / (-0.3)
        C = 0.085 * (1 ** (-1.3) - 0.5 ** (-1.3)) / (-1.3)
    
        # Define the conditions for each segment:
        mask1 = (z < A)
        mask2 = (z >= A) & (z < A+B)
        mask3 = (z >= A+B) # For the third segment
        
        # Apply the inverse CDF transformations for each mass range
        masses[mask1] = (0.7 * z[mask1] / K1 + 0.01**0.7)**(1/0.7)
        masses[mask2] = (-0.3 * (z[mask2] - A) / K2 + 0.08**(-0.3))**(-1/0.3)
        masses[mask3] = (-1.3 * (z[mask3] - A - B) / K3 + 0.5**(-1.3))**(-1/1.3)
        
        return masses
    
    

    st.markdown(r'''With this setup, the `sample_kroupa_imf` function will generate masses sampled from the Kroupa IMF. Below we will visualise the generated samples alongside the actual Kroupa IMF for validation. Plotting in a log-log scale can be very informative, especially for power-law distributions like the Kroupa IMF, so the plot on the right hand side might look more familiar to you.''')

    st.code('''samples = sample_kroupa_imf(size=100000)

# define the kroupa imf
def kroupa_imf(m):
    K1 = 2.116
    K2 = 0.169
    K3 = 0.085

    if m < 0.01:
        return 0
    elif m < 0.08:
        return K1 * m**(-0.3)
    elif m < 0.5:
        return K2 * m**(-1.3)
    elif m <= 1.0:
        return K3 * m**(-2.3)
    else:
        return 0
    
# Regular Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]

ax.hist(samples, bins=np.logspace(np.log10(0.01), np.log10(1.0), 50), density=True, alpha=0.7, label="Samples")
mass_range = np.linspace(0.01, 1, 400)
pdf_values = np.vectorize(kroupa_imf)(mass_range)
ax.plot(mass_range, pdf_values, 'r-', label="Kroupa IMF")
ax.set_xlabel("Stellar Mass")
ax.set_ylabel("Probability Density")
ax.legend()
ax.set_title("Regular Scale")

# Log-log Plot
ax = axes[1]
plt.hist(samples, bins=np.logspace(np.log10(0.01), np.log10(1.0), 50), density=True, alpha=0.7, label="Samples", log=True)
plt.plot(mass_range, pdf_values, 'r-', label="Kroupa IMF")
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel("Stellar Mass (Log Scale)")
ax.set_ylabel("Probability Density (Log Scale)")
ax.legend()
ax.set_title("Log-log Scale")

plt.show()
''', language='python')

    samples = sample_kroupa_imf(size=100000)
    
    # define the kroupa imf
    def kroupa_imf(m):
        K1 = 2.116
        K2 = 0.169
        K3 = 0.085
    
        if m < 0.01:
            return 0
        elif m < 0.08:
            return K1 * m**(-0.3)
        elif m < 0.5:
            return K2 * m**(-1.3)
        elif m <= 1.0:
            return K3 * m**(-2.3)
        else:
            return 0
        
    # Regular Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    ax = axes[0]
    
    ax.hist(samples, bins=np.logspace(np.log10(0.01), np.log10(1.0), 50), density=True, alpha=0.7, label="Samples")
    mass_range = np.linspace(0.01, 1, 400)
    pdf_values = np.vectorize(kroupa_imf)(mass_range)
    ax.plot(mass_range, pdf_values, 'r-', label="Kroupa IMF")
    ax.set_xlabel("Stellar Mass")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.set_title("Regular Scale")
    
    # Log-log Plot
    ax = axes[1]
    ax.hist(samples, bins=np.logspace(np.log10(0.01), np.log10(1.0), 50), density=True, alpha=0.7, label="Samples", log=True)
    ax.plot(mass_range, pdf_values, 'r-', label="Kroupa IMF")
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlabel("Stellar Mass (Log Scale)")
    ax.set_ylabel("Probability Density (Log Scale)")
    ax.legend()
    ax.set_title("Log-log Scale")
    st.pyplot(fig)
    
    

    st.markdown(r'''--- 

## Rejection Sampling

Rejection sampling simplifies the process of drawing samples from a complex distribution, $ p(z) $, using an easier-to-handle proposal distribution, $ q(z) $. A key criterion is ensuring that, for every $ z $ and a constant multiplier $ k > 0 $, our scaled proposal distribution $ q(z) $ always envelopes $ p(z) $:
$$ 
k \cdot q(z) \geq p(z) 
$$

Our sampling mechanism is as follows:

1. **Proposal**: Randomly choose a sample $ z_0 $ from $ q(z) $.
2. **Height Selection**: For the chosen $ z_0 $, pick a random height $ u_0 $ from the interval [0, $ k \cdot q(z_0) $].
3. **Sample Decision**: Accept $ z_0 $ if $ u_0 $ lies below $ p(z_0) $; otherwise, reject it.

The samples we end up accepting, $ z_0 $, follow the desired distribution $ p(z) $.

### The Kroupa IMF and the Exponential Proposal Function

For our example, we're venturing to sample from the Kroupa IMF. While there are more efficient methods, such as inverse transform sampling, rejection sampling offers a unique perspective. We'll use the exponential distribution as our proposal function, given its simplicity and the ease with which we can sample from it using the inverse CDF method.

Let's dive into the implementation:
''')

    st.code('''import numpy as np

def sample_exponential(l, size=1):
    """Sample from the exponential distribution using the inverse CDF method."""
    z = np.random.uniform(size=size)
    y = -(1 / l) * np.log(1 - z)
    return y

def p(z):
    """Kroupa IMF pdf."""
    if z < 0.08:
        return 2.116 * z**(-0.3)
    elif z < 0.5:
        return 0.169 * z**(-1.3)
    else:
        return 0.085 * z**(-2.3)

def q(z, lambd):
    """Exponential pdf."""
    return lambd * np.exp(-lambd * z)

def rejection_samples(lambd, k, num_samples=10000):
    """Generate samples using rejection sampling."""
    
    acc_samples_x = []
    acc_samples_y = []
    rej_samples_x = []
    rej_samples_y = []
    
    for _ in range(num_samples):
        
        # Step 1: draw z from the exponential distribution q(z)
        z = sample_exponential(lambd, 1)[0]
        
        # Step 2: draw u from U[0, k*q(z)]
        qz = q(z, lambd)
        u = np.random.uniform(0, k * qz)
        
        # Step 3: accept z if u is below p(z)
        pz = p(z)
        if u <= pz and z <= 1 and z >= 0.01:
            acc_samples_x.append(z)
            acc_samples_y.append(u)
        else:
            rej_samples_x.append(z)
            rej_samples_y.append(u)
    
    return acc_samples_x, acc_samples_y, rej_samples_x, rej_samples_y
''', language='python')

    import numpy as np
    
    def sample_exponential(l, size=1):
        """Sample from the exponential distribution using the inverse CDF method."""
        z = np.random.uniform(size=size)
        y = -(1 / l) * np.log(1 - z)
        return y
    
    def p(z):
        """Kroupa IMF pdf."""
        if z < 0.08:
            return 2.116 * z**(-0.3)
        elif z < 0.5:
            return 0.169 * z**(-1.3)
        else:
            return 0.085 * z**(-2.3)
    
    def q(z, lambd):
        """Exponential pdf."""
        return lambd * np.exp(-lambd * z)
    
    def rejection_samples(lambd, k, num_samples=10000):
        """Generate samples using rejection sampling."""
        
        acc_samples_x = []
        acc_samples_y = []
        rej_samples_x = []
        rej_samples_y = []
        
        for _ in range(num_samples):
            
            # Step 1: draw z from the exponential distribution q(z)
            z = sample_exponential(lambd, 1)[0]
            
            # Step 2: draw u from U[0, k*q(z)]
            qz = q(z, lambd)
            u = np.random.uniform(0, k * qz)
            
            # Step 3: accept z if u is below p(z)
            pz = p(z)
            if u <= pz and z <= 1 and z >= 0.01:
                acc_samples_x.append(z)
                acc_samples_y.append(u)
            else:
                rej_samples_x.append(z)
                rej_samples_y.append(u)
        
        return acc_samples_x, acc_samples_y, rej_samples_x, rej_samples_y
    
    

    st.markdown(r'''#### Visualizing Rejection Sampling

The plot below showcases the mechanics of rejection sampling using the Kroupa IMF (in red) and a scaled exponential function (in green dashed line). The stars indicate samples drawn using our proposal distribution - the exponential function. 

Now, what do we see? 

- **Blue stars**: These are the samples we've accepted. Notice how they lie beneath the curve of the Kroupa IMF? This is by design. In rejection sampling, we accept samples that fall under the curve of our target distribution.
- **Cyan stars**: These samples, on the other hand, have been rejected. They lie between the Kroupa IMF and our scaled exponential function. 

Another observation is that no samples, neither accepted nor rejected, appear above the scaled exponential function. This is a fundamental property of our chosen proposal distribution and the value of $ k $ we've defined. Remember, $ k \times q(z) $ must always be greater than or equal to $ p(z) $ for all $ z $. This is crucial for rejection sampling to work.
''')

    st.code('''import numpy as np
import matplotlib.pyplot as plt

# Use the rejection sampling method
lambd = 4  # Adjust as needed; represents the rate for the exponential proposal distribution.
k = 2.2  # Scaling factor
num_samples = 100000
acc_samples_x, acc_samples_y, rej_samples_x, rej_samples_y = rejection_samples(lambd, k, num_samples)

# Visualization
plt.figure(figsize=(7, 5))

plt.scatter(rej_samples_x, rej_samples_y, s=1, color='cyan', label='Rejected Samples')
plt.scatter(acc_samples_x, acc_samples_y, s=1, color='blue', label='Accepted Samples')
mass_range = np.linspace(0.01, 1., 400)
pdf_values_imf = np.vectorize(p)(mass_range)
pdf_values_exp = k * np.vectorize(lambda z: q(z, lambd))(mass_range)
plt.plot(mass_range, pdf_values_imf, 'r-', label="Kroupa IMF")
plt.plot(mass_range, pdf_values_exp, 'g--', label="Scaled Exponential")
plt.xlim([0.0, 1.1])
plt.xlabel("Stellar Mass")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Regular Scale")

print(f'Number of accepted samples = {{}len(acc_samples_x){}}')''', language='python')

    import numpy as np
    import matplotlib.pyplot as plt
    
    # Use the rejection sampling method
    lambd = 4  # Adjust as needed; represents the rate for the exponential proposal distribution.
    k = 2.2  # Scaling factor
    num_samples = 100000
    acc_samples_x, acc_samples_y, rej_samples_x, rej_samples_y = rejection_samples(lambd, k, num_samples)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(7, 5))
    
    scatter = ax.scatter(rej_samples_x, rej_samples_y, s=1, color='cyan', label='Rejected Samples')
    scatter = ax.scatter(acc_samples_x, acc_samples_y, s=1, color='blue', label='Accepted Samples')
    mass_range = np.linspace(0.01, 1., 400)
    pdf_values_imf = np.vectorize(p)(mass_range)
    pdf_values_exp = k * np.vectorize(lambda z: q(z, lambd))(mass_range)
    ax.plot(mass_range, pdf_values_imf, 'r-', label="Kroupa IMF")
    ax.plot(mass_range, pdf_values_exp, 'g--', label="Scaled Exponential")
    plt.xlim([0.0, 1.1])
    ax.set_xlabel("Stellar Mass")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.set_title("Regular Scale")
    
    st.write(f'Number of accepted samples = {len(acc_samples_x)}')
    st.pyplot(fig)
    
    

    st.markdown(r'''While rejection sampling is a powerful tool, it comes with a caveat: computational efficiency. Rejected samples, by definition, are computational efforts that didn't yield usable results. If our proposal distribution is poorly chosen, we may end up with a large number of rejected samples, which wastes computational resources. 

Consider our example. If our scaled exponential doesn't closely follow the Kroupa IMF, especially in regions where the IMF has significant mass, we'll reject more samples. So, choosing a good proposal distribution is crucial to minimize computational waste. 

You might ask: *How do we make our rejection sampling more efficient?* The answer lies in fine-tuning our proposal distribution and the scaling factor $ k $. By optimizing these parameters, we can reduce the number of rejected samples, making our sampling process more efficient.

Explore and experiment! Adjust the values of $ k $ and $ \lambda $ in the exponential function. Your objective? Maximize the number of accepted samples. This exercise will give you a hands-on experience of the delicate balance between proposal distributions and computational efficiency in rejection sampling.

---''')

    st.markdown(r'''## Importance Sampling

In the vast cosmic theatre, stars of different masses play their unique roles, radiating light and energy into space. The Kroupa Initial Mass Function (IMF) is a pivotal tool in astrophysics that helps us understand this distribution of stellar masses. The 'average' stellar mass, a concept derived from the IMF, represents the most typical star you'd expect to find in a random cosmic sample. But why is this average significant? Well, by discerning the 'typical' stellar mass, we gain deeper insights into the predominant stellar formations in specific regions of our galaxy, and even in remote star clusters.

### A Sampling Quest

Our mathematical quest is to compute:
$$
\mathbb{E}_{p(m)}[m] = \int m \times p(m) dm
$$

Here, $ \mathbb{E}_{p(m)}[m] $ is the expected value of the stellar mass as per the Kroupa IMF, represented by $ p(m) $. However, the multi-segment nature of the Kroupa IMF makes direct sampling a challenging endeavor as we have seen. This is where importance sampling comes to our rescue! It turns out that we can calculate the mean without even doing the sampling.

### The Essence of Importance Sampling

Rather than wrestling with direct samples from the Kroupa IMF, we employ a nifty trick: use a simpler, friendlier distribution called the proposal distribution, denoted as $ q(m) $. The magic here is in judiciously choosing this proposal distribution, ensuring that it mirrors the significant areas of the Kroupa IMF, thereby capturing its essence.

--- 

### Navigating the Math

Let's dive a bit deeper into the math. Often, we're not blessed with normalized distributions. Imagine having an unnormalized version of our Kroupa IMF, termed as $ \widetilde{p}(z) $, and an unnormalized proposal distribution, $ \widetilde{q}(z) $. 

Our aspiration to find the expectation morphs into:
$$
\mathbb{E}_{p(z)}[f(z)] = \frac{\int f(z) \times \widetilde{p}(z) dz}{\int \widetilde{p}(z) dz}
$$

Harnessing the power of importance sampling, we can express this expectation using the samples from $ \widetilde{q}(z) $:
$$
\mathbb{E}_{p(z)}[f(z)] \approx \frac{\sum_{i=1}^{N} f(z_i) \times w(z_i)}{\sum_{i=1}^{N} w(z_i)}
$$

Here, each sample $ z_i $ originates from $ \widetilde{q}(z) $, and the importance weight $ w(z_i) $ is determined by:
$$
w(z_i) = \frac{\widetilde{p}(z_i)}{\widetilde{q}(z_i)}
$$

Breaking it down:

1. **Sampling**: Generate $ N $ samples $ \{z_1, z_2, ... z_N\} $ from $ \widetilde{q}(z) $.
2. **Weighting**: Assign a weight, $ w(z_i) $, to each sample. This weight signifies the importance of that sample in terms of the target and proposal distributions.
3. **Estimation**: With weights in hand, compute the weighted mean of the function's values, adjusting for the total weight.

A beautiful facet of this approach is its independence from the normalizing constants of the distributions. Thus, even when normalization seems like a Herculean task, importance sampling glides through.

However, remember: the choice of $ \widetilde{q}(z) $ holds the key. A thoughtfully selected proposal promises efficient, low-variance estimates. A hasty choice? Well, that might lead us astray with high variances and potentially misleading results.

---''')

    st.markdown(r'''### Importance Sampling with an Exponential

Before delving deep into the mechanics of importance sampling for our Kroupa IMF, let's reflect on the choice of the proposal distribution. As you may recall, our proposal distribution needs to mimic the shape of our target distribution as closely as possible, primarily where the target function has significant mass. As we have seen above, on the context of the Kroupa IMF, an exponential distribution with parameter $ \lambda = 5 $ seems to offer that balance. It smoothly captures the declining nature of the Kroupa IMF without adding unnecessary complexity.

Let's embark on our importance sampling journey using this exponential distribution as our guide. Recall that, to evaluate and sample from the exponential function, we can also call the `scipy.stat` package to simplify the codes.
''')

    st.code('''from scipy.stats import expon

def importance_exponential(num_samples, lambda_val=5):
    """ Compute the expectation of m over the Kroupa IMF.
    Uses an exponential distribution as a proposal distribution.
    """
    expectation_f = 0
    denom = 0
    
    # Define the exponential proposal distribution
    rv_m = expon(scale=1/lambda_val)
    
    for _ in range(num_samples):

        # only keep samples in the range [0.01, 1.0]
        m = rv_m.rvs()
        while m < 0.01 or m > 1.0:
            m = rv_m.rvs()
        
        p_tilde_val = kroupa_imf(m)  # Computing the value of the Kroupa IMF
        q_val = rv_m.pdf(m)
        
        weight = p_tilde_val / q_val  # Computing the importance weight
        expectation_f += m * weight  # Weighted sample
        denom += weight  # Accumulating the weight for normalization

    return expectation_f / denom  # Returning the normalized weighted average''', language='python')

    from scipy.stats import expon
    
    def importance_exponential(num_samples, lambda_val=5):
        """ Compute the expectation of m over the Kroupa IMF.
        Uses an exponential distribution as a proposal distribution.
        """
        expectation_f = 0
        denom = 0
        
        # Define the exponential proposal distribution
        rv_m = expon(scale=1/lambda_val)
        
        for _ in range(num_samples):
    
            # only keep samples in the range [0.01, 1.0]
            m = rv_m.rvs()
            while m < 0.01 or m > 1.0:
                m = rv_m.rvs()
            
            p_tilde_val = kroupa_imf(m)  # Computing the value of the Kroupa IMF
            q_val = rv_m.pdf(m)
            
            weight = p_tilde_val / q_val  # Computing the importance weight
            expectation_f += m * weight  # Weighted sample
            denom += weight  # Accumulating the weight for normalization
    
        return expectation_f / denom  # Returning the normalized weighted average
    
    

    st.markdown(r'''Now, let's break down the function and what's happening under the hood. Here we assume Kroupa IMF to be $\tilde{p}(m)$.

1. **Proposal Sampling**: For each iteration, we draw a sample $ m $ from our exponential proposal distribution.
2. **Compute Importance Weight**: The weight for each sample is determined by the ratio $ \frac{\tilde{p}(m)}{q(m)} $, where $ \tilde{p}(m) $ is the value of the Kroupa IMF at that sample, and $ q(m) $ is the value of our exponential proposal distribution.
3. **Accumulate Weighted Sample**: The sample is then multiplied by its weight and accumulated. This gives us a weighted estimate of the stellar mass for that sample.
4. **Normalize**: Finally, we normalize our accumulated estimate by the total accumulated weight. This normalization step ensures that our estimator remains unbiased even when dealing with unnormalized distributions.

With this setup, we can now proceed to perform importance sampling on the Kroupa IMF and retrieve an accurate estimate of the average stellar mass.''')

    st.code('''num_samples = 100000  # The number of samples we'll draw from our proposal distribution
expected_mass = importance_exponential(num_samples)

print(f"Estimated Expectation (Average Stellar Mass) under Kroupa IMF: {{}expected_mass:.4f{}}")
''', language='python')

    num_samples = 100000  # The number of samples we'll draw from our proposal distribution
    expected_mass = importance_exponential(num_samples)
    
    st.write(f"Estimated Expectation (Average Stellar Mass) under Kroupa IMF: {expected_mass:.4f}")
    
    

    st.markdown(r'''--- 

### Calculating the 'Exact' Expectation of Stellar Mass

Given our function $f(m) = m$ and the Kroupa IMF $p(m)$, we can calculate the expectation analytically by:

$$
\mathbf{E}_{p(m)}[m] = \int_{0.01}^{1} m \times p(m) \, \mathrm{d}m
$$

While we provided this formula for illustrative purposes, directly evaluating it might be challenging due to the piecewise nature of the Kroupa IMF. Thankfully, computational tools like SciPy provide powerful integration capabilities.

Let's compute the numerical expectation using SciPy:
''')

    st.code('''from scipy import integrate

def integrand(m):
    return m * kroupa_imf(m)

# We integrate from the lower limit (0.01) to the upper limit (1) of our Kroupa IMF.
exact_expectation, _ = integrate.quad(integrand, 0.01, 1)

print(f"Exact Expectation (Average Stellar Mass) under Kroupa IMF: {{}exact_expectation:.4f{}}")''', language='python')

    from scipy import integrate
    
    def integrand(m):
        return m * kroupa_imf(m)
    
    # We integrate from the lower limit (0.01) to the upper limit (1) of our Kroupa IMF.
    exact_expectation, _ = integrate.quad(integrand, 0.01, 1)
    
    st.write(f"Exact Expectation (Average Stellar Mass) under Kroupa IMF: {exact_expectation:.4f}")
    
    

    st.markdown(r'''--- 

### Comparing Empirical and Analytical Results

Having computed both the empirical expectation (using importance sampling) and the exact expectation, we can now compare the two to assess the accuracy of our importance sampling method.

In the scenario where we want to investigate the convergence rate of our importance sampling method, we can repeatedly use the method with increasing numbers of samples and plot the estimated expectation against the number of samples. Overlaying this with the exact expectation will provide a visual assessment of how quickly our empirical method approaches the true value as the number of samples increases.

From this plot, we should observe that as the number of samples increases, the empirical expectation (derived from importance sampling) converges to the exact expectation. This offers a compelling visual confirmation of the accuracy and utility of the importance sampling method.
''')

    st.code('''from tqdm import tqdm

# Generate a range of sample sizes in log space
sample_sizes = np.logspace(2, 5, 20).astype(int)

# Calculate the importance sampling estimate for each sample size
empirical_expectations = [importance_exponential(size) for size in tqdm(sample_sizes)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, empirical_expectations, label="Empirical (Importance Sampling)", marker='o')
plt.axhline(y=exact_expectation, color='r', linestyle='-', label="Analytical")
plt.xlabel("Number of Samples")
plt.ylabel("Expectation (Average Stellar Mass)")
plt.title("Convergence of Empirical Expectation with Sample Size")
plt.legend()
plt.xscale('log')
plt.ylim([0.15, 0.25])
plt.grid(True)
''', language='python')

    from tqdm import tqdm
    
    # Generate a range of sample sizes in log space
    sample_sizes = np.logspace(2, 5, 20).astype(int)
    
    # Calculate the importance sampling estimate for each sample size
    empirical_expectations = [importance_exponential(size) for size in tqdm(sample_sizes)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, empirical_expectations, label="Empirical (Importance Sampling)", marker='o')
    plt.axhline(y=exact_expectation, color='r', linestyle='-', label="Analytical")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Expectation (Average Stellar Mass)")
    ax.set_title("Convergence of Empirical Expectation with Sample Size")
    ax.legend()
    plt.xscale('log')
    plt.ylim([0.15, 0.25])
    plt.grid(True)
    st.pyplot(fig)
    
    

    st.markdown(r'''### Importance Sampling with a Uniform Distribution

When it comes to selecting a proposal distribution for importance sampling, the Uniform distribution is often one of the first distributions to be considered due to its simplicity. However, simplicity isn't always beneficial. The uniform distribution assigns equal probability over a specified range, and therefore, may not mimic the nuances of more complex target distributions like the Kroupa IMF.

For this demonstration, let's use a uniform distribution spanning the range of our Kroupa IMF, i.e., [0.01, 1.0], as our proposal distribution. This choice will help us highlight the challenges and inefficiencies that might arise from using a distribution that doesn't align well with our target.

Let's go ahead and implement importance sampling with this uniform proposal:
''')

    st.code('''from scipy.stats import uniform

def importance_uniform(num_samples):
    """ Compute the expectation of m over the Kroupa IMF.
    Uses a uniform distribution as a proposal distribution.
    """
    expectation_f = 0
    denom = 0
    
    # Define the uniform proposal distribution over [0.01, 1.0]
    rv_m = uniform(loc=0.0, scale=1.)  # scale is the range, so 1.0 - 0.01
    
    for _ in range(num_samples):
        m = rv_m.rvs()
        p_tilde_val = kroupa_imf(m)  # Computing the value of the Kroupa IMF
        q_val = rv_m.pdf(m)
        
        weight = p_tilde_val / q_val  # Computing the importance weight
        expectation_f += m * weight  # Weighted sample
        denom += weight  # Accumulating the weight for normalization

    return expectation_f / denom  # Returning the normalized weighted average''', language='python')

    from scipy.stats import uniform
    
    def importance_uniform(num_samples):
        """ Compute the expectation of m over the Kroupa IMF.
        Uses a uniform distribution as a proposal distribution.
        """
        expectation_f = 0
        denom = 0
        
        # Define the uniform proposal distribution over [0.01, 1.0]
        rv_m = uniform(loc=0.0, scale=1.)  # scale is the range, so 1.0 - 0.01
        
        for _ in range(num_samples):
            m = rv_m.rvs()
            p_tilde_val = kroupa_imf(m)  # Computing the value of the Kroupa IMF
            q_val = rv_m.pdf(m)
            
            weight = p_tilde_val / q_val  # Computing the importance weight
            expectation_f += m * weight  # Weighted sample
            denom += weight  # Accumulating the weight for normalization
    
        return expectation_f / denom  # Returning the normalized weighted average
    
    

    st.markdown(r'''

While using the uniform distribution might seem straightforward, it's crucial to note the potential pitfalls. Since the Kroupa IMF has regions where its value is significantly higher than in other areas, using a uniform proposal might lead to a high variance in our estimates. This is because the uniformly drawn samples may not adequately capture the peaks and nuances of the Kroupa IMF, resulting in a lot of wasted samples in low-probability regions and inadequate sampling in high-probability regions.

By comparing the results obtained using the uniform proposal to those from the exponential proposal and the exact expectation, we'll observe firsthand the impact of the choice of proposal distribution on the efficiency and accuracy of importance sampling.''')

    st.code('''from tqdm import tqdm

# Generate a range of sample sizes in log space
sample_sizes = np.logspace(2, 5, 20).astype(int)

# Calculate the importance sampling estimate for each sample size
empirical_expectations = [importance_uniform(size) for size in tqdm(sample_sizes)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, empirical_expectations, label="Empirical (Importance Sampling)", marker='o')
plt.axhline(y=exact_expectation, color='r', linestyle='-', label="Analytical")
plt.xlabel("Number of Samples")
plt.ylabel("Expectation (Average Stellar Mass)")
plt.title("Convergence of Empirical Expectation with Sample Size")
plt.legend()
plt.xscale('log')
plt.ylim([0.15, 0.25])
plt.grid(True)
plt.show()
''', language='python')

    from tqdm import tqdm
    
    # Generate a range of sample sizes in log space
    sample_sizes = np.logspace(2, 5, 20).astype(int)
    
    # Calculate the importance sampling estimate for each sample size
    empirical_expectations = [importance_uniform(size) for size in tqdm(sample_sizes)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, empirical_expectations, label="Empirical (Importance Sampling)", marker='o')
    plt.axhline(y=exact_expectation, color='r', linestyle='-', label="Analytical")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Expectation (Average Stellar Mass)")
    ax.set_title("Convergence of Empirical Expectation with Sample Size")
    ax.legend()
    plt.xscale('log')
    plt.ylim([0.15, 0.25])
    plt.grid(True)
    st.pyplot(fig)
    
    

    st.markdown(r'''You'll notice the fluctuations in the empirical expectations as the number of samples increases. While the general trend suggests that the estimates do approach the true analytical expectation, the oscillations are evident. This waviness symbolizes the variability introduced by the uniform distribution's inability to appropriately capture the Kroupa IMF's behavior.

In essence, the larger variance in our importance sampling estimates emphasizes the significance of the proposal distribution choice. When the proposal distribution doesn't align well with the target distribution, as in the case of the uniform proposal for the Kroupa IMF, our "effective sample size" decreases. That means a lot of our samples might be from regions where the Kroupa IMF has little to no mass, making them virtually ineffective in our estimates.

This demonstration underscores the art and science of Monte Carlo methods, where thoughtful considerations, like selecting an apt proposal distribution, can drastically impact the efficiency and accuracy of our approximations.

---''')

    st.markdown(r'''### Summary:

The Kroupa Initial Mass Function (IMF) is a pivotal distribution in astrophysics, describing the distribution of stellar masses in stellar populations. Its intricate, piecewise structure provided an ideal testbed for exploring various statistical sampling techniques.

1. **Inverse CDF Sampling**: We initiated our exploration with this method, which is grounded in the principle of transforming uniform random samples. By harnessing the cumulative distribution function associated with the Kroupa IMF, we demonstrated a systematic process for deriving samples that adhere to the IMF's distribution.

2. **Rejection Sampling**: This technique illuminated the process of generating samples from intricate distributions using a supplementary proposal distribution. The core idea revolves around proposing samples and then either accepting or rejecting them based on the target distribution. Through this method, we visualized the synergy between the Kroupa IMF and our proposal distribution, offering insights into the efficacy and efficiency of the sampling process.

3. **Importance Sampling**: Our final foray was into the domain of Importance Sampling, a method especially pertinent for approximating the expectations of functions with respect to specific probability distributions. Utilizing the Kroupa IMF, we elucidated the mechanics of how a judiciously chosen proposal distribution can aid in approximating integrals, even if it does not precisely match the target distribution. This method showcased the significance of importance weights in rectifying the biases introduced by the proposal distribution.

In summary, we have navigated the nuances and applicabilities of various sampling methods in this tutorial. These methodologies, while presented in the context of astrophysics, have broader applications across numerous domains in science and engineering. The endeavor underscores the importance of rigorous statistical techniques in extracting meaningful information from complex distributions.''')

#if __name__ == '__main__':
show_page()
