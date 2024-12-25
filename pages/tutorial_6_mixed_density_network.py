import streamlit as st
from streamlit_app import navigation_menu

def show_page():
    st.set_page_config(page_title="Comp Astro",
                       page_icon="./image/tutor_favicon.png", layout="wide")

    navigation_menu()
    st.markdown(r'''# Mixed Density Network''')

    st.markdown(r'''In astronomy, we often encounter problems where we have to predict multiple possible outcomes for a given input. While traditional neural networks, such as multilayer perceptrons, output a deterministic result, Mixed Density Networks (MDNs) offer a more sophisticated approach, allowing predictions of entire probability distributions. This enables us to capture the inherent uncertainty and multimodal nature of many real-world problems.

In this tutorial, you'll explore the intriguing domain of Mixed Density Networks.

#### Key Topics

- **Designing an MDN**: A step-by-step guide on constructing a Mixed Density Network tailored for specific astronomical problems.

- **Probability distributions**: Understand how MDNs can predict a range of outcomes and the math behind it.

- **Training MDNs**: Dive into the intricacies of training an MDN, ensuring convergence and accurate predictions that capture the complexities of astronomical data.

- **Interpreting MDN outputs**: Techniques and best practices for visualizing and making sense of the probability distributions predicted by the MDN.

By mastering these concepts and techniques, you'll be equipped to handle astronomical problems that demand a richer form of prediction than what standard neural networks can offer.''')

    st.code('''# import the relevant packages
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader''', language='python')

    # import the relevant packages
    import numpy as np
    import matplotlib.pyplot as plt
    
    import torch
    import torch.nn as nn
    
    from torch.utils.data import TensorDataset, DataLoader
    
    

    st.markdown(r'''---

### Toy Example: Moon-shaped Dataset

Before diving deep into the intricacies of Mixed Density Networks, it's crucial to grasp why we might need them. Let's start with a simple toy dataset – the moon-shaped data from Scikit-Learn.

The moon-shaped dataset consists of two interleaving half circles (or moons), making it an example of non-linear and non-convex data. Traditional linear regression models would struggle with this dataset because for a given 'x' value, there are potentially multiple 'y' values, resulting in uncertainty in the predictions. This is where the idea of predicting a probability distribution (offered by MDNs) becomes attractive.

''')

    st.code('''# Importing the necessary library to generate moon-shaped data
from sklearn.datasets import make_moons

# Generate moon-shaped data with added noise for realism
X, _ = make_moons(n_samples=1000, noise=0.1, random_state=0)

# Redefining x and y for clarity and simplicity. 
x_train = X[:, 0]
y_train = X[:, 1]

# Plotting the moon-shaped data
plt.scatter(x_train, y_train, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Moon-shaped Dataset')''', language='python')

    fig, ax = plt.subplots()
    
    # Importing the necessary library to generate moon-shaped data
    from sklearn.datasets import make_moons
    
    # Generate moon-shaped data with added noise for realism
    X, _ = make_moons(n_samples=1000, noise=0.1, random_state=0)
    
    # Redefining x and y for clarity and simplicity. 
    x_train = X[:, 0]
    y_train = X[:, 1]
    
    # Plotting the moon-shaped data
    scatter = ax.scatter(x_train, y_train, s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Moon-shaped Dataset')
    st.pyplot(fig)
    
    

    st.markdown(r'''---

### Introduction to Mixed Density Networks (MDN)

A Mixed Density Network (MDN) is a specialized type of neural network designed to predict multiple potential outputs for a given input. It accomplishes this by integrating the modeling power of standard neural networks with a probabilistic framework, typically a mixture of Gaussian distributions.

At its core, an MDN doesn't just provide a deterministic output. Instead, it provides an entire probability distribution over potential outputs, giving a richer, more nuanced prediction. This is particularly useful in situations where an input can correspond to multiple plausible outputs.

#### The Mathematics Behind MDN

The primary components of an MDN's predictions are:

1. **Means (µ)**: Represents the central tendency of each Gaussian distribution.
2. **Standard Deviations (σ)**: Indicates the dispersion or spread of each Gaussian distribution. 
3. **Mixing Coefficients (w)**: Demonstrates the weight or importance of each Gaussian distribution in the mixture. All mixing coefficients sum up to 1.

The predicted output probability distribution $ P(y|x) $ for a given input $ x $ is a weighted sum of several Gaussian distributions:

$$
 P(y|x) = \sum_{k=1}^{M} w_k(x) \cdot \mathcal{N}(y| \mu_k(x), \sigma_k^2(x)) 
$$

where:
- $ M $ is the total number of Gaussian distributions.
- $ \mathcal{N} $ is the Gaussian (or normal) distribution.
- $ w_k, \mu_k, \sigma_k $ are the mixing coefficient, mean, and standard deviation of the $ k^{th} $ Gaussian, respectively, which are predicted by the neural network based on the input $ x $.

#### MDN Architecture

Given the components, our MDN needs to predict three values for each Gaussian in the mixture: $ \mu $, $ \sigma $, and $ w $. For $ K $ Gaussians, this requires the network to produce $ 3 \times K $ outputs.

In our implementation:

- The network has two hidden layers with `ReLU` activations, which help in capturing non-linear relationships.
- The output layer (`fc3`) produces $ 3 \times K $ values. We then split these values to get $ \mu $, $ \sigma $, and $ w $.
- We further ensure $ \sigma $ is positive (using the exponential function) and that $ w $ forms a valid probability distribution (using the softmax function).

By understanding the architecture and the mathematics behind MDN, we can better appreciate its ability to model complex, multi-modal data relationships, capturing the inherent uncertainty in many real-world problems.''')

    st.code('''class MDN(nn.Module):
    def __init__(self, n_hidden=20, n_gaussians=2, n_inputs=1):
        super().__init__()

        # Define internal parameters
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_inputs = n_inputs
        
        # Define the linear layers
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        
        # The output layer predicts µ, σ and w for each Gaussian.
        self.fc3 = nn.Linear(n_hidden, n_gaussians * 3)
        
    def forward(self, x):
        # Passing the input through the first two hidden layers with relu activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Passing through the final layer
        x = self.fc3(x)
        
        # Splitting the output to retrieve each parameter
        mu = x[:, :self.n_gaussians]
        sigma = x[:, self.n_gaussians:2*self.n_gaussians]
        w = x[:, 2*self.n_gaussians:]
        
        # Ensure that sigma is positive and w is a valid probability distribution across Gaussians.
        sigma = torch.exp(sigma)
        w = torch.softmax(w, dim=1)
        
        return mu, sigma, w
''', language='python')

    class MDN(nn.Module):
        def __init__(self, n_hidden=20, n_gaussians=2, n_inputs=1):
            super().__init__()
    
            # Define internal parameters
            self.n_hidden = n_hidden
            self.n_gaussians = n_gaussians
            self.n_inputs = n_inputs
            
            # Define the linear layers
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            
            # The output layer predicts µ, σ and w for each Gaussian.
            self.fc3 = nn.Linear(n_hidden, n_gaussians * 3)
            
        def forward(self, x):
            # Passing the input through the first two hidden layers with relu activation
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            
            # Passing through the final layer
            x = self.fc3(x)
            
            # Splitting the output to retrieve each parameter
            mu = x[:, :self.n_gaussians]
            sigma = x[:, self.n_gaussians:2*self.n_gaussians]
            w = x[:, 2*self.n_gaussians:]
            
            # Ensure that sigma is positive and w is a valid probability distribution across Gaussians.
            sigma = torch.exp(sigma)
            w = torch.softmax(w, dim=1)
            
            return mu, sigma, w
    
    

    st.markdown(r'''### Data Preprocessing and Model Initialization

Before we start training, we need to convert our data to PyTorch tensors and initialize the model.
''')

    st.code('''# Convert data to PyTorch tensors
x = torch.from_numpy(x_train).float().reshape(-1, 1)
y = torch.from_numpy(y_train).float().reshape(-1, 1)

# Create a dataset and dataloader for batching and shuffling
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Instantiate the MDN model and define the optimizer
model = MDN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)''', language='python')

    # Convert data to PyTorch tensors
    x = torch.from_numpy(x_train).float().reshape(-1, 1)
    y = torch.from_numpy(y_train).float().reshape(-1, 1)
    
    # Create a dataset and dataloader for batching and shuffling
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Instantiate the MDN model and define the optimizer
    model = MDN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    

    st.markdown(r'''### MDN Loss Function and Training

For MDNs, we utilize a custom loss function that measures how well the predicted Gaussian mixtures match the observed data. Specifically, we use the logarithm of the likelihood of the observed data under the Gaussian mixture model. Taking the negative log of the likelihood provides a quantity to minimize.

The reason we take the logarithm is primarily for numerical stability. Products of probabilities can be very small, leading to underflow. Summing logarithms of probabilities, however, avoids this issue.

In this training loop, we compute the model's predictions for each batch, calculate the loss, and then use backpropagation to adjust the model's weights.''')

    st.code('''# Training loss
def mdn_loss_fn(y, mu, sigma, w):
    gaussian = torch.distributions.Normal(mu, sigma)

    # Compute the likelihood of y under each Gaussian
    likelihood = torch.exp(gaussian.log_prob(y)) * w

    # Take the negative log likelihood
    loss = -torch.log(torch.sum(likelihood, dim=1))

    # Average over the entire batch
    loss = torch.mean(loss)
    
    return loss

# Training loop
for epoch in range(1000):
    for x_batch, y_batch in dataloader:
        mu, sigma, w = model(x_batch)
        loss = mdn_loss_fn(y_batch, mu, sigma, w)
        
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
    if epoch % 100 == 0:
        print('Epoch: {{}{}}, Loss: {{}:.4f{}}'.format(epoch, loss.item()))
''', language='python')

    # Training loss
    def mdn_loss_fn(y, mu, sigma, w):
        gaussian = torch.distributions.Normal(mu, sigma)
    
        # Compute the likelihood of y under each Gaussian
        likelihood = torch.exp(gaussian.log_prob(y)) * w
    
        # Take the negative log likelihood
        loss = -torch.log(torch.sum(likelihood, dim=1))
    
        # Average over the entire batch
        loss = torch.mean(loss)
        
        return loss
    
    # Training loop
    for epoch in range(1000):
        for x_batch, y_batch in dataloader:
            mu, sigma, w = model(x_batch)
            loss = mdn_loss_fn(y_batch, mu, sigma, w)
            
            optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
        if epoch % 100 == 0:
            st.write('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
    
    

    st.markdown(r'''### Visualizing the Gaussian Predictions

Once we've trained our Mixed Density Network (MDN), it's important to visually inspect how the predicted Gaussian distributions fit our data. Visualization not only provides insight into the model's predictions but also helps identify areas of uncertainty and multi-modal behaviors.

The first step is to generate predictions using a set of input values that span the range of our dataset.

- `torch.linspace(-2, 3, 100)` creates 100 evenly spaced values between -2 and 3. These values span the $x$ range of our dataset and will be used as inputs to the model.

- We then use our trained model to predict the parameters $ \mu $ (mean), $ \sigma $ (standard deviation), and $ w $ (mixing coefficient) for each input $x$.''')

    st.code('''# Generate a series of input values for prediction
x_pred = torch.linspace(-2, 3, 100).reshape(-1, 1)

# Obtain the predicted parameters of the Gaussian distributions from the model
mu, sigma, alpha = model(x_pred)''', language='python')

    # Generate a series of input values for prediction
    x_pred = torch.linspace(-2, 3, 100).reshape(-1, 1)
    
    # Obtain the predicted parameters of the Gaussian distributions from the model
    mu, sigma, alpha = model(x_pred)
    
    

    st.markdown(r'''The MDN predicts parameters for multiple Gaussians. Above, we have assume two Gaussians, we'll plot each Gaussian (the mean and the 1 $\sigma$ range) separately to understand the individual contribution of each.

By plotting the Gaussians separately, we can discern the contribution and importance of each Gaussian to the overall model. It provides insight into regions where there is more uncertainty (wider sigma) or where multiple outputs (modes) are plausible. This visualization helps us appreciate the power and flexibility of MDNs in modeling complex relationships.''')

    st.code('''# Define colors for each Gaussian
color = ['r', 'g']

# Plot the 1-sigma range for each Gaussian
for i in range(2):

    # Plot the mean of the Gaussian
    plt.plot(x_pred, mu[:, i].detach().numpy(), c=color[i], label='Gaussian Component {{}{}}'.format(i+1)) 
       
    # Plot the 1-sigma range around the mean, creating a shaded region
    plt.fill_between(x_pred.ravel(),
                        mu[:, i].detach().numpy() - sigma[:, i].detach().numpy(),
                        mu[:, i].detach().numpy() + sigma[:, i].detach().numpy(),
                        alpha=0.5, color=color[i])

# Plot the original data points
plt.scatter(x_train.ravel(), y_train.ravel(), s=5, alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')

plt.legend(frameon=False)''', language='python')

    fig, ax = plt.subplots()
    
    # Define colors for each Gaussian
    color = ['r', 'g']
    
    # Plot the 1-sigma range for each Gaussian
    for i in range(2):
    
        # Plot the mean of the Gaussian
        ax.plot(x_pred, mu[:, i].detach().numpy(), c=color[i], label='Gaussian Component {}'.format(i+1)) 
           
        # Plot the 1-sigma range around the mean, creating a shaded region
        plt.fill_between(x_pred.ravel(),
                            mu[:, i].detach().numpy() - sigma[:, i].detach().numpy(),
                            mu[:, i].detach().numpy() + sigma[:, i].detach().numpy(),
                            alpha=0.5, color=color[i])
    
    # Plot the original data points
    scatter = ax.scatter(x_train.ravel(), y_train.ravel(), s=5, alpha=0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.legend(frameon=False)
    st.pyplot(fig)
    
    

    st.markdown(r'''The ability of Mixed Density Networks (MDN) to model complex, multimodal data distributions lies not just in predicting multiple Gaussian parameters but also in determining the relative importance of each Gaussian. This importance is given by the mixing coefficients, often denoted $ w $.

While we've visualized the means and standard deviations of the Gaussians in the previous plots, it's equally crucial to understand how the mixing coefficients change based on the input. By doing so, we gain insight into which Gaussian mode the network deems more probable for a given input.

The plot below shows the network dynamically adjusts the mixing coefficients based on input data. In regions where there's a clear dominance of one mode over the other, the respective weight will be significantly higher. This ability to adjust the importance of each mode ensures that the MDN captures the intricate patterns and behaviors present in our data.''')

    st.code('''# Plot the mixing coefficients for each Gaussian
for i in range(2):
    plt.plot(x_pred, alpha[:, i].detach().numpy(), c=color[i], label=f'Weight of Gaussian {{}i+1{}}')
             
plt.xlim(-1, 2)
plt.legend(frameon=False)

plt.xlabel('x')
plt.ylabel('Weight')
''', language='python')

    fig, ax = plt.subplots()
    
    # Plot the mixing coefficients for each Gaussian
    for i in range(2):
        ax.plot(x_pred, alpha[:, i].detach().numpy(), c=color[i], label=f'Weight of Gaussian {i+1}')
                 
    plt.xlim(-1, 2)
    ax.legend(frameon=False)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Weight')
    st.pyplot(fig)
    
    

    st.markdown(r'''### Visualizing the Conditional Distribution $ p(y|x) $

One of the powerful features of MDNs is their ability to capture conditional distributions. This distribution represents the range and likelihood of potential outcomes $ y $ for a given input $ x $. By sampling from this distribution, we can visualize the range of predictions and their respective probabilities.

To visualize this distribution, we will sample from the predicted Gaussians based on their parameters and mixing coefficients. By doing so, we obtain a representation of the possible values of $ y $ and their likelihoods for different values of $ x $.

- We pick three $x$-values for which we want to visualize the conditional distributions.

- For each $x$-value:
  - We obtain the predicted Gaussian parameters from our MDN.
  - We draw a large number of samples from each Gaussian.
  - To represent the relative importance of each Gaussian, we subsample the drawn samples based on their respective mixing coefficients.
  - We then combine the subsampled values from both Gaussians to get the final sample representation for the current $x$-value.
  - A Kernel Density Estimation (KDE) plot is then used to visualize the combined samples. This gives a smoother representation of the underlying distribution.

By inspecting these plots, you'll notice how the conditional distribution changes with different $x$-values, reflecting the multimodal behavior captured by the MDN. This helps us understand the range and likelihood of possible outcomes for various inputs.''')

    st.code('''# We'll use seaborn for enhanced visualization
import seaborn as sns

# We choose three x-values to inspect the conditional distributions
x_pred = torch.linspace(-1,1,3).reshape(-1, 1)
mu, sigma, alpha = model(x_pred)

# Colors for the different x-values' distributions
color = ['r', 'g', 'b']

# Loop over the three x-values
for i in range(3):
    list_samples = []

    # For each Gaussian component
    for j in range(2):
        gaussian = torch.distributions.Normal(mu[i, j], sigma[i, j])

        # Draw a large number of samples from the Gaussian
        samples = gaussian.sample((10000,))

        # Filter or subsample these based on the mixing coefficient alpha
        subsamples = samples[torch.rand(samples.shape[0]) < alpha[i, j]]
        
        # Gather the samples from both Gaussians
        list_samples.append(subsamples.detach().numpy())

    # We then combine the samples from both Gaussians
    list_samples = np.concatenate(list_samples)

    # Use kernel density estimation to plot the estimated distribution of y for the current x
    sns.kdeplot(list_samples, color=color[i], label='x={{}:.1f{}}'.format(x_pred[i, 0]))

plt.xlabel('y')
plt.ylabel('p(y|x)')
plt.legend(frameon=False)''', language='python')

    fig, ax = plt.subplots()
    
    # We'll use seaborn for enhanced visualization
    import seaborn as sns
    
    # We choose three x-values to inspect the conditional distributions
    x_pred = torch.linspace(-1,1,3).reshape(-1, 1)
    mu, sigma, alpha = model(x_pred)
    
    # Colors for the different x-values' distributions
    color = ['r', 'g', 'b']
    
    # Loop over the three x-values
    for i in range(3):
        list_samples = []
    
        # For each Gaussian component
        for j in range(2):
            gaussian = torch.distributions.Normal(mu[i, j], sigma[i, j])
    
            # Draw a large number of samples from the Gaussian
            samples = gaussian.sample((10000,))
    
            # Filter or subsample these based on the mixing coefficient alpha
            subsamples = samples[torch.rand(samples.shape[0]) < alpha[i, j]]
            
            # Gather the samples from both Gaussians
            list_samples.append(subsamples.detach().numpy())
    
        # We then combine the samples from both Gaussians
        list_samples = np.concatenate(list_samples)
    
        # Use kernel density estimation to plot the estimated distribution of y for the current x
        sns.kdeplot(list_samples, color=color[i], label='x={:.1f}'.format(x_pred[i, 0]))
    
    ax.set_xlabel('y')
    ax.set_ylabel('p(y|x)')
    ax.legend(frameon=False)
    st.pyplot(fig)
    
    

    st.markdown(r'''---

## Investigating Lithium Abundance in Stars

In this section, we dive into a fascinating aspect of stellar astrophysics: the abundance of the element Lithium in stars, and how it evolves based on specific stellar characteristics. 

### Why is Lithium Abundance Important?

Lithium offers a unique window into the interior workings of a star.

1. **Fragility of Lithium**: Lithium is a rather delicate element. It gets destroyed when it reaches the base of a star's convective layer, a turbulent region where energy is transported by bulk fluid movements. The destruction of Lithium in this layer gives insight into the depth and nature of these convective motions.
  
2. **Indicator of Stellar Interior Structures**: A star with deeper convective layers (typically cooler stars with lower effective temperatures $T_{\rm eff}$) will destroy more Lithium. As such, the amount of Lithium present can offer clues about a star's internal structure.

3. **Chronometer of Stellar Ages**: The amount of Lithium depletion also correlates with a star's age. Older stars have had more time to destroy their Lithium, especially if they have active convective layers.

### However, It's Complicated!

While the factors mentioned above play significant roles, Lithium depletion is intricate. It's influenced by other factors besides just the effective temperature and a star's age. Moreover, the amount of Lithium in stars can have some randomness due to the stochastic nature of Lithium production in prior generations of stars.

Thus, we shouldn't expect a straightforward deterministic relationship of the form:

$$
 (T_{\rm eff}, {\rm \tau}) \rightarrow A_{\rm Li}
$$

where $ \tau $ signifies the star's age, and $ A_{\rm Li} $ represents its absolute Lithium abundance.

Instead, given the inherent complexities and uncertainties, it's more apt to model this as a probabilistic relationship:

$$
 p(A_{\rm Li} | \tau, T_{\rm eff}) 
$$

some that which we will model through an MDN.

### Training Dataset: The Role of Open Clusters

For our investigation, we'll employ a dataset derived from four open star clusters.

Open clusters are groups of stars born from the same giant molecular cloud, so they share a similar age and initial elemental composition. This makes them invaluable for our study:

- **Age Determination**: Determining the age of individual stars can be challenging. However, for stars in an open cluster, their collective age can be deduced using the Color-Magnitude Diagram, a tool unavailable for isolated field stars.

Using open clusters offers a more controlled dataset, minimizing some variables and letting us focus on the relationship between Lithium abundance, age, and effective temperature.

Let's load in the dataset and visualize how the Lithium abundance in stars varies with their effective temperature.
''')

    st.code('''# Load the Lithium dataset
data = np.load('lithium_sample_tutorial_week10a.npz')

# Extract individual data columns for age, effective temperature, and Lithium abundance
age = data['age']
teff = data['teff']
ALi = data['ALi']

# Create a scatter plot of Lithium Abundance (ALi) vs. Effective Temperature (Teff)
# The color of each point represents the age of the star (color-coded by age).
plt.scatter(teff, ALi, c=age, cmap='viridis', s=10)

# Add a colorbar to indicate ages and label axes for clarity
plt.colorbar().set_label('Age (Gyr)')
plt.xlabel('Effective Temperature (K)')
plt.ylabel('Lithium Abundance')
plt.show()''', language='python')

    fig, ax = plt.subplots()
    
    # Load the Lithium dataset
    import requests
    from io import BytesIO
    
    # Load the dataset using np.load
    response = requests.get('https://storage.googleapis.com/compute_astro/lithium_sample_tutorial_week10a.npz')
    f = BytesIO(response.content)
    data = np.load(f, allow_pickle=False)
    
    # Extract individual data columns for age, effective temperature, and Lithium abundance
    age = data['age']
    teff = data['teff']
    ALi = data['ALi']
    
    # Create a scatter plot of Lithium Abundance (ALi) vs. Effective Temperature (Teff)
    # The color of each point represents the age of the star (color-coded by age).
    scatter = ax.scatter(teff, ALi, c=age, cmap='viridis', s=10)
    
    # Add a colorbar to indicate ages and label axes for clarity
    fig.colorbar(scatter, ax=ax, ).set_label('Age (Gyr)')
    ax.set_xlabel('Effective Temperature (K)')
    ax.set_ylabel('Lithium Abundance')
    st.pyplot(fig)
    

    st.markdown(r'''As expected the Lithium abundance is depleted more for cooler and older stars. 

However, given the scatter in the data, it's evident that a single deterministic model might not capture the underlying relationship between effective temperature ($ T_{\rm eff} $), age, and Lithium abundance ($ A_{\rm Li} $) well. Instead, a probabilistic approach, which estimates a range of possible $ A_{\rm Li} $ values for a given $ T_{\rm eff} $ and age, is more apt. This is where an MDN shines.

An MDN returns a mixture of Gaussian distributions for each input. In our context:

1. **Inputs**:
   - Effective temperature ($ T_{\rm eff} $).
   - Age of the star.

2. **Output**:
   - A mixture of Gaussian distributions representing possible values of $ A_{\rm Li} $.

The goal of our MDN is not to predict a single value but rather to estimate a range (distribution) of $ A_{\rm Li} $ values for given inputs.

Let's structure our training process as before. By the end of the training, our MDN will offer a probabilistic understanding of $ A_{\rm Li} $ given a star's effective temperature and age, reflecting the inherent complexities in the processes determining Lithium abundance in stars.
''')

    st.code('''# Preparing the input and output data for our model
# Our input data consists of both effective temperature (Teff) and age
x = np.stack([teff, age], axis=1)
y = ALi.reshape(-1, 1)

# Normalize the temperature by 1000K so that it is of the same order of magnitude as age
x[:, 0] /= 1000

# Convert the data into PyTorch tensors
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float().reshape(-1, 1)

# Preparing a dataset and dataloader for efficient batching during training
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Defining our MDN model
model = MDN(n_hidden=50, n_gaussians=5, n_inputs=2)

# We'll use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training our model
for epoch in range(5000):
    for x_batch, y_batch in dataloader:
        mu, sigma, alpha = model(x_batch)
        loss = mdn_loss_fn(y_batch, mu, sigma, alpha)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 500 == 0:
        print(f'Epoch: {{}epoch{}}, Loss: {{}loss.item():.4f{}}')
''', language='python')

    # Preparing the input and output data for our model
    # Our input data consists of both effective temperature (Teff) and age
    x = np.stack([teff, age], axis=1)
    y = ALi.reshape(-1, 1)
    
    # Normalize the temperature by 1000K so that it is of the same order of magnitude as age
    x[:, 0] /= 1000
    
    # Convert the data into PyTorch tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    
    # Preparing a dataset and dataloader for efficient batching during training
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Defining our MDN model
    model = MDN(n_hidden=50, n_gaussians=5, n_inputs=2)
    
    # We'll use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training our model
    for epoch in range(5000):
        for x_batch, y_batch in dataloader:
            mu, sigma, alpha = model(x_batch)
            loss = mdn_loss_fn(y_batch, mu, sigma, alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 500 == 0:
            st.write(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
    
    

    st.markdown(r'''### Visualizing the Results of the MDN

After training our MDN on the training data, our aim is to visually ascertain the relationship between effective temperature, age, and Lithium abundance. The primary purpose of this visualization is twofold:

1. **Mean Trend**: Understanding how the mean Lithium abundance changes with effective temperature at different ages. The gradient in color signifies the mean Lithium abundance across effective temperatures for various ages. It provides a holistic picture of how the mean abundance evolves.

2. **Scatter (Variability)**: Capturing the spread or variability in the Lithium abundance predictions for a given effective temperature and age. The plot represents the scatter or variability in the predictions. Brighter regions denote higher variability in Lithium abundance across the stellar population.

Let's delve into these visualizations:''')

    st.code('''# Define age grid for predictions and get a color map for visualization
age_list = np.linspace(0.1, 1, 100)
cm = plt.cm.get_cmap('viridis')

# Lists to store predictions for each age
all_mus = []     # Stores means of A_Li predictions
all_sigmas = []  # Stores scatter (standard deviations) of A_Li predictions

#-----------------------------
# Iterate over age values
for age_choose in age_list:

    # Create a tensor of effective temperatures (Teff) to predict A_Li values
    x_pred = torch.linspace(4., 8., 100).reshape(-1, 1)
    x_pred = torch.cat([x_pred, torch.ones_like(x_pred) * age_choose], dim=1)
    
    # Use the model to predict A_Li
    mu, sigma, alpha = model(x_pred)

    list_mu = []
    list_1sigma = []

    # For each Teff, sampler from weighted mixture of Gaussians to get mean and scatter
    for i in range(len(x_pred)):
        list_samples = []

        # For each Gaussian in the mixture
        for j in range(2):
            gaussian = torch.distributions.Normal(mu[i, j], sigma[i, j])
            samples = gaussian.sample((10000,))
            subsamples = samples[torch.rand(samples.shape[0]) < alpha[i, j]]
            list_samples.append(subsamples.detach().numpy())

        list_samples = np.concatenate(list_samples)
        list_mu.append(np.mean(list_samples))
        list_1sigma.append(np.std(list_samples))
    
    all_mus.append(list_mu)
    all_sigmas.append(list_1sigma)

#-----------------------------
# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Plot A_Li vs Teff, where color represents age
for i, age_choose in enumerate(age_list):
    ax1.plot(x_pred[:, 0].detach().numpy() * 1000., all_mus[i], c=cm(age_choose))
sc1 = ax1.scatter(teff, ALi, c=age, cmap='viridis', s=15, zorder=10, edgecolor='w') # plot data points
ax1.set_xlabel('Teff')
ax1.set_ylabel('A(Li)')
ax1.set_title('Mean A(Li) vs Teff (Color-coded by Age)')
cbar1 = fig.colorbar(sc1, ax=ax1)
cbar1.set_label('Age (Gyr)')

# Right plot: Plot A_Li vs Teff, where color represents the scatter in predictions
teff_grid = x_pred[:, 0].detach().numpy() * 1000.
teff_repeated = np.tile(teff_grid, len(age_list))
scatter_flat = np.array(all_sigmas).flatten()
norm = plt.Normalize(0.2, 0.5)
sc2 = ax2.scatter(teff_repeated, np.array(all_mus).flatten(),
                  c=scatter_flat, cmap='viridis', norm=norm, s=15)
ax2.set_xlabel('Teff')
ax2.set_ylabel('A(Li)')
ax2.set_title('Mean A(Li) vs Teff (Color-coded by Scatter)')
cbar2 = fig.colorbar(sc2, ax=ax2)
cbar2.set_label('Scatter (dex)')

plt.tight_layout()
''', language='python')

    # Define age grid for predictions and get a color map for visualization
    age_list = np.linspace(0.1, 1, 100)
    cm = plt.cm.get_cmap('viridis')
    
    # Lists to store predictions for each age
    all_mus = []     # Stores means of A_Li predictions
    all_sigmas = []  # Stores scatter (standard deviations) of A_Li predictions
    
    #-----------------------------
    # Iterate over age values
    for age_choose in age_list:
    
        # Create a tensor of effective temperatures (Teff) to predict A_Li values
        x_pred = torch.linspace(4., 8., 100).reshape(-1, 1)
        x_pred = torch.cat([x_pred, torch.ones_like(x_pred) * age_choose], dim=1)
        
        # Use the model to predict A_Li
        mu, sigma, alpha = model(x_pred)
    
        list_mu = []
        list_1sigma = []
    
        # For each Teff, sampler from weighted mixture of Gaussians to get mean and scatter
        for i in range(len(x_pred)):
            list_samples = []
    
            # For each Gaussian in the mixture
            for j in range(2):
                gaussian = torch.distributions.Normal(mu[i, j], sigma[i, j])
                samples = gaussian.sample((10000,))
                subsamples = samples[torch.rand(samples.shape[0]) < alpha[i, j]]
                list_samples.append(subsamples.detach().numpy())
    
            list_samples = np.concatenate(list_samples)
            list_mu.append(np.mean(list_samples))
            list_1sigma.append(np.std(list_samples))
        
        all_mus.append(list_mu)
        all_sigmas.append(list_1sigma)
    
    #-----------------------------
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Plot A_Li vs Teff, where color represents age
    for i, age_choose in enumerate(age_list):
        ax1.plot(x_pred[:, 0].detach().numpy() * 1000., all_mus[i], c=cm(age_choose))
    sc1 = ax1.scatter(teff, ALi, c=age, cmap='viridis', s=15, zorder=10, edgecolor='w') # plot data points
    ax1.set_xlabel('Teff')
    ax1.set_ylabel('A(Li)')
    ax1.set_title('Mean A(Li) vs Teff (Color-coded by Age)')
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label('Age (Gyr)')
    
    # Right plot: Plot A_Li vs Teff, where color represents the scatter in predictions
    teff_grid = x_pred[:, 0].detach().numpy() * 1000.
    teff_repeated = np.tile(teff_grid, len(age_list))
    scatter_flat = np.array(all_sigmas).flatten()
    norm = plt.Normalize(0.2, 0.5)
    sc2 = ax2.scatter(teff_repeated, np.array(all_mus).flatten(),
                      c=scatter_flat, cmap='viridis', norm=norm, s=15)
    ax2.set_xlabel('Teff')
    ax2.set_ylabel('A(Li)')
    ax2.set_title('Mean A(Li) vs Teff (Color-coded by Scatter)')
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Scatter (dex)')
    
    st.pyplot(fig)
    
    

    st.markdown(r'''In this script, we aim to visualize how the predicted lithium abundance ($ A_{\rm Li} $) varies with effective temperature ($T_{\rm eff}$) based on a Gaussian mixture model. The key features of this script are:

1. **Creating a Prediction Grid**: For each age in `age_list`, we create a tensor `x_pred` that encompasses a range of effective temperatures at that age. This allows us to see how $ A_{\rm Li} $ predictions vary for a specific age across different $T_{\rm eff}$ values.

2. **Model Predictions**: For each `x_pred`, our model predicts three things: `mu` (the means of the Gaussian mixture), `sigma` (standard deviations), and `alpha` (mixture weights). These are used to compute a weighted average of $ A_{\rm Li} $ for the given $T_{\rm eff}$ and age.

3. **Visualizing Predictions**: 
   - **Left Plot**: Shows the mean $ A_{\rm Li} $ against $T_{\rm eff}$. The color of each line represents a specific age, making it easy to see how age influences $ A_{\rm Li} $ for a given $T_{\rm eff}$.
   - **Right Plot**: Visualizes the scatter (or standard deviation) in the $ A_{\rm Li} $ predictions. The scatter provides insights into the uncertainty or variation in our predictions for a given $T_{\rm eff}$ and age.

By analyzing these plots, one can obtain a holistic view of how $ A_{\rm Li} $ varies with $T_{\rm eff}$ and understand the underlying uncertainties in our model's predictions as well as the evolution of Lithium.
''')

    st.markdown(r'''---
    
## Conclusion

In this tutorial, we embarked on an exploration of Mixture Density Networks starting from foundational concepts to advanced applications. 

We began our journey with a simple but illustrative toy example using the double moon shape dataset from `sklearn`. This dataset served as a tangible visualization tool to highlight the limitations of conventional neural networks when modeling multi-modal data distributions. Through this, we observed the power of MDNs in capturing complex data structures.

Building on this foundation, we transitioned to a more sophisticated use case - studying Lithium abundance in stars of open clusters. By employing MDNs, we aimed to decipher intricate relationships possibly influenced by factors like age and effective temperature. MDN is a compelling way to comprehend the intricate relations and uncertainties in the predictions, demonstrating the prowess of MDNs in practical scientific applications.

In essence, MDNs provide a versatile tool to model complex distributions in a wide range of scenarios, bridging the gap between deterministic predictions and probabilistic understanding.
''')

#if __name__ == '__main__':
show_page()
