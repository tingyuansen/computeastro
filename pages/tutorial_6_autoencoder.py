import streamlit as st
from streamlit_app import navigation_menu

def show_page():
    st.set_page_config(page_title="Comp Astro",
                       page_icon="./image/tutor_favicon.png", layout="wide")

    navigation_menu()
    st.markdown(r'''# Autoencoder''')

    st.markdown(r'''In today's data-driven world, we are frequently presented with large and intricate datasets. Directly dissecting this information is not only challenging but can be quite taxing. This is where neural network-based dimensionality reduction techniques, like autoencoders, become invaluable. These methods enable us to encode vast amounts of data into a more digestible format, making it easier to process and interpret.

In this tutorial, you'll delve deep into the world of autoencoders.

#### Key Topics

- **Understanding autoencoders**: Delve into the mechanics behind these neural networks.

- **Step-by-step guide on implementing autoencoders**: Learn the nuances of designing and training an autoencoder for data compression.

- **Visualization techniques**: Explore how to represent high-dimensional data in a simplified manner using the encoded representations.

- **Reconstruction of data**: Understand how to decode the compact representation back to its original form, and assess the quality of the reconstruction.

- **Interpreting results**: Best practices for analyzing the efficiency of your autoencoder and the insights it provides.

By grasping these concepts and methodologies, you'll be poised to navigate vast and intricate datasets efficiently in your upcoming projects. This tutorial caters to all – whether you're stepping into the realm of neural networks for the first time or are looking to expand your knowledge.
''')

    st.code('''import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
%matplotlib inline''', language='python')

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.optimize as opt
    
    

    st.markdown(r'''--- 

## Encoding Pokémon: Diving into Autoencoders

Recall our exploration with the Pokémon dataset using PCA in our previous tutorial? Today, we'll revisit that same dataset, but this time we'll be using autoencoders to grasp its intricacies. When dealing with image datasets, like the ones of Pokémon, the encoded representations through autoencoders often capture significant attributes or patterns central to the images.

In the PCA tutorial, you might remember the visual patterns, or 'Eigen-Pokémon', that represented the essence of the Pokémon variations. Similarly, with autoencoders, we can capture essential characteristics that define and differentiate these Pokémon images. To help in our exploration, we will visualize these compressed representations and see how they compare and contrast with the PCA approach.

Let's first import the data and plot the data as last time.

- **Data Structure**: The dataset is a 2D array, where each row represents a greyscale image of a Pokemon sprite.
- **Image Dimensions**: Each image is 64x64 pixels, resulting in a total of 4096 pixels per image.
- **Pixel Values**: Pixel intensity values are floating-point numbers between 0 and 1, where 0 stands for white and 1 for black.''')

    st.code('''# Load the dataset using np.loadtxt
images = np.loadtxt('pokemon_sample_tutorial_week10a.csv')

# Check the shape of the loaded data
print("Shape of the loaded images:", images.shape)''', language='python')

    # load data
    import requests
    from io import BytesIO
    
    # Load the dataset using np.load
    response = requests.get('https://storage.googleapis.com/compute_astro/pokemon_sample_tutorial_week9a.npz')
    f = BytesIO(response.content)
    data = np.load(f, allow_pickle=True)
    images = data["images"]
    
    # Check the shape of the loaded data
    st.write("Shape of the loaded images:", images.shape)
    
    

    st.code('''def plot_gallery(images, titles, h, w, n_row=2, n_col=6):
    """
    Plot a gallery of images.
    
    Parameters:
        images (numpy.ndarray): Each row is a flattened image.
        titles (list): Titles for each subplot.
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        n_row (int): Number of rows in the gallery.
        n_col (int): Number of columns in the gallery.
    """
    # Assert to make sure we have enough images and titles for the grid
    assert len(images) >= n_row * n_col
    assert len(titles) >= n_row * n_col
    
    # Initialize the plot
    fig, axes = plt.subplots(n_row, n_col, figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    
    axes = axes.flatten()

    # Loop to populate the gallery
    for i in range(n_row * n_col):
        ax = axes[i]
        
        # Reshape the flattened image data to 2D and plot it
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        
        # Add title and remove axis ticks
        ax.set_title(titles[i], size=12)
        ax.set_xticks(())
        ax.set_yticks(())
        
    plt.show()''', language='python')

    def plot_gallery(images, titles, h, w, n_row=2, n_col=6):
        """
        Plot a gallery of images.
        
        Parameters:
            images (numpy.ndarray): Each row is a flattened image.
            titles (list): Titles for each subplot.
            h (int): Image height in pixels.
            w (int): Image width in pixels.
            n_row (int): Number of rows in the gallery.
            n_col (int): Number of columns in the gallery.
        """
        # Assert to make sure we have enough images and titles for the grid
        assert len(images) >= n_row * n_col
        assert len(titles) >= n_row * n_col
        
        # Initialize the plot
        fig, axes = plt.subplots(n_row, n_col, figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        
        axes = axes.flatten()
    
        # Loop to populate the gallery
        for i in range(n_row * n_col):
            ax = axes[i]
            
            # Reshape the flattened image data to 2D and plot it
            ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            
            # Add title and remove axis ticks
            ax.set_title(titles[i], size=12)
            ax.set_xticks(())
            ax.set_yticks(())
            
        st.pyplot(fig)
    

    st.markdown(r'''To use `plot_gallery`, we will simply pass in the first 30 Pokémon images and their corresponding labels. The function will handle the plotting for us.

This will give you a nice gallery view of the first 30 Pokémon images in your dataset, helping you to visually comprehend the diversity and features in your data.''')

    st.code('''plot_gallery(images, np.arange(30), 64, 64, 5, 6)''', language='python')

    plot_gallery(images, np.arange(30), 64, 64, 5, 6)
    
    

    st.markdown(r'''---

## Implementing Autoencoders in PyTorch: A Deep Dive

In our prior discussions, we delved into dimensionality reduction using Principal Component Analysis (PCA) as a technique to represent Pokémon images in a lower-dimensional space. While PCA offers linear projections of data to capture the most variance, neural-based methods, like autoencoders, offer more nuanced and intricate representations by leveraging non-linear transformations.

Autoencoders are neural networks designed to encode data into a compressed representation and subsequently decode it to reconstruct the input. This dual mechanism of encoding and decoding ensures that the compressed representation captures the most salient features of the data.

### Weight Initialization: Xavier Initialization

Training deep neural networks, like autoencoders, presents challenges. Among them, the initialization of network weights is paramount. Improper initialization can lead to erratic learning dynamics, with the network suffering from vanishing or exploding gradients, thereby hindering convergence.

Xavier Initialization, named after its primary author Xavier Glorot, addresses this challenge. The central tenet of Xavier Initialization is to maintain the variance of activations across layers, ensuring that neither the activations nor their gradients reach extremely high or low values during forward and backward passes, respectively.

Mathematically, if a layer has $ n_{in} $ incoming connections and $ n_{out} $ outgoing connections, Xavier initialization samples the weights $ W $ from a distribution with a variance given by:

$$
 \mathrm{Var}(W) = \dfrac{2}{n_{in} + n_{out}} 
$$

The rationale behind this distribution is derived from ensuring that the variance remains consistent throughout the network layers, thus fostering stable learning dynamics.


### Autoencoder Architecture in PyTorch

Autoencoders consist of two main components: the **encoder**, which reduces input data into a lower-dimensional encoded representation, and the **decoder**, which reconstructs the input data from this representation. This symmetry in design ensures that the autoencoder learns a compact representation that retains as much information about the original data as possible.

Let's dive deeper into the structure and implementation.

- **`nn.Module`**: This is the base class for all neural network modules in PyTorch. Our `AutoEncoder` class inherits from it, giving us access to various useful methods and attributes.

- **Encoder**: The encoder begins with an input size of 4096, which could represent a flattened 64x64 image. It compresses this input in stages: first to 512 nodes, then to 128, and finally to a mere 32 nodes. This progressive compression forces the network to learn a compact representation of the input data. The `nn.ReLU()` activations introduce non-linearity after each linear transformation, enabling the network to learn more complex patterns.

- **Decoder**: The decoder's role is to take the compressed representation (32 nodes) and reconstruct the original input. It mirrors the encoder's structure but in reverse, progressively expanding the representation until it reaches the original size of 4096.

- **`forward` Method**: This method defines the forward pass of the network. Data flows through the encoder, getting compressed, and then through the decoder, where it's reconstructed. This method is called when we pass input data to an instance of our `AutoEncoder` class.

With this architecture, our autoencoder is set to learn to represent the data in a condensed form, with the ability to reconstruct the input from this compressed representation. Proper training, evaluation, and tuning can further enhance its performance, providing a robust tool for dimensionality reduction tasks.''')

    st.code('''import torch
import torch.nn as nn

# set random seed for reproducibility
torch.manual_seed(42)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(4096, 512),   # Layer 1: Linear transformation from 4096 input features to 512 nodes
            nn.ReLU(),              # Layer 1: Activation function to introduce non-linearity
            nn.Linear(512, 128),    # Layer 2: Further compressing from 512 nodes to 128 nodes
            nn.ReLU(),              # Layer 2: Activation
            nn.Linear(128, 32),     # Layer 3: The densest compression, reducing the data to just 32 nodes
            nn.ReLU()               # Layer 3: Activation
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),     # Layer 1: Begin the reconstruction by expanding from 32 to 128 nodes
            nn.ReLU(),              # Layer 1: Activation
            nn.Linear(128, 512),    # Layer 2: Expand from 128 nodes to 512 nodes
            nn.ReLU(),              # Layer 2: Activation
            nn.Linear(512, 4096),   # Layer 3: Final reconstruction to the original 4096 input features
          )
    
    def forward(self, x):
        x = self.encoder(x)        # Pass the input through the encoder
        x = self.decoder(x)        # Pass the encoded representation through the decoder
        return x                   # Return the reconstructed data
''', language='python')

    import torch
    import torch.nn as nn
    
    # set random seed for reproducibility
    torch.manual_seed(42)
    
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()
    
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(4096, 512),   # Layer 1: Linear transformation from 4096 input features to 512 nodes
                nn.ReLU(),              # Layer 1: Activation function to introduce non-linearity
                nn.Linear(512, 128),    # Layer 2: Further compressing from 512 nodes to 128 nodes
                nn.ReLU(),              # Layer 2: Activation
                nn.Linear(128, 32),     # Layer 3: The densest compression, reducing the data to just 32 nodes
                nn.ReLU()               # Layer 3: Activation
            )
    
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(32, 128),     # Layer 1: Begin the reconstruction by expanding from 32 to 128 nodes
                nn.ReLU(),              # Layer 1: Activation
                nn.Linear(128, 512),    # Layer 2: Expand from 128 nodes to 512 nodes
                nn.ReLU(),              # Layer 2: Activation
                nn.Linear(512, 4096),   # Layer 3: Final reconstruction to the original 4096 input features
              )
        
        def forward(self, x):
            x = self.encoder(x)        # Pass the input through the encoder
            x = self.decoder(x)        # Pass the encoded representation through the decoder
            return x                   # Return the reconstructed data
    
    

    st.markdown(r'''PyTorch provides built-in functions to simplify the application of Xavier initialization. Let's dissect the following code segment to grasp its workings:

- **`model.modules()`**:
  - This is a generator method for the model that iterates through all its sub-modules (layers). For our `AutoEncoder` class, it would include the encoder and decoder, as well as the individual linear layers and activation functions.


- **`if isinstance(module, nn.Linear)`**:
  - We only want to apply Xavier initialization to fully connected (linear) layers. This check ensures that we're currently looking at a linear layer.


- **`nn.init.xavier_uniform_(module.weight)`**:
  - This is where the magic happens. PyTorch's `nn.init` module has a method `xavier_uniform_` that applies Xavier initialization with uniform distribution to the weights of the provided module. The underscore at the end signifies that this method operates in-place on the tensor.


- **`nn.init.constant_(module.bias, 0)`**:
  - This initializes the bias values of the linear layer to zero. While there are various strategies for bias initialization, setting them to zero is a common and often effective method.

Adopting Xavier initialization in this manner gives our neural network a head start. It mitigates challenges like the vanishing or exploding gradient problems, which can severely hamper training, especially in deep networks.''')

    st.code('''def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

model = AutoEncoder()
initialize_weights(model)''', language='python')

    def initialize_weights(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    model = AutoEncoder()
    initialize_weights(model)
    
    

    st.markdown(r'''### CPU vs GPU in PyTorch

For our tutorial, we'll stick with CPU computation, given the simplicity of our network. However, PyTorch offers seamless integration with GPUs. If you're using platforms like Google Colab, you can even access GPUs for free!

To check for GPU availability in PyTorch:''')

    st.code('''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)''', language='python')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(device)
    
    

    st.markdown(r'''Moving the model between CPU and GPU:

''')

    st.code('''model.to(device)''', language='python')

    model.to(device)
    
    

    st.markdown(r'''For users on Google Colab, switching to GPU computation is easy. Go to `Runtime` > `Change runtime type` and select `GPU` from the dropdown. This choice can significantly expedite the training process for complex networks.''')

    st.markdown(r'''### ReLU Activation Function

Before we move on you might also notice that we chose to use the ReLU activation function instead of the sigmoid activation function. Let's recall why this is the case.

ReLU, or Rectified Linear Unit, has become one of the most widely used activation functions in deep neural networks, especially within the hidden layers. The functional form of ReLU is deceptively simple:

$$
 h(x) = \max(0, x) 
$$

This essentially means that for all positive values of $ x $, $ h(x) $ equals $ x $, and for non-positive values, $ h(x) $ is zero.

#### Gradients of ReLU

The gradient or derivative of the ReLU function is critical for backpropagation during training:

$$
 h'(x) = \begin{cases} 
0 & \mathrm{if } x \leq 0 \\
1 & \mathrm{if } x > 0 
\end{cases} 
$$

This gradient serves two important roles:

1. **Combatting the Vanishing Gradient Problem**: Traditional activation functions like the sigmoid or hyperbolic tangent (tanh) squash their inputs into a small range between 0 and 1 or -1 and 1 respectively. During backpropagation, especially in deeper networks, these small gradients can multiply, leading to an even smaller gradient—a phenomenon called the vanishing gradient problem. This causes the weights of initial layers to change very little, effectively halting their learning. ReLU's gradient is either 0 (for negative values) or 1 (for positive values), and thus it doesn't squash the gradient.

2. **Promotion of Sparsity**: ReLU activation introduces sparsity. When the output is 0, it's essentially ignoring that particular input, leading to a sparser representation. Sparsity is beneficial because it makes the network easier to optimize and can lead to a more efficient model.

However, it's worth noting that ReLU is not without its challenges, such as the dying ReLU problem where neurons can sometimes get stuck and stop updating, but variants like LeakyReLU and ParametricReLU have been introduced to address this.

''')

    st.markdown(r'''In the context of our autoencoder, the goal is to reconstruct Pokémon images from a compressed representation. As we discuss in the lecture, the Mean Squared Error (MSE) is a fitting choice that can be derived from maximum likelihood formalis. MSE will measure the squared difference between our original images and their reconstructed counterparts, effectively capturing how well our autoencoder is performing.

In PyTorch, MSE loss can be easily instantiated using:''')

    st.code('''loss_function = nn.MSELoss()''', language='python')

    loss_function = nn.MSELoss()
    
    

    st.markdown(r'''### Benefits of Mini-batch Training

When training deep learning models, one of the critical choices to make is how many data samples to use for each weight update. While using the entire dataset for every update (known as batch gradient descent) may seem straightforward, it's rarely the most efficient method in practice. Enter mini-batch training.

**What is Mini-batch Training?**  
In mini-batch training, rather than using the entire dataset, we divide the data into small, manageable batches and update our model's weights after each batch. This approach strikes a balance between the computational efficiency of stochastic gradient descent (where the weights are updated after every single example) and the stability of batch gradient descent.

**Why Use Mini-batches?**  
1. **Computational Efficiency**: Mini-batch training leads to faster convergence as weight updates are more frequent. It's a pragmatic middle-ground to leverage both the benefits of batch and stochastic gradient descent.
  
2. **Noise as a Regularizer**: While noise in gradient updates might sound detrimental, in practice, it has been observed that this noise can prevent the model from settling into suboptimal local minima or saddle points. This random noise can push the gradients out of these unfavorable spots, potentially leading to better model convergence.

3. **Memory Efficiency**: When dealing with massive datasets, loading the entire dataset into memory can be infeasible. Mini-batches circumvent this issue, allowing for out-of-core or online training.

While mini-batches offer these advantages, the choice of batch size can significantly affect the model's performance and training dynamics. It's a hyperparameter that often requires some experimentation to find the optimal value.

### The Adam Optimizer

Optimizing deep neural networks is a nuanced task. While the Stochastic Gradient Descent (SGD) algorithm has its merits, more sophisticated optimization methods can better navigate the loss landscapes of deep models. One such optimizer is Adam.

**Understanding Adam**:  
Adam, a contraction of **Ada**ptive **M**oment Estimation, combines the best properties of two popular optimization algorithms: AdaGrad and RMSProp. 

1. **Momentum**: Like a heavy ball rolling downhill, momentum ensures that our optimizer gathers speed in consistent gradient directions while dampening oscillations in inconsistent directions. Adam maintains an exponential moving average of past gradients to achieve this.

2. **Adaptive Learning Rates**: Not all parameters in a model should be updated at the same rate. Some might need more substantial adjustments, while others require fine-tuning. Adam calculates adaptive learning rates for each parameter by keeping an exponential moving average of past squared gradients.

3. **Bias Correction**: At the start of training, the moving averages are initialized at zero, leading to a bias towards zero. Adam corrects this initial bias, ensuring that the moving averages are unbiased.

With these features, Adam often converges faster and requires less hyperparameter tuning than SGD, making it a popular choice in the deep learning community.
''')

    st.markdown(r'''---

### Training the Autoencoder in PyTorch

In this section, we will demonstrate the concept of overfitting by training our autoencoder to its limit without early termination. Even though the model will be overfitting towards the end of its training cycle, we will ensure we retain the best model from the perspective of validation performance.

We split our dataset into:
- **Training set (80%)**: This set will be used to train our model.
- **Validation set (20%)**: This will be used to gauge how well our model performs on unseen data.

During the training process:
1. We calculate the reconstruction loss on the training set.

2. We periodically check the model's performance on the validation set.

3. If the model achieves a lower validation loss than previously observed, we save it as our "best model".

4. We continue training the model for the entire set number of epochs, despite potentially observing increases in validation loss (indicative of overfitting).

Every 30 epochs, we visualize how the current model reconstructs images from the training set. At the conclusion of training, we plot the training and validation loss curves to observe the model's learning progression and the onset of overfitting.

''')

    st.code('''import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

# Parameters
training_epoch = 200
batch_size = 64
learning_rate = 1e-3

# Splitting dataset into training and validation
dataset = TensorDataset(torch.tensor(images, dtype=torch.float32))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(training_epoch):
    epoch_losses = []
    val_epoch_losses = []

    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        X = batch[0]
        X_hat = model(X)
        loss = loss_function(X_hat, X)
        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            X = batch[0]
            X_hat = model(X)
            val_loss = loss_function(X_hat, X)
            val_epoch_losses.append(val_loss.item())

    # Log the losses for this epoch for later visualization
    avg_train_loss = sum(epoch_losses) / len(epoch_losses)
    avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()

    # Visualization of reconstructions every 20 epochs
    if epoch % 20 == 0:
        sample_data = next(iter(train_loader))[0]
        reconstructed_data = model(sample_data)
        
        print(f"Epoch {{}epoch+1{}}/{{}training_epoch{}}, Training Loss: {{}avg_train_loss:.4f{}}, Validation Loss: {{}avg_val_loss:.4f{}}")

        plot_gallery(sample_data.numpy(), ['True Pokemon {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
        plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
''', language='python')

    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset, random_split
    
    # Parameters
    training_epoch = 200
    batch_size = 64
    learning_rate = 1e-3
    
    # Splitting dataset into training and validation
    dataset = TensorDataset(torch.tensor(images, dtype=torch.float32))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(training_epoch):
        epoch_losses = []
        val_epoch_losses = []
    
        # Training phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch[0]
            X_hat = model(X)
            loss = loss_function(X_hat, X)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
    
        # Validation phase
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0]
                X_hat = model(X)
                val_loss = loss_function(X_hat, X)
                val_epoch_losses.append(val_loss.item())
    
        # Log the losses for this epoch for later visualization
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
    
        # Visualization of reconstructions every 20 epochs
        if epoch % 20 == 0:
            sample_data = next(iter(train_loader))[0]
            reconstructed_data = model(sample_data)
            
            st.write(f"Epoch {epoch+1}/{training_epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
            plot_gallery(sample_data.numpy(), ['True Pokemon {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
            plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
    
    # Plot the training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)
    
    

    st.markdown(r'''When we observe the plotted loss curves, it provides a clear picture of how our model performed over the epochs. 

**Training Loss**: As expected, the training loss consistently decreases as the epochs progress, eventually approaching zero. This is an indication that the model is becoming adept at reconstructing the training images it has been exposed to. 

**Validation Loss**: Initially, the validation loss also decreases, indicating that the model's capability to generalize to unseen data is improving. However, at a certain point, this trend reverses. Instead of continuing to decrease, the validation loss plateaus and then starts to rise. This rise, while the training loss still drops, is a classic sign of overfitting. It implies that while our model gets better and better at handling the training data, its performance on new, unseen data (i.e., validation data) deteriorates.

The reconstructed images from the training set further substantiate the findings from the loss curves. Early in training, the reconstructed images might show some differences from the true images. As training progresses, especially when the model starts to overfit, the reconstructed images from the training set become almost indistinguishable from the true images. This perfection in reconstruction is a tangible sign of overfitting: the model has become too attuned to the training data, potentially memorizing it, rather than learning the broader patterns.

#### The Challenge of Small Datasets

With approximately 1000 training samples, we don't anticipate the reconstructions to be perfect. Such a limited dataset means the deep learning model can easily memorize specific training examples instead of understanding and learning the broader features and patterns of the data. When a model can nearly perfectly reproduce training samples but struggles on validation data, it's a strong indicator that the model might be memorizing the training data rather than generalizing from it.

While it's tempting to assume a model is excellent because it performs well on training data, it's vital to always check its performance on unseen data. Our experiment clearly shows that a model can become near-perfect on its training data while simultaneously losing its ability to generalize effectively to new data. This phenomenon, known as overfitting, is especially prevalent with smaller datasets where memorization is easier for a small dataset (nb. for a sufficiently large dataset. this might not be a problem -- recall the discussion about grokking in lecture).

To address this, during our training loop, we've periodically saved the state of our model — but not just any state. We specifically saved the state when our model performed the best on the validation data, which serves as a proxy for unseen, real-world data.

The function `load_state_dict` plays a pivotal role here. It allows us to load previously saved parameters, effectively reverting the model to its most generalized state. This state isn't swayed by the idiosyncrasies of the training data but is shaped by its broader understanding, as evidenced by its validation performance.

As you'll observe, while the reconstructions from our training data might appear nearly perfect — a testament to the model's memorization — the reconstructions from validation data offer a dose of realism. They might not mirror the originals as closely, but they provide a more genuine measure of the model's prowess. This distinction underscores the importance of validation sets and external benchmarks.

Lastly, while this sobering comparison is beneficial, it's also essential to acknowledge the success. The reconstructions achieved via the autoencoder are still significantly superior to simpler methods like PCA.
''')

    st.code('''# Load the best model state
model.load_state_dict(best_model_state)

# Switch the model to evaluation mode
model.eval()

# Obtain a batch of validation images
sample_data = next(iter(val_loader))[0]

# Get the reconstructions using the best model state
with torch.no_grad():
    reconstructed_data = model(sample_data)

# Plot the original and reconstructed validation images
plot_gallery(sample_data.numpy(), ['True Pokemon {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
''', language='python')

    # Load the best model state
    model.load_state_dict(best_model_state)
    
    # Switch the model to evaluation mode
    model.eval()
    
    # Obtain a batch of validation images
    sample_data = next(iter(val_loader))[0]
    
    # Get the reconstructions using the best model state
    with torch.no_grad():
        reconstructed_data = model(sample_data)
    
    # Plot the original and reconstructed validation images
    plot_gallery(sample_data.numpy(), ['True Pokemon {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
    plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
    
    

    st.markdown(r'''### Visualising the Latent Representation 

We can also further visualise the "bottleneck" or latent representation of the Autoencoder. The bottleneck should capture essential high level features of the data. By observing this code, we gain insights into what the autoencoder deems as significant in representing the input. A simple but effective method to explore these features is to decode specially crafted codes. When we use a hidden unit vector of the form $(0,\dots,0,1,0,\dots,0)$, where the `1` represents the activation of a single feature, we are effectively trying to visualize what this single activated feature would look like when decoded.
''')

    st.code('''num_features = model.encoder[-2].out_features  # Get the number of features from the second-to-last layer

for factor in [1, 5, 20]:
    code = factor * torch.eye(num_features).to(device)  # Create scaled identity matrices
    with torch.no_grad():
        recon = model.decoder(code)  # Decoding the code using the decoder part of the autoencoder

    plot_gallery(recon.cpu().numpy(), ['{{}:0.2f{}}xFeature {{}:d{}}'.format(factor, i) for i in range(num_features)], 64, 64, 4, num_features//4)
''', language='python')

    num_features = model.encoder[-2].out_features  # Get the number of features from the second-to-last layer
    
    for factor in [1, 5, 20]:
        code = factor * torch.eye(num_features).to(device)  # Create scaled identity matrices
        with torch.no_grad():
            recon = model.decoder(code)  # Decoding the code using the decoder part of the autoencoder
    
        plot_gallery(recon.cpu().numpy(), ['{:0.2f}xFeature {:d}'.format(factor, i) for i in range(num_features)], 64, 64, 4, num_features//4)
    
    

    st.markdown(r'''In this section, we will create identity matrices, each scaled by different factors, and pass them through our decoder. By doing so and visualizing the output, we can gain insight into the features that our autoencoder has learned when they are maximally activated and amplified.

Utilizing this technique provides a visually engaging means to unravel the intricacies of our autoencoder. It's particularly insightful when juxtaposed with techniques like PCA, allowing us to highlight and appreciate the nuanced non-linearities introduced by the autoencoder.

At first glance, the individual features may not appear highly informative. This is primarily because we've chosen a relatively large representation dimension of 32. As a result, each feature has the liberty to learn distinct characteristics. By experimenting with a smaller bottleneck dimension, you'll find that the derived representations become considerably more insightful.

This observation segues into an important consideration: the choice of hyperparameters. For the autoencoder in this tutorial, we've yet to dive deep into hyperparameter tuning—factors like learning rate, batch size, and the neural architecture itself. Each of these plays a pivotal role in determining the quality of our reconstructed outputs.

Moreover, as touched upon in our lecture, the MSE loss function treats all pixels with equal importance, neglecting the coherence of global features. This can pose a challenge, especially when our aim is to reconstruct intricate details characteristic of subjects like Pokémon. The field of computer vision has recognized this limitation, leading to the proposition of various techniques, ranging from the classical Generative Adversarial Network (GAN) to the avant-garde Diffusion Model. However, delving into these methodologies is beyond the scope of this tutorial.

---''')

    st.markdown(r'''## Galaxy Images

Pokémon images, with their intricate featres, present a complex challenge for our autoencoder, especially when trained on a limited dataset of just about 1,000 samples. Realistically, with such a limited dataset, capturing more than the basic features becomes an uphill task.

To demonstrate the efficacy of our autoencoder on a slightly simpler dataset, let's turn our attention to a collection of galaxy images, a more relevant task for this course. These are sourced from the Hyper Suprime-Cam (HSC). Given our intent to keep this exercise accessible even for those running on CPUs, we've scaled down the resolution of these images to 64 x 64 pixels. This dataset consists of about 2,000 images.

Our approach remains consistent: we'll execute our autoencoder code just as we did with the Pokémon dataset and then analyze the reconstructed features.
''')

    st.code('''# Loading the galaxy images dataset
images = np.load('galaxy_sample_tutorial_week10a.npy')

# Let's verify the shape of our loaded data to ensure everything's in order
print("Shape of the loaded images:", images.shape)''', language='python')

    # Loading the galaxy images dataset
    response = requests.get('https://storage.googleapis.com/compute_astro/galaxy_sample_tutorial_week10a.npy')
    f = BytesIO(response.content)
    images = np.load(f, allow_pickle=True)
    
    # Let's verify the shape of our loaded data to ensure everything's in order
    st.write("Shape of the loaded images:", images.shape)
    
    

    st.markdown(r'''To get a feel for our dataset, let's visualize a subset of these images, just as we did with our Pokémon samples.''')

    st.code('''# Displaying a selection of our galaxy images
plot_gallery(images, np.arange(30), 64, 64, 5, 6)''', language='python')

    # Displaying a selection of our galaxy images
    plot_gallery(images, np.arange(30), 64, 64, 5, 6)
    
    

    st.markdown(r'''With our dataset ready, we'll reinitialize our autoencoder and proceed with the training process in a manner identical to our previous example.''')

    st.code('''# reinstantiate the model
model = AutoEncoder()

# initialize the weights
initialize_weights(model)
''', language='python')

    # reinstantiate the model
    model = AutoEncoder()
    
    # initialize the weights
    initialize_weights(model)
    
    

    st.code('''import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

# Parameters
training_epoch = 200
batch_size = 64
learning_rate = 1e-3

# Splitting dataset into training and validation
dataset = TensorDataset(torch.tensor(images, dtype=torch.float32))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(training_epoch):
    epoch_losses = []
    val_epoch_losses = []

    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        X = batch[0]
        X_hat = model(X)
        loss = loss_function(X_hat, X)
        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            X = batch[0]
            X_hat = model(X)
            val_loss = loss_function(X_hat, X)
            val_epoch_losses.append(val_loss.item())

    # Log the losses for this epoch for later visualization
    avg_train_loss = sum(epoch_losses) / len(epoch_losses)
    avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()

    # Visualization of reconstructions every 20 epochs
    if epoch % 20 == 0:
        sample_data = next(iter(train_loader))[0]
        reconstructed_data = model(sample_data)
        
        print(f"Epoch {{}epoch+1{}}/{{}training_epoch{}}, Training Loss: {{}avg_train_loss:.4f{}}, Validation Loss: {{}avg_val_loss:.4f{}}")

        plot_gallery(sample_data.numpy(), ['True Galaxy Images {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
        plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
''', language='python')

    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset, random_split
    
    # Parameters
    training_epoch = 200
    batch_size = 64
    learning_rate = 1e-3
    
    # Splitting dataset into training and validation
    dataset = TensorDataset(torch.tensor(images, dtype=torch.float32))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(training_epoch):
        epoch_losses = []
        val_epoch_losses = []
    
        # Training phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch[0]
            X_hat = model(X)
            loss = loss_function(X_hat, X)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
    
        # Validation phase
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0]
                X_hat = model(X)
                val_loss = loss_function(X_hat, X)
                val_epoch_losses.append(val_loss.item())
    
        # Log the losses for this epoch for later visualization
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
    
        # Visualization of reconstructions every 20 epochs
        if epoch % 20 == 0:
            sample_data = next(iter(train_loader))[0]
            reconstructed_data = model(sample_data)
            
            st.write(f"Epoch {epoch+1}/{training_epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
            plot_gallery(sample_data.numpy(), ['True Galaxy Images {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
            plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
    
    # Plot the training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)
    
    

    st.markdown(r'''Just as with the Pokémon images, the limited dataset for our galaxy images has a caveat. After a certain number of epochs, our autoencoder struggles to generalize and instead starts to memorize the training data. This leads to overfitting. To counter this, we'll revert to the model's state from when it performed best on the validation data.''')

    st.code('''# Load the state of the best-performing model
model.load_state_dict(best_model_state)

# Switch the model to evaluation mode
model.eval()

# Fetch a batch of validation images
sample_data = next(iter(val_loader))[0]

# Using our best model state, let's reconstruct these images
with torch.no_grad():
    reconstructed_data = model(sample_data)

# Visualizing both the original and reconstructed validation images side by side
plot_gallery(sample_data.numpy(), ['True Galaxy Images {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {{}:d{}}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)''', language='python')

    # Load the state of the best-performing model
    model.load_state_dict(best_model_state)
    
    # Switch the model to evaluation mode
    model.eval()
    
    # Fetch a batch of validation images
    sample_data = next(iter(val_loader))[0]
    
    # Using our best model state, let's reconstruct these images
    with torch.no_grad():
        reconstructed_data = model(sample_data)
    
    # Visualizing both the original and reconstructed validation images side by side
    plot_gallery(sample_data.numpy(), ['True Galaxy Images {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
    plot_gallery(reconstructed_data.detach().numpy(), ['Reconstructed {:d}'.format(i) for i in range(batch_size)], 64, 64, min(2, int(np.ceil(batch_size/6))), 5)
    
    

    st.markdown(r'''Now, let's delve deeper into the representations our model has learned.''')

    st.code('''# Retrieve the number of features from the second-to-last layer of the encoder
num_features = model.encoder[-2].out_features  

for factor in [1, 5, 20]:
    code = factor * torch.eye(num_features).to(device)  # Creating scaled identity matrices
    with torch.no_grad():
        recon = model.decoder(code)  # Decode these representations

    # Visualize the decoded images
    plot_gallery(recon.cpu().numpy(), ['{{}:0.2f{}}xFeature {{}:d{}}'.format(factor, i) for i in range(num_features)], 64, 64, 4, num_features//4)''', language='python')

    # Retrieve the number of features from the second-to-last layer of the encoder
    num_features = model.encoder[-2].out_features  
    
    for factor in [1, 5, 20]:
        code = factor * torch.eye(num_features).to(device)  # Creating scaled identity matrices
        with torch.no_grad():
            recon = model.decoder(code)  # Decode these representations
    
        # Visualize the decoded images
        plot_gallery(recon.cpu().numpy(), ['{:0.2f}xFeature {:d}'.format(factor, i) for i in range(num_features)], 64, 64, 4, num_features//4)
    
    

    st.markdown(r'''Though the reconstructions are not perfect due to the constraints of our limited dataset, they reveal fascinating insights. Given the restricted representation dimensions, our autoencoder prioritizes capturing the most prominent features in the images, overlooking minor details. This behavior aligns with our prior discussion on how autoencoders inherently denoise data, focusing mainly on salient features, disregarding the smaller galaxies around the main galaxy.

Furthermore, the dominant features learned resonate with our intuition. The neural network appears to be capturing key attributes of galaxies such as their disk-like structures, cores, and oblateness. Additionally, a few components seem to be devoted to capturing the more diffuse aspects of these galaxies.

---''')

    st.markdown(r'''### Conclusion

Throughout this tutorial, we journeyed through the intricacies of using autoencoders for image reconstruction, particularly for Pokémon and galaxy images. Here's a summary of what we covered:

1. **Autoencoder Architecture**: At its core, an autoencoder consists of an encoder and a decoder. The encoder compresses the input into a lower-dimensional latent space or "bottleneck", and the decoder then reconstructs the input from this compressed representation.

2. **Training and Validation**: It's crucial to monitor both training and validation performance. The essence of this approach is to ensure the model doesn't overfit to the training data. With our limited dataset, the risk of memorization is high, but by focusing on validation performance, we've enforced a model that generalizes better.

3. **Model State and Overfitting**: PyTorch's `load_state_dict` function proved invaluable in reverting our model back to its most effective state, countering the common issue of overfitting. While our training reconstructions might have seemed almost flawless, validation reconstructions grounded our perspective, emphasizing the model's true capability on unseen data.

4. **Latent Space Visualization**: We dived deep into the latent space, decoding specific features to understand their visual significance.

5. **Comparison with PCA**: Autoencoders and PCA both strive for dimensionality reduction, but our observations indicated that autoencoders can capture more intricate patterns in the data, resulting in superior reconstructions.

In essence, autoencoders are a potent tool in the machine learning toolkit. While our focus here was on image reconstruction, their applications span far wider, from anomaly detection to generative models. The principles we discussed – like monitoring validation performance, countering overfitting, and understanding feature significance – remain central in many machine learning contexts.

Thank you for joining us on this exploration of autoencoders. We hope you found it enlightening and are now equipped with deeper insights and tools for your own projects!
''')

#if __name__ == '__main__':
show_page()
