import streamlit as st

def show_page():

    st.markdown(r'''# Markov Chain Monte Carlo''')

    st.markdown(r'''### Assumed knowledge

- **Sampling and Bayesian Methods (lectures)**: You should have attended and understood the main concepts from the lectures on sampling and Bayesian methods. This includes understanding why we use sampling in Bayesian contexts, the basics of posterior distributions, and an introduction to the Markov chain Monte Carlo (MCMC) techniques. If you're unsure about these foundational ideas, consider revisiting the lecture materials.

### Objectives

By the end of this lab, you'll grasp the workings and nuances of advanced sampling techniques in Bayesian contexts, especially within the realm of astronomy. Specifically:

1. **Metropolis-Hastings Algorithm**:

   - *What it means*: An MCMC method to obtain a sequence of random samples from a probability distribution for which direct sampling is challenging.

   - *Why it's important*: It offers a versatile approach to exploring complex and high-dimensional distributions, forming the bedrock for many modern Bayesian analyses in varied fields, astronomy included.

2. **Gibbs Sampling**:

   - *What it means*: An MCMC technique where each variable is sampled sequentially, conditional on the current values of the other variables.

   - *Why it's important*: In situations where conditional distributions are more accessible or can be sampled more efficiently than the joint distribution, Gibbs Sampling offers a compelling strategy, often yielding faster convergence.

3. **Burn-in, Thinning, Effective Sample Size, and Sample Correlation**:

   - *What it means*: Concepts that delve into the quality and independence of the generated MCMC samples. Burn-in refers to the initial set of samples discarded, ensuring the chain has reached its stationary distribution. Effective sample size gives a measure of independent samples, and the correlation between samples indicates the extent of autocorrelation.

   - *Why it's important*: To guarantee reliable and unbiased inferences, it's essential to assess and ensure the quality of the generated samples. These metrics provide the tools to make such assessments.

---''')

    st.markdown(r'''## Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is an MCMC method that allows us to obtain a sequence of random samples from a probability distribution, especially when direct sampling from this distribution is challenging. The main idea behind this algorithm is to construct a Markov chain that, in the long run, converges to the desired distribution.

### Principles Underlying the Metropolis-Hastings Algorithm

#### 1. Detailed Balance

A key principle ensuring the convergence of the chain to the desired stationary distribution is the concept of detailed balance. It requires that, for any two states $ x $ and $ x^* $, the probability of being in state $ x $ and transitioning to state $ x^* $ is the same as the probability of being in state $ x^* $ and transitioning to state $ x $.

Mathematically, this can be expressed as:

$$
 P(x) T(x \rightarrow x^*) = P(x^*) T(x^* \rightarrow x) 
$$

Where $ P $ represents the stationary distribution, and $ T $ denotes the transition probabilities between states.

#### 2. Ergodicity

The chain is ergodic if every state can be reached from any other state in a finite number of steps with a positive probability. Ergodicity ensures that the chain does not get trapped in any subset of the state space and can explore the entire state space given enough time.

If a Markov chain is both ergodic and satisfies detailed balance concerning the desired distribution $ P $, it will converge to $ P $ as its stationary distribution.



### Mathematical Details:

1. **Proposal Distribution**: This is a distribution we can easily sample from. We use it to suggest a new point $ x^* $ in the parameter space based on the current point $ x $.

2. **Acceptance Criterion**: Given the current point $ x $ and the proposed point $ x^* $, we calculate the acceptance ratio $ \alpha $ defined as:
    $$
        \alpha = \min \bigg( 1, \dfrac{P(x^*)}{P(x)} \times \dfrac{Q(x|x^*)}{Q(x^*|x)} \bigg) 
    $$
    when the proposal distribution $ Q $ is symmetric (i.e., $ Q(x | x^*) = Q(x^* | x) $), the acceptance criterion simplifies. This is the special case known as the Metropolis algorithm. For symmetric proposals, the acceptance ratio $ \alpha $ reduces to:
    $$
        \alpha = \min \bigg( 1, \dfrac{P(x^*)}{P(x)} \bigg) 
    $$
    where:
        - $ P $ is the target distribution (posterior in Bayesian analysis).
        - $ Q $ is the proposal distribution.

This ensures that the acceptance ratio $ \alpha $ never exceeds 1. If the term $ \dfrac{P(x^*)}{P(x)} \times \dfrac{Q(x|x^*)}{Q(x^*|x)} $ (or $ \dfrac{P(x^*)}{P(x)} $ for the symmetric case) is greater than 1, we always accept the proposed state, hence the use of the minimum function. If $ \alpha $ is greater than a uniform random number between 0 and 1, we accept $ x^* $ as the next sample. Otherwise, the next sample remains $ x $.
''')

    st.code('''import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tqdm
import seaborn as sns''', language='python')

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import tqdm
    import seaborn as sns
    
    

    st.markdown(r'''--- 

## Toy Example: Estimating the Sample Mean

Understanding the mean of a distribution is crucial in many scientific inquiries. In this tutorial, we'll delve into how the Metropolis Algorithm—a Markov chain Monte Carlo method—can help us estimate the mean of a sample drawn from a complicated distribution, especially when direct sampling isn't feasible.

### Introducing the Challenge

We wish to estimate the mean of a sample drawn from a distribution that's a combination of a Gaussian and a Cauchy distribution. This merged distribution isn't easy to sample from directly, which is where the power of the Metropolis Algorithm shines.

### Key Distributions:

**1. Gaussian Distribution**: It's defined by its mean $ \mu $ and variance $ \sigma^2 $. Given a sample, we know the variance $ \sigma = 1 $, but the mean $ \mu $ is unknown and what we wish to estimate.

**2. Cauchy Distribution**: Our prior beliefs about the mean are characterized using the Cauchy distribution. It is defined as:
    $$
        P(\mu | \mu_0, \gamma) = \dfrac{1}{\pi\gamma} \dfrac{1}{1 + \Big(\dfrac{\mu-\mu_0}{\gamma}\Big)^2} 
    $$
    where:
    - $ \mu_0 $ is the location parameter.
    - $ \gamma $ is the scale parameter, influencing the width of the distribution.

The choice of the Cauchy distribution is deliberate. Its long tails allow for greater flexibility, ensuring that even if our initial beliefs about the mean are off-mark, with a sufficient number of samples, we can still approximate the true mean accurately.

### Formulating the Problem

Mathematically, the challenge boils down to this: Given our sample, we wish to compute the posterior distribution for the mean $ \mu $. This posterior is proportional to the product of our sample's likelihood (assumed Gaussian) and our prior (the Cauchy distribution). Let $\{ x_i \}$ be the sample,

$$
    P(\mu | \{ x_i \}, \sigma) \propto \prod_{i=1}^N \mathcal{N} (x_i | \mu, \sigma=1) \times \mathrm{Cauchy}(\mu | \mu_0 = 0, \gamma =3) 
$$

In the upcoming sections, we will delve into the mechanics of the Metropolis Algorithm, see how it helps in estimating $ \mu $, and analyze the results to understand the efficacy of our approach.
''')

    st.code('''# Generate a sample from a shifted normal distribution with a fixed variance of one
n = 1000
loc_true = 1
x_sample = np.random.normal(loc=loc_true, scale=1, size=n)
''', language='python')

    # Generate a sample from a shifted normal distribution with a fixed variance of one
    n = 1000
    loc_true = 1
    x_sample = np.random.normal(loc=loc_true, scale=1, size=n)
    
    

    st.markdown(r'''### Defining the Posterior Distribution

Our posterior is the product of our likelihood (the Gaussian distribution with a varying mean and known variance) and our prior (the Cauchy distribution). By taking the natural logarithm of this product, we can work with summing log probabilities, which is computationally more stable.
''')

    st.code('''def log_posterior(mu, x_sample):
    likelihood = stats.norm.pdf(x_sample, loc=mu, scale=1)
    prior = stats.cauchy.pdf(mu, loc=1, scale=3)

    # Summing log likelihood and log prior
    return np.sum(np.log(likelihood)) + np.log(prior)''', language='python')

    def log_posterior(mu, x_sample):
        likelihood = stats.norm.pdf(x_sample, loc=mu, scale=1)
        prior = stats.cauchy.pdf(mu, loc=1, scale=3)
    
        # Summing log likelihood and log prior
        return np.sum(np.log(likelihood)) + np.log(prior)
    
    

    st.markdown(r'''For those new to the Metropolis Algorithm, the method might seem like magic. But it's rooted in solid statistical principles. Let's walk through the two essential components of the method: the proposal distribution and the sampling process.

### Proposal Distribution: Generating Candidate Samples

The Metropolis Algorithm operates in a series of steps, where in each step, we suggest a new candidate sample (or point in parameter space). This candidate is generated from a **proposal distribution**. The choice of proposal distribution can affect the efficiency of the sampling but not the representativeness when the chain is long enough.

For our problem, we will use a normal (Gaussian) distribution centered at the current sample value $ \mu $ with a variance $ \nu $ to generate these candidates. Here's the function that achieves this:

''')

    st.code('''def proposal(mu, nu):
    return stats.norm.rvs(loc=mu, scale=nu, size=1).item()''', language='python')

    def proposal(mu, nu):
        return stats.norm.rvs(loc=mu, scale=nu, size=1).item()
    
    

    st.markdown(r'''This function returns a value sampled from a normal distribution centered at the current value of $ \mu $ and with variance $ \nu $.

--- 

### The Metropolis-Hastings Sampling Process

Now that we have a way to generate candidate samples, let's move on to the sampling process itself. In each iteration:

1. A new candidate $ \mu_* $ is proposed.

2. We compute the acceptance ratio $ \alpha $, which is the ratio of the likelihood of the proposed candidate to the current value. If the proposal distribution is symmetric (as in our case), $ \alpha $ simplifies to the ratio of the target posterior values for the candidate and current sample. Mathematically, the acceptance ratio is:   
   $$
      \alpha = \min ( 1, \alpha' ) 
    $$
   where
   $$
      \alpha' = \frac{P(\mu_* | x_{\mathrm{sample}})}{P(\mu | x_{\mathrm{sample}})} 
    $$
   For computational stability and efficiency, it's more suitable to work in the log space, especially when dealing with probabilities. Thus, our acceptance criterion in the log space is:
   $$
      \log(\alpha') = \log P(\mu_* | x_{\mathrm{sample}}) - \log P(\mu | x_{\mathrm{sample}}) 
   $$
   To obtain $ \alpha' $ from this log scale, we exponentiate: 
   $$
      \alpha' = e^{\log P(\mu_* | x_{\mathrm{sample}}) - \log P(\mu | x_{\mathrm{sample}})} 
   $$
   Finally, our refined acceptance probability is:
   $$
      \alpha = \min \bigg( 1, e^{\log P(\mu_* | x_{\mathrm{sample}}) - \log P(\mu | x_{\mathrm{sample}})} \bigg) 
   $$
   This criterion ensures a balance between exploring new regions of the parameter space (proposed values) and exploiting regions where we've found high posterior values.

3. We randomly decide if we should accept this new candidate as the next value in our chain, based on $ \alpha $.

This process will result in a series of $ \mu $ values, which is our **Markov chain**. This chain, when long enough, will represent samples from our desired posterior distribution. By recognizing the reasons behind each mathematical decision in the Metropolis-Hastings algorithm, we can confidently apply it to various situations and understand its behavior.

Here's our function encapsulating the entire Metropolis-Hastings sampling process:
''')

    st.code('''def metropolis_hastings(mu_init, nu, n):
    chain = [mu_init]
    mu = mu_init
    accept = 0

    for i in tqdm.tqdm(range(n)):
        
        # Propose a new value
        mu_star = proposal(mu, nu)

        # Calculate acceptance ratio
        alpha = min(1, np.exp(log_posterior(mu_star, x_sample) - log_posterior(mu, x_sample)))

        # Accept or reject
        u = np.random.uniform()
        if u < alpha:
            mu = mu_star
            accept += 1
        chain.append(mu)

    print("Acceptance ratio: ", accept/n*100, "%")
    return chain, accept/n*100
''', language='python')

    def metropolis_hastings(mu_init, nu, n):
        chain = [mu_init]
        mu = mu_init
        accept = 0
    
        for i in tqdm.tqdm(range(n)):
            
            # Propose a new value
            mu_star = proposal(mu, nu)
    
            # Calculate acceptance ratio
            alpha = min(1, np.exp(log_posterior(mu_star, x_sample) - log_posterior(mu, x_sample)))
    
            # Accept or reject
            u = np.random.uniform()
            if u < alpha:
                mu = mu_star
                accept += 1
            chain.append(mu)
    
        st.write("Acceptance ratio: ", accept/n*100, "%")
        return chain, accept/n*100
    
    

    st.markdown(r'''A critical point to note: the **acceptance ratio** gives us a sense of how often we're accepting the proposed candidates. A very high or low acceptance rate might indicate issues with the choice of proposal distribution (e.g., too wide or too narrow).

''')

    st.markdown(r'''### Tuning the Proposal Distribution

The proposal distribution plays a pivotal role in the Metropolis-Hastings algorithm. It is the mechanism by which we propose new candidate values in the sampling process. However, the width of this proposal distribution, represented by $\nu$, is crucial:

- **Too large**: The proposed values might jump too far away, resulting in a low acceptance ratio because the likelihood might decrease drastically.

- **Too small**: The proposed values might not venture far from the current value. Although this might result in a high acceptance ratio, the samples will be very correlated, meaning the Markov chain might not explore the entire parameter space efficiently.

Let's consider three different values for $\nu$: 0.01, 0.1, and 0.5 to understand the effects of tuning this parameter.

''')

    st.code('''nu_values = [0.01, 0.1, 0.5]
mu_init = 0
n = 10000
results = {{}{}}

for nu in nu_values:
    mu, accept_ratio = metropolis_hastings(mu_init, nu, n)
    results[nu] = (mu, accept_ratio)
''', language='python')

    nu_values = [0.01, 0.1, 0.5]
    mu_init = 0
    n = 10000
    results = {}
    
    for nu in nu_values:
        mu, accept_ratio = metropolis_hastings(mu_init, nu, n)
        results[nu] = (mu, accept_ratio)
    
    

    st.markdown(r'''We will plot the first 1000 iterations to see the initial behavior and burn-in of our Markov chain.''')

    st.code('''# Setting up high resolution plots with clear font
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 20
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for index, nu in enumerate(nu_values):
    ax = axes[index]
    mu, accept_ratio = results[nu]
    ax.plot(mu[:1000]) # just plotting the first 1000 samples
    ax.set_title(r"$\nu$ = " + str(nu))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Estimated Mean")
    ax.text(350, 0.1, "Acceptance ratio: " + str(round(accept_ratio, 1)) + "%", fontsize=20)

plt.tight_layout()
''', language='python')

    # Setting up high resolution plots with clear font
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    for index, nu in enumerate(nu_values):
        ax = axes[index]
        mu, accept_ratio = results[nu]
        ax.plot(mu[:1000]) # just plotting the first 1000 samples
        ax.set_title(r"$\nu$ = " + str(nu))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Estimated Mean")
        ax.text(350, 0.1, "Acceptance ratio: " + str(round(accept_ratio, 1)) + "%", fontsize=20)
    
    st.pyplot(fig)
    
    

    st.markdown(r'''--- 

### Understanding the Impact of Proposal Width on the Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is influenced significantly by the choice of the proposal distribution, specifically its width or variance, denoted by $ \nu $ in our example. This width determines how far we're likely to propose jumps in each iteration of the algorithm.

From our plots, we can derive crucial insights:

1. **Small Proposal Width ($ \nu = 0.01 $)**:

    - **High Acceptance Rate**: This might seem like a good thing initially since it indicates that the majority of our proposed jumps are being accepted.

    - **High Correlation Between Samples**: However, the catch is that because we're proposing very small jumps, our samples end up being very close to each other. Consequently, they are highly correlated, reducing the effective number of independent samples.

    - **Implication**: Although we are making many samples, they don't give us a broad view of the parameter space. Thus, a large portion of the samples might be redundant in conveying information about the posterior distribution.

2. **Large Proposal Width ($ \nu = 0.5 $)**:

    - **Low Acceptance Rate**: Here, the algorithm often suggests aggressive jumps in the parameter space, which are frequently not in regions of high posterior probability. Consequently, many proposals get rejected.

    - **Infrequent Updates**: This is evident from the jagged nature of the plot. The chain only updates occasionally, spending long intervals stuck at a particular value.

    - **Implication**: This behavior can make the algorithm inefficient. Not only does it not explore the parameter space well, but it also wastes computational resources on rejected proposals.

3. **Moderate Proposal Width ($ \nu = 0.1 $)**:

    - This appears to strike a balance between exploration and exploitation. The algorithm manages to explore the parameter space while also accepting a decent proportion of proposals.
    
    - The acceptance rate around 0.1 seems to offer a good trade-off in this example.

The acceptance ratio is a diagnostic tool. It provides insights into how often the proposed values were accepted. But it's essential not to chase high acceptance rates blindly. Instead, one must ensure that the Markov chain is both exploring new regions and exploiting areas of high probability. The proposal width plays a pivotal role in balancing these aspects. Properly tuning it can dramatically improve the efficiency of the Metropolis-Hastings algorithm and the quality of the samples it produces.''')

    st.markdown(r'''--- 

### Autocorrelation and its Role in MCMC

Beside effective sample, autocorrelation is a powerful diagnostic tool for assessing the performance of Markov Chain Monte Carlo (MCMC) methods, such as Metropolis-Hastings. It indicates the correlation between different samples in the chain at different lags. 

In the context of MCMC, autocorrelation gives us insight into the correlation between samples at various points in our Markov chain. Specifically, for a given lag $ k $, the autocorrelation tells us how correlated a sample is with the sample $ k $ steps before it. 

In time series analysis, the autocorrelation of a series at lag $ k $ is formally defined as:

$$
 R(k) = \frac{\sum_{i=1}^{N-k} ( X_i - \bar{X} ) ( X_{i+k} - \bar{X} )}{\sum_{i=1}^N ( X_i - \bar{X} )^2} 
$$

where:
- $ R(k) $ is the autocorrelation at lag $ k $.
- $ X_i $ is the sample at position $ i $.
- $ \bar{X} $ is the mean of the samples.
- $ N $ is the number of samples.

The value of $ R(k) $ lies between -1 and 1. A value close to 1 indicates that the samples $ k $ apart are positively correlated, a value close to -1 means they are negatively correlated, and a value close to 0 suggests no correlation.

Ideally, in MCMC, we want our samples to be as independent as possible, meaning low autocorrelation.

1. **Why Autocorrelation Matters**: 

    - Autocorrelation can significantly affect the convergence of MCMC algorithms. High autocorrelation means that our chain is making slow explorations of the parameter space, which can lead to biased inferences.

    - Highly correlated samples can give a false sense of precision. Even though you might have many samples, if they're all highly correlated, the effective sample size (the number of independent samples) is much smaller.


2. **How to Measure Autocorrelation**: 

    - One common way to visualize autocorrelation is with an autocorrelation plot. This plot displays the autocorrelation coefficient for each lag. A rapidly declining plot indicates low autocorrelation, which is what we aim for in MCMC.


3. **Effective Sample Size (ESS)**:

    - The ESS accounts for autocorrelation and can be considerably smaller than the actual number of samples taken. Mathematically, the ESS is given by: $ \mathrm{ESS} = \dfrac{N}{1 + 2 \sum_{k=1}^{N} R(k)} $

    where:
    - $ N $ is the total number of samples.
    - $ R(k) $ is the autocorrelation at lag $ k $.

    The autocorrelation for large lags tends to be very small or statistically indistinguishable from zero for many sequences, so we can set `nlags` to be much smaller than `N`, in our case 100, we re effectively truncating the computation to only those lags where the autocorrelation is meaningful, which is usually what is desired in practice to avoid the numerical issue with FFT.

The ESS gives a clearer picture of how much information is present in the chain. For instance, a chain of length 10,000 with an ESS of 1,000 means we effectively have only 1,000 independent samples.

As stated before, autocorrelation plots can be immensely helpful. A decaying autocorrelation plot suggests that as the lag increases, samples become less correlated, which is a sign of a well-mixed chain.
''')

    st.code('''import numpy as np
import statsmodels.api as sm

def calculate_ess(samples):
    """Calculate Effective Sample Size."""
    N = len(samples)
    autocorr_values = sm.tsa.acf(samples, nlags=100, fft=True)
    return N / (1 + 2 * sum(autocorr_values))

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for index, nu in enumerate(nu_values):
    ax = axes[index]
    mu, _ = results[nu]

    # discard the first 500 sample as burn-in
    mu = mu[500:]
    
    # Plot the autocorrelation
    sm.graphics.tsa.plot_acf(mu, lags=100, ax=ax)
    ax.set_title(r"Autocorrelation for $\nu$ = " + str(nu))
    ax.set_xlabel("Lag")

    if index == 0:
        ax.set_ylabel("Autocorrelation")

    # Calculate and display ESS
    ess = calculate_ess(mu)
    ax.annotate(f"ESS: {{}ess:.1f{}}", xy=(0.6, 0.9), xycoords='axes fraction')
''', language='python')

    import numpy as np
    import statsmodels.api as sm
    
    def calculate_ess(samples):
        """Calculate Effective Sample Size."""
        N = len(samples)
        autocorr_values = sm.tsa.acf(samples, nlags=100, fft=True)
        return N / (1 + 2 * sum(autocorr_values))
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    for index, nu in enumerate(nu_values):
        ax = axes[index]
        mu, _ = results[nu]
    
        # discard the first 500 sample as burn-in
        mu = mu[500:]
        
        # Plot the autocorrelation
        sm.graphics.tsa.plot_acf(mu, lags=100, ax=ax)
        ax.set_title(r"Autocorrelation for $\nu$ = " + str(nu))
        ax.set_xlabel("Lag")
    
        if index == 0:
            ax.set_ylabel("Autocorrelation")
    
        # Calculate and display ESS
        ess = calculate_ess(mu)
        ax.annotate(f"ESS: {ess:.1f}", xy=(0.6, 0.9), xycoords='axes fraction')
    

    st.markdown(r'''### Insights from Autocorrelation Plots

1. **Small Proposal Width ($ \nu = 0.01 $)**:

    - The autocorrelation plot shows significant correlation for higher lags, reiterating that the samples are not independent, leading to a small effective sample.


2. **Large Proposal Width ($ \nu = 0.5 $)**:

    - Due to the aggressive jumps, the autocorrelation might be inconsistent. There might be lower autocorrelation for smaller lags, but occasional long-lasting correlations for some higher lags because the chain occasionally gets stuck at specific values, also leading to a small effective sample.


3. **Moderate Proposal Width ($ \nu = 0.1 $)**:

    - **Balanced Autocorrelation**: The autocorrelation should decrease more rapidly than in the small proposal width scenario, indicating a more efficient exploration of the parameter space.

By combining visual inspection of our chain, the acceptance ratio, and the autocorrelation plots, we can develop a comprehensive understanding of how well our MCMC method is performing. Each of these tools provides a different lens through which we can assess and improve our sampling strategy.''')

    st.markdown(r'''---

### Post-Processing and Evaluating the Markov Chain

After running the Metropolis-Hastings algorithm and obtaining a Markov chain, post-processing steps are crucial to extract meaningful insights from the chain. Here, we'll discuss the concepts of **burn-in**, **thinning**, and the **histogram representation** of the samples.

#### 1. Burn-In:
The beginning of the Markov chain may not be representative of the true distribution, especially if the starting point, or initial value, is far from regions of high probability. Hence, it's a common practice to discard a certain initial portion of the chain, which is referred to as the **burn-in**. In our case, we've discarded the first 500 samples.

#### 2. Thinning:
Even after discarding the burn-in, successive samples in the chain can be correlated. This correlation can be reduced by "thinning", which means only keeping every $k$-th sample. Here, we've decided to keep every 10th sample, thereby reducing any residual autocorrelation in the chain.

#### 3. Histogram Representation:
To visually represent the spread and concentration of the samples, we plot a histogram. This histogram gives a non-parametric estimate of the distribution of the parameter (mean in our case). By overlaying the true mean, we can compare how well our Markov chain has captured the true characteristics of the distribution.

We will study on the sample from $\nu=0.1$ as it is the most adequate sampling as we have discussed above.''')

    st.code('''plt.figure(figsize=(10, 10))

# Choose the chain sampling with nu=0.01
mu_sample = results[0.01][0] 

# Burn in
mu_sample = mu_sample[500:]

# Thinning
mu_sample = mu_sample[::10]

plt.hist(mu_sample, bins=50, density=True, color="green", alpha=0.5, label='Posterior')
plt.xlabel("Estimated Mean")
plt.ylabel("Density")
plt.axvline(x=loc_true, color='red', label='True Mean')
plt.legend()
plt.title("Posterior Samples")''', language='python')

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Choose the chain sampling with nu=0.01
    mu_sample = results[0.01][0] 
    
    # Burn in
    mu_sample = mu_sample[500:]
    
    # Thinning
    mu_sample = mu_sample[::10]
    
    ax.hist(mu_sample, bins=50, density=True, color="green", alpha=0.5, label='Posterior')
    ax.set_xlabel("Estimated Mean")
    ax.set_ylabel("Density")
    plt.axvline(x=loc_true, color='red', label='True Mean')
    ax.legend()
    ax.set_title("Posterior Samples")
    st.pyplot(fig)
    
    

    st.markdown(r'''By examining this histogram, we can grasp both the central location and the variability of the estimated mean. The mean of the sample acts as a point estimate, offering a singular value to represent the probable location of the true mean. Impressively, our posterior captures this true mean quite effectively. As the size of "training/constraint sample" `x_sample` increases - that is, as we accumulate more data - you'll observe the posterior becoming narrower. This tightening of the distribution signifies increasing confidence in our estimate, nudging it ever closer to the true value.

---''')

    st.markdown(r'''## Gibbs Sampling

Gibbs sampling is a type of Markov Chain Monte Carlo (MCMC) method that's used to obtain a sequence of samples from a multivariate distribution. The primary principle behind Gibbs sampling is relatively simple: 

Given a multivariate distribution, we sample from each variable's conditional distribution while holding the others fixed, cycling through each variable in turn. This sequence of samples forms a Markov chain. As the number of iterations increases, this chain converges to the target multivariate distribution.

### Gibbs Sampling - The Math

The main idea of Gibbs sampling is to sample from the full conditional distributions of each variable in turn, treating all other variables as known. For a set of variables $X_1, X_2, ..., X_n$, the Gibbs sampling algorithm iterates as:

1. Choose an initial state for the variables.
2. For each $X_i$:
   - Sample $X_i \sim p(X_i | X_1, X_2, ..., X_{i-1}, X_{i+1}, ..., X_n)$

Given the full joint distribution $p(X_1, X_2, ..., X_n)$, the challenge often lies in deriving these full conditional distributions. 

### Benefits of Gibbs Sampling:

1. **Simplicity**: Gibbs sampling can be more straightforward to implement for specific problems than other MCMC methods.
  
2. **Efficiency**: In cases where the conditional distributions are known and easy to sample from, Gibbs sampling can be efficient.

3. **Deterministic**: Unlike Metropolis-Hastings, which has an inherent randomness due to its acceptance-rejection step, Gibbs sampling is deterministic and always accept the proposed sample.


Even if Gibbs sampling always accept the proposed sample, as we will see in the example below, it does not necessary mean the effective sample is high. Gibbs sampling can lead to high autocorrelation between successive samples, especially in high-dimensional distributions or when variables are highly correlated.


### Toy Example: Sampling from a 2D Gaussian Distribution

The given code provides a demonstration of Gibbs sampling on a bivariate Gaussian distribution. Although it is already trivial to sample a bivariate Gaussian distribution from numpy, in the following, we will nonetheless show how this can be done with Gibbs sampling. The primary aspect being illustrated is the difference in the convergence and autocorrelation of the Gibbs sampler when the two variables have high correlation vs. low correlation.

### Gibbs Sampling for Bivariate Gaussian

For a bivariate Gaussian distribution with parameters:

- Mean: $ \mu = \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix} $
- Covariance matrix: $ \Sigma = \begin{bmatrix} \sigma_x^2 & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma_y^2 \end{bmatrix} $

The conditional distributions are:

$$
 p(x|y) = \mathcal{N}(\mu_x + \rho \sigma_x \dfrac{y - \mu_y}{\sigma_y}, (1 - \rho^2) \sigma_x^2) 
$$

$$
 p(y|x) = \mathcal{N}(\mu_y + \rho \sigma_y \dfrac{x - \mu_x}{\sigma_x}, (1 - \rho^2) \sigma_y^2) 
$$

In the example below, we will assume both $ \mu_x $ and $ \mu_y $ are 0, and $ \sigma_x^2 = \sigma_y^2 = 1 $, thus simplifying these formulas.

$$
 p(x|y) = \mathcal{N}(\rho y, 1 - \rho^2) 
$$

$$
 p(y|x) = \mathcal{N}(\rho x, 1 - \rho^2) 
$$

where $ \rho $ is the correlation coefficient.

In the Gibbs sampler below, these distributions are being sampled from in the loop. When sampling $ x $ given $ y $, the mean becomes $ \rho y $ and the variance becomes $ 1 - \rho^2 $, and vice versa for $ y $ given $ x $.
''')

    st.code('''# Define a Gibbs sampler function for a bivariate Gaussian distribution
def gibbs_sampler(n_samples, rho):
    """
    Perform Gibbs sampling for a bivariate Gaussian distribution.
    
    Parameters:
    - n_samples: Number of samples to draw
    - rho: Correlation coefficient of the bivariate Gaussian
    
    Returns:
    - samples: Array of sampled points
    """

    # Initialize storage for samples
    samples = np.zeros((n_samples, 2))

    # Start from an arbitrary point, here (100,100)
    x = 100
    y = 100

    # Begin Gibbs sampling iterations
    for i in range(n_samples):

        # Sample x from its conditional distribution given y
        x = stats.multivariate_normal.rvs(mean=rho*y, cov=1-rho**2, size=1)
    
        # Sample y from its conditional distribution given x
        y = stats.multivariate_normal.rvs(mean=rho*x, cov=1-rho**2, size=1)
    
        # Store the sample
        samples[i, :] = np.array([x, y])

    return samples
''', language='python')

    # Define a Gibbs sampler function for a bivariate Gaussian distribution
    def gibbs_sampler(n_samples, rho):
        """
        Perform Gibbs sampling for a bivariate Gaussian distribution.
        
        Parameters:
        - n_samples: Number of samples to draw
        - rho: Correlation coefficient of the bivariate Gaussian
        
        Returns:
        - samples: Array of sampled points
        """
    
        # Initialize storage for samples
        samples = np.zeros((n_samples, 2))
    
        # Start from an arbitrary point, here (100,100)
        x = 100
        y = 100
    
        # Begin Gibbs sampling iterations
        for i in range(n_samples):
    
            # Sample x from its conditional distribution given y
            x = stats.multivariate_normal.rvs(mean=rho*y, cov=1-rho**2, size=1)
        
            # Sample y from its conditional distribution given x
            y = stats.multivariate_normal.rvs(mean=rho*x, cov=1-rho**2, size=1)
        
            # Store the sample
            samples[i, :] = np.array([x, y])
    
        return samples
    
    

    st.code('''# Configure plotting parameters for better clarity
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 20
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

# --------------------------------------------------------------------------------
# Define a function to visualize the samples
def plot_samples(samples, n_burn, n_thin):
    """
    Plot the samples obtained from Gibbs sampling.
    
    Parameters:
    - samples: Array of sampled points
    - n_burn: Number of initial samples to discard to ensure convergence (burn-in period)
    - n_thin: Thinning factor to reduce autocorrelation. E.g., n_thin=10 means taking every 10th sample.
    """
    
    plt.figure(figsize=(10, 10))

    # plot scatter plot of the samples including the marginal distributions with seaborn
    sns.set_style('white')
    sns.jointplot(x=samples[n_burn::n_thin, 0], y=samples[n_burn::n_thin, 1], kind='scatter', alpha=0.5)

    # including axes
    plt.xlabel('x')
    plt.ylabel('y')

    # compare with the true distribution 
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = stats.multivariate_normal.pdf(np.dstack((X, Y)), mean=[0, 0], cov=[[1, rho], [rho, 1]])
    plt.contour(X, Y, Z, colors='k', alpha=0.5)


# ---------------------------------------------------------------------------------
# Define a function to visualize the progression of the Markov chain
def plot_chains(samples, n_burn):
    """
    Visualize the progression of the Markov chain to diagnose convergence.
    
    Parameters:
    - samples: Array of sampled points
    - n_burn: Number of initial samples to consider for burn-in visualization
    """
    # plot the chains and show burn-in and correlation
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(samples[:n_burn, 0])
    plt.plot(samples[:n_burn, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend(['x', 'y'], frameon=False)

    # show the chain after burn-in
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(n_burn, n_burn+100), samples[n_burn:n_burn+100, 0])
    plt.plot(np.arange(n_burn, n_burn+100), samples[n_burn:n_burn+100, 1])
    plt.xlabel('Iteration')
    plt.legend(['x', 'y'], frameon=False)
''', language='python')

    # Configure plotting parameters for better clarity
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    
    # --------------------------------------------------------------------------------
    # Define a function to visualize the samples
    def plot_samples(samples, n_burn, n_thin):
        """
        Plot the samples obtained from Gibbs sampling.
        
        Parameters:
        - samples: Array of sampled points
        - n_burn: Number of initial samples to discard to ensure convergence (burn-in period)
        - n_thin: Thinning factor to reduce autocorrelation. E.g., n_thin=10 means taking every 10th sample.
        """
        
        fig, ax = plt.subplots(figsize=(10, 10))
    
        # plot scatter plot of the samples including the marginal distributions with seaborn
        sns.set_style('white')
        sns.jointplot(x=samples[n_burn::n_thin, 0], y=samples[n_burn::n_thin, 1], kind='scatter', alpha=0.5)
    
        # including axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
        # compare with the true distribution 
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = stats.multivariate_normal.pdf(np.dstack((X, Y)), mean=[0, 0], cov=[[1, rho], [rho, 1]])
        plt.contour(X, Y, Z, colors='k', alpha=0.5)
    
    # ---------------------------------------------------------------------------------
    # Define a function to visualize the progression of the Markov chain
    def plot_chains(samples, n_burn):
        """
        Visualize the progression of the Markov chain to diagnose convergence.
        
        Parameters:
        - samples: Array of sampled points
        - n_burn: Number of initial samples to consider for burn-in visualization
        """
        # plot the chains and show burn-in and correlation
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        ax.plot(samples[:n_burn, 0])
        ax.plot(samples[:n_burn, 1])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.legend(['x', 'y'], frameon=False)
    
        # show the chain after burn-in
        plt.subplot(1, 2, 2)
        ax.plot(np.arange(n_burn, n_burn+100), samples[n_burn:n_burn+100, 0])
        ax.plot(np.arange(n_burn, n_burn+100), samples[n_burn:n_burn+100, 1])
        ax.set_xlabel('Iteration')
        ax.legend(['x', 'y'], frameon=False)
    st.pyplot(fig)
    
    

    st.markdown(r'''### High Correlation ($\rho = 0.95$)

When the correlation coefficient $\rho$ is set to 0.95, the two Gaussian variables are highly correlated. Gibbs sampling can face challenges in such situations due to the inherent nature of its sampling mechanism.

1. **Influence of Conditional Sampling**: In Gibbs sampling, each variable is sampled based on the current value of the other variable(s). High correlation means that when one variable is given, there's less uncertainty about the other. This interdependence can impact the sampler's behavior in several ways:

   - **Slower convergence**: The Markov chain may take more iterations to converge to the stationary distribution. This is because each sampling step is heavily influenced by the current state due to the high correlation.

   - **Higher autocorrelation**: As consecutive samples are more influenced by their preceding values, they tend to be more similar to each other. This high autocorrelation means that even if you draw many samples, the effective number of independent samples may be much lower.

   - **Longer burn-in period**: A longer initial period might be required before the chain reaches its stationary distribution. This is the 'burn-in' period, during which samples are often discarded.

Let's draw the samples for this high-correlation scenario and observe these behaviors:

''')

    st.code('''# draw samples with high-correlation
n_samples = 10000
rho = 0.95
samples = gibbs_sampler(n_samples, rho)
plot_samples(samples, n_burn=100, n_thin=1)''', language='python')

    # draw samples with high-correlation
    n_samples = 10000
    rho = 0.95
    samples = gibbs_sampler(n_samples, rho)
    plot_samples(samples, n_burn=100, n_thin=1)
    
    

    st.code('''# plot the chains to visualize the sampling behavior
plot_chains(samples, n_burn=100)''', language='python')

    # plot the chains to visualize the sampling behavior
    plot_chains(samples, n_burn=100)
    
    

    st.markdown(r'''### Diving into Autocorrelation and Effective Sample Size

As evident from the preceding figure, the Gibbs sampler, when grappling with our high-correlation Gaussian, takes roughly 50 iterations just to settle into the accurate region of the distribution. Yet, even after that initial settling, there's still a pronounced correlation between successive iterations.

To get a clearer grasp of the sampler's behavior in this scenario, we're going to delve into the autocorrelation of the drawn samples as before. This will provide insights into the independence (or lack thereof) between samples. On top of that, we'll determine the Effective Sample Size. The ESS highlights how many of our drawn samples can be considered as truly 'independent' in statistical terms.
''')

    st.code('''import statsmodels.api as sm

def calculate_ess(samples):
    """Calculate Effective Sample Size."""
    N = len(samples)
    autocorr_values = sm.tsa.acf(samples, nlags=100, fft=True)
    return N / (1 + 2 * sum(autocorr_values))

# Extract the samples for 'x' and 'y' variables
samples_x = samples[:, 0]
samples_y = samples[:, 1]

# Plot autocorrelation for 'x'
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sm.graphics.tsa.plot_acf(samples_x[100:], lags=100, ax=axes[0])
axes[0].set_title("Autocorrelation for x")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Autocorrelation")

# Calculate and display ESS for 'x'
ess_x = calculate_ess(samples_x[100:])
axes[0].annotate(f"ESS (x): {{}ess_x:.1f{}}", xy=(0.6, 0.9), xycoords='axes fraction')

# Plot autocorrelation for 'y'
sm.graphics.tsa.plot_acf(samples_y[100:], lags=100, ax=axes[1])
axes[1].set_title("Autocorrelation for y")
axes[1].set_xlabel("Lag")

# Calculate and display ESS for 'y'
ess_y = calculate_ess(samples_y[100:])
axes[1].annotate(f"ESS (y): {{}ess_y:.1f{}}", xy=(0.6, 0.9), xycoords='axes fraction')

plt.tight_layout()
''', language='python')

    import statsmodels.api as sm
    
    def calculate_ess(samples):
        """Calculate Effective Sample Size."""
        N = len(samples)
        autocorr_values = sm.tsa.acf(samples, nlags=100, fft=True)
        return N / (1 + 2 * sum(autocorr_values))
    
    # Extract the samples for 'x' and 'y' variables
    samples_x = samples[:, 0]
    samples_y = samples[:, 1]
    
    # Plot autocorrelation for 'x'
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    sm.graphics.tsa.plot_acf(samples_x[100:], lags=100, ax=axes[0])
    axes[0].set_title("Autocorrelation for x")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    
    # Calculate and display ESS for 'x'
    ess_x = calculate_ess(samples_x[100:])
    axes[0].annotate(f"ESS (x): {ess_x:.1f}", xy=(0.6, 0.9), xycoords='axes fraction')
    
    # Plot autocorrelation for 'y'
    sm.graphics.tsa.plot_acf(samples_y[100:], lags=100, ax=axes[1])
    axes[1].set_title("Autocorrelation for y")
    axes[1].set_xlabel("Lag")
    
    # Calculate and display ESS for 'y'
    ess_y = calculate_ess(samples_y[100:])
    axes[1].annotate(f"ESS (y): {ess_y:.1f}", xy=(0.6, 0.9), xycoords='axes fraction')
    
    st.pyplot(fig)
    
    

    st.markdown(r'''### Understanding Low Correlation ($\rho = 0.1$) in Gibbs Sampling

When dealing with Gaussian variables that have a correlation coefficient ($\rho$) of just 0.1, we're essentially exploring a scenario where the two variables exhibit only slight correlation. This has profound implications for the Gibbs sampler's performance, providing certain benefits:

- **Quicker Convergence**: One of the inherent features of low correlation is that the variables don't significantly "pull" or influence each other. As a result, the Gibbs sampler doesn't "wander around" as much trying to understand the joint distribution, leading the Markov chain to reach the target distribution more promptly.

- **Reduced Autocorrelation**: A low correlation means that knowing the value of one variable gives you very little information about the value of the other variable. This leads to more independent sampling. In turn, this means that consecutive samples generated by the Gibbs sampler have a higher chance of being distinct, reducing the issue of autocorrelation.

- **Trimmed Burn-in Phase**: Given the sampler's faster convergence and the reduced influence between variables, it becomes logical that we can confidently use the chain's samples sooner. This means the need to discard a large chunk of initial samples (burn-in) diminishes.

Now, let's visualize this behavior through some graphical representations and computations.

''')

    st.code('''# Drawing samples with low correlation
n_samples = 10000
rho = 0.1
samples_low_corr = gibbs_sampler(n_samples, rho)

# Plotting the samples
plot_samples(samples_low_corr, n_burn=100, n_thin=10)
''', language='python')

    # Drawing samples with low correlation
    n_samples = 10000
    rho = 0.1
    samples_low_corr = gibbs_sampler(n_samples, rho)
    
    # Plotting the samples
    plot_samples(samples_low_corr, n_burn=100, n_thin=10)
    
    

    st.code('''# Visualizing the chains
plot_chains(samples_low_corr, n_burn=100)''', language='python')

    # Visualizing the chains
    plot_chains(samples_low_corr, n_burn=100)
    
    

    st.code('''# Extract the samples for 'x' and 'y' variables
samples_x = samples_low_corr[:, 0]
samples_y = samples_low_corr[:, 1]

# Plot autocorrelation for 'x'
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sm.graphics.tsa.plot_acf(samples_x[100:], lags=100, ax=axes[0])
axes[0].set_title("Autocorrelation for x")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Autocorrelation")

# Calculate and display ESS for 'x'
ess_x = calculate_ess(samples_x[100:])
axes[0].annotate(f"ESS (x): {{}ess_x:.1f{}}", xy=(0.6, 0.9), xycoords='axes fraction')

# Plot autocorrelation for 'y'
sm.graphics.tsa.plot_acf(samples_y[100:], lags=100, ax=axes[1])
axes[1].set_title("Autocorrelation for y")
axes[1].set_xlabel("Lag")

# Calculate and display ESS for 'y'
ess_y = calculate_ess(samples_y[100:])
axes[1].annotate(f"ESS (y): {{}ess_y:.1f{}}", xy=(0.6, 0.9), xycoords='axes fraction')

plt.tight_layout()
''', language='python')

    # Extract the samples for 'x' and 'y' variables
    samples_x = samples_low_corr[:, 0]
    samples_y = samples_low_corr[:, 1]
    
    # Plot autocorrelation for 'x'
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    sm.graphics.tsa.plot_acf(samples_x[100:], lags=100, ax=axes[0])
    axes[0].set_title("Autocorrelation for x")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    
    # Calculate and display ESS for 'x'
    ess_x = calculate_ess(samples_x[100:])
    axes[0].annotate(f"ESS (x): {ess_x:.1f}", xy=(0.6, 0.9), xycoords='axes fraction')
    
    # Plot autocorrelation for 'y'
    sm.graphics.tsa.plot_acf(samples_y[100:], lags=100, ax=axes[1])
    axes[1].set_title("Autocorrelation for y")
    axes[1].set_xlabel("Lag")
    
    # Calculate and display ESS for 'y'
    ess_y = calculate_ess(samples_y[100:])
    axes[1].annotate(f"ESS (y): {ess_y:.1f}", xy=(0.6, 0.9), xycoords='axes fraction')
    
    st.pyplot(fig)
    
    

    st.markdown(r'''Gibbs sampling, a type of Markov Chain Monte Carlo (MCMC) method, reveals its strengths and weaknesses when confronted with variables of varying correlations. In the regime of high-correlation, Gibbs sampling faces challenges. It exhibits slower convergence, higher autocorrelation between samples, and demands a more extended burn-in phase. This behavior stems from the strong interdependency between variables, causing the sampler to move more cautiously in the state space. 

In the regime of low correlation, the method shines. The sampler converges more rapidly, samples have reduced autocorrelation, and there's a shortened burn-in period. The minimal influence between variables grants the sampler more freedom, resulting in more efficient sampling.

In a nutshell, while Gibbs sampling is a powerful tool, its efficacy is influenced by the correlation structure of the target distribution. Recognizing and understanding these nuances is pivotal for effective sampling and model diagnostics.

---''')

    st.markdown(r'''### Summary

In our exploration of the expansive landscape of Markov Chain Monte Carlo methods, two standout techniques captured our attention: Metropolis-Hastings and Gibbs Sampling.

**Metropolis-Hastings**:

- A versatile and widely-used MCMC algorithm, Metropolis-Hastings offers flexibility in navigating complex posterior distributions. Its core revolves around a proposal distribution and an acceptance mechanism, allowing it to sample from intricate, high-dimensional spaces.

- As we saw through our Gaussian-Cauchy example, proposal distribution tuning, burn-in phases, and autocorrelation diagnostics are crucial elements in harnessing the algorithm's full potential.

**Gibbs Sampling**:

- Gibbs Sampling epitomizes the "divide and conquer" paradigm in the world of MCMC. By breaking down a multivariate distribution into simpler conditional distributions, this method promises deterministic sampling with a characteristic 100% acceptance rate.

- Our dive into bivariate Gaussian distributions illuminated Gibbs Sampling's nuances — from its efficient stride in low-correlation terrains to its cautious treading in highly correlated scenarios.

Understanding the underlying mechanics, strengths, and limitations of these algorithms is paramount in the realm of Bayesian inference. Both techniques offer unique perspectives and tools for addressing complex sampling challenges. While Metropolis-Hastings provides a broad, adaptable framework, Gibbs Sampling shines in contexts where conditional distributions are more tractable.

Through the specific examples and broader discussions presented, this tutorial underscores the essential role of MCMC methods in modern statistics and the fascinating interplay of theory and practice they bring to the table.
''')

if __name__ == '__main__':
    show_page()
