<!-- ---
title: "Probability Distributions"
description: "Understanding how probability distributions model uncertainty and randomness in data and decision-making processes"
tags: ["mathematics", "statistics", "probability", "data-science", "programming"]
difficulty: "intermediate"
category: "concept"
symbol: "P(X), f(x), μ, σ"
prerequisites: ["descriptive-stats", "basic-arithmetic", "functions"]
related_concepts: ["hypothesis-testing", "confidence-intervals", "bayes-theorem", "central-limit-theorem"]
applications: ["data-analysis", "machine-learning", "risk-assessment", "quality-control"]
interactive: true
code_examples: true
complexity_analysis: true
real_world_examples: true
layout: "concept-page"
date_created: "2024-01-01"
last_updated: "2024-01-01"
author: "Snake Math Team"
reviewers: []
version: "1.0"
--- -->

# Probability Distributions (P(X), f(x), μ, σ)

Think of probability distributions as weather forecasts for data! Just like meteorologists predict the likelihood of rain, sunshine, or snow, probability distributions tell us how likely different outcomes are for any random process. They're the mathematical crystal balls that help us understand and quantify uncertainty.

## Understanding Probability Distributions

**Probability distributions** describe how likely different outcomes are for a random variable. They're like blueprints that show us the shape of randomness - some outcomes cluster around the middle, others spread out evenly, and some have long tails of rare but possible events.

The fundamental types include:

$$
\begin{align}
\text{Discrete (Binomial): } P(X = k) &= \binom{n}{k}p^k(1-p)^{n-k} \\
\text{Continuous (Normal): } f(x) &= \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\end{align}
$$

Think of probability distributions like different personality types for randomness - the normal distribution is the "balanced" type that likes to stay near the average, while the uniform distribution is the "rebel" that treats all outcomes equally:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Different "personalities" of randomness
np.random.seed(42)

# Normal: "The Balanced Type" - loves the middle
normal_data = np.random.normal(100, 15, 1000)

# Uniform: "The Rebel" - all outcomes are equal
uniform_data = np.random.uniform(70, 130, 1000)

# Exponential: "The Dramatic Type" - lots of small values, few big ones
exponential_data = np.random.exponential(20, 1000) + 70

print(f"Normal distribution: mean={np.mean(normal_data):.1f}, std={np.std(normal_data):.1f}")
print(f"Uniform distribution: mean={np.mean(uniform_data):.1f}, std={np.std(uniform_data):.1f}")
print(f"Exponential distribution: mean={np.mean(exponential_data):.1f}, std={np.std(exponential_data):.1f}")
```

## Why Probability Distributions Matter for Programmers

Probability distributions are essential for modeling uncertainty in algorithms, building robust machine learning models, conducting A/B tests, estimating confidence intervals, and making data-driven decisions under uncertainty.

Understanding distributions helps you choose appropriate statistical methods, generate realistic test data, model user behavior, assess risks, and build systems that handle uncertainty gracefully.


## Interactive Exploration

<ProbabilitySimulator />

```plaintext
Component conceptualization:
Create an interactive probability distributions explorer where users can:
- Select different distribution types (normal, binomial, poisson, exponential, uniform) with parameter sliders
- Visualize probability density/mass functions with real-time parameter changes
- Compare multiple distributions side-by-side with overlay capabilities
- Demonstrate Central Limit Theorem with sample size and sample count controls
- Interactive A/B testing simulator with statistical significance calculations
- Bayes' theorem calculator with visual probability tree diagrams
- Confidence interval visualization showing coverage probability
- Monte Carlo simulation tools for complex probability scenarios
- Generate random samples and analyze their statistical properties
The component should provide both theoretical curves and empirical demonstrations.
```

Experiment with different distributions and their parameters to understand how they model various types of uncertainty and randomness.


## Probability Distribution Techniques and Efficiency

Understanding different approaches to working with probability distributions helps optimize calculations and choose appropriate methods for different scenarios.

### Method 1: Manual Mathematical Implementation

**Pros**: Educational value, complete understanding, no dependencies\
**Complexity**: Varies by distribution, O(1) for simple PDFs, O(k) for combinatorial calculations

```python
import math

def manual_probability_calculations():
    """Implement probability distributions from mathematical definitions"""
    
    def factorial(n):
        """Calculate factorial manually"""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def binomial_coefficient(n, k):
        """Calculate n choose k efficiently"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        # Use multiplicative formula to avoid large factorials
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def binomial_pmf(k, n, p):
        """Calculate P(X = k) for binomial distribution"""
        coeff = binomial_coefficient(n, k)
        prob = coeff * (p ** k) * ((1 - p) ** (n - k))
        return prob
    
    def normal_pdf(x, mu, sigma):
        """Calculate probability density for normal distribution"""
        coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coefficient * math.exp(exponent)
    
    def poisson_pmf(k, lambda_param):
        """Calculate P(X = k) for Poisson distribution"""
        return (math.exp(-lambda_param) * (lambda_param ** k)) / factorial(k)
    
    def exponential_pdf(x, lambda_param):
        """Calculate probability density for exponential distribution"""
        if x < 0:
            return 0
        return lambda_param * math.exp(-lambda_param * x)
    
    # Demonstration
    print("Manual Probability Distribution Calculations")
    print("=" * 50)
    
    # Binomial example: 10 coin flips, P(X = 6 heads)
    n, k, p = 10, 6, 0.5
    binom_prob = binomial_pmf(k, n, p)
    print(f"Binomial(n={n}, p={p}): P(X = {k}) = {binom_prob:.6f}")
    
    # Normal example: height distribution
    mu, sigma = 170, 10
    heights = [160, 170, 180, 190]
    print(f"\nNormal(μ={mu}, σ={sigma}) PDF values:")
    for height in heights:
        pdf_val = normal_pdf(height, mu, sigma)
        print(f"  f({height}) = {pdf_val:.6f}")
    
    # Poisson example: events per hour
    lambda_param = 3
    events = [0, 1, 2, 3, 4, 5]
    print(f"\nPoisson(λ={lambda_param}) probabilities:")
    for event_count in events:
        poisson_prob = poisson_pmf(event_count, lambda_param)
        print(f"  P(X = {event_count}) = {poisson_prob:.6f}")
    
    # Exponential example: waiting times
    lambda_param = 0.5
    times = [0.5, 1.0, 2.0, 3.0]
    print(f"\nExponential(λ={lambda_param}) PDF values:")
    for time in times:
        exp_pdf = exponential_pdf(time, lambda_param)
        print(f"  f({time}) = {exp_pdf:.6f}")
    
    return binom_prob, heights

manual_probability_calculations()
```

### Method 2: SciPy Statistical Functions

**Pros**: Highly optimized, extensive functionality, numerical stability\
**Complexity**: O(1) for most operations, optimized implementations

```python
from scipy import stats
import numpy as np
import time

def scipy_probability_calculations():
    """Demonstrate SciPy's statistical functions for probability distributions"""
    
    print("SciPy Probability Distribution Analysis")
    print("=" * 45)
    
    # Normal distribution analysis
    mu, sigma = 100, 15
    normal_dist = stats.norm(mu, sigma)
    
    print(f"Normal Distribution (μ={mu}, σ={sigma}):")
    
    # PDF and CDF values
    x_values = [70, 85, 100, 115, 130]
    print(f"{'x':>4} {'PDF':>10} {'CDF':>10} {'SF':>10}")
    print("-" * 38)
    
    for x in x_values:
        pdf_val = normal_dist.pdf(x)
        cdf_val = normal_dist.cdf(x)
        sf_val = normal_dist.sf(x)  # Survival function: 1 - CDF
        print(f"{x:4d} {pdf_val:10.6f} {cdf_val:10.6f} {sf_val:10.6f}")
    
    # Percentiles and quantiles
    percentiles = [5, 25, 50, 75, 95]
    print(f"\nPercentiles:")
    for p in percentiles:
        quantile = normal_dist.ppf(p / 100)  # Percent point function (inverse CDF)
        print(f"  {p:2d}th percentile: {quantile:.2f}")
    
    # Discrete distributions
    print(f"\nBinomial Distribution (n=20, p=0.3):")
    n, p = 20, 0.3
    binom_dist = stats.binom(n, p)
    
    k_values = range(0, 11)
    print(f"{'k':>3} {'PMF':>10} {'CDF':>10}")
    print("-" * 26)
    
    for k in k_values:
        pmf_val = binom_dist.pmf(k)
        cdf_val = binom_dist.cdf(k)
        print(f"{k:3d} {pmf_val:10.6f} {cdf_val:10.6f}")
    
    # Distribution properties
    print(f"\nDistribution Properties:")
    distributions = {
        'Normal(100, 15)': stats.norm(100, 15),
        'Binomial(20, 0.3)': stats.binom(20, 0.3),
        'Poisson(5)': stats.poisson(5),
        'Exponential(0.5)': stats.expon(scale=1/0.5)
    }
    
    print(f"{'Distribution':>20} {'Mean':>10} {'Variance':>10} {'Std Dev':>10}")
    print("-" * 53)
    
    for name, dist in distributions.items():
        mean, var = dist.stats(moments='mv')
        std_dev = np.sqrt(var)
        print(f"{name:>20} {mean:10.3f} {var:10.3f} {std_dev:10.3f}")
    
    # Performance comparison
    print(f"\nPerformance Test (1,000,000 calculations):")
    n_calc = 1000000
    x_test = np.random.normal(100, 15, n_calc)
    
    # SciPy timing
    start_time = time.time()
    scipy_results = stats.norm.pdf(x_test, 100, 15)
    scipy_time = time.time() - start_time
    
    print(f"SciPy vectorized: {scipy_time:.4f} seconds")
    print(f"Throughput: {n_calc/scipy_time:,.0f} calculations/second")
    
    return normal_dist, binom_dist

scipy_probability_calculations()
```

### Method 3: Monte Carlo Simulation

**Pros**: Handles complex scenarios, intuitive approach, parallelizable\
**Complexity**: O(n) where n is number of simulations, can be distributed

```python
def monte_carlo_probability_estimation():
    """Use Monte Carlo simulation to estimate probabilities"""
    
    print("Monte Carlo Probability Estimation")
    print("=" * 40)
    
    def estimate_pi():
        """Estimate π using random points in a circle"""
        n_simulations = 1000000
        np.random.seed(42)
        
        # Generate random points in unit square
        x = np.random.uniform(-1, 1, n_simulations)
        y = np.random.uniform(-1, 1, n_simulations)
        
        # Count points inside unit circle
        inside_circle = (x**2 + y**2) <= 1
        pi_estimate = 4 * np.sum(inside_circle) / n_simulations
        
        print(f"Estimating π with {n_simulations:,} random points:")
        print(f"  Estimated π: {pi_estimate:.6f}")
        print(f"  Actual π: {math.pi:.6f}")
        print(f"  Error: {abs(pi_estimate - math.pi):.6f}")
        
        return pi_estimate
    
    def probability_of_sum():
        """Estimate probability of dice sum using simulation"""
        n_rolls = 100000
        np.random.seed(42)
        
        # Roll two dice many times
        die1 = np.random.randint(1, 7, n_rolls)
        die2 = np.random.randint(1, 7, n_rolls)
        sums = die1 + die2
        
        print(f"\nDice Sum Probabilities ({n_rolls:,} rolls):")
        print(f"{'Sum':>4} {'Simulated':>12} {'Theoretical':>12} {'Difference':>12}")
        print("-" * 44)
        
        for target_sum in range(2, 13):
            # Simulated probability
            simulated_prob = np.sum(sums == target_sum) / n_rolls
            
            # Theoretical probability
            if target_sum <= 7:
                ways = target_sum - 1
            else:
                ways = 13 - target_sum
            theoretical_prob = ways / 36
            
            difference = abs(simulated_prob - theoretical_prob)
            
            print(f"{target_sum:4d} {simulated_prob:12.6f} {theoretical_prob:12.6f} {difference:12.6f}")
    
    def central_limit_theorem():
        """Demonstrate Central Limit Theorem with simulation"""
        n_samples = 1000
        sample_sizes = [1, 5, 10, 30, 50]
        
        print(f"\nCentral Limit Theorem Demonstration:")
        print(f"Drawing from uniform distribution [0, 1]")
        print(f"{'Sample Size':>12} {'Mean of Means':>15} {'Std of Means':>15}")
        print("-" * 45)
        
        np.random.seed(42)
        
        for sample_size in sample_sizes:
            # Generate many samples and calculate their means
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.uniform(0, 1, sample_size)
                sample_means.append(np.mean(sample))
            
            sample_means = np.array(sample_means)
            mean_of_means = np.mean(sample_means)
            std_of_means = np.std(sample_means, ddof=1)
            
            # Theoretical standard error
            theoretical_se = (1/12)**0.5 / np.sqrt(sample_size)  # σ/√n for uniform[0,1]
            
            print(f"{sample_size:12d} {mean_of_means:15.6f} {std_of_means:15.6f}")
            print(f"{'':12} {'(≈ 0.5)':>15} {'(≈ ' + f'{theoretical_se:.6f})':>15}")
    
    def bootstrap_confidence_interval():
        """Calculate confidence interval using bootstrap resampling"""
        # Original sample
        np.random.seed(42)
        original_sample = np.random.normal(100, 15, 50)
        original_mean = np.mean(original_sample)
        
        # Bootstrap resampling
        n_bootstrap = 10000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence intervals
        ci_95_lower = np.percentile(bootstrap_means, 2.5)
        ci_95_upper = np.percentile(bootstrap_means, 97.5)
        
        print(f"\nBootstrap Confidence Interval:")
        print(f"Original sample mean: {original_mean:.3f}")
        print(f"Bootstrap estimates: {np.mean(bootstrap_means):.3f} ± {np.std(bootstrap_means):.3f}")
        print(f"95% CI: [{ci_95_lower:.3f}, {ci_95_upper:.3f}]")
        
        return bootstrap_means
    
    # Run demonstrations
    estimate_pi()
    probability_of_sum()
    central_limit_theorem()
    bootstrap_ci = bootstrap_confidence_interval()
    
    return bootstrap_ci

monte_carlo_probability_estimation()
```


## Why Bayes' Theorem Works

Bayes' theorem provides a mathematical framework for updating our beliefs when we receive new evidence. It's the foundation of rational decision-making under uncertainty:

```python
def explain_bayes_theorem():
    """Demonstrate how Bayes' theorem works with intuitive examples"""
    
    print("Understanding Bayes' Theorem")
    print("P(A|B) = P(B|A) × P(A) / P(B)")
    print("=" * 35)
    
    def medical_diagnosis_example():
        """Medical diagnosis: updating probability with test results"""
        
        print("Medical Diagnosis Example:")
        print("A rare disease affects 1 in 1000 people")
        print("Test is 99% accurate (both sensitivity and specificity)")
        
        # Prior probabilities
        P_disease = 0.001  # 1 in 1000
        P_no_disease = 0.999
        
        # Test characteristics
        P_pos_given_disease = 0.99    # Sensitivity
        P_neg_given_no_disease = 0.99  # Specificity
        P_pos_given_no_disease = 0.01  # False positive rate
        
        # Calculate P(positive test) using law of total probability
        P_positive = (P_pos_given_disease * P_disease + 
                     P_pos_given_no_disease * P_no_disease)
        
        # Apply Bayes' theorem
        P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive
        
        print(f"\nBefore test (prior): P(disease) = {P_disease:.3f} ({P_disease*100:.1f}%)")
        print(f"After positive test: P(disease|+) = {P_disease_given_pos:.3f} ({P_disease_given_pos*100:.1f}%)")
        print(f"Even with 99% accurate test, only ~9.0% chance of actually having disease!")
        
        # Show the calculation step by step
        print(f"\nStep-by-step calculation:")
        print(f"P(+|disease) × P(disease) = {P_pos_given_disease} × {P_disease} = {P_pos_given_disease * P_disease:.6f}")
        print(f"P(+) = {P_positive:.6f}")
        print(f"P(disease|+) = {P_pos_given_disease * P_disease:.6f} / {P_positive:.6f} = {P_disease_given_pos:.6f}")
        
        return P_disease_given_pos
    
    def spam_filter_example():
        """Email spam filtering: combining multiple pieces of evidence"""
        
        print(f"\n" + "="*50)
        print("Spam Filter Example:")
        print("Combining evidence from multiple words")
        
        # Prior probabilities
        P_spam = 0.4  # 40% of emails are spam
        P_legitimate = 0.6
        
        # Word probabilities
        words = ['FREE', 'URGENT', 'Click']
        P_word_given_spam = [0.8, 0.7, 0.6]
        P_word_given_legit = [0.05, 0.02, 0.1]
        
        print(f"\nPrior probability: P(spam) = {P_spam}")
        print(f"Email contains words: {words}")
        
        # Start with prior
        current_prob_spam = P_spam
        current_prob_legit = P_legitimate
        
        print(f"\nSequential Bayesian updating:")
        print(f"Initial: P(spam) = {current_prob_spam:.4f}")
        
        # Update probability for each word
        for i, (word, p_word_spam, p_word_legit) in enumerate(zip(words, P_word_given_spam, P_word_given_legit)):
            # Calculate evidence (total probability of seeing this word)
            P_word = p_word_spam * current_prob_spam + p_word_legit * current_prob_legit
            
            # Update using Bayes' theorem
            new_prob_spam = (p_word_spam * current_prob_spam) / P_word
            new_prob_legit = (p_word_legit * current_prob_legit) / P_word
            
            print(f"After '{word}': P(spam) = {new_prob_spam:.4f}")
            
            current_prob_spam = new_prob_spam
            current_prob_legit = new_prob_legit
        
        print(f"\nFinal classification:")
        if current_prob_spam > 0.5:
            print(f"SPAM (confidence: {current_prob_spam:.1%})")
        else:
            print(f"LEGITIMATE (confidence: {current_prob_legit:.1%})")
        
        return current_prob_spam
    
    def ab_testing_example():
        """A/B testing: updating conversion rate beliefs"""
        
        print(f"\n" + "="*50)
        print("A/B Testing Example:")
        print("Updating beliefs about conversion rates")
        
        # Prior belief: conversion rate around 10% ± 3%
        # Using Beta distribution as conjugate prior
        alpha_prior = 10  # "successes"
        beta_prior = 90   # "failures"
        
        print(f"Prior belief: Beta({alpha_prior}, {beta_prior})")
        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        print(f"Prior mean conversion rate: {prior_mean:.1%}")
        
        # Observed data
        visitors = 1000
        conversions = 120
        
        print(f"\nObserved data: {conversions} conversions out of {visitors} visitors")
        
        # Update using Bayesian inference (Beta-Binomial conjugacy)
        alpha_post = alpha_prior + conversions
        beta_post = beta_prior + (visitors - conversions)
        
        posterior_mean = alpha_post / (alpha_post + beta_post)
        
        # Calculate credible interval
        from scipy.stats import beta
        post_dist = beta(alpha_post, beta_post)
        ci_lower = post_dist.ppf(0.025)
        ci_upper = post_dist.ppf(0.975)
        
        print(f"\nPosterior: Beta({alpha_post}, {beta_post})")
        print(f"Updated conversion rate: {posterior_mean:.1%}")
        print(f"95% credible interval: [{ci_lower:.1%}, {ci_upper:.1%}]")
        
        # Compare with frequentist estimate
        freq_estimate = conversions / visitors
        freq_se = np.sqrt(freq_estimate * (1 - freq_estimate) / visitors)
        freq_ci_lower = freq_estimate - 1.96 * freq_se
        freq_ci_upper = freq_estimate + 1.96 * freq_se
        
        print(f"\nFrequentist comparison:")
        print(f"Point estimate: {freq_estimate:.1%}")
        print(f"95% confidence interval: [{freq_ci_lower:.1%}, {freq_ci_upper:.1%}]")
        
        return posterior_mean, (ci_lower, ci_upper)
    
    # Run examples
    medical_prob = medical_diagnosis_example()
    spam_prob = spam_filter_example()
    conversion_rate, ci = ab_testing_example()
    
    print(f"\n" + "="*50)
    print("Key Insights about Bayes' Theorem:")
    print("• Updates prior beliefs with new evidence")
    print("• Accounts for both test accuracy and base rates")
    print("• Provides probabilistic (not just yes/no) answers")
    print("• Allows sequential updating as more evidence arrives")
    print("• Quantifies uncertainty in our conclusions")
    
    return medical_prob, spam_prob, conversion_rate

explain_bayes_theorem()
```

### Normal Distribution

```python
import numpy as np
from scipy import stats
import math

def normal_distribution_analysis():
    """Comprehensive normal distribution analysis"""
    
    mu, sigma = 100, 15  # Mean and standard deviation
    
    print(f"Normal Distribution Analysis")
    print(f"μ = {mu}, σ = {sigma}")
    print("-" * 30)
    
    # Probability density function
    def normal_pdf(x, mu, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Calculate probabilities
    x_values = [85, 100, 115, 130]
    
    print("Probability Density:")
    for x in x_values:
        pdf_value = normal_pdf(x, mu, sigma)
        print(f"  f({x}) = {pdf_value:.6f}")
    
    # Cumulative probabilities
    print("\nCumulative Probabilities:")
    for x in x_values:
        cdf_value = stats.norm.cdf(x, mu, sigma)
        print(f"  P(X ≤ {x}) = {cdf_value:.4f}")
    
    # Empirical Rule (68-95-99.7 rule)
    print(f"\nEmpirical Rule:")
    print(f"  68% of data: [{mu - sigma:.1f}, {mu + sigma:.1f}]")
    print(f"  95% of data: [{mu - 2*sigma:.1f}, {mu + 2*sigma:.1f}]")
    print(f"  99.7% of data: [{mu - 3*sigma:.1f}, {mu + 3*sigma:.1f}]")
    
    # Standardization (Z-scores)
    print(f"\nStandardization:")
    for x in x_values:
        z = (x - mu) / sigma
        print(f"  Z-score for {x}: {z:.2f}")
    
    return mu, sigma

normal_distribution_analysis()
```

### Binomial Distribution

```python
def binomial_distribution_analysis():
    """Binomial distribution calculations and applications"""
    
    n, p = 20, 0.3  # Number of trials, probability of success
    
    print(f"Binomial Distribution Analysis")
    print(f"n = {n} trials, p = {p} probability of success")
    print("-" * 45)
    
    # Manual calculation
    def binomial_coefficient(n, k):
        """Calculate n choose k"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        # Use multiplicative formula to avoid large factorials
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def binomial_pmf(k, n, p):
        """Calculate P(X = k) for binomial distribution"""
        return binomial_coefficient(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    # Calculate specific probabilities
    k_values = [0, 5, 6, 7, 10, 15]
    
    print("Probability Mass Function:")
    for k in k_values:
        pmf_manual = binomial_pmf(k, n, p)
        pmf_scipy = stats.binom.pmf(k, n, p)
        print(f"  P(X = {k:2d}) = {pmf_manual:.6f} (manual) | {pmf_scipy:.6f} (scipy)")
    
    # Expected value and variance
    expected_value = n * p
    variance = n * p * (1 - p)
    std_dev = math.sqrt(variance)
    
    print(f"\nDistribution Properties:")
    print(f"  Expected value: E[X] = np = {expected_value:.2f}")
    print(f"  Variance: Var(X) = np(1-p) = {variance:.2f}")
    print(f"  Standard deviation: σ = {std_dev:.2f}")
    
    # Cumulative probabilities
    print(f"\nCumulative Probabilities:")
    print(f"  P(X ≤ 5) = {stats.binom.cdf(5, n, p):.4f}")
    print(f"  P(X ≥ 10) = {1 - stats.binom.cdf(9, n, p):.4f}")
    print(f"  P(5 ≤ X ≤ 8) = {stats.binom.cdf(8, n, p) - stats.binom.cdf(4, n, p):.4f}")

binomial_distribution_analysis()
```

### Central Limit Theorem Demonstration

The ProbabilitySimulator component includes an interactive demonstration of the Central Limit Theorem, showing how sample means approach a normal distribution regardless of the original population distribution.

## Probability Calculations

### Bayes' Theorem

```python
def bayes_theorem_examples():
    """Practical applications of Bayes' Theorem"""
    
    print("Bayes' Theorem Applications")
    print("P(A|B) = P(B|A) × P(A) / P(B)")
    print("=" * 40)
    
    # Medical diagnosis example
    print("\nExample 1: Medical Diagnosis")
    print("Disease affects 1% of population")
    print("Test has 95% sensitivity (detects disease when present)")
    print("Test has 90% specificity (negative when disease absent)")
    
    # Given information
    P_disease = 0.01  # Prior probability of disease
    P_no_disease = 1 - P_disease
    P_pos_given_disease = 0.95  # Sensitivity
    P_neg_given_no_disease = 0.90  # Specificity
    P_pos_given_no_disease = 1 - P_neg_given_no_disease  # False positive rate
    
    # Calculate P(positive test)
    P_positive = (P_pos_given_disease * P_disease + 
                 P_pos_given_no_disease * P_no_disease)
    
    # Bayes' theorem: P(disease | positive test)
    P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive
    
    print(f"P(disease | positive test) = {P_disease_given_pos:.4f} ({P_disease_given_pos*100:.1f}%)")
    print("Despite positive test, only ~8.7% chance of actually having disease!")
    
    # Spam filter example
    print("\nExample 2: Email Spam Filter")
    print("30% of emails are spam")
    print("Word 'FREE' appears in 60% of spam emails")
    print("Word 'FREE' appears in 5% of legitimate emails")
    
    P_spam = 0.30
    P_legitimate = 0.70
    P_free_given_spam = 0.60
    P_free_given_legit = 0.05
    
    # P(contains 'FREE')
    P_free = P_free_given_spam * P_spam + P_free_given_legit * P_legitimate
    
    # P(spam | contains 'FREE')
    P_spam_given_free = (P_free_given_spam * P_spam) / P_free
    
    print(f"P(spam | contains 'FREE') = {P_spam_given_free:.4f} ({P_spam_given_free*100:.1f}%)")

bayes_theorem_examples()
```

### Confidence Intervals

```python
def confidence_intervals():
    """Calculate and interpret confidence intervals"""
    
    # Sample data
    np.random.seed(123)
    sample_data = np.random.normal(100, 15, 25)  # Sample of 25 observations
    
    n = len(sample_data)
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    
    print("Confidence Interval Calculation")
    print(f"Sample size: {n}")
    print(f"Sample mean: {sample_mean:.2f}")
    print(f"Sample std: {sample_std:.2f}")
    print("-" * 35)
    
    # Confidence levels and corresponding z-scores
    confidence_levels = [0.90, 0.95, 0.99]
    z_scores = [1.645, 1.96, 2.576]
    
    print("Confidence Intervals for Population Mean:")
    
    for conf_level, z_score in zip(confidence_levels, z_scores):
        # Standard error
        standard_error = sample_std / math.sqrt(n)
        
        # Margin of error
        margin_of_error = z_score * standard_error
        
        # Confidence interval
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        
        print(f"{conf_level*100:4.0f}% CI: [{lower_bound:.2f}, {upper_bound:.2f}] (±{margin_of_error:.2f})")
    
    # t-distribution for small samples
    print(f"\nUsing t-distribution (df = {n-1}):")
    
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin_of_error_t = t_critical * (sample_std / math.sqrt(n))
        lower_bound_t = sample_mean - margin_of_error_t
        upper_bound_t = sample_mean + margin_of_error_t
        
        print(f"{conf_level*100:4.0f}% CI: [{lower_bound_t:.2f}, {upper_bound_t:.2f}] (±{margin_of_error_t:.2f})")
    
    print(f"\nInterpretation:")
    print(f"• We are 95% confident the true population mean is between the CI bounds")
    print(f"• If we repeated this process many times, 95% of intervals would contain μ")
    print(f"• Higher confidence = wider interval")

confidence_intervals()
```

## Real-World Applications

### A/B Testing

The ProbabilitySimulator component includes comprehensive A/B testing analysis with statistical significance testing, confidence intervals, and power analysis visualization.

## Key Takeaways

1. **Probability distributions** model uncertainty in different scenarios
2. **Normal distribution** is fundamental due to Central Limit Theorem
3. **Bayes' theorem** updates probabilities with new evidence
4. **Confidence intervals** quantify estimation uncertainty
5. **Hypothesis testing** provides framework for decision-making
6. **A/B testing** applies statistics to business decisions

## Next Steps

- Study **hypothesis testing** in depth
- Learn **regression analysis** for prediction
- Explore **time series analysis** for temporal data
- Apply probability to **machine learning** algorithms