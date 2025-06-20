# Probability Distributions

## Mathematical Concept

**Probability distributions** describe how likely different outcomes are for a random variable. Key types include:

**Discrete Distributions:**
- **Binomial**: $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$
- **Poisson**: $P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}$

**Continuous Distributions:**
- **Normal**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- **Uniform**: $f(x) = \frac{1}{b-a}$ for $a \leq x \leq b$

## Interactive Distribution Explorer

<ProbabilitySimulator />

## Python Implementation

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