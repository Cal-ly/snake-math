<!-- ---
title: "Descriptive Statistics"
description: "Understanding how to summarize and describe data using measures of central tendency, variability, and distribution shape"
tags: ["mathematics", "statistics", "data-analysis", "programming", "data-science"]
difficulty: "beginner"
category: "concept"
symbol: "x̄, s, σ, Q₁, Q₃"
prerequisites: ["basic-arithmetic", "variables-expressions", "data-structures"]
related_concepts: ["probability", "hypothesis-testing", "regression", "data-visualization"]
applications: ["data-analysis", "quality-control", "research", "business-intelligence"]
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

# Descriptive Statistics (x̄, s, σ, Q₁, Q₃)

Think of descriptive statistics as your data's biography - they tell you everything important about your dataset's personality: where it likes to hang out (central tendency), how spread out it is (variability), and what shape it prefers (distribution). It's like getting to know your data before asking it any serious questions!

## Understanding Descriptive Statistics

**Descriptive statistics** summarize and describe data using key numerical measures that capture the essential characteristics of a dataset. Just like you might describe a person by their height, weight, and personality, we describe data using central tendency, variability, and shape.

The fundamental measures include:

$$
\begin{align}
\text{Mean: } \bar{x} &= \frac{1}{n}\sum_{i=1}^{n} x_i \\
\text{Variance: } s^2 &= \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2 \\
\text{Standard deviation: } s &= \sqrt{s^2}
\end{align}
$$

Think of descriptive statistics like a restaurant review - the mean tells you the average rating, the standard deviation tells you how much opinions varied, and the median tells you the "typical" experience most people had:

```python
import numpy as np

# Restaurant ratings from 1-10
ratings = [8, 9, 7, 10, 8, 6, 9, 8, 7, 10, 9, 8, 5, 9, 8]

# The "biography" of our ratings data
mean_rating = np.mean(ratings)          # Average experience
median_rating = np.median(ratings)      # Typical experience  
std_rating = np.std(ratings, ddof=1)    # How much opinions varied
min_rating = np.min(ratings)            # Worst experience
max_rating = np.max(ratings)            # Best experience

print(f"Restaurant Data Biography:")
print(f"Average rating: {mean_rating:.1f}/10")
print(f"Typical rating: {median_rating:.1f}/10") 
print(f"Opinion spread: ±{std_rating:.1f} points")
print(f"Rating range: {min_rating}-{max_rating}")
```

## Why Descriptive Statistics Matter for Programmers

Descriptive statistics are essential for data-driven programming, helping you understand datasets before building models, detecting anomalies in system logs, monitoring application performance, and making informed decisions about data processing strategies.

Understanding these concepts helps you write better data analysis pipelines, implement quality checks, and communicate findings effectively to stakeholders who need actionable insights from your code.


## Interactive Exploration

<StatisticsCalculator />

```plaintext
Component conceptualization:
Create an interactive descriptive statistics explorer where users can:
- Input datasets directly or load sample datasets from different domains
- Calculate and display all major statistics with real-time updates
- Visualize distributions with histograms, box plots, and violin plots
- Compare multiple datasets side-by-side with comparative statistics
- Demonstrate the effect of outliers on different measures interactively
- Show various distribution shapes (normal, skewed, bimodal) and their characteristics
- Interactive quartile and percentile exploration with visual indicators
- Step-by-step calculation breakdowns for educational purposes
- Export statistical summaries and visualizations for further analysis
The component should provide both numerical results and rich visualizations with interpretive guidance.
```

Experiment with different datasets to see how various factors affect statistical measures and learn to interpret data distributions effectively.


## Descriptive Statistics Techniques and Efficiency

Understanding different approaches to calculating and applying descriptive statistics helps optimize performance and choose appropriate methods for different types of data.

### Method 1: Manual Implementation

**Pros**: Educational value, complete control, understanding of calculations\
**Complexity**: O(n) for most measures, O(n log n) for median with sorting

```python
import math
from collections import Counter

def manual_descriptive_statistics(data):
    """Calculate descriptive statistics from scratch for educational purposes"""
    
    if not data:
        raise ValueError("Dataset cannot be empty")
    
    # Ensure data is numeric
    data = [float(x) for x in data]
    n = len(data)
    
    # Central Tendency
    mean = sum(data) / n
    
    # Median (requires sorting)
    sorted_data = sorted(data)
    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]
    
    # Mode
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [value for value, count in counts.items() if count == max_count]
    mode = modes[0] if len(modes) == 1 else modes
    
    # Variability
    data_range = max(data) - min(data)
    
    # Variance (sample variance using n-1)
    mean_deviations_squared = [(x - mean)**2 for x in data]
    variance = sum(mean_deviations_squared) / (n - 1) if n > 1 else 0
    std_deviation = math.sqrt(variance)
    
    # Quartiles
    def calculate_quartile(sorted_data, q):
        """Calculate quartile using the interpolation method"""
        n = len(sorted_data)
        index = q * (n - 1)
        lower_index = int(index)
        upper_index = lower_index + 1
        
        if upper_index >= n:
            return sorted_data[lower_index]
        
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    q1 = calculate_quartile(sorted_data, 0.25)
    q3 = calculate_quartile(sorted_data, 0.75)
    iqr = q3 - q1
    
    # Shape measures
    def calculate_skewness():
        """Calculate sample skewness"""
        if std_deviation == 0:
            return 0
        skew_sum = sum(((x - mean) / std_deviation)**3 for x in data)
        return (n * skew_sum) / ((n - 1) * (n - 2)) if n > 2 else 0
    
    def calculate_kurtosis():
        """Calculate sample excess kurtosis"""
        if std_deviation == 0:
            return 0
        kurt_sum = sum(((x - mean) / std_deviation)**4 for x in data)
        if n > 3:
            return (n * (n + 1) * kurt_sum) / ((n - 1) * (n - 2) * (n - 3)) - (3 * (n - 1)**2) / ((n - 2) * (n - 3))
        return 0
    
    skewness = calculate_skewness()
    kurtosis = calculate_kurtosis()
    
    # Outlier detection using IQR method
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_fence or x > upper_fence]
    
    return {
        'count': n,
        'mean': mean,
        'median': median,
        'mode': mode,
        'range': data_range,
        'min': min(data),
        'max': max(data),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'variance': variance,
        'std_deviation': std_deviation,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'outliers': outliers,
        'outlier_bounds': (lower_fence, upper_fence)
    }

# Example usage
sample_data = [2, 4, 4, 4, 5, 5, 7, 9, 12, 15, 18]
stats = manual_descriptive_statistics(sample_data)

print("Manual Descriptive Statistics:")
print("=" * 35)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")
```

### Method 2: NumPy/SciPy Optimized

**Pros**: Highly optimized, handles large datasets efficiently, extensive functionality\
**Complexity**: Optimized implementations, typically O(n) or O(n log n)

```python
import numpy as np
from scipy import stats
import time

def optimized_descriptive_statistics(data):
    """Calculate descriptive statistics using optimized NumPy/SciPy functions"""
    
    data = np.array(data, dtype=float)
    
    # Basic measures
    basic_stats = {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std_deviation': np.std(data, ddof=1),
        'variance': np.var(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.ptp(data),  # peak-to-peak (max - min)
    }
    
    # Percentiles and quartiles
    percentiles = np.percentile(data, [25, 50, 75])
    basic_stats.update({
        'q1': percentiles[0],
        'q3': percentiles[2],
        'iqr': percentiles[2] - percentiles[0]
    })
    
    # Mode (using scipy.stats)
    mode_result = stats.mode(data, keepdims=True)
    basic_stats['mode'] = mode_result.mode[0] if len(mode_result.mode) > 0 else None
    
    # Shape statistics
    basic_stats.update({
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)  # excess kurtosis
    })
    
    # Outlier detection
    q1, q3 = basic_stats['q1'], basic_stats['q3']
    iqr = basic_stats['iqr']
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    
    basic_stats.update({
        'outliers': outliers.tolist(),
        'outlier_bounds': (lower_fence, upper_fence),
        'outlier_count': len(outliers)
    })
    
    return basic_stats

def performance_comparison():
    """Compare performance between manual and optimized implementations"""
    
    # Generate large dataset for performance testing
    np.random.seed(42)
    large_dataset = np.random.normal(100, 15, 100000)
    
    print("Performance Comparison (100,000 data points):")
    print("=" * 50)
    
    # Manual implementation timing
    start_time = time.time()
    manual_stats = manual_descriptive_statistics(large_dataset)
    manual_time = time.time() - start_time
    
    # Optimized implementation timing
    start_time = time.time()
    optimized_stats = optimized_descriptive_statistics(large_dataset)
    optimized_time = time.time() - start_time
    
    print(f"Manual implementation: {manual_time:.4f} seconds")
    print(f"Optimized implementation: {optimized_time:.4f} seconds")
    print(f"Speedup: {manual_time/optimized_time:.1f}x faster")
    
    # Verify results are similar
    print(f"\nResults comparison:")
    print(f"Mean - Manual: {manual_stats['mean']:.3f}, Optimized: {optimized_stats['mean']:.3f}")
    print(f"Std Dev - Manual: {manual_stats['std_deviation']:.3f}, Optimized: {optimized_stats['std_deviation']:.3f}")
    
    return manual_time, optimized_time

performance_comparison()
```

### Method 3: Streaming/Online Statistics

**Pros**: Memory efficient for large datasets, real-time updates, constant space\
**Complexity**: O(1) per update for most measures

```python
class StreamingStatistics:
    """Calculate descriptive statistics for streaming data with constant memory"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.n = 0
        self.sum = 0
        self.sum_squares = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # For running variance (Welford's algorithm)
        self.mean = 0
        self.m2 = 0  # sum of squares of deviations
        
        # For approximate quantiles (P² algorithm simplified)
        self.sorted_sample = []  # Keep small sample for quantile estimation
        self.sample_size = 1000  # Maximum sample size for quantiles
    
    def update(self, value):
        """Add a new value and update statistics"""
        value = float(value)
        self.n += 1
        
        # Update basic statistics
        self.sum += value
        self.sum_squares += value * value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Update running mean and variance (Welford's algorithm)
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        # Maintain sample for quantile estimation
        if len(self.sorted_sample) < self.sample_size:
            self.sorted_sample.append(value)
            self.sorted_sample.sort()
        else:
            # Replace random element to maintain representative sample
            import random
            replace_idx = random.randint(0, self.sample_size - 1)
            self.sorted_sample[replace_idx] = value
            self.sorted_sample.sort()
    
    def get_statistics(self):
        """Get current statistics"""
        if self.n == 0:
            return {}
        
        variance = self.m2 / (self.n - 1) if self.n > 1 else 0
        std_dev = math.sqrt(variance)
        
        # Approximate quantiles from sample
        if self.sorted_sample:
            sample_array = np.array(self.sorted_sample)
            q1 = np.percentile(sample_array, 25)
            median = np.percentile(sample_array, 50)
            q3 = np.percentile(sample_array, 75)
        else:
            q1 = median = q3 = self.mean
        
        return {
            'count': self.n,
            'mean': self.mean,
            'variance': variance,
            'std_deviation': std_dev,
            'min': self.min_val,
            'max': self.max_val,
            'range': self.max_val - self.min_val,
            'median_approx': median,
            'q1_approx': q1,
            'q3_approx': q3,
            'iqr_approx': q3 - q1
        }

def streaming_statistics_demo():
    """Demonstrate streaming statistics with simulated data stream"""
    
    print("Streaming Statistics Demo:")
    print("=" * 30)
    
    # Simulate streaming data
    np.random.seed(42)
    stream = StreamingStatistics()
    
    # Process data in chunks to simulate streaming
    chunk_sizes = [100, 500, 1000, 5000, 10000]
    
    for chunk_size in chunk_sizes:
        # Generate and process new data
        new_data = np.random.normal(50, 10, chunk_size)
        for value in new_data:
            stream.update(value)
        
        stats = stream.get_statistics()
        print(f"\nAfter {stats['count']:,} values:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std Dev: {stats['std_deviation']:.2f}")
        print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
        print(f"  Median (approx): {stats['median_approx']:.2f}")
    
    return stream

streaming_demo = streaming_statistics_demo()
```


## Why Robust Statistics Work

Robust statistics are less sensitive to outliers and provide more reliable measures for real-world data that often contains extreme values or errors:

```python
def demonstrate_robust_vs_traditional():
    """Show the difference between robust and traditional statistics"""
    
    print("Robust vs Traditional Statistics:")
    print("=" * 40)
    
    # Clean dataset
    clean_data = [12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    # Same dataset with outliers
    outlier_data = clean_data + [100, 150]  # Add extreme outliers
    
    datasets = {
        'Clean Data': clean_data,
        'Data with Outliers': outlier_data
    }
    
    for name, data in datasets.items():
        data_array = np.array(data)
        
        print(f"\n{name}: {data}")
        
        # Traditional statistics
        trad_mean = np.mean(data_array)
        trad_std = np.std(data_array, ddof=1)
        
        # Robust statistics
        robust_median = np.median(data_array)
        robust_mad = np.median(np.abs(data_array - robust_median))  # Median Absolute Deviation
        robust_iqr = np.percentile(data_array, 75) - np.percentile(data_array, 25)
        
        # Trimmed mean (remove top and bottom 10%)
        trim_percent = 0.1
        trimmed_mean = stats.trim_mean(data_array, trim_percent)
        
        print(f"Traditional - Mean: {trad_mean:.1f}, Std Dev: {trad_std:.1f}")
        print(f"Robust - Median: {robust_median:.1f}, MAD: {robust_mad:.1f}, IQR: {robust_iqr:.1f}")
        print(f"Trimmed Mean (10%): {trimmed_mean:.1f}")
        
        # Show impact of outliers
        if 'Outliers' in name:
            clean_mean = np.mean(clean_data)
            clean_median = np.median(clean_data)
            
            mean_change = abs(trad_mean - clean_mean)
            median_change = abs(robust_median - clean_median)
            
            print(f"Impact of outliers:")
            print(f"  Mean changed by: {mean_change:.1f}")
            print(f"  Median changed by: {median_change:.1f}")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plots
    ax1.boxplot([clean_data, outlier_data], labels=['Clean', 'With Outliers'])
    ax1.set_title('Box Plot Comparison')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Histograms
    ax2.hist(clean_data, bins=10, alpha=0.7, label='Clean Data', color='blue')
    ax2.hist(outlier_data, bins=15, alpha=0.7, label='With Outliers', color='red')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return clean_data, outlier_data

demonstrate_robust_vs_traditional()
```

### Basic Statistics Functions

```python
import numpy as np
from collections import Counter
import math

def basic_statistics(data):
    """Calculate basic descriptive statistics manually"""
    data = sorted(data)  # Sort for median calculation
    n = len(data)
    
    # Mean
    mean = sum(data) / n
    
    # Median
    if n % 2 == 0:
        median = (data[n//2 - 1] + data[n//2]) / 2
    else:
        median = data[n//2]
    
    # Mode
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    mode = modes[0] if len(modes) == 1 else modes  # Single mode or multiple modes
    
    # Range
    data_range = max(data) - min(data)
    
    # Variance and standard deviation
    variance = sum((x - mean)**2 for x in data) / (n - 1)  # Sample variance
    std_dev = math.sqrt(variance)
    
    return {
        'mean': mean,
        'median': median, 
        'mode': mode,
        'range': data_range,
        'variance': variance,
        'std_dev': std_dev,
        'n': n
    }

# Example
data = [2, 4, 4, 4, 5, 5, 7, 9]
stats = basic_statistics(data)

print("Manual Calculation Results:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")
```

### Advanced Measures

```python
def advanced_statistics(data):
    """Calculate advanced descriptive measures"""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    
    # Percentiles and quartiles
    percentiles = {
        '5th': np.percentile(data, 5),
        '25th (Q1)': np.percentile(data, 25),
        '50th (median)': np.percentile(data, 50),
        '75th (Q3)': np.percentile(data, 75),
        '95th': np.percentile(data, 95)
    }
    
    # Interquartile range
    iqr = percentiles['75th (Q3)'] - percentiles['25th (Q1)']
    
    # Five-number summary
    five_number = {
        'Minimum': np.min(data),
        'Q1': percentiles['25th (Q1)'],
        'Median': percentiles['50th (median)'],
        'Q3': percentiles['75th (Q3)'],
        'Maximum': np.max(data)
    }
    
    # Outlier detection using IQR method
    lower_fence = percentiles['25th (Q1)'] - 1.5 * iqr
    upper_fence = percentiles['75th (Q3)'] + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    
    # Shape measures
    def skewness(data):
        """Calculate skewness manually"""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    
    def kurtosis(data):
        """Calculate excess kurtosis manually"""  
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n-1)**2 / ((n-2) * (n-3))
    
    skew = skewness(data)
    kurt = kurtosis(data)
    
    # Coefficient of variation
    cv = (std_dev / abs(mean)) * 100 if mean != 0 else float('inf')
    
    print("Advanced Statistics:")
    print("-" * 30)
    
    print("Percentiles:")
    for label, value in percentiles.items():
        print(f"  {label}: {value:.2f}")
    
    print(f"\nIQR: {iqr:.2f}")
    
    print("\nFive-Number Summary:")
    for label, value in five_number.items():
        print(f"  {label}: {value:.2f}")
    
    print(f"\nOutliers: {list(outliers) if len(outliers) > 0 else 'None detected'}")
    print(f"Outlier bounds: [{lower_fence:.2f}, {upper_fence:.2f}]")
    
    print(f"\nShape measures:")
    print(f"  Skewness: {skew:.3f}")
    print(f"  Kurtosis: {kurt:.3f}")
    print(f"  Coefficient of Variation: {cv:.1f}%")
    
    return {
        'percentiles': percentiles,
        'iqr': iqr,
        'five_number': five_number,
        'outliers': outliers,  
        'skewness': skew,
        'kurtosis': kurt,
        'cv': cv
    }

# Example with outliers
data_with_outliers = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7, 15, 20]  # 15, 20 are outliers
advanced_statistics(data_with_outliers)
```

## Data Visualization and Interpretation

### Distribution Shapes

The StatisticsCalculator component includes visualization of different distribution shapes and their characteristics, including normal, skewed, uniform, and bimodal distributions with statistical measures.

### Comparative Analysis

```python
def compare_datasets():
    """Compare statistics between different datasets"""
    
    # Create sample datasets
    np.random.seed(123)
    
    dataset_A = np.random.normal(75, 8, 50)   # Class A test scores
    dataset_B = np.random.normal(72, 12, 50)  # Class B test scores  
    dataset_C = np.random.normal(78, 5, 50)   # Class C test scores
    
    datasets = {
        'Class A': dataset_A,
        'Class B': dataset_B, 
        'Class C': dataset_C
    }
    
    print("Comparative Analysis: Test Scores by Class")
    print("=" * 50)
    
    comparison_stats = {}
    
    for name, data in datasets.items():
        stats = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std_dev': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75)
        }
        comparison_stats[name] = stats
        
        print(f"\n{name}:")
        print(f"  Mean: {stats['mean']:.1f}")
        print(f"  Median: {stats['median']:.1f}") 
        print(f"  Std Dev: {stats['std_dev']:.1f}")
        print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
    
    # Visual comparison
    plt.figure(figsize=(15, 10))
    
    # Box plots
    plt.subplot(2, 2, 1)
    plt.boxplot([datasets[name] for name in datasets.keys()], 
                labels=list(datasets.keys()))
    plt.title('Box Plot Comparison')
    plt.ylabel('Test Score')
    plt.grid(True, alpha=0.3)
    
    # Histograms
    plt.subplot(2, 2, 2)
    colors = ['red', 'blue', 'green']
    for i, (name, data) in enumerate(datasets.items()):
        plt.hist(data, bins=15, alpha=0.6, label=name, color=colors[i])
    plt.title('Distribution Comparison')
    plt.xlabel('Test Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Means comparison
    plt.subplot(2, 2, 3)
    names = list(comparison_stats.keys())
    means = [comparison_stats[name]['mean'] for name in names]
    std_devs = [comparison_stats[name]['std_dev'] for name in names]
    
    plt.bar(names, means, yerr=std_devs, capsize=5, alpha=0.7, color=colors)
    plt.title('Mean ± Standard Deviation')
    plt.ylabel('Test Score')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot of mean vs std dev
    plt.subplot(2, 2, 4)
    for i, name in enumerate(names):
        plt.scatter(means[i], std_devs[i], s=100, color=colors[i], label=name)
        plt.text(means[i], std_devs[i], name, fontsize=9, ha='center', va='bottom')
    
    plt.xlabel('Mean Score')
    plt.ylabel('Standard Deviation')
    plt.title('Mean vs Variability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    print("\nInterpretation:")
    highest_mean = max(names, key=lambda x: comparison_stats[x]['mean'])
    lowest_variability = min(names, key=lambda x: comparison_stats[x]['std_dev'])
    
    print(f"• Highest average performance: {highest_mean}")
    print(f"• Most consistent performance: {lowest_variability}")
    print(f"• Class B has the highest variability (largest spread)")

compare_datasets()
```

## Real-World Applications

### Quality Control Analysis

```python
def quality_control_analysis():
    """Statistical process control using descriptive statistics"""
    
    print("Quality Control Analysis")
    print("=" * 30)
    
    # Simulate manufacturing data (widget weights in grams)
    np.random.seed(456)
    target_weight = 100
    
    # Normal production (in control)
    normal_production = np.random.normal(target_weight, 2, 30)
    
    # Production with shift in mean (out of control)
    shifted_production = np.random.normal(target_weight + 3, 2, 20)
    
    # Production with increased variability
    variable_production = np.random.normal(target_weight, 5, 25)
    
    all_data = np.concatenate([normal_production, shifted_production, variable_production])
    
    print(f"Target weight: {target_weight}g ± 5g")
    print(f"Total samples: {len(all_data)}")
    
    # Calculate control limits (±3 standard deviations)
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data, ddof=1)
    
    upper_control_limit = overall_mean + 3 * overall_std
    lower_control_limit = overall_mean - 3 * overall_std
    
    upper_spec_limit = target_weight + 5
    lower_spec_limit = target_weight - 5
    
    print(f"\nProcess statistics:")
    print(f"  Mean: {overall_mean:.2f}g")
    print(f"  Std Dev: {overall_std:.2f}g")
    print(f"  Control limits: {lower_control_limit:.2f}g to {upper_control_limit:.2f}g")
    print(f"  Specification limits: {lower_spec_limit}g to {upper_spec_limit}g")
    
    # Count out-of-spec items
    out_of_spec = np.sum((all_data < lower_spec_limit) | (all_data > upper_spec_limit))
    out_of_control = np.sum((all_data < lower_control_limit) | (all_data > upper_control_limit))
    
    print(f"\nQuality metrics:")
    print(f"  Out of specification: {out_of_spec}/{len(all_data)} ({out_of_spec/len(all_data)*100:.1f}%)")
    print(f"  Out of control: {out_of_control}/{len(all_data)} ({out_of_control/len(all_data)*100:.1f}%)")
    
    # Capability analysis
    process_capability = min(
        (upper_spec_limit - overall_mean) / (3 * overall_std),
        (overall_mean - lower_spec_limit) / (3 * overall_std)
    )
    
    print(f"  Process capability (Cpk): {process_capability:.3f}")
    if process_capability >= 1.33:
        print("    Process is capable")
    elif process_capability >= 1.0:
        print("    Process is marginally capable")
    else:
        print("    Process is not capable")
    
    # Control chart visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(all_data) + 1), all_data, 'bo-', markersize=4)
    plt.axhline(target_weight, color='green', linewidth=2, label='Target')
    plt.axhline(upper_spec_limit, color='red', linestyle='--', label='Spec Limits')
    plt.axhline(lower_spec_limit, color='red', linestyle='--')
    plt.axhline(upper_control_limit, color='orange', linestyle=':', label='Control Limits')
    plt.axhline(lower_control_limit, color='orange', linestyle=':')
    plt.ylabel('Weight (g)')
    plt.title('Quality Control Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram with specifications
    plt.subplot(2, 1, 2)
    plt.hist(all_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(target_weight, color='green', linewidth=2, label='Target')
    plt.axvline(upper_spec_limit, color='red', linestyle='--', linewidth=2, label='Spec Limits')
    plt.axvline(lower_spec_limit, color='red', linestyle='--', linewidth=2)
    plt.axvline(overall_mean, color='blue', linestyle='-', linewidth=2, label='Process Mean')
    plt.xlabel('Weight (g)')
    plt.ylabel('Frequency')
    plt.title('Process Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return all_data, overall_mean, overall_std

quality_control_analysis()
```

### Sports Performance Analysis

```python
def sports_performance_analysis():
    """Analyze athlete performance using descriptive statistics"""
    
    print("Basketball Performance Analysis")
    print("=" * 35)
    
    # Sample data: Basketball player statistics over 10 games
    players = {
        'Player A': {
            'points': [25, 18, 32, 28, 15, 22, 35, 19, 27, 24],
            'rebounds': [8, 12, 6, 9, 14, 7, 5, 11, 8, 10],
            'assists': [5, 7, 4, 6, 8, 5, 3, 9, 6, 7]
        },
        'Player B': {
            'points': [22, 24, 26, 20, 28, 25, 23, 27, 21, 29],
            'rebounds': [6, 7, 8, 5, 9, 7, 6, 8, 7, 7],
            'assists': [12, 10, 11, 13, 9, 12, 14, 8, 11, 10]
        },
        'Player C': {
            'points': [30, 35, 28, 40, 25, 32, 38, 33, 29, 36],
            'rebounds': [4, 3, 5, 2, 6, 4, 3, 5, 4, 4],
            'assists': [3, 2, 4, 1, 5, 3, 2, 4, 3, 3]
        }
    }
    
    performance_summary = {}
    
    for player_name, stats in players.items():
        points = np.array(stats['points'])
        rebounds = np.array(stats['rebounds'])
        assists = np.array(stats['assists'])
        
        player_stats = {
            'points_mean': np.mean(points),
            'points_std': np.std(points, ddof=1),
            'points_cv': (np.std(points, ddof=1) / np.mean(points)) * 100,
            'rebounds_mean': np.mean(rebounds),
            'assists_mean': np.mean(assists),
            'total_mean': np.mean(points + rebounds + assists)
        }
        
        performance_summary[player_name] = player_stats
        
        print(f"\n{player_name}:")
        print(f"  Points: {player_stats['points_mean']:.1f} ± {player_stats['points_std']:.1f}")
        print(f"  Rebounds: {player_stats['rebounds_mean']:.1f}")
        print(f"  Assists: {player_stats['assists_mean']:.1f}")
        print(f"  Consistency (CV): {player_stats['points_cv']:.1f}%")
        print(f"  Overall impact: {player_stats['total_mean']:.1f}")
    
    # Performance visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Points comparison
    points_data = [players[player]['points'] for player in players.keys()]
    ax1.boxplot(points_data, labels=list(players.keys()))
    ax1.set_title('Points per Game Distribution')
    ax1.set_ylabel('Points')
    ax1.grid(True, alpha=0.3)
    
    # Rebounds comparison  
    rebounds_data = [players[player]['rebounds'] for player in players.keys()]
    ax2.boxplot(rebounds_data, labels=list(players.keys()))
    ax2.set_title('Rebounds per Game Distribution')
    ax2.set_ylabel('Rebounds')
    ax2.grid(True, alpha=0.3)
    
    # Performance trends
    games = range(1, 11)
    colors = ['red', 'blue', 'green']
    for i, (player, stats) in enumerate(players.items()):
        ax3.plot(games, stats['points'], 'o-', color=colors[i], label=player, linewidth=2)
    ax3.set_title('Points Trend Over Games')
    ax3.set_xlabel('Game Number')
    ax3.set_ylabel('Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Consistency analysis
    player_names = list(performance_summary.keys())
    cv_values = [performance_summary[player]['points_cv'] for player in player_names]
    mean_points = [performance_summary[player]['points_mean'] for player in player_names]
    
    ax4.scatter(mean_points, cv_values, s=100, c=colors, alpha=0.7)
    for i, name in enumerate(player_names):
        ax4.annotate(name, (mean_points[i], cv_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Average Points')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.set_title('Scoring Average vs Consistency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance ranking
    print("\nPerformance Summary:")
    best_scorer = max(player_names, key=lambda x: performance_summary[x]['points_mean'])
    most_consistent = min(player_names, key=lambda x: performance_summary[x]['points_cv'])
    best_overall = max(player_names, key=lambda x: performance_summary[x]['total_mean'])
    
    print(f"• Highest scoring average: {best_scorer}")
    print(f"• Most consistent scorer: {most_consistent}")
    print(f"• Best overall impact: {best_overall}")
    
    return players, performance_summary

sports_performance_analysis()
```

### Application 3: Financial Risk Assessment

```python
def financial_risk_assessment():
    """Analyze investment portfolio risk using descriptive statistics"""
    
    print("Investment Portfolio Risk Analysis")
    print("=" * 40)
    
    # Simulate daily returns for different assets (252 trading days)
    np.random.seed(789)
    days = 252
    
    # Different asset types with varying risk profiles
    stock_a_returns = np.random.normal(0.0008, 0.02, days)  # High volatility stock
    stock_b_returns = np.random.normal(0.0005, 0.015, days)  # Medium volatility stock
    bonds_returns = np.random.normal(0.0003, 0.005, days)   # Low volatility bonds
    
    # Portfolio allocation (40% A, 40% B, 20% bonds)
    portfolio_returns = 0.4 * stock_a_returns + 0.4 * stock_b_returns + 0.2 * bonds_returns
    
    assets = {
        'Stock A': stock_a_returns,
        'Stock B': stock_b_returns,
        'Bonds': bonds_returns,
        'Portfolio': portfolio_returns
    }
    
    risk_metrics = {}
    
    print("Annual Risk Metrics (252 trading days):")
    print("=" * 45)
    
    for asset_name, returns in assets.items():
        # Annualized statistics
        annual_return = np.mean(returns) * 252 * 100
        annual_volatility = np.std(returns, ddof=1) * np.sqrt(252) * 100
        
        # Risk metrics
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        # Maximum drawdown simulation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        risk_metrics[asset_name] = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'positive_days': np.sum(returns > 0) / len(returns) * 100
        }
        
        print(f"\n{asset_name}:")
        print(f"  Annual Return: {annual_return:.2f}%")
        print(f"  Annual Volatility: {annual_volatility:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  VaR (95%): {var_95:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Positive Days: {risk_metrics[asset_name]['positive_days']:.1f}%")
    
    # Risk visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns distribution
    for asset_name, returns in assets.items():
        ax1.hist(returns * 100, bins=30, alpha=0.6, label=asset_name)
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Return Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Risk-Return scatter
    asset_names = list(risk_metrics.keys())
    returns_list = [risk_metrics[name]['annual_return'] for name in asset_names]
    volatility_list = [risk_metrics[name]['annual_volatility'] for name in asset_names]
    
    colors = ['red', 'blue', 'green', 'purple']
    for i, name in enumerate(asset_names):
        ax2.scatter(volatility_list[i], returns_list[i], s=100, c=colors[i], label=name)
        ax2.annotate(name, (volatility_list[i], returns_list[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Annual Volatility (%)')
    ax2.set_ylabel('Annual Return (%)')
    ax2.set_title('Risk-Return Profile')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative returns
    for i, (asset_name, returns) in enumerate(assets.items()):
        cumulative = np.cumprod(1 + returns)
        ax3.plot(cumulative, color=colors[i], label=asset_name, linewidth=2)
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Cumulative Return')
    ax3.set_title('Cumulative Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Risk comparison
    metrics = ['annual_volatility', 'var_95', 'max_drawdown']
    metric_labels = ['Volatility (%)', 'VaR 95% (%)', 'Max Drawdown (%)']
    
    x = np.arange(len(asset_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [abs(risk_metrics[name][metric]) for name in asset_names]
        ax4.bar(x + i * width, values, width, label=metric_labels[i])
    
    ax4.set_xlabel('Assets')
    ax4.set_ylabel('Risk Measure')
    ax4.set_title('Risk Metrics Comparison')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(asset_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Investment recommendation
    print("\nInvestment Analysis:")
    best_sharpe = max(asset_names, key=lambda x: risk_metrics[x]['sharpe_ratio'])
    lowest_risk = min(asset_names, key=lambda x: risk_metrics[x]['annual_volatility'])
    
    print(f"• Best risk-adjusted return: {best_sharpe}")
    print(f"• Lowest risk asset: {lowest_risk}")
    print(f"• Portfolio diversification reduces risk vs individual stocks")
    
    return assets, risk_metrics

financial_risk_assessment()
```


## Common Statistics Patterns

Standard statistical measures and patterns that appear frequently in data analysis:

- **Five-Number Summary:**\
  \(\text{Min, Q₁, Median, Q₃, Max}\)

- **Empirical Rule (68-95-99.7):**\
  \(\text{For normal distributions: 68% within 1σ, 95% within 2σ, 99.7% within 3σ}\)

- **Coefficient of Variation:**\
  \(CV = \frac{s}{\bar{x}} \times 100\%\)

- **Z-Score Standardization:**\
  \(z = \frac{x - \bar{x}}{s}\)

Python implementations demonstrating these patterns:

```python
def statistics_patterns_library():
    """Collection of common statistical patterns and calculations"""
    
    def five_number_summary(data):
        """Calculate five-number summary"""
        data = np.array(data)
        return {
            'minimum': np.min(data),
            'q1': np.percentile(data, 25),
            'median': np.median(data),
            'q3': np.percentile(data, 75),
            'maximum': np.max(data)
        }
    
    def empirical_rule_check(data):
        """Check if data follows empirical rule (normal distribution)"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Count data within each range
        within_1_std = np.sum(np.abs(data - mean) <= std) / len(data)
        within_2_std = np.sum(np.abs(data - mean) <= 2*std) / len(data)
        within_3_std = np.sum(np.abs(data - mean) <= 3*std) / len(data)
        
        return {
            'within_1_std': within_1_std,
            'within_2_std': within_2_std,
            'within_3_std': within_3_std,
            'expected_1_std': 0.68,
            'expected_2_std': 0.95,
            'expected_3_std': 0.997
        }
    
    def coefficient_of_variation(data):
        """Calculate coefficient of variation (relative variability)"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if mean == 0:
            return float('inf')
        
        return (std / abs(mean)) * 100
    
    def z_score_standardization(data):
        """Standardize data using z-scores"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        z_scores = (data - mean) / std
        
        return {
            'z_scores': z_scores,
            'mean_z': np.mean(z_scores),  # Should be ~0
            'std_z': np.std(z_scores, ddof=1)  # Should be ~1
        }
    
    def percentile_ranks(data, values):
        """Calculate percentile ranks for given values"""
        data = np.array(data)
        ranks = []
        
        for value in values:
            # Percentage of data less than or equal to value
            rank = (np.sum(data <= value) / len(data)) * 100
            ranks.append(rank)
        
        return ranks
    
    # Demonstrate patterns
    print("Statistics Patterns Library")
    print("=" * 30)
    
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    
    print("Sample Data Statistics:")
    print(f"Mean: {np.mean(normal_data):.2f}")
    print(f"Std Dev: {np.std(normal_data, ddof=1):.2f}")
    
    # Five-number summary
    summary = five_number_summary(normal_data)
    print(f"\nFive-Number Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    # Empirical rule check
    empirical = empirical_rule_check(normal_data)
    print(f"\nEmpirical Rule Check:")
    print(f"  Within 1σ: {empirical['within_1_std']:.1%} (expected: {empirical['expected_1_std']:.1%})")
    print(f"  Within 2σ: {empirical['within_2_std']:.1%} (expected: {empirical['expected_2_std']:.1%})")
    print(f"  Within 3σ: {empirical['within_3_std']:.1%} (expected: {empirical['expected_3_std']:.1%})")
    
    # Coefficient of variation
    cv = coefficient_of_variation(normal_data)
    print(f"\nCoefficient of Variation: {cv:.1f}%")
    
    # Z-score standardization
    z_info = z_score_standardization(normal_data[:10])  # First 10 values
    print(f"\nZ-Score Standardization (first 10 values):")
    print(f"  Original: {normal_data[:10]}")
    print(f"  Z-scores: {z_info['z_scores']}")
    print(f"  Z-score mean: {z_info['mean_z']:.6f}")
    print(f"  Z-score std: {z_info['std_z']:.6f}")
    
    # Percentile ranks
    test_values = [85, 100, 115]
    ranks = percentile_ranks(normal_data, test_values)
    print(f"\nPercentile Ranks:")
    for value, rank in zip(test_values, ranks):
        print(f"  Value {value}: {rank:.1f}th percentile")
    
    return normal_data, summary, empirical

statistics_patterns_library()
```


## Practical Real-world Applications

Descriptive statistics aren't just academic - they're essential tools for solving real-world data problems across multiple domains:

### Application 1: Quality Control and Process Monitoring

```python
def quality_control_analysis():
    """Statistical process control using descriptive statistics"""
    
    print("Quality Control Analysis")
    print("=" * 30)
    
    # Simulate manufacturing data (widget weights in grams)
    np.random.seed(456)
    target_weight = 100
    
    # Normal production (in control)
    normal_production = np.random.normal(target_weight, 2, 30)
    
    # Production with shift in mean (out of control)
    shifted_production = np.random.normal(target_weight + 3, 2, 20)
    
    # Production with increased variability
    variable_production = np.random.normal(target_weight, 5, 25)
    
    all_data = np.concatenate([normal_production, shifted_production, variable_production])
    
    print(f"Target weight: {target_weight}g ± 5g")
    print(f"Total samples: {len(all_data)}")
    
    # Calculate control limits (±3 standard deviations)
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data, ddof=1)
    
    upper_control_limit = overall_mean + 3 * overall_std
    lower_control_limit = overall_mean - 3 * overall_std
    
    upper_spec_limit = target_weight + 5
    lower_spec_limit = target_weight - 5
    
    print(f"\nProcess statistics:")
    print(f"  Mean: {overall_mean:.2f}g")
    print(f"  Std Dev: {overall_std:.2f}g")
    print(f"  Control limits: {lower_control_limit:.2f}g to {upper_control_limit:.2f}g")
    print(f"  Specification limits: {lower_spec_limit}g to {upper_spec_limit}g")
    
    # Count out-of-spec items
    out_of_spec = np.sum((all_data < lower_spec_limit) | (all_data > upper_spec_limit))
    out_of_control = np.sum((all_data < lower_control_limit) | (all_data > upper_control_limit))
    
    print(f"\nQuality metrics:")
    print(f"  Out of specification: {out_of_spec}/{len(all_data)} ({out_of_spec/len(all_data)*100:.1f}%)")
    print(f"  Out of control: {out_of_control}/{len(all_data)} ({out_of_control/len(all_data)*100:.1f}%)")
    
    # Capability analysis
    process_capability = min(
        (upper_spec_limit - overall_mean) / (3 * overall_std),
        (overall_mean - lower_spec_limit) / (3 * overall_std)
    )
    
    print(f"  Process capability (Cpk): {process_capability:.3f}")
    if process_capability >= 1.33:
        print("    Process is capable")
    elif process_capability >= 1.0:
        print("    Process is marginally capable")
    else:
        print("    Process is not capable")
    
    # Control chart visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(all_data) + 1), all_data, 'bo-', markersize=4)
    plt.axhline(target_weight, color='green', linewidth=2, label='Target')
    plt.axhline(upper_spec_limit, color='red', linestyle='--', label='Spec Limits')
    plt.axhline(lower_spec_limit, color='red', linestyle='--')
    plt.axhline(upper_control_limit, color='orange', linestyle=':', label='Control Limits')
    plt.axhline(lower_control_limit, color='orange', linestyle=':')
    plt.ylabel('Weight (g)')
    plt.title('Quality Control Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram with specifications
    plt.subplot(2, 1, 2)
    plt.hist(all_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(target_weight, color='green', linewidth=2, label='Target')
    plt.axvline(upper_spec_limit, color='red', linestyle='--', linewidth=2, label='Spec Limits')
    plt.axvline(lower_spec_limit, color='red', linestyle='--', linewidth=2)
    plt.axvline(overall_mean, color='blue', linestyle='-', linewidth=2, label='Process Mean')
    plt.xlabel('Weight (g)')
    plt.ylabel('Frequency')
    plt.title('Process Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return all_data, overall_mean, overall_std

quality_control_analysis()
```

### Application 2: Sports Performance Analytics

```python
def sports_performance_analysis():
    """Analyze athlete performance using descriptive statistics"""
    
    print("Basketball Performance Analysis")
    print("=" * 35)
    
    # Sample data: Basketball player statistics over 10 games
    players = {
        'Player A': {
            'points': [25, 18, 32, 28, 15, 22, 35, 19, 27, 24],
            'rebounds': [8, 12, 6, 9, 14, 7, 5, 11, 8, 10],
            'assists': [5, 7, 4, 6, 8, 5, 3, 9, 6, 7]
        },
        'Player B': {
            'points': [22, 24, 26, 20, 28, 25, 23, 27, 21, 29],
            'rebounds': [6, 7, 8, 5, 9, 7, 6, 8, 7, 7],
            'assists': [12, 10, 11, 13, 9, 12, 14, 8, 11, 10]
        },
        'Player C': {
            'points': [30, 35, 28, 40, 25, 32, 38, 33, 29, 36],
            'rebounds': [4, 3, 5, 2, 6, 4, 3, 5, 4, 4],
            'assists': [3, 2, 4, 1, 5, 3, 2, 4, 3, 3]
        }
    }
    
    performance_summary = {}
    
    for player_name, stats in players.items():
        points = np.array(stats['points'])
        rebounds = np.array(stats['rebounds'])
        assists = np.array(stats['assists'])
        
        player_stats = {
            'points_mean': np.mean(points),
            'points_std': np.std(points, ddof=1),
            'points_cv': (np.std(points, ddof=1) / np.mean(points)) * 100,
            'rebounds_mean': np.mean(rebounds),
            'assists_mean': np.mean(assists),
            'total_mean': np.mean(points + rebounds + assists)
        }
        
        performance_summary[player_name] = player_stats
        
        print(f"\n{player_name}:")
        print(f"  Points: {player_stats['points_mean']:.1f} ± {player_stats['points_std']:.1f}")
        print(f"  Rebounds: {player_stats['rebounds_mean']:.1f}")
        print(f"  Assists: {player_stats['assists_mean']:.1f}")
        print(f"  Consistency (CV): {player_stats['points_cv']:.1f}%")
        print(f"  Overall impact: {player_stats['total_mean']:.1f}")
    
    # Performance visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Points comparison
    points_data = [players[player]['points'] for player in players.keys()]
    ax1.boxplot(points_data, labels=list(players.keys()))
    ax1.set_title('Points per Game Distribution')
    ax1.set_ylabel('Points')
    ax1.grid(True, alpha=0.3)
    
    # Rebounds comparison  
    rebounds_data = [players[player]['rebounds'] for player in players.keys()]
    ax2.boxplot(rebounds_data, labels=list(players.keys()))
    ax2.set_title('Rebounds per Game Distribution')
    ax2.set_ylabel('Rebounds')
    ax2.grid(True, alpha=0.3)
    
    # Performance trends
    games = range(1, 11)
    colors = ['red', 'blue', 'green']
    for i, (player, stats) in enumerate(players.items()):
        ax3.plot(games, stats['points'], 'o-', color=colors[i], label=player, linewidth=2)
    ax3.set_title('Points Trend Over Games')
    ax3.set_xlabel('Game Number')
    ax3.set_ylabel('Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Consistency analysis
    player_names = list(performance_summary.keys())
    cv_values = [performance_summary[player]['points_cv'] for player in player_names]
    mean_points = [performance_summary[player]['points_mean'] for player in player_names]
    
    ax4.scatter(mean_points, cv_values, s=100, c=colors, alpha=0.7)
    for i, name in enumerate(player_names):
        ax4.annotate(name, (mean_points[i], cv_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Average Points')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.set_title('Scoring Average vs Consistency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance ranking
    print("\nPerformance Summary:")
    best_scorer = max(player_names, key=lambda x: performance_summary[x]['points_mean'])
    most_consistent = min(player_names, key=lambda x: performance_summary[x]['points_cv'])
    best_overall = max(player_names, key=lambda x: performance_summary[x]['total_mean'])
    
    print(f"• Highest scoring average: {best_scorer}")
    print(f"• Most consistent scorer: {most_consistent}")
    print(f"• Best overall impact: {best_overall}")
    
    return players, performance_summary

sports_performance_analysis()
```

### Application 3: Financial Risk Assessment

```python
def financial_risk_assessment():
    """Analyze investment portfolio risk using descriptive statistics"""
    
    print("Investment Portfolio Risk Analysis")
    print("=" * 40)
    
    # Simulate daily returns for different assets (252 trading days)
    np.random.seed(789)
    days = 252
    
    # Different asset types with varying risk profiles
    stock_a_returns = np.random.normal(0.0008, 0.02, days)  # High volatility stock
    stock_b_returns = np.random.normal(0.0005, 0.015, days)  # Medium volatility stock
    bonds_returns = np.random.normal(0.0003, 0.005, days)   # Low volatility bonds
    
    # Portfolio allocation (40% A, 40% B, 20% bonds)
    portfolio_returns = 0.4 * stock_a_returns + 0.4 * stock_b_returns + 0.2 * bonds_returns
    
    assets = {
        'Stock A': stock_a_returns,
        'Stock B': stock_b_returns,
        'Bonds': bonds_returns,
        'Portfolio': portfolio_returns
    }
    
    risk_metrics = {}
    
    print("Annual Risk Metrics (252 trading days):")
    print("=" * 45)
    
    for asset_name, returns in assets.items():
        # Annualized statistics
        annual_return = np.mean(returns) * 252 * 100
        annual_volatility = np.std(returns, ddof=1) * np.sqrt(252) * 100
        
        # Risk metrics
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        # Maximum drawdown simulation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        risk_metrics[asset_name] = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'positive_days': np.sum(returns > 0) / len(returns) * 100
        }
        
        print(f"\n{asset_name}:")
        print(f"  Annual Return: {annual_return:.2f}%")
        print(f"  Annual Volatility: {annual_volatility:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  VaR (95%): {var_95:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Positive Days: {risk_metrics[asset_name]['positive_days']:.1f}%")
    
    # Risk visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns distribution
    for asset_name, returns in assets.items():
        ax1.hist(returns * 100, bins=30, alpha=0.6, label=asset_name)
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Return Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Risk-Return scatter
    asset_names = list(risk_metrics.keys())
    returns_list = [risk_metrics[name]['annual_return'] for name in asset_names]
    volatility_list = [risk_metrics[name]['annual_volatility'] for name in asset_names]
    
    colors = ['red', 'blue', 'green', 'purple']
    for i, name in enumerate(asset_names):
        ax2.scatter(volatility_list[i], returns_list[i], s=100, c=colors[i], label=name)
        ax2.annotate(name, (volatility_list[i], returns_list[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Annual Volatility (%)')
    ax2.set_ylabel('Annual Return (%)')
    ax2.set_title('Risk-Return Profile')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative returns
    for i, (asset_name, returns) in enumerate(assets.items()):
        cumulative = np.cumprod(1 + returns)
        ax3.plot(cumulative, color=colors[i], label=asset_name, linewidth=2)
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Cumulative Return')
    ax3.set_title('Cumulative Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Risk comparison
    metrics = ['annual_volatility', 'var_95', 'max_drawdown']
    metric_labels = ['Volatility (%)', 'VaR 95% (%)', 'Max Drawdown (%)']
    
    x = np.arange(len(asset_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [abs(risk_metrics[name][metric]) for name in asset_names]
        ax4.bar(x + i * width, values, width, label=metric_labels[i])
    
    ax4.set_xlabel('Assets')
    ax4.set_ylabel('Risk Measure')
    ax4.set_title('Risk Metrics Comparison')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(asset_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Investment recommendation
    print("\nInvestment Analysis:")
    best_sharpe = max(asset_names, key=lambda x: risk_metrics[x]['sharpe_ratio'])
    lowest_risk = min(asset_names, key=lambda x: risk_metrics[x]['annual_volatility'])
    
    print(f"• Best risk-adjusted return: {best_sharpe}")
    print(f"• Lowest risk asset: {lowest_risk}")
    print(f"• Portfolio diversification reduces risk vs individual stocks")
    
    return assets, risk_metrics

financial_risk_assessment()
```


## Try it Yourself

Ready to master descriptive statistics? Here are some hands-on challenges:

- **Data Explorer:** Build a comprehensive tool that loads datasets and automatically generates statistical summaries and visualizations.
- **Outlier Detective:** Create an outlier detection system using multiple statistical methods (IQR, z-score, modified z-score).
- **Distribution Analyzer:** Implement a tool that identifies data distribution types and suggests appropriate statistical measures.
- **Comparative Dashboard:** Build a dashboard that compares statistics across multiple groups or time periods.
- **Real-time Monitor:** Create a streaming statistics monitor for live data feeds (server logs, sensor data, etc.).
- **Statistical Storyteller:** Develop a system that automatically generates narrative descriptions of statistical findings.


## Key Takeaways

- Descriptive statistics provide essential summaries of data characteristics through measures of central tendency, variability, and shape.
- Different measures serve different purposes: means for typical values, medians for robust centers, standard deviations for spread.
- Robust statistics (median, IQR, MAD) are less sensitive to outliers than traditional measures (mean, standard deviation).
- Visualization enhances statistical understanding and reveals patterns not apparent in numerical summaries alone.
- Performance optimization matters for large datasets - streaming algorithms enable real-time analysis with constant memory.
- Context determines which statistics are most appropriate for analysis and decision-making.
- Statistical literacy helps avoid common misinterpretations and guides better data-driven decisions.


## Next Steps & Further Exploration

Ready to build on your understanding of descriptive statistics?

- Explore **Probability Distributions** to understand the theoretical foundations underlying statistical measures.
- Study **Inferential Statistics** to learn how sample statistics relate to population parameters.
- Learn **Correlation and Regression** to analyze relationships between variables.
- Investigate **Hypothesis Testing** to make statistical decisions and validate assumptions.
- Apply statistics to **Experimental Design** and **A/B Testing** for controlled studies.
- Explore **Time Series Analysis** for analyzing data that changes over time.