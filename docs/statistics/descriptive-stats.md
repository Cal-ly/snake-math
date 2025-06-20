# Descriptive Statistics

## Mathematical Concept  

**Descriptive statistics** summarize and describe data using measures of central tendency, variability, and distribution shape. Key measures include:

**Central Tendency:**
- Mean: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- Median: middle value when data is ordered
- Mode: most frequently occurring value

**Variability:**
- Range: max - min
- Variance: $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
- Standard deviation: $s = \sqrt{s^2}$

## Interactive Statistics Calculator

<StatisticsCalculator />

## Python Implementation

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
    
    print("Quality Control Analysis")
    print("Target weight: 100g ± 5g")
    print("-" * 30)
    
    # Calculate control limits (±3 standard deviations)
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data, ddof=1)
    
    upper_control_limit = overall_mean + 3 * overall_std
    lower_control_limit = overall_mean - 3 * overall_std
    
    upper_spec_limit = target_weight + 5
    lower_spec_limit = target_weight - 5
    
    print(f"Process statistics:")
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
    
    # Control chart
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

quality_control_analysis()
```

### Sports Performance Analysis

```python
def sports_performance_analysis():
    """Analyze athlete performance using descriptive statistics"""
    
    # Sample data: Basketball player statistics
    players = {
        'Player A': {'points': [25, 18, 32, 28, 15, 22, 35, 19, 27, 24],
                    'rebounds': [8, 12, 6, 9, 14, 7, 5, 11, 8, 10]},
        'Player B': {'points': [22, 24, 26, 20, 28, 25, 23, 27, 21, 29],
                    'rebounds': [6, 7, 8, 5, 9, 7, 6, 8, 7, 7]},
        'Player C': {'points': [30, 35, 28, 40, 25, 32, 38, 33, 29, 36],
                    'rebounds': [4, 3, 5, 2, 6, 4, 3, 5, 4, 4]}
    }
    
    print("Basketball Performance Analysis")
    print("=" * 40)
    
    for player_name, stats in players.items():
        points = np.array(stats['points'])
        rebounds = np.array(stats['rebounds'])
        
        print(f"\n{player_name}:")
        print(f"Points per game:")
        print(f"  Mean: {np.mean(points):.1f}")
        print(f"  Median: {np.median(points):.1f}")
        print(f"  Std Dev: {np.std(points, ddof=1):.1f}")
        print(f"  Range: {np.min(points)} - {np.max(points)}")
        print(f"  Consistency (CV): {(np.std(points, ddof=1)/np.mean(points))*100:.1f}%")
        
        print(f"Rebounds per game:")
        print(f"  Mean: {np.mean(rebounds):.1f}")
        print(f"  Std Dev: {np.std(rebounds, ddof=1):.1f}")
    
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
    
    # Scatter plot: Points vs Rebounds correlation
    for i, (player, stats) in enumerate(players.items()):
        ax4.scatter(stats['rebounds'], stats['points'], 
                   color=colors[i], label=player, s=60, alpha=0.7)
    ax4.set_xlabel('Rebounds')
    ax4.set_ylabel('Points')
    ax4.set_title('Points vs Rebounds Relationship')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance ranking
    print("\nPerformance Summary:")
    print("• Player C: Highest scoring average but most variable")
    print("• Player B: Most consistent scorer (lowest CV)")
    print("• Player A: Best rebounder with high variability")

sports_performance_analysis()
```

## Key Takeaways

1. **Central tendency** measures locate the "center" of data
2. **Variability** measures describe data spread and consistency  
3. **Distribution shape** affects interpretation of statistics
4. **Outliers** can significantly impact statistical measures
5. **Comparative analysis** reveals patterns between groups
6. **Visualization** enhances understanding of statistical summaries

## Next Steps

- Study **probability distributions** and their parameters
- Learn **correlation and regression** for relationships
- Explore **hypothesis testing** with sample statistics
- Apply statistics to **experimental design** and **A/B testing**