# Summation Notation (Σ)

## Mathematical Concept

**Summation notation** uses the Greek letter Σ (sigma) to represent the sum of a sequence of terms. The general form is:

$$\sum_{i=1}^{n} i = 1 + 2 + 3 + \ldots + n$$

This means: **add up all values of i from 1 to n**

## Interactive Exploration

<SummationDemo />

## Python Implementation

### Method 1: Using `sum()` and `range()`

```python
def summation_builtin(n):
    """Calculate sum using Python's built-in functions"""
    return sum(range(1, n + 1))

# Example usage
n = 10
result = summation_builtin(n)
print(f"Sum of 1 to {n} = {result}")  # Output: 55
```

### Method 2: Using a For Loop

```python
def summation_loop(n):
    """Calculate sum using a manual loop"""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

# Example usage  
n = 10
result = summation_loop(n)
print(f"Sum of 1 to {n} = {result}")  # Output: 55
```

### Method 3: Using the Mathematical Formula

```python
def summation_formula(n):
    """Calculate sum using the closed-form formula"""
    return n * (n + 1) // 2

# Example usage
n = 10
result = summation_formula(n)
print(f"Sum of 1 to {n} = {result}")  # Output: 55
```

## Why the Formula Works

The formula **n(n+1)/2** comes from a clever observation:

```python
# For n = 5:
# Forward:  1 + 2 + 3 + 4 + 5 = 15
# Backward: 5 + 4 + 3 + 2 + 1 = 15
# Combined: 6 + 6 + 6 + 6 + 6 = 30
# Each pair sums to (n+1), and we have n pairs
# So total = n × (n+1), but we counted twice
# Therefore: sum = n × (n+1) / 2

def explain_formula(n):
    pairs_sum = n + 1
    num_pairs = n
    double_counted = num_pairs * pairs_sum
    actual_sum = double_counted // 2
    
    print(f"For n = {n}:")
    print(f"Each pair sums to: {pairs_sum}")
    print(f"Number of pairs: {num_pairs}")
    print(f"Double-counted total: {double_counted}")
    print(f"Actual sum: {actual_sum}")

explain_formula(5)
```

## Common Summation Patterns

```python
# Sum of first n positive integers
def sum_integers(n):
    return n * (n + 1) // 2

# Sum of first n squares: 1² + 2² + 3² + ... + n²
def sum_squares(n):
    return n * (n + 1) * (2*n + 1) // 6

# Sum of first n cubes: 1³ + 2³ + 3³ + ... + n³
def sum_cubes(n):
    return (n * (n + 1) // 2) ** 2

# Examples
n = 5
print(f"Sum of integers 1 to {n}: {sum_integers(n)}")    # 15
print(f"Sum of squares 1 to {n}: {sum_squares(n)}")     # 55  
print(f"Sum of cubes 1 to {n}: {sum_cubes(n)}")         # 225
```

## Real-World Applications

### Calculating Averages
```python
def calculate_average(data):
    """Calculate the mean of a dataset"""
    total = sum(data)  # This is summation!
    count = len(data)
    return total / count

scores = [85, 92, 78, 96, 88]
average = calculate_average(scores)
print(f"Average score: {average}")
```

### Total Distance Traveled
```python
def total_distance(speeds, time_intervals):
    """Calculate total distance from speed × time"""
    return sum(speed * time for speed, time in zip(speeds, time_intervals))

speeds = [60, 45, 70]  # mph
times = [2, 1, 3]      # hours
distance = total_distance(speeds, times)
print(f"Total distance: {distance} miles")
```

## Key Takeaways

1. **Summation notation** provides a compact way to represent adding sequences
2. **Multiple approaches** exist: built-in functions, loops, and formulas
3. **Mathematical formulas** can be much more efficient than loops
4. **Pattern recognition** helps find closed-form solutions
5. **Real applications** appear everywhere in data analysis and mathematics

## Next Steps

- Explore **product notation (Π)** for multiplication sequences
- Learn about **infinite series** and convergence
- Apply summation to **calculus** (Riemann sums)
- Use summation in **statistics** for variance and standard deviation