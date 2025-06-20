# Functions and Plotting

## Mathematical Concept

A **function** maps input values to output values using a specific rule. In mathematical notation, we write $f(x) = y$ where $x$ is the input and $y$ is the output.

Common function types:
- Linear: $f(x) = mx + b$
- Quadratic: $f(x) = ax^2 + bx + c$
- Exponential: $f(x) = a \cdot b^x$

## Interactive Function Plotter

<FunctionPlotter />

## Python Implementation

### Defining Functions

```python
def f(x):
    """Simple function definition"""
    return 2 * x + 1

# Call the function
result = f(5)
print(f"f(5) = {result}")  # Output: f(5) = 11
```

### Functions with Multiple Parameters

```python
def linear_function(x, slope=1, intercept=0):
    """Linear function: f(x) = mx + b"""
    return slope * x + intercept

# Different ways to call
print(linear_function(3))           # f(3) = 3 (default slope=1, intercept=0)
print(linear_function(3, 2))        # f(3) = 6 (slope=2, intercept=0)
print(linear_function(3, 2, 5))     # f(3) = 11 (slope=2, intercept=5)
```

### Function Composition

```python
def f(x):
    return 2 * x + 1

def g(x):
    return x**2

# Composition: (f ∘ g)(x) = f(g(x))
def f_of_g(x):
    return f(g(x))

# Composition: (g ∘ f)(x) = g(f(x))
def g_of_f(x):
    return g(f(x))

x = 3
print(f"f(g({x})) = {f_of_g(x)}")  # f(g(3)) = f(9) = 19
print(f"g(f({x})) = {g_of_f(x)}")  # g(f(3)) = g(7) = 49
```

## Domain and Range

### Finding Domain Restrictions

```python
import math

def safe_sqrt(x):
    """Square root function with domain checking"""
    if x < 0:
        return None  # Undefined for negative numbers
    return math.sqrt(x)

def safe_divide(x, a=1):
    """Division function: f(x) = a/x"""
    if x == 0:
        return float('inf')  # Undefined at x=0
    return a / x

# Test domain restrictions
print(safe_sqrt(4))    # 2.0
print(safe_sqrt(-1))   # None (outside domain)
print(safe_divide(2))  # 0.5
print(safe_divide(0))  # inf (asymptote)
```

### Analyzing Range

```python
def analyze_quadratic(a, b, c):
    """Analyze quadratic function properties"""
    def quadratic(x):
        return a * x**2 + b * x + c
    
    # Vertex (gives min/max value)
    vertex_x = -b / (2 * a)
    vertex_y = quadratic(vertex_x)
    
    if a > 0:
        print(f"Parabola opens upward")
        print(f"Minimum value: {vertex_y} at x = {vertex_x}")
        print(f"Range: [{vertex_y}, ∞)")
    else:
        print(f"Parabola opens downward")
        print(f"Maximum value: {vertex_y} at x = {vertex_x}")
        print(f"Range: (-∞, {vertex_y}]")
    
    return quadratic

# Example: f(x) = x² - 4x + 3
f = analyze_quadratic(1, -4, 3)
```

## Inverse Functions

### Finding Inverse Functions

```python
def find_linear_inverse(m, b):
    """Find inverse of f(x) = mx + b"""
    if m == 0:
        return None  # No inverse if slope is 0
    
    def f(x):
        return m * x + b
    
    def f_inverse(y):
        # Solve y = mx + b for x
        # x = (y - b) / m
        return (y - b) / m
    
    return f, f_inverse

# Example
f, f_inv = find_linear_inverse(2, 3)
x = 5
y = f(x)        # f(5) = 2(5) + 3 = 13
x_back = f_inv(y)  # f⁻¹(13) = (13-3)/2 = 5

print(f"f({x}) = {y}")
print(f"f⁻¹({y}) = {x_back}")
print(f"Verification: x = {x} matches f⁻¹(f(x)) = {x_back}")
```

## Real-World Function Applications

### Physics: Motion Functions

```python
def position_function(t, initial_pos=0, initial_vel=0, acceleration=0):
    """Position as function of time: s(t) = s₀ + v₀t + ½at²"""
    return initial_pos + initial_vel * t + 0.5 * acceleration * t**2

def velocity_function(t, initial_vel=0, acceleration=0):
    """Velocity as function of time: v(t) = v₀ + at"""
    return initial_vel + acceleration * t

# Example: Free fall (acceleration = -9.8 m/s²)
def free_fall_height(t, initial_height=100):
    return position_function(t, initial_height, 0, -9.8)

# When does object hit ground?
import numpy as np
times = np.linspace(0, 5, 100)
heights = [free_fall_height(t) for t in times]

for i, h in enumerate(heights):
    if h <= 0:
        print(f"Object hits ground at t ≈ {times[i]:.2f} seconds")
        break
```

### Economics: Cost Functions

```python
def linear_cost(quantity, fixed_cost=1000, variable_cost=5):
    """Linear cost function: C(q) = FC + VC·q"""
    return fixed_cost + variable_cost * quantity

def average_cost(quantity, fixed_cost=1000, variable_cost=5):
    """Average cost per unit: AC(q) = C(q)/q"""
    if quantity == 0:
        return float('inf')
    return linear_cost(quantity, fixed_cost, variable_cost) / quantity

# Break-even analysis
def break_even_point(fixed_cost, variable_cost, price_per_unit):
    """Find break-even quantity"""
    if price_per_unit <= variable_cost:
        return None  # No profit possible
    return fixed_cost / (price_per_unit - variable_cost)

# Example
be_point = break_even_point(1000, 5, 15)
print(f"Break-even point: {be_point} units")
```

## Key Takeaways

1. **Functions** are mathematical mappings from inputs to outputs
2. **Python functions** directly implement mathematical functions
3. **Domain** and **range** define where functions are valid
4. **Composition** combines functions to create new relationships
5. **Inverse functions** "undo" the original function
6. **Visualization** helps understand function behavior

## Next Steps

- Explore **linear equations** and their solutions
- Study **quadratic functions** and parabolas  
- Learn about **exponential growth** and decay
- Apply functions to **data modeling**