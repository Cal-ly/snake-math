# Variables and Expressions

## Mathematical Concept

In mathematics, we use **variables** to represent unknown or changing values, and **expressions** to show relationships between these variables.

For example: $y = 2x + 1$

## Interactive Example

<VariablesDemo />

## Python Implementation

### Basic Variable Assignment

```python
# Assign values to variables
x = 5
y = 2 * x + 1

print(f"When x = {x}, y = {y}")
# Output: When x = 5, y = 11
```

### Working with Different Values

```python
# Try different values of x
values = [1, 2, 3, 4, 5]

for x in values:
    y = 2 * x + 1
    print(f"x = {x}, y = {y}")

# Output:
# x = 1, y = 3
# x = 2, y = 5  
# x = 3, y = 7
# x = 4, y = 9
# x = 5, y = 11
```

### Creating Functions

```python
def linear_function(x):
    """Calculate y = 2x + 1 for any value of x"""
    return 2 * x + 1

# Use the function
x = 10
y = linear_function(x)
print(f"f({x}) = {y}")
# Output: f(10) = 21
```

## Types of Mathematical Expressions

### Linear Expressions
```python
def linear(x, m=2, b=1):
    """Linear function: y = mx + b"""
    return m * x + b

# Examples
print(linear(5))        # y = 2(5) + 1 = 11
print(linear(3, 4, 2))  # y = 4(3) + 2 = 14
```

### Quadratic Expressions
```python
def quadratic(x, a=1, b=0, c=0):
    """Quadratic function: y = ax² + bx + c"""
    return a * x**2 + b * x + c

# Examples
print(quadratic(3))           # y = 1(3)² + 0(3) + 0 = 9
print(quadratic(2, 2, -3, 1)) # y = 2(2)² - 3(2) + 1 = 3
```

### Exponential Expressions
```python
import math

def exponential(x, base=2):
    """Exponential function: y = base^x"""
    return base ** x

def natural_exponential(x):
    """Natural exponential: y = e^x"""
    return math.exp(x)

# Examples  
print(exponential(3))         # y = 2³ = 8
print(natural_exponential(1)) # y = e¹ ≈ 2.718
```

## Real-World Applications

### Temperature Conversion
```python
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit: F = (9/5)C + 32"""
    return (9/5) * celsius + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius: C = (5/9)(F - 32)"""
    return (5/9) * (fahrenheit - 32)

# Examples
print(f"0°C = {celsius_to_fahrenheit(0)}°F")   # 32°F
print(f"100°F = {fahrenheit_to_celsius(100)}°C") # 37.78°C
```

### Compound Interest
```python
def compound_interest(principal, rate, time):
    """Calculate compound interest: A = P(1 + r)^t"""
    return principal * (1 + rate) ** time

# Example: $1000 at 5% for 10 years
amount = compound_interest(1000, 0.05, 10)
print(f"$1000 at 5% for 10 years = ${amount:.2f}")
# Output: $1628.89
```

### Distance and Speed
```python
def distance(speed, time):
    """Calculate distance: d = st"""
    return speed * time

def speed(distance, time):
    """Calculate speed: s = d/t"""
    return distance / time

# Examples
d = distance(60, 2.5)  # 60 mph for 2.5 hours
print(f"Distance traveled: {d} miles")

s = speed(150, 3)      # 150 miles in 3 hours
print(f"Average speed: {s} mph")
```

## Key Takeaways

1. **Variables** represent values that can change
2. **Expressions** show mathematical relationships
3. **Functions** make expressions reusable and flexible
4. **Python syntax** closely mirrors mathematical notation
5. **Real applications** are everywhere in science and engineering

## Next Steps

- Learn about **functions and graphs** 
- Explore **algebraic manipulation**
- Study **summation notation** for sequences
- Apply expressions to **data analysis**