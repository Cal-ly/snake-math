# Number Types and Data Types

Understanding different types of numbers and how to represent them in code is fundamental to programming. This section covers the mathematical classification of numbers and how to work with them as data types in programming:

- Natural numbers, integers, and real numbers
- Data type declarations and constraints
- Range validation and bounds checking
- Precision and overflow considerations
- Type conversion and casting

## Mathematical Number Classifications

Understanding how numbers are categorized mathematically helps us choose appropriate data types:

### Natural Numbers (ℕ)
Counting numbers: 1, 2, 3, 4, ...

```python
# Natural numbers are typically represented as positive integers
count = 5  # Natural number
items = [1, 2, 3, 4, 5]  # List of natural numbers
```

### Integers (ℤ)
Whole numbers including negative values: ..., -2, -1, 0, 1, 2, ...

```python
# Integers can be positive, negative, or zero
temperature = -15  # Integer
score_difference = 0  # Integer
population = 1000000  # Integer
```

### Real Numbers (ℝ)
Numbers that can have decimal points, including irrational numbers.

```python
# Real numbers are represented as floats
pi = 3.14159  # Real number
height = 5.75  # Real number
```

## Data Type Declarations

Different programming languages handle number types differently:

```python
# Python (dynamic typing)
x = 42        # int
y = 3.14      # float
z = True      # bool (subclass of int)

# Type hints for clarity
def calculate_area(radius: float) -> float:
    return 3.14159 * radius * radius
```

```javascript
// JavaScript (dynamic typing)
let count = 10;          // number
let price = 19.99;       // number
let isValid = true;      // boolean
```

```java
// Java (static typing)
int count = 10;
double price = 19.99;
boolean isValid = true;
long bigNumber = 1000000000L;
```

## Range Validation and Bounds Checking

When working with different number types, it's important to validate that values fall within expected ranges:

```python
def validate_age(age: int) -> bool:
    """Validate that age is a reasonable natural number"""
    return 0 <= age <= 150

def validate_temperature(temp: float) -> bool:
    """Validate temperature in Celsius"""
    return -273.15 <= temp <= 5778  # Absolute zero to Sun's surface

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Constrain a value to be within specified bounds"""
    return max(min_val, min(value, max_val))
```

## Precision and Overflow Considerations

Different data types have different precision limits and overflow behaviors:

```python
import sys

# Integer limits in Python (arbitrary precision)
big_int = 10**100  # Python handles arbitrarily large integers

# Float precision limits
print(f"Float max: {sys.float_info.max}")
print(f"Float epsilon: {sys.float_info.epsilon}")

# Precision comparison
def safe_float_compare(a: float, b: float, tolerance: float = 1e-9) -> bool:
    """Compare floats with tolerance for precision errors"""
    return abs(a - b) < tolerance
```

```java
// Java overflow behavior
public class NumberLimits {
    public static void main(String[] args) {
        // Integer overflow wraps around
        int maxInt = Integer.MAX_VALUE;  // 2,147,483,647
        int overflow = maxInt + 1;       // -2,147,483,648
        
        // Use BigInteger for arbitrary precision
        BigInteger bigNum = new BigInteger("12345678901234567890");
    }
}
```

## Type Conversion and Casting

Converting between number types requires careful consideration of precision loss:

```python
# Safe conversions
def safe_int_conversion(value: float) -> int:
    """Convert float to int with validation"""
    if value.is_integer():
        return int(value)
    raise ValueError(f"Cannot safely convert {value} to integer")

# Explicit casting with awareness of precision loss
price = 19.99
price_cents = int(price * 100)  # Convert to cents to avoid precision issues

# Type checking before operations
def divide_safely(a: float, b: float) -> float:
    """Divide with type and zero checking"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
```
