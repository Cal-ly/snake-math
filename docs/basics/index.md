# Mathematical Foundations

## PyScript Integration Demo

This page demonstrates interactive Python mathematics running directly in your browser using PyScript 2025.5.1.

<PyScriptDemo />

## How It Works

The interactive demo above shows:

1. **Basic PyScript functionality** - Python running in the browser
2. **Mathematical computations** - Using Python's `math` module  
3. **Interactive input handling** - Real-time calculations
4. **JavaScript integration** - Seamless DOM manipulation

## Mathematical Concepts

### Variables and Expressions

In mathematics, we write: *y = 2x + 1*

In Python, this becomes a simple function that we can evaluate for any value of x:

```python
def linear_function(x):
    return 2 * x + 1

# Example usage
x = 5
y = linear_function(x)
print(f"When x = {x}, y = {y}")
```

### Real-time Calculations

The slider above demonstrates how mathematical functions can be visualized interactively. As you change the input value, the output updates immediately, showing the relationship between variables.

## Next Steps

With PyScript working, we can now build more complex examples:

- **Quadratic function plotter** with matplotlib
- **Trigonometric visualizations** 
- **Calculus demonstrations** (limits, derivatives)
- **Linear algebra operations** (matrices, transformations)
- **Statistical analysis** with data visualization

## Technical Notes

This implementation:
- ✅ Works with VitePress build system
- ✅ Integrates PyScript 2025.5.1 with Vue 3
- ✅ Provides real-time interactive math
- ✅ Maintains clean separation between Python and JavaScript
- ✅ Supports complex mathematical computations