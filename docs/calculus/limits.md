# Limits and Continuity

## Mathematical Concept

A **limit** describes the behavior of a function as its input approaches a particular value. We write:

$$\lim_{x \to a} f(x) = L$$

This means "as $x$ gets arbitrarily close to $a$, $f(x)$ gets arbitrarily close to $L$."

**Continuity** occurs when $\lim_{x \to a} f(x) = f(a)$, meaning there are no jumps, holes, or asymptotes at that point.

## Interactive Limit Explorer

<LimitsExplorer />

## Python Implementation

### Numerical Limit Calculation

```python
import numpy as np

def numerical_limits():
    """Calculate limits numerically by approaching the target value"""
    
    print("Numerical Limit Calculation")
    print("=" * 35)
    
    def calculate_limit(func, target, direction='both', tolerance=1e-10):
        """Calculate limit numerically"""
        
        print(f"\nCalculating limit as x approaches {target}")
        print(f"{'h':>12} {'x':>12} {'f(x)':>15}")
        print("-" * 42)
        
        # Approach from both sides with decreasing step sizes
        for i in range(1, 8):
            h = 10**(-i)
            
            if direction in ['both', 'right']:
                x_right = target + h
                try:
                    fx_right = func(x_right)
                    print(f"{h:12.0e} {x_right:12.6f} {fx_right:15.8f}")
                except:
                    print(f"{h:12.0e} {x_right:12.6f} {'undefined':>15}")
            
            if direction in ['both', 'left']:
                x_left = target - h
                try:
                    fx_left = func(x_left)
                    print(f"{-h:12.0e} {x_left:12.6f} {fx_left:15.8f}")
                except:
                    print(f"{-h:12.0e} {x_left:12.6f} {'undefined':>15}")
    
    # Example 1: Simple polynomial (continuous)
    print("Example 1: lim(x→2) (x² + 1)")
    def f1(x):
        return x**2 + 1
    
    calculate_limit(f1, 2)
    print(f"Exact value: f(2) = {f1(2)}")
    
    # Example 2: Indeterminate form 0/0
    print("\n" + "="*50)
    print("Example 2: lim(x→2) (x² - 4)/(x - 2)")
    def f2(x):
        if abs(x - 2) < 1e-15:
            return float('nan')  # Avoid division by zero
        return (x**2 - 4) / (x - 2)
    
    calculate_limit(f2, 2)
    print("Analytical solution: (x² - 4)/(x - 2) = (x + 2)(x - 2)/(x - 2) = x + 2")
    print("So the limit is 2 + 2 = 4")
    
    # Example 3: One-sided limits
    print("\n" + "="*50)
    print("Example 3: lim(x→0) 1/x (one-sided limits)")
    def f3(x):
        return 1/x
    
    print("From the right:")
    calculate_limit(f3, 0, 'right')
    print("\nFrom the left:")
    calculate_limit(f3, 0, 'left')
    print("Left and right limits are different → limit does not exist")

numerical_limits()
```

### L'Hôpital's Rule

```python
def lhopital_rule_examples():
    """Demonstrate L'Hôpital's rule for indeterminate forms"""
    
    print("L'Hôpital's Rule Examples")
    print("=" * 30)
    print("For indeterminate forms 0/0 or ∞/∞:")
    print("lim[x→a] f(x)/g(x) = lim[x→a] f'(x)/g'(x)")
    
    # Example 1: sin(x)/x as x → 0
    print(f"\nExample 1: lim(x→0) sin(x)/x")
    
    def f1_original(x):
        return np.sin(x) / x if x != 0 else np.nan
    
    def f1_derivative(x):
        return np.cos(x) / 1  # d/dx[sin(x)] = cos(x), d/dx[x] = 1
    
    print("Original form at x=0: 0/0 (indeterminate)")
    print("Applying L'Hôpital's rule:")
    print("lim(x→0) sin(x)/x = lim(x→0) cos(x)/1 = cos(0)/1 = 1")
    
    # Numerical verification
    x_values = [0.1, 0.01, 0.001, 0.0001]
    print(f"\nNumerical verification:")
    print(f"{'x':>8} {'sin(x)/x':>12} {'cos(x)/1':>12}")
    for x in x_values:
        original = f1_original(x)
        derivative = f1_derivative(x)
        print(f"{x:8.4f} {original:12.8f} {derivative:12.8f}")
    
    # Example 2: (e^x - 1)/x as x → 0
    print(f"\nExample 2: lim(x→0) (e^x - 1)/x")
    
    def f2_original(x):
        return (np.exp(x) - 1) / x if x != 0 else np.nan
    
    def f2_derivative(x):
        return np.exp(x) / 1  # d/dx[e^x - 1] = e^x, d/dx[x] = 1
    
    print("Original form at x=0: 0/0 (indeterminate)")
    print("Applying L'Hôpital's rule:")
    print("lim(x→0) (e^x - 1)/x = lim(x→0) e^x/1 = e^0/1 = 1")
    
    print(f"\nNumerical verification:")
    print(f"{'x':>8} {'(e^x-1)/x':>12} {'e^x/1':>12}")
    for x in x_values:
        original = f2_original(x)
        derivative = f2_derivative(x)
        print(f"{x:8.4f} {original:12.8f} {derivative:12.8f}")
    
    # Example 3: x²/e^x as x → ∞ (∞/∞ form)
    print(f"\nExample 3: lim(x→∞) x²/e^x")
    
    def f3_original(x):
        return x**2 / np.exp(x)
    
    print("Original form as x→∞: ∞/∞ (indeterminate)")
    print("First application: lim(x→∞) x²/e^x = lim(x→∞) 2x/e^x")
    print("Still ∞/∞, apply again: lim(x→∞) 2x/e^x = lim(x→∞) 2/e^x = 0")
    
    x_large = [10, 20, 30, 40]
    print(f"\nNumerical verification:")
    print(f"{'x':>4} {'x²/e^x':>15}")
    for x in x_large:
        value = f3_original(x)
        print(f"{x:4.0f} {value:15.2e}")

lhopital_rule_examples()
```

## Continuity Analysis

```python
def continuity_analysis():
    """Analyze different types of continuity and discontinuity"""
    
    print("Continuity Analysis")
    print("=" * 25)
    
    def test_continuity(func, point, func_name, epsilon=1e-6):
        """Test if a function is continuous at a given point"""
        
        print(f"\nTesting continuity of {func_name} at x = {point}")
        print("-" * 50)
        
        try:
            # Function value at the point
            f_at_point = func(point)
            print(f"f({point}) = {f_at_point}")
        except:
            print(f"f({point}) is undefined")
            f_at_point = None
        
        # Left limit
        try:
            x_left = point - epsilon
            left_limit = func(x_left)
            print(f"Left limit ≈ {left_limit:.6f}")
        except:
            left_limit = None
            print("Left limit does not exist")
        
        # Right limit  
        try:
            x_right = point + epsilon
            right_limit = func(x_right)
            print(f"Right limit ≈ {right_limit:.6f}")
        except:
            right_limit = None
            print("Right limit does not exist")
        
        # Check continuity conditions
        if left_limit is not None and right_limit is not None:
            if abs(left_limit - right_limit) < 1e-10:
                limit_exists = True
                limit_value = left_limit
                print(f"Limit exists: {limit_value:.6f}")
            else:
                limit_exists = False
                print("Limit does not exist (left ≠ right)")
        else:
            limit_exists = False
            print("Limit does not exist")
        
        # Determine continuity
        if limit_exists and f_at_point is not None:
            if abs(limit_value - f_at_point) < 1e-10:
                print("✓ CONTINUOUS: lim f(x) = f(a)")
            else:
                print("✗ DISCONTINUOUS: lim f(x) ≠ f(a) (removable)")
        elif limit_exists and f_at_point is None:
            print("✗ DISCONTINUOUS: f(a) undefined (removable)")
        else:
            print("✗ DISCONTINUOUS: limit does not exist")
    
    # Test cases
    
    # 1. Continuous function
    def f1(x):
        return x**2 + 2*x + 1
    
    test_continuity(f1, 1, "f(x) = x² + 2x + 1")
    
    # 2. Removable discontinuity
    def f2(x):
        if abs(x - 2) < 1e-15:
            return 10  # Different value at x = 2
        return (x**2 - 4) / (x - 2)
    
    test_continuity(f2, 2, "f(x) = (x² - 4)/(x - 2) with f(2) = 10")
    
    # 3. Jump discontinuity
    def f3(x):
        return 1 if x >= 0 else -1
    
    test_continuity(f3, 0, "f(x) = 1 if x ≥ 0, -1 if x < 0")
    
    # 4. Infinite discontinuity
    def f4(x):
        return 1 / (x - 1)
    
    test_continuity(f4, 1, "f(x) = 1/(x - 1)")
    
    # 5. Oscillatory behavior
    def f5(x):
        if x == 0:
            return 0
        return x * np.sin(1/x)
    
    test_continuity(f5, 0, "f(x) = x·sin(1/x) with f(0) = 0")

continuity_analysis()
```

## Applications

### Optimization and Root Finding

```python
def optimization_applications():
    """Apply limits to optimization and root-finding problems"""
    
    print("Applications: Optimization and Root Finding")
    print("=" * 45)
    
    # Newton's Method for root finding
    print("1. Newton's Method")
    print("Uses limits to find roots: x_{n+1} = x_n - f(x_n)/f'(x_n)")
    
    def newton_method(f, df, x0, tolerance=1e-10, max_iterations=20):
        """Newton's method for finding roots"""
        x = x0
        
        print(f"{'Iteration':>9} {'x':>15} {'f(x)':>15} {'Error':>15}")
        print("-" * 60)
        
        for i in range(max_iterations):
            fx = f(x)
            dfx = df(x)
            
            if abs(dfx) < 1e-15:
                print("Derivative too small - method may fail")
                break
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            print(f"{i:9d} {x:15.10f} {fx:15.2e} {error:15.2e}")
            
            if error < tolerance:
                print(f"\nConverged to root: x = {x_new:.10f}")
                return x_new
            
            x = x_new
        
        print("Did not converge within maximum iterations")
        return x
    
    # Example: Find square root of 2 (root of x² - 2 = 0)
    def f_sqrt2(x):
        return x**2 - 2
    
    def df_sqrt2(x):
        return 2*x
    
    print(f"\nFinding √2 (root of x² - 2 = 0):")
    root = newton_method(f_sqrt2, df_sqrt2, 1.5)
    print(f"Actual √2 = {np.sqrt(2):.10f}")
    print(f"Error: {abs(root - np.sqrt(2)):.2e}")
    
    # 2. Limits in optimization
    print("\n" + "="*50)
    print("2. Optimization using Limits")
    print("Finding critical points where f'(x) = 0")
    
    def optimize_function():
        """Find maximum of f(x) = -x² + 4x + 1"""
        
        def f(x):
            return -x**2 + 4*x + 1
        
        def df(x):
            return -2*x + 4
        
        # Critical point where f'(x) = 0
        critical_point = newton_method(df, lambda x: -2, 1)
        
        print(f"\nCritical point: x = {critical_point:.6f}")
        print(f"Function value: f({critical_point:.6f}) = {f(critical_point):.6f}")
        
        # Verify it's a maximum using second derivative
        def d2f(x):
            return -2
        
        second_deriv = d2f(critical_point)
        if second_deriv < 0:
            print("Second derivative < 0: This is a maximum")
        elif second_deriv > 0:
            print("Second derivative > 0: This is a minimum")
        else:
            print("Second derivative = 0: Test is inconclusive")
    
    optimize_function()
    
    # 3. Limits in numerical integration
    print("\n" + "="*50)
    print("3. Numerical Integration using Limits")
    print("Riemann sums approach definite integrals as Δx → 0")
    
    def riemann_sum_demo():
        """Demonstrate how Riemann sums approach the integral"""
        
        def f(x):
            return x**2
        
        a, b = 0, 2  # Integrate from 0 to 2
        
        print(f"\nApproximating ∫₀² x² dx using Riemann sums")
        print("Exact value = [x³/3]₀² = 8/3 ≈ 2.6667")
        
        print(f"\n{'n':>6} {'Δx':>12} {'Riemann Sum':>15} {'Error':>12}")
        print("-" * 50)
        
        for n in [10, 50, 100, 500, 1000, 5000]:
            dx = (b - a) / n
            x_values = np.linspace(a, b - dx, n)  # Left endpoints
            riemann_sum = dx * np.sum(f(x_values))
            exact_value = 8/3
            error = abs(riemann_sum - exact_value)
            
            print(f"{n:6d} {dx:12.6f} {riemann_sum:15.8f} {error:12.2e}")
        
        print(f"\nAs n → ∞ (Δx → 0), Riemann sum → exact integral value")
    
    riemann_sum_demo()

optimization_applications()
```

## Key Takeaways

1. **Limits** describe function behavior near specific points
2. **Continuity** requires the limit to equal the function value
3. **L'Hôpital's rule** resolves indeterminate forms 0/0 and ∞/∞
4. **Epsilon-delta definition** provides rigorous foundation
5. **Applications** include optimization, root finding, and integration
6. **Numerical methods** approximate limits when analytical solutions are difficult

## Next Steps

- Study **derivatives** as limits of difference quotients
- Learn **integration** through limits of Riemann sums
- Explore **infinite series** and convergence tests
- Apply limits to **differential equations** and **mathematical modeling**