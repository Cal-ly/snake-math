<!-- ---
title: "Quadratic Functions"
description: "Understanding quadratic functions and their parabolic graphs, from solving equations to modeling real-world phenomena"
tags: ["mathematics", "algebra", "functions", "optimization", "physics"]
difficulty: "intermediate"
category: "concept"
symbol: "f(x) = ax² + bx + c"
prerequisites: ["linear-functions", "basic-algebra", "coordinate-geometry"]
related_concepts: ["polynomials", "optimization", "calculus", "physics-motion"]
applications: ["physics", "optimization", "computer-graphics", "data-modeling"]
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

# Quadratic Functions (f(x) = ax² + bx + c)

Think of quadratic functions as the mathematical DNA of curves! They're everywhere - from the graceful arc of a basketball shot to the optimization curves that maximize profits. A quadratic function is nature's way of describing anything that accelerates or decelerates smoothly.

## Understanding Quadratic Functions

A **quadratic function** is a mathematical expression that creates the iconic U-shaped (or upside-down U) curve called a parabola:

$$f(x) = ax^2 + bx + c \quad \text{where} \quad a \neq 0$$

Think of it like a mathematical recipe for curves: the **a** controls how "wide" or "narrow" the parabola is, **b** shifts it left or right, and **c** moves it up or down. The graph is a **parabola** — a symmetric curve that opens upward if **a > 0** (like a smile) or downward if **a < 0** (like a frown).

Key anatomical features of every parabola:
- **Vertex**: The tip of the parabola at x = -b/(2a) — the highest or lowest point
- **Axis of symmetry**: The vertical line x = -b/(2a) that splits the parabola in half
- **Discriminant**: Δ = b² - 4ac — tells us how many real solutions exist

It's like having a GPS for parabolas - these formulas tell you exactly where the important points are:

```python
import math
import matplotlib.pyplot as plt
import numpy as np

def quadratic_anatomy(a, b, c):
    """Dissect a quadratic function to reveal its key features"""
    
    print(f"Analyzing f(x) = {a}x² + {b}x + {c}")
    print("=" * 40)
    
    # Vertex coordinates
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c
    
    # Discriminant and roots
    discriminant = b**2 - 4*a*c
    
    print(f"Vertex: ({vertex_x:.2f}, {vertex_y:.2f})")
    print(f"Axis of symmetry: x = {vertex_x:.2f}")
    print(f"Opens: {'upward' if a > 0 else 'downward'}")
    print(f"Discriminant: {discriminant}")
    
    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        print(f"Two real roots: x = {root1:.2f}, x = {root2:.2f}")
    elif discriminant == 0:
        root = -b / (2*a)
        print(f"One real root (double): x = {root:.2f}")
    else:
        real_part = -b / (2*a)
        imaginary_part = math.sqrt(abs(discriminant)) / (2*a)
        print(f"Complex roots: {real_part:.2f} ± {imaginary_part:.2f}i")
    
    return vertex_x, vertex_y, discriminant

# Example: Analyze the quadratic x² - 5x + 6
vertex_x, vertex_y, disc = quadratic_anatomy(1, -5, 6)
```

## Why Quadratic Functions Matter for Programmers

Quadratic functions are the Swiss Army knife of mathematical modeling! They appear in physics simulations, optimization algorithms, computer graphics, machine learning, and data analysis. Understanding quadratics helps you model acceleration, optimize performance, create smooth animations, and solve complex algorithmic problems.

Mastering quadratics enables you to build realistic physics engines, optimize business processes, create beautiful curves in graphics, analyze algorithm complexity, and solve optimization problems efficiently.


## Interactive Exploration

<!-- <QuadraticExplorer /> -->

Explore how changing coefficients a, b, and c affects the graph, vertex, and roots of the quadratic function.


## Quadratic Function Techniques and Efficiency

Understanding different approaches to solving and analyzing quadratic functions helps optimize performance and choose appropriate methods for different scenarios.

### Method 1: Quadratic Formula

**Pros**: Always works for any quadratic, handles real and complex roots\
**Complexity**: O(1) constant time

```python
import math
import cmath

def quadratic_formula_comprehensive(a, b, c):
    """Solve quadratic equation using the quadratic formula"""
    
    print(f"Solving {a}x² + {b}x + {c} = 0")
    print("Using quadratic formula: x = [-b ± √(b² - 4ac)] / (2a)")
    
    discriminant = b**2 - 4*a*c
    print(f"Discriminant: {discriminant}")
    
    if discriminant > 0:
        # Two distinct real roots
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        
        print(f"Two real roots:")
        print(f"  x₁ = {root1:.6f}")
        print(f"  x₂ = {root2:.6f}")
        
        return root1, root2
        
    elif discriminant == 0:
        # One repeated real root
        root = -b / (2*a)
        print(f"One repeated real root:")
        print(f"  x = {root:.6f}")
        
        return root, root
        
    else:
        # Complex conjugate roots
        sqrt_disc = cmath.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        
        print(f"Two complex roots:")
        print(f"  x₁ = {root1}")
        print(f"  x₂ = {root2}")
        
        return root1, root2

# Test different cases
quadratic_formula_comprehensive(1, -5, 6)    # Two real roots
quadratic_formula_comprehensive(1, -4, 4)    # One repeated root
quadratic_formula_comprehensive(1, 0, 1)     # Complex roots
```

### Method 2: Completing the Square

**Pros**: Reveals vertex form intuitively, great for optimization problems\
**Complexity**: O(1) constant time with clear geometric interpretation

```python
def complete_the_square(a, b, c):
    """Convert quadratic to vertex form by completing the square"""
    
    print(f"Converting {a}x² + {b}x + {c} to vertex form")
    print("=" * 45)
    
    if a != 1:
        print(f"Step 1: Factor out {a} from x² and x terms")
        print(f"{a}(x² + {b/a}x) + {c}")
    
    # Complete the square for the x terms
    h = -b / (2*a)  # x-coordinate of vertex
    k = c - (b**2) / (4*a)  # y-coordinate of vertex
    
    print(f"Step 2: Complete the square")
    print(f"h = -b/(2a) = -({b})/(2×{a}) = {h}")
    print(f"k = c - b²/(4a) = {c} - ({b})²/(4×{a}) = {k}")
    
    print(f"\nVertex form: f(x) = {a}(x - {h})² + {k}")
    print(f"Vertex: ({h}, {k})")
    
    # Verify by expanding back
    expanded_a = a
    expanded_b = -2*a*h
    expanded_c = a*h**2 + k
    
    print(f"\nVerification by expanding:")
    print(f"{a}(x - {h})² + {k}")
    print(f"= {a}(x² - {2*h}x + {h**2}) + {k}")
    print(f"= {expanded_a}x² + {expanded_b}x + {expanded_c}")
    
    # Check if it matches original
    matches = (expanded_a == a and abs(expanded_b - b) < 1e-10 and abs(expanded_c - c) < 1e-10)
    print(f"Matches original: {matches}")
    
    return h, k

# Example: Complete the square for x² - 6x + 5
h, k = complete_the_square(1, -6, 5)
```

### Method 3: Factoring (when possible)

**Pros**: Simple and fast if roots are rational, provides immediate insight\
**Complexity**: O(1) if factorable, may require trial and error

```python
def factor_quadratic_comprehensive(a, b, c):
    """Attempt to factor quadratic into (px + q)(rx + s) form"""
    
    print(f"Attempting to factor {a}x² + {b}x + {c}")
    print("=" * 40)
    
    # Check if discriminant is a perfect square
    discriminant = b**2 - 4*a*c
    sqrt_disc = math.sqrt(abs(discriminant))
    
    if discriminant < 0:
        print("Cannot factor over real numbers (complex roots)")
        return None
    
    if not sqrt_disc.is_integer():
        print("Roots are irrational - factoring not simple")
        print("Using quadratic formula for exact roots:")
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        print(f"f(x) = {a}(x - {root1:.6f})(x - {root2:.6f})")
        return root1, root2
    
    # Rational roots - attempt to factor nicely
    sqrt_disc = int(sqrt_disc)
    root1 = (-b + sqrt_disc) / (2*a)
    root2 = (-b - sqrt_disc) / (2*a)
    
    print(f"Roots: x = {root1}, x = {root2}")
    
    # Express in factored form
    if a == 1:
        print(f"Factored form: (x - {root1})(x - {root2})")
    else:
        print(f"Factored form: {a}(x - {root1})(x - {root2})")
    
    # Alternative factoring approach for integer coefficients
    if all(isinstance(x, int) for x in [a, b, c]):
        print(f"\nAlternative integer factoring:")
        factoring_by_grouping(a, b, c)
    
    return root1, root2

def factoring_by_grouping(a, b, c):
    """Demonstrate factoring by grouping method"""
    
    print(f"Looking for factors of ac = {a}×{c} = {a*c}")
    print(f"That add up to b = {b}")
    
    # Find factor pairs of ac
    ac = a * c
    factors = []
    for i in range(1, int(abs(ac)**0.5) + 1):
        if ac % i == 0:
            factors.extend([(i, ac//i), (-i, -ac//i)])
    
    # Find pair that adds to b
    for f1, f2 in factors:
        if f1 + f2 == b:
            print(f"Found factors: {f1} and {f2}")
            print(f"Rewrite middle term: {a}x² + {f1}x + {f2}x + {c}")
            break
    else:
        print("No integer factor pair found")

# Test factoring
factor_quadratic_comprehensive(1, -5, 6)   # x² - 5x + 6 = (x-2)(x-3)
factor_quadratic_comprehensive(2, -7, 3)   # 2x² - 7x + 3 = (2x-1)(x-3)
factor_quadratic_comprehensive(1, 0, -4)   # x² - 4 = (x-2)(x+2)
```


## Why the Quadratic Formula Works

The quadratic formula is derived from completing the square - it's like having a universal key that unlocks any quadratic equation:

```python
def derive_quadratic_formula():
    """Step-by-step derivation of the quadratic formula"""
    
    print("Deriving the Quadratic Formula")
    print("=" * 35)
    
    print("Start with the general quadratic equation:")
    print("ax² + bx + c = 0")
    
    print("\nStep 1: Divide everything by 'a'")
    print("x² + (b/a)x + (c/a) = 0")
    
    print("\nStep 2: Move constant term to right side")
    print("x² + (b/a)x = -(c/a)")
    
    print("\nStep 3: Complete the square")
    print("Add (b/2a)² to both sides:")
    print("x² + (b/a)x + (b/2a)² = -(c/a) + (b/2a)²")
    
    print("\nStep 4: Factor left side as perfect square")
    print("(x + b/2a)² = -(c/a) + (b/2a)²")
    
    print("\nStep 5: Simplify right side")
    print("(x + b/2a)² = -(c/a) + b²/(4a²)")
    print("(x + b/2a)² = (-4ac + b²)/(4a²)")
    print("(x + b/2a)² = (b² - 4ac)/(4a²)")
    
    print("\nStep 6: Take square root of both sides")
    print("x + b/2a = ±√(b² - 4ac)/(2a)")
    
    print("\nStep 7: Solve for x")
    print("x = -b/2a ± √(b² - 4ac)/(2a)")
    print("x = [-b ± √(b² - 4ac)]/(2a)")
    
    print("\n🎉 The Quadratic Formula!")
    
    # Demonstrate with specific example
    print("\nExample: x² - 5x + 6 = 0")
    a, b, c = 1, -5, 6
    discriminant = b**2 - 4*a*c
    
    print(f"a = {a}, b = {b}, c = {c}")
    print(f"Discriminant = {b}² - 4({a})({c}) = {discriminant}")
    
    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        
        print(f"x = [-({b}) ± √{discriminant}]/(2×{a})")
        print(f"x = [{-b} ± {sqrt_disc}]/{2*a}")
        print(f"x = {root1} or x = {root2}")
        
        # Verify solutions
        print(f"\nVerification:")
        for i, root in enumerate([root1, root2], 1):
            result = a*root**2 + b*root + c
            print(f"  x{i} = {root}: {a}({root})² + {b}({root}) + {c} = {result:.10f}")

derive_quadratic_formula()
```


## Common Quadratic Patterns

Standard quadratic forms and transformations that appear frequently in mathematics and programming:

- **Standard Form:**\
  \(f(x) = ax^2 + bx + c\)

- **Vertex Form:**\
  \(f(x) = a(x - h)^2 + k\) where (h, k) is the vertex

- **Factored Form:**\
  \(f(x) = a(x - r_1)(x - r_2)\) where r₁, r₂ are the roots

- **Completed Square Form:**\
  \(f(x) = a\left(x + \frac{b}{2a}\right)^2 - \frac{b^2-4ac}{4a}\)

Python implementations demonstrating these patterns:

```python
def quadratic_forms_library():
    """Collection of common quadratic forms and conversions"""
    
    def standard_to_vertex(a, b, c):
        """Convert standard form to vertex form"""
        h = -b / (2*a)
        k = a*h**2 + b*h + c
        return a, h, k
    
    def vertex_to_standard(a, h, k):
        """Convert vertex form to standard form"""
        # f(x) = a(x - h)² + k = a(x² - 2hx + h²) + k = ax² - 2ahx + ah² + k
        new_a = a
        new_b = -2*a*h
        new_c = a*h**2 + k
        return new_a, new_b, new_c
    
    def standard_to_factored(a, b, c):
        """Convert standard form to factored form if possible"""
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None  # No real factors
        
        sqrt_disc = math.sqrt(discriminant)
        r1 = (-b + sqrt_disc) / (2*a)
        r2 = (-b - sqrt_disc) / (2*a)
        return a, r1, r2
    
    def factored_to_standard(a, r1, r2):
        """Convert factored form to standard form"""
        # f(x) = a(x - r1)(x - r2) = a(x² - (r1+r2)x + r1*r2) = ax² - a(r1+r2)x + a*r1*r2
        new_a = a
        new_b = -a * (r1 + r2)
        new_c = a * r1 * r2
        return new_a, new_b, new_c
    
    # Demonstration
    print("Quadratic Form Conversions")
    print("=" * 30)
    
    # Start with standard form: x² - 5x + 6
    original_a, original_b, original_c = 1, -5, 6
    print(f"Original: f(x) = {original_a}x² + {original_b}x + {original_c}")
    
    # Convert to vertex form
    a_v, h, k = standard_to_vertex(original_a, original_b, original_c)
    print(f"Vertex form: f(x) = {a_v}(x - {h})² + {k}")
    
    # Convert back to standard form
    back_a, back_b, back_c = vertex_to_standard(a_v, h, k)
    print(f"Back to standard: f(x) = {back_a}x² + {back_b}x + {back_c}")
    
    # Convert to factored form
    factored_result = standard_to_factored(original_a, original_b, original_c)
    if factored_result:
        a_f, r1, r2 = factored_result
        print(f"Factored form: f(x) = {a_f}(x - {r1})(x - {r2})")
        
        # Convert back to standard form
        back2_a, back2_b, back2_c = factored_to_standard(a_f, r1, r2)
        print(f"Back to standard: f(x) = {back2_a}x² + {back2_b}x + {back2_c}")
    
    # Verify all forms are equivalent
    print(f"\nVerification at x = 2:")
    x = 2
    
    # Standard form
    std_result = original_a*x**2 + original_b*x + original_c
    print(f"Standard form: {std_result}")
    
    # Vertex form
    vertex_result = a_v*(x - h)**2 + k
    print(f"Vertex form: {vertex_result}")
    
    # Factored form (if exists)
    if factored_result:
        factored_eval = a_f*(x - r1)*(x - r2)
        print(f"Factored form: {factored_eval}")
    
    return original_a, original_b, original_c

quadratic_forms_library()
```


## Practical Real-world Applications

Quadratic functions aren't just academic exercises - they're essential tools for solving real-world problems across physics, business, and engineering:

### Application 1: Projectile Motion and Physics Simulations

```python
def projectile_motion_analysis():
    """Model projectile motion using quadratic functions"""
    
    print("Projectile Motion Analysis")
    print("=" * 30)
    
    def projectile_trajectory(v0, angle_deg, h0=0):
        """
        Calculate projectile trajectory
        h(t) = h0 + v0*sin(θ)*t - (1/2)*g*t²
        This is a quadratic function in t!
        """
        import math
        
        angle_rad = math.radians(angle_deg)
        g = 9.81  # acceleration due to gravity (m/s²)
        
        # Vertical component of initial velocity
        v0_y = v0 * math.sin(angle_rad)
        
        # Horizontal component of initial velocity
        v0_x = v0 * math.cos(angle_rad)
        
        print(f"Projectile Parameters:")
        print(f"  Initial velocity: {v0} m/s")
        print(f"  Launch angle: {angle_deg}°")
        print(f"  Initial height: {h0} m")
        print(f"  Vertical velocity component: {v0_y:.2f} m/s")
        print(f"  Horizontal velocity component: {v0_x:.2f} m/s")
        
        # Height as a function of time: h(t) = h0 + v0_y*t - 0.5*g*t²
        # This is a quadratic: h(t) = -0.5*g*t² + v0_y*t + h0
        a_coeff = -0.5 * g
        b_coeff = v0_y
        c_coeff = h0
        
        print(f"\nHeight equation: h(t) = {a_coeff}t² + {b_coeff}t + {c_coeff}")
        
        # Find when projectile hits ground (h(t) = 0)
        discriminant = b_coeff**2 - 4*a_coeff*c_coeff
        
        if discriminant >= 0:
            t1 = (-b_coeff + math.sqrt(discriminant)) / (2*a_coeff)
            t2 = (-b_coeff - math.sqrt(discriminant)) / (2*a_coeff)
            
            # Choose positive time
            flight_time = max(t1, t2)
            print(f"Flight time: {flight_time:.2f} seconds")
            
            # Maximum height (vertex of parabola)
            t_max = -b_coeff / (2*a_coeff)
            h_max = h0 + v0_y*t_max - 0.5*g*t_max**2
            
            print(f"Maximum height: {h_max:.2f} m at t = {t_max:.2f} s")
            
            # Range (horizontal distance)
            range_distance = v0_x * flight_time
            print(f"Range: {range_distance:.2f} m")
            
            return flight_time, h_max, range_distance
        else:
            print("No real solution - projectile never lands!")
            return None, None, None
    
    def basketball_shot_analysis():
        """Analyze a basketball shot using quadratic motion"""
        
        print(f"\nBasketball Shot Analysis:")
        
        # Basketball shot parameters
        v0 = 7.5  # m/s
        angle = 45  # degrees
        release_height = 2.0  # meters
        basket_height = 3.05  # meters (10 feet)
        basket_distance = 4.6  # meters (15 feet)
        
        flight_time, max_height, total_range = projectile_trajectory(v0, angle, release_height)
        
        if flight_time:
            # Check if shot makes it to basket
            v0_x = v0 * math.cos(math.radians(angle))
            time_to_basket = basket_distance / v0_x
            
            if time_to_basket <= flight_time:
                # Calculate height at basket
                v0_y = v0 * math.sin(math.radians(angle))
                g = 9.81
                height_at_basket = release_height + v0_y*time_to_basket - 0.5*g*time_to_basket**2
                
                print(f"\nShot Analysis:")
                print(f"  Time to reach basket: {time_to_basket:.2f} s")
                print(f"  Height at basket: {height_at_basket:.2f} m")
                print(f"  Basket height: {basket_height} m")
                
                if abs(height_at_basket - basket_height) < 0.1:
                    print("  Result: PERFECT SHOT! 🏀")
                elif height_at_basket > basket_height:
                    print(f"  Result: Shot too high by {height_at_basket - basket_height:.2f} m")
                else:
                    print(f"  Result: Shot too low by {basket_height - height_at_basket:.2f} m")
            else:
                print("  Result: Shot doesn't reach the basket")
    
    # Run demonstrations
    projectile_trajectory(20, 30, 2)  # General projectile
    basketball_shot_analysis()  # Basketball-specific
    
    return v0, angle_deg, h0

projectile_motion_analysis()
```

### Application 2: Business Optimization and Revenue Maximization

```python
def business_optimization():
    """Use quadratic functions for business optimization problems"""
    
    print("\nBusiness Optimization with Quadratics")
    print("=" * 40)
    
    def revenue_optimization():
        """Find optimal pricing to maximize revenue"""
        
        print("Revenue Optimization Problem:")
        print("A company sells widgets. Market research shows:")
        print("- At $10 per widget, they sell 1000 widgets")
        print("- For every $1 price increase, they lose 50 customers")
        print("- For every $1 price decrease, they gain 50 customers")
        
        # Let x = price increase from $10
        # Price = 10 + x
        # Quantity = 1000 - 50x
        # Revenue = Price × Quantity = (10 + x)(1000 - 50x)
        
        def revenue_function(x):
            price = 10 + x
            quantity = 1000 - 50*x
            return price * quantity
        
        # Expand: R(x) = (10 + x)(1000 - 50x) = 10000 - 500x + 1000x - 50x²
        # R(x) = -50x² + 500x + 10000
        
        a, b, c = -50, 500, 10000
        print(f"\nRevenue function: R(x) = {a}x² + {b}x + {c}")
        print("where x is the price increase from $10")
        
        # Find maximum revenue (vertex of parabola)
        optimal_x = -b / (2*a)
        optimal_price = 10 + optimal_x
        optimal_quantity = 1000 - 50*optimal_x
        max_revenue = revenue_function(optimal_x)
        
        print(f"\nOptimal Analysis:")
        print(f"  Optimal price change: ${optimal_x:+.2f}")
        print(f"  Optimal selling price: ${optimal_price:.2f}")
        print(f"  Optimal quantity: {optimal_quantity} widgets")
        print(f"  Maximum revenue: ${max_revenue:,.2f}")
        
        # Compare with other pricing strategies
        print(f"\nComparison with other prices:")
        for price_change in [-2, -1, 0, 1, 2]:
            price = 10 + price_change
            revenue = revenue_function(price_change)
            print(f"  Price ${price:.2f}: Revenue ${revenue:,.2f}")
        
        return optimal_price, max_revenue
    
    def profit_maximization():
        """Find optimal production level to maximize profit"""
        
        print(f"\nProfit Maximization Problem:")
        print("A factory has the following cost and revenue structure:")
        print("- Fixed costs: $5,000")
        print("- Variable cost per unit: $15")
        print("- Revenue per unit decreases with quantity: R(q) = 50q - 0.01q²")
        
        # Cost function: C(q) = 5000 + 15q (linear)
        # Revenue function: R(q) = 50q - 0.01q² (quadratic)
        # Profit function: P(q) = R(q) - C(q) = 50q - 0.01q² - 5000 - 15q
        # P(q) = -0.01q² + 35q - 5000
        
        def cost_function(q):
            return 5000 + 15*q
        
        def revenue_function(q):
            return 50*q - 0.01*q**2
        
        def profit_function(q):
            return revenue_function(q) - cost_function(q)
        
        # Profit: P(q) = -0.01q² + 35q - 5000
        a_profit, b_profit, c_profit = -0.01, 35, -5000
        
        print(f"Profit function: P(q) = {a_profit}q² + {b_profit}q + {c_profit}")
        
        # Find maximum profit (vertex)
        optimal_quantity = -b_profit / (2*a_profit)
        max_profit = profit_function(optimal_quantity)
        optimal_revenue = revenue_function(optimal_quantity)
        optimal_cost = cost_function(optimal_quantity)
        
        print(f"\nOptimal Production Analysis:")
        print(f"  Optimal quantity: {optimal_quantity} units")
        print(f"  Revenue at optimal quantity: ${optimal_revenue:,.2f}")
        print(f"  Cost at optimal quantity: ${optimal_cost:,.2f}")
        print(f"  Maximum profit: ${max_profit:,.2f}")
        
        # Break-even analysis (when profit = 0)
        # -0.01q² + 35q - 5000 = 0
        discriminant = b_profit**2 - 4*a_profit*c_profit
        
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            q1 = (-b_profit + sqrt_disc) / (2*a_profit)
            q2 = (-b_profit - sqrt_disc) / (2*a_profit)
            
            print(f"\nBreak-even points:")
            print(f"  Break-even at {min(q1, q2):.0f} units")
            print(f"  Break-even at {max(q1, q2):.0f} units")
            print(f"  Profitable range: {min(q1, q2):.0f} to {max(q1, q2):.0f} units")
        
        return optimal_quantity, max_profit
    
    # Run optimization analyses
    optimal_price, max_revenue = revenue_optimization()
    optimal_qty, max_profit = profit_maximization()
    
    return optimal_price, optimal_qty

business_optimization()
```

### Application 3: Computer Graphics and Animation

```python
def graphics_and_animation():
    """Apply quadratic functions to computer graphics and animation"""
    
    print("\nQuadratic Functions in Computer Graphics")
    print("=" * 45)
    
    def parabolic_curve_generation():
        """Generate smooth parabolic curves for graphics"""
        
        print("Parabolic Curve Generation:")
        
        # Generate points for a parabolic arch
        def generate_arch(width, height, num_points=50):
            """Generate points for a parabolic arch"""
            
            # Arch from x = 0 to x = width, max height at center
            # Use vertex form: y = a(x - h)² + k
            # where h = width/2 (center), k = height (max height)
            # At x = 0 and x = width, y should be 0
            # So: 0 = a(0 - width/2)² + height
            # a = -height / (width/2)²
            
            h = width / 2
            k = height
            a = -height / (h**2)
            
            print(f"Arch parameters:")
            print(f"  Width: {width} units")
            print(f"  Height: {height} units")
            print(f"  Equation: y = {a:.4f}(x - {h})² + {k}")
            
            points = []
            for i in range(num_points + 1):
                x = (width * i) / num_points
                y = a * (x - h)**2 + k
                points.append((x, y))
            
            return points
        
        # Generate arch points
        arch_points = generate_arch(10, 5, 20)
        print(f"Generated {len(arch_points)} points for arch")
        
        return arch_points
    
    def easing_functions():
        """Create smooth animation easing using quadratic functions"""
        
        print(f"\nAnimation Easing Functions:")
        
        def ease_in_quad(t):
            """Quadratic ease-in: slow start, fast finish"""
            return t * t
        
        def ease_out_quad(t):
            """Quadratic ease-out: fast start, slow finish"""
            return 1 - (1 - t)**2
        
        def ease_in_out_quad(t):
            """Quadratic ease-in-out: slow start and finish"""
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t)**2
        
        print("Easing function examples (t from 0 to 1):")
        print(f"{'t':>6} {'Linear':>8} {'Ease In':>10} {'Ease Out':>10} {'Ease In-Out':>12}")
        print("-" * 50)
        
        for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            linear = t
            ease_in = ease_in_quad(t)
            ease_out = ease_out_quad(t)
            ease_in_out = ease_in_out_quad(t)
            
            print(f"{t:>6.1f} {linear:>8.3f} {ease_in:>10.3f} {ease_out:>10.3f} {ease_in_out:>12.3f}")
        
        return ease_in_quad, ease_out_quad, ease_in_out_quad
    
    def bezier_curves():
        """Generate Bezier curves using quadratic interpolation"""
        
        print(f"\nQuadratic Bezier Curves:")
        
        def quadratic_bezier(p0, p1, p2, t):
            """
            Calculate point on quadratic Bezier curve
            B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            This is a quadratic function in t!
            """
            return (
                (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0],
                (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            )
        
        # Define control points
        p0 = (0, 0)    # Start point
        p1 = (5, 8)    # Control point
        p2 = (10, 0)   # End point
        
        print(f"Control points:")
        print(f"  P₀ (start): {p0}")
        print(f"  P₁ (control): {p1}")
        print(f"  P₂ (end): {p2}")
        
        # Generate curve points
        curve_points = []
        num_points = 10
        
        print(f"\nBezier curve points:")
        for i in range(num_points + 1):
            t = i / num_points
            point = quadratic_bezier(p0, p1, p2, t)
            curve_points.append(point)
            print(f"  t={t:.1f}: ({point[0]:.2f}, {point[1]:.2f})")
        
        return curve_points
    
    def projectile_path_animation():
        """Create realistic projectile path for game physics"""
        
        print(f"\nGame Physics: Projectile Path")
        
        def calculate_trajectory_points(v0, angle, num_frames=60, dt=0.1):
            """Calculate trajectory points for animation"""
            
            import math
            
            angle_rad = math.radians(angle)
            v0_x = v0 * math.cos(angle_rad)
            v0_y = v0 * math.sin(angle_rad)
            g = 9.81
            
            points = []
            for frame in range(num_frames):
                t = frame * dt
                
                # Position equations (quadratic in t)
                x = v0_x * t
                y = v0_y * t - 0.5 * g * t**2
                
                if y < 0:  # Hit ground
                    break
                    
                points.append((x, y))
            
            return points
        
        # Calculate trajectory for a cannonball
        trajectory = calculate_trajectory_points(30, 45, 100, 0.05)
        
        print(f"Generated {len(trajectory)} trajectory points")
        print(f"Sample points:")
        for i in range(0, len(trajectory), 10):
            x, y = trajectory[i]
            print(f"  Frame {i}: ({x:.2f}, {y:.2f})")
        
        return trajectory
    
    # Run all graphics demonstrations
    arch_points = parabolic_curve_generation()
    easing_funcs = easing_functions()
    bezier_points = bezier_curves()
    projectile_points = projectile_path_animation()
    
    print(f"\nGraphics Applications Summary:")
    print(f"• Parabolic arches for architectural visualization")
    print(f"• Quadratic easing for smooth animation transitions")
    print(f"• Bezier curves for smooth path generation")
    print(f"• Projectile physics for realistic game motion")
    
    return arch_points, bezier_points, projectile_points

graphics_and_animation()
```


## Try it Yourself

Ready to master quadratic functions? Here are some hands-on challenges:

- **Interactive Quadratic Explorer:** Build a dynamic graphing tool where users can adjust coefficients and see real-time changes to the parabola, vertex, and roots.
- **Physics Simulator:** Create a projectile motion simulator with adjustable launch angles and velocities, showing trajectory paths.
- **Business Optimizer:** Develop a profit/revenue optimization calculator that finds optimal pricing or production levels.
- **Animation Framework:** Implement smooth animation easing functions using quadratic curves for natural motion.
- **Curve Designer:** Build a tool for creating parabolic arches, bridges, or decorative curves for design applications.
- **Game Physics Engine:** Create a realistic ballistics system for games with gravity, air resistance, and collision detection.


## Key Takeaways

- Quadratic functions create parabolic curves that model acceleration, optimization, and many natural phenomena.
- The quadratic formula provides a universal solution method, derived from completing the square.
- Vertex form reveals optimization points (maximum or minimum values) crucial for business and physics problems.
- Different solution methods (formula, factoring, completing square) offer various insights and computational advantages.
- Real-world applications span physics simulations, business optimization, computer graphics, and game development.
- Understanding the geometric meaning of coefficients a, b, c helps visualize and predict parabolic behavior.
- Quadratics bridge algebra and calculus, providing foundation for advanced mathematical modeling.


## Next Steps & Further Exploration

Ready to dive deeper into the world of functions and mathematical modeling?

- Explore **Higher-Degree Polynomials** for more complex curve behaviors and multiple turning points.
- Study **Exponential and Logarithmic Functions** for growth, decay, and scaling phenomena.
- Learn **Trigonometric Functions** for periodic and oscillating behaviors in waves and rotations.
- Investigate **Parametric Equations** for advanced curve generation and 3D modeling.
- Apply quadratics to **Calculus** concepts like derivatives for optimization and physics analysis.
- Explore **Differential Equations** where quadratics appear as solutions to acceleration problems.