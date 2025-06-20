# Quadratic Functions

## Mathematical Concept

A **quadratic function** has the form $f(x) = ax^2 + bx + c$ where $a \neq 0$. The graph is a **parabola** that opens upward if $a > 0$ or downward if $a < 0$.

Key properties:
- **Vertex**: $x = -\frac{b}{2a}$, minimum/maximum point
- **Axis of symmetry**: $x = -\frac{b}{2a}$
- **Discriminant**: $\Delta = b^2 - 4ac$ determines number of real roots

## Interactive Quadratic Explorer

<QuadraticExplorer />

## Python Implementation

### Quadratic Formula

```python
import math
import cmath  # For complex numbers

def quadratic_formula(a, b, c):
    """
    Solve ax² + bx + c = 0 using the quadratic formula
    Returns real or complex roots
    """
    if a == 0:
        return "Not a quadratic equation (a = 0)"
    
    discriminant = b**2 - 4*a*c
    
    if discriminant >= 0:
        # Real roots
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        return root1, root2
    else:
        # Complex roots
        sqrt_disc = cmath.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2*a)
        root2 = (-b - sqrt_disc) / (2*a)
        return root1, root2

# Examples
print("Solving x² - 5x + 6 = 0:")
roots = quadratic_formula(1, -5, 6)
print(f"Roots: {roots}")

print("\nSolving x² + 2x + 5 = 0:")
roots = quadratic_formula(1, 2, 5)
print(f"Roots: {roots}")
```

### Vertex Form Conversion

```python
def standard_to_vertex_form(a, b, c):
    """
    Convert ax² + bx + c to vertex form: a(x - h)² + k
    where (h, k) is the vertex
    """
    h = -b / (2*a)  # x-coordinate of vertex
    k = a*h**2 + b*h + c  # y-coordinate of vertex
    
    print(f"Standard form: {a}x² + {b}x + {c}")
    print(f"Vertex form: {a}(x - {h})² + {k}")
    print(f"Vertex: ({h}, {k})")
    
    return a, h, k

def vertex_to_standard_form(a, h, k):
    """
    Convert a(x - h)² + k to standard form ax² + bx + c
    """
    # Expand: a(x - h)² + k = a(x² - 2hx + h²) + k = ax² - 2ahx + ah² + k
    b = -2*a*h
    c = a*h**2 + k
    
    print(f"Vertex form: {a}(x - {h})² + {k}")
    print(f"Standard form: {a}x² + {b}x + {c}")
    
    return a, b, c

# Examples
print("Converting to vertex form:")
standard_to_vertex_form(1, -6, 8)

print("\nConverting back to standard form:")
vertex_to_standard_form(1, 3, -1)
```

### Factoring Quadratics

```python
def factor_quadratic(a, b, c):
    """
    Attempt to factor ax² + bx + c
    """
    # Find roots first
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return f"{a}x² + {b}x + {c} cannot be factored over real numbers"
    
    sqrt_disc = math.sqrt(discriminant)
    root1 = (-b + sqrt_disc) / (2*a)
    root2 = (-b - sqrt_disc) / (2*a)
    
    if discriminant == 0:
        # Perfect square
        if a == 1:
            return f"x² + {b}x + {c} = (x - {root1})²"
        else:
            return f"{a}x² + {b}x + {c} = {a}(x - {root1})²"
    else:
        # Two distinct factors
        if a == 1:
            return f"x² + {b}x + {c} = (x - {root1})(x - {root2})"
        else:
            return f"{a}x² + {b}x + {c} = {a}(x - {root1})(x - {root2})"

# Examples
print(factor_quadratic(1, -5, 6))     # x² - 5x + 6
print(factor_quadratic(1, -6, 9))     # x² - 6x + 9 (perfect square)
print(factor_quadratic(2, -8, 6))     # 2x² - 8x + 6
print(factor_quadratic(1, 0, 1))      # x² + 1 (no real factors)
```

## Interactive Parameter Exploration

The QuadraticExplorer component above provides interactive parameter controls to explore how changing the coefficients a, b, and c affects the quadratic function's graph, vertex, roots, and other properties.

## Applications and Word Problems

### Projectile Motion

```python
def projectile_motion(v0, angle_deg, h0=0):
    """
    Calculate projectile trajectory
    v0: initial velocity (m/s)
    angle_deg: launch angle (degrees)
    h0: initial height (m)
    """
    import math
    
    angle_rad = math.radians(angle_deg)
    g = 9.81  # acceleration due to gravity
    
    # Components of initial velocity
    v0x = v0 * math.cos(angle_rad)
    v0y = v0 * math.sin(angle_rad)
    
    print(f"Projectile Motion Analysis:")
    print(f"Initial velocity: {v0} m/s at {angle_deg}°")
    print(f"Initial height: {h0} m")
    
    # Time to reach maximum height
    t_max = v0y / g
    h_max = h0 + v0y * t_max - 0.5 * g * t_max**2
    
    print(f"Maximum height: {h_max:.1f} m at t = {t_max:.1f} s")
    
    # Time to hit ground (solve: h0 + v0y*t - 0.5*g*t² = 0)
    # -0.5*g*t² + v0y*t + h0 = 0
    a, b, c = -0.5*g, v0y, h0
    discriminant = b**2 - 4*a*c
    
    if discriminant >= 0:
        t_land = (-b - math.sqrt(discriminant)) / (2*a)  # Take positive root
        range_x = v0x * t_land
        
        print(f"Time of flight: {t_land:.1f} s")
        print(f"Horizontal range: {range_x:.1f} m")
        
        # Plot trajectory
        t = np.linspace(0, t_land, 100)
        x = v0x * t
        y = h0 + v0y * t - 0.5 * g * t**2
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')
        plt.plot(range_x, 0, 'ro', markersize=8, label=f'Landing: ({range_x:.1f}, 0)')
        plt.plot(v0x * t_max, h_max, 'go', markersize=8, label=f'Max height: ({v0x * t_max:.1f}, {h_max:.1f})')
        plt.xlabel('Horizontal distance (m)')
        plt.ylabel('Height (m)')
        plt.title(f'Projectile Motion: v₀ = {v0} m/s, θ = {angle_deg}°')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, max(h_max * 1.2, 10))
        plt.show()

# Example: Ball thrown at 30° with initial velocity 20 m/s
projectile_motion(20, 30, 2)
```

### Optimization Problems

```python
def fence_optimization():
    """
    Problem: A farmer has 100 meters of fence to enclose a rectangular area.
    What dimensions give the maximum area?
    
    Let width = w, length = l
    Constraint: 2w + 2l = 100 → l = 50 - w
    Area: A = w × l = w(50 - w) = 50w - w²
    """
    print("Fence Optimization Problem:")
    print("Perimeter constraint: 2w + 2l = 100")
    print("Area function: A(w) = w(50 - w) = 50w - w²")
    
    # This is a quadratic: A = -w² + 50w
    # Maximum occurs at vertex: w = -b/(2a) = -50/(2×(-1)) = 25
    
    optimal_width = 25
    optimal_length = 50 - optimal_width
    max_area = optimal_width * optimal_length
    
    print(f"Optimal dimensions: {optimal_width} × {optimal_length} meters")
    print(f"Maximum area: {max_area} square meters")
    
    # Plot area function
    w = np.linspace(0, 50, 100)
    A = 50 * w - w**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(w, A, 'b-', linewidth=2, label='Area = 50w - w²')
    plt.plot(optimal_width, max_area, 'ro', markersize=8, label=f'Maximum: ({optimal_width}, {max_area})')
    plt.xlabel('Width (m)')
    plt.ylabel('Area (m²)')
    plt.title('Area vs Width for Fixed Perimeter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

fence_optimization()
```

### Revenue and Profit Models

```python
def revenue_analysis():
    """
    Business problem: Price vs demand relationship
    Demand: q = 1000 - 10p (quantity sold at price p)
    Revenue: R = p × q = p(1000 - 10p) = 1000p - 10p²
    """
    print("Revenue Analysis:")
    print("Demand function: q = 1000 - 10p")
    print("Revenue function: R(p) = p × q = 1000p - 10p²")
    
    # Revenue is quadratic in price: R = -10p² + 1000p
    # Maximum at p = -b/(2a) = -1000/(2×(-10)) = 50
    
    optimal_price = 50
    optimal_quantity = 1000 - 10 * optimal_price
    max_revenue = optimal_price * optimal_quantity
    
    print(f"Optimal price: ${optimal_price}")
    print(f"Quantity sold: {optimal_quantity} units")
    print(f"Maximum revenue: ${max_revenue}")
    
    # Plot revenue curve
    p = np.linspace(0, 100, 200)
    R = 1000 * p - 10 * p**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, R, 'b-', linewidth=2, label='Revenue = 1000p - 10p²')
    plt.plot(optimal_price, max_revenue, 'ro', markersize=8, 
             label=f'Maximum: (${optimal_price}, ${max_revenue})')
    plt.xlabel('Price ($)')
    plt.ylabel('Revenue ($)')
    plt.title('Revenue vs Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, max_revenue * 1.1)
    plt.show()

revenue_analysis()
```

## Key Takeaways

1. **Quadratic functions** create parabolic graphs with one turning point
2. **Vertex form** makes it easy to identify maximum/minimum values
3. **Discriminant** determines the nature of roots without solving
4. **Factoring** provides insight into x-intercepts and behavior
5. **Applications** include physics (projectile motion) and optimization
6. **Python tools** make analysis and visualization straightforward

## Next Steps

- Learn about **exponential functions** and logarithms
- Study **polynomial functions** of higher degree
- Explore **trigonometric functions** and their graphs
- Apply quadratic models to **real-world data**