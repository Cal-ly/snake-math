---
title: "Trigonometric Functions and the Unit Circle"
description: "Understanding trigonometric functions through the unit circle and their applications in waves, rotations, and periodic phenomena"
tags: ["mathematics", "trigonometry", "geometry", "programming", "physics"]
difficulty: "intermediate"
category: "concept"
symbol: "sin, cos, tan, θ"
prerequisites: ["basic-arithmetic", "coordinate-systems", "functions"]
related_concepts: ["vectors", "complex-numbers", "fourier-series", "wave-analysis"]
applications: ["computer-graphics", "signal-processing", "physics", "game-development"]
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
---

# Trigonometric Functions and the Unit Circle (sin, cos, tan, θ)

Think of trigonometric functions as the mathematical DNA of waves, rotations, and cycles! The unit circle is like a universal coordinate system where every angle tells a story through its x and y coordinates. It's the bridge between geometry and the rhythmic patterns that govern everything from sound waves to planetary motion.

## Understanding Trigonometric Functions

**Trigonometric functions** relate angles to ratios in right triangles and coordinates on the **unit circle** (radius = 1, centered at origin). The unit circle is a powerful visualization tool that reveals how angles connect to coordinates and helps us understand the periodic nature of trigonometric functions.

The fundamental relationships:

$$
\begin{align}
\sin(\theta) &= \frac{\text{opposite}}{\text{hypotenuse}} = y\text{-coordinate} \\
\cos(\theta) &= \frac{\text{adjacent}}{\text{hypotenuse}} = x\text{-coordinate} \\
\tan(\theta) &= \frac{\sin(\theta)}{\cos(\theta)} = \frac{y}{x}
\end{align}
$$

Think of the unit circle like a clock face where every "time" (angle) has a unique fingerprint given by its (x, y) coordinates. As you walk around this circle, the x-coordinate dances like a horizontal wave while the y-coordinate creates a vertical wave:

```python
import math
import numpy as np

# Walking around the unit circle
angles = np.linspace(0, 2*math.pi, 8)  # 8 points around the circle

print("Unit Circle Journey:")
print(f"{'Angle (rad)':>12} {'Angle (deg)':>12} {'x (cos)':>10} {'y (sin)':>10}")
print("-" * 50)

for angle in angles:
    x_coord = math.cos(angle)  # x-coordinate
    y_coord = math.sin(angle)  # y-coordinate
    angle_deg = math.degrees(angle)
    
    print(f"{angle:12.3f} {angle_deg:12.1f} {x_coord:10.3f} {y_coord:10.3f}")

# The fundamental identity emerges from the unit circle!
print(f"\nFundamental identity: sin²(θ) + cos²(θ) = 1")
for angle in [math.pi/6, math.pi/4, math.pi/3]:
    identity_check = math.sin(angle)**2 + math.cos(angle)**2
    print(f"At θ = {angle:.3f}: {identity_check:.6f}")
```

## Why Trigonometric Functions Matter for Programmers

Trigonometric functions are essential for computer graphics, game development, signal processing, animation, physics simulations, and machine learning. They help you rotate objects, generate smooth animations, analyze periodic data, and model wave-like phenomena.

Understanding these functions enables you to work with 2D/3D transformations, implement smooth interpolations, analyze audio signals, create realistic physics, and build algorithms that handle periodic or oscillatory behavior.


## Interactive Exploration

<UnitCircleExplorer />

```plaintext
Component conceptualization:
Create an interactive unit circle and trigonometric functions explorer where users can:
- Visualize the unit circle with draggable point to explore angle relationships
- Display real-time sin, cos, and tan values as the angle changes
- Show animated graphs of sine, cosine, and tangent functions
- Demonstrate phase relationships between different trigonometric functions
- Interactive transformation controls (amplitude, frequency, phase shift)
- Special angle calculator with exact values (30°, 45°, 60°, etc.)
- Triangle overlay showing right triangle relationships
- Wave visualization showing how circular motion creates sinusoidal waves
- Angle conversion between radians and degrees with visual feedback
- Trigonometric identity verification with dynamic calculations
The component should provide both geometric intuition and analytical understanding.
```

Experiment with different angles to see how trigonometric functions behave and understand their periodic nature and geometric relationships.


## Trigonometric Functions Techniques and Efficiency

Understanding different approaches to calculating and applying trigonometric functions helps optimize performance and choose appropriate methods for different scenarios.

### Method 1: Built-in Math Library Functions

**Pros**: Highly optimized, accurate, standard library availability\
**Complexity**: O(1) for individual calculations, optimized algorithms

```python
import math
import time

def builtin_trigonometric_functions():
    """Use Python's built-in math library for trigonometric calculations"""
    
    print("Built-in Math Library Trigonometric Functions")
    print("=" * 50)
    
    # Convert between degrees and radians
    def deg_to_rad(degrees):
        return math.radians(degrees)  # or degrees * math.pi / 180
    
    def rad_to_deg(radians):
        return math.degrees(radians)  # or radians * 180 / math.pi
    
    # Test angles in degrees
    test_angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
    
    print(f"{'Angle (°)':>8} {'Radians':>10} {'sin':>8} {'cos':>8} {'tan':>10}")
    print("-" * 50)
    
    for angle_deg in test_angles_deg:
        angle_rad = deg_to_rad(angle_deg)
        
        sin_val = math.sin(angle_rad)
        cos_val = math.cos(angle_rad)
        
        # Handle tan(90°) and tan(270°) special cases
        if abs(cos_val) < 1e-10:  # cos ≈ 0
            tan_val = "undefined"
        else:
            tan_val = f"{math.tan(angle_rad):8.3f}"
        
        print(f"{angle_deg:>6} {angle_rad:>10.3f} {sin_val:>8.3f} {cos_val:>8.3f} {tan_val:>10}")
    
    # Performance test
    print(f"\nPerformance Test (1,000,000 calculations):")
    n_calculations = 1000000
    test_angle = math.pi / 4  # 45 degrees
    
    start_time = time.time()
    for _ in range(n_calculations):
        result = math.sin(test_angle)
    sin_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(n_calculations):
        result = math.cos(test_angle)
    cos_time = time.time() - start_time
    
    print(f"sin() calculations: {sin_time:.4f} seconds")
    print(f"cos() calculations: {cos_time:.4f} seconds")
    print(f"Rate: ~{n_calculations/sin_time:,.0f} calculations/second")
    
    return test_angles_deg

builtin_trigonometric_functions()
```

### Method 2: NumPy Vectorized Operations

**Pros**: Handles arrays efficiently, broadcasting capabilities, optimized for large datasets\
**Complexity**: O(n) for n elements, highly optimized vectorized operations

```python
import numpy as np
import matplotlib.pyplot as plt

def numpy_trigonometric_operations():
    """Demonstrate NumPy's vectorized trigonometric operations"""
    
    print("NumPy Vectorized Trigonometric Operations")
    print("=" * 45)
    
    # Create angle arrays
    angles_deg = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180, 270, 360])
    angles_rad = np.radians(angles_deg)
    
    # Vectorized calculations
    sin_vals = np.sin(angles_rad)
    cos_vals = np.cos(angles_rad)
    tan_vals = np.tan(angles_rad)
    
    print(f"Vectorized calculations for {len(angles_deg)} angles:")
    print(f"{'Angle (°)':>8} {'sin':>8} {'cos':>8} {'tan':>10}")
    print("-" * 38)
    
    for i, angle_deg in enumerate(angles_deg):
        tan_str = f"{tan_vals[i]:8.3f}" if abs(tan_vals[i]) < 1000 else "undefined"
        print(f"{angle_deg:>6} {sin_vals[i]:>8.3f} {cos_vals[i]:>8.3f} {tan_str:>10}")
    
    # Generate smooth curves
    print(f"\nGenerating smooth trigonometric curves:")
    
    # High-resolution angle array for smooth plotting
    t = np.linspace(0, 4*np.pi, 1000)  # 0 to 720 degrees
    
    # Calculate trigonometric functions
    sin_curve = np.sin(t)
    cos_curve = np.cos(t)
    tan_curve = np.tan(t)
    
    # Limit tangent for visualization (clip extreme values)
    tan_curve_clipped = np.clip(tan_curve, -10, 10)
    
    print(f"Generated {len(t)} points for smooth curves")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    large_array = np.linspace(0, 2*np.pi, 1000000)
    
    # NumPy vectorized
    start_time = time.time()
    numpy_result = np.sin(large_array)
    numpy_time = time.time() - start_time
    
    # Loop-based calculation
    start_time = time.time()
    loop_result = [math.sin(x) for x in large_array]
    loop_time = time.time() - start_time
    
    print(f"NumPy vectorized: {numpy_time:.4f} seconds")
    print(f"Python loop: {loop_time:.4f} seconds")
    print(f"Speedup: {loop_time/numpy_time:.1f}x faster")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Sine and Cosine
    plt.subplot(2, 2, 1)
    plt.plot(t, sin_curve, 'b-', linewidth=2, label='sin(x)')
    plt.plot(t, cos_curve, 'r-', linewidth=2, label='cos(x)')
    plt.title('Sine and Cosine Functions')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4*np.pi)
    plt.ylim(-1.5, 1.5)
    
    # Tangent
    plt.subplot(2, 2, 2)
    plt.plot(t, tan_curve_clipped, 'g-', linewidth=2, label='tan(x)')
    plt.title('Tangent Function (clipped)')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4*np.pi)
    plt.ylim(-10, 10)
    
    # Unit circle
    plt.subplot(2, 2, 3)
    circle_t = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(circle_t)
    circle_y = np.sin(circle_t)
    
    plt.plot(circle_x, circle_y, 'k-', linewidth=2, label='Unit Circle')
    
    # Mark special angles
    special_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi]
    special_x = np.cos(special_angles)
    special_y = np.sin(special_angles)
    
    plt.plot(special_x, special_y, 'ro', markersize=8, label='Special Angles')
    plt.title('Unit Circle with Special Angles')
    plt.xlabel('x (cos θ)')
    plt.ylabel('y (sin θ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Phase relationships
    plt.subplot(2, 2, 4)
    t_phase = np.linspace(0, 2*np.pi, 100)
    sin_phase = np.sin(t_phase)
    cos_phase = np.cos(t_phase)
    sin_shifted = np.sin(t_phase + np.pi/2)  # sin(x + π/2) = cos(x)
    
    plt.plot(t_phase, sin_phase, 'b-', linewidth=2, label='sin(x)')
    plt.plot(t_phase, cos_phase, 'r-', linewidth=2, label='cos(x)')
    plt.plot(t_phase, sin_shifted, 'b--', linewidth=2, alpha=0.7, label='sin(x + π/2)')
    plt.title('Phase Relationships')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return t, sin_curve, cos_curve

numpy_trigonometric_operations()
```

### Method 3: Taylor Series Approximation

**Pros**: Educational value, customizable precision, works without math libraries\
**Complexity**: O(n) where n is number of terms in series

```python
def taylor_series_trigonometry():
    """Implement trigonometric functions using Taylor series approximation"""
    
    print("Taylor Series Trigonometric Approximations")
    print("=" * 45)
    
    def factorial(n):
        """Calculate factorial"""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def sin_taylor(x, terms=10):
        """
        Calculate sin(x) using Taylor series:
        sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
        """
        # Normalize x to [-π, π] for better convergence
        while x > math.pi:
            x -= 2 * math.pi
        while x < -math.pi:
            x += 2 * math.pi
        
        result = 0
        for n in range(terms):
            power = 2 * n + 1
            sign = (-1) ** n
            term = sign * (x ** power) / factorial(power)
            result += term
        
        return result
    
    def cos_taylor(x, terms=10):
        """
        Calculate cos(x) using Taylor series:
        cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
        """
        # Normalize x to [-π, π] for better convergence
        while x > math.pi:
            x -= 2 * math.pi
        while x < -math.pi:
            x += 2 * math.pi
        
        result = 0
        for n in range(terms):
            power = 2 * n
            sign = (-1) ** n
            term = sign * (x ** power) / factorial(power)
            result += term
        
        return result
    
    def tan_taylor(x, terms=10):
        """Calculate tan(x) = sin(x) / cos(x)"""
        sin_val = sin_taylor(x, terms)
        cos_val = cos_taylor(x, terms)
        
        if abs(cos_val) < 1e-10:
            return float('inf') if sin_val > 0 else float('-inf')
        
        return sin_val / cos_val
    
    # Test accuracy with different numbers of terms
    test_angles = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi]
    
    print(f"Accuracy comparison with different numbers of terms:")
    print(f"{'Angle':>10} {'Built-in':>12} {'5 terms':>12} {'10 terms':>12} {'15 terms':>12}")
    print("-" * 62)
    
    for angle in test_angles:
        builtin_sin = math.sin(angle)
        taylor_5 = sin_taylor(angle, 5)
        taylor_10 = sin_taylor(angle, 10)
        taylor_15 = sin_taylor(angle, 15)
        
        print(f"{angle:10.3f} {builtin_sin:12.6f} {taylor_5:12.6f} {taylor_10:12.6f} {taylor_15:12.6f}")
    
    # Error analysis
    print(f"\nError analysis for sin(π/4) ≈ 0.707107:")
    exact_value = math.sin(math.pi/4)
    
    for terms in range(1, 11):
        approx_value = sin_taylor(math.pi/4, terms)
        error = abs(exact_value - approx_value)
        print(f"  {terms:2d} terms: {approx_value:.8f}, error: {error:.2e}")
    
    # Convergence visualization
    plt.figure(figsize=(12, 8))
    
    # Taylor series convergence for sin(x)
    plt.subplot(2, 2, 1)
    x_vals = np.linspace(-2*np.pi, 2*np.pi, 1000)
    exact_sin = np.sin(x_vals)
    
    plt.plot(x_vals, exact_sin, 'k-', linewidth=3, label='Exact sin(x)')
    
    for terms in [3, 5, 7, 9]:
        taylor_approx = [sin_taylor(x, terms) for x in x_vals]
        plt.plot(x_vals, taylor_approx, '--', linewidth=2, label=f'{terms} terms')
    
    plt.title('Taylor Series Convergence for sin(x)')
    plt.xlabel('x (radians)')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2, 2)
    
    # Error vs number of terms
    plt.subplot(2, 2, 2)
    terms_range = range(1, 21)
    test_angle = math.pi/3
    exact_val = math.sin(test_angle)
    
    errors = []
    for terms in terms_range:
        approx_val = sin_taylor(test_angle, terms)
        error = abs(exact_val - approx_val)
        errors.append(error)
    
    plt.semilogy(terms_range, errors, 'bo-', linewidth=2, markersize=6)
    plt.title(f'Error vs Number of Terms (x = π/3)')
    plt.xlabel('Number of Terms')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Comparison of all three functions
    plt.subplot(2, 1, 2)
    x_range = np.linspace(-np.pi, np.pi, 200)
    
    sin_exact = np.sin(x_range)
    cos_exact = np.cos(x_range)
    
    sin_approx = [sin_taylor(x, 10) for x in x_range]
    cos_approx = [cos_taylor(x, 10) for x in x_range]
    
    plt.plot(x_range, sin_exact, 'b-', linewidth=3, label='sin(x) exact')
    plt.plot(x_range, cos_exact, 'r-', linewidth=3, label='cos(x) exact')
    plt.plot(x_range, sin_approx, 'b--', linewidth=2, alpha=0.7, label='sin(x) Taylor (10 terms)')
    plt.plot(x_range, cos_approx, 'r--', linewidth=2, alpha=0.7, label='cos(x) Taylor (10 terms)')
    
    plt.title('Taylor Series vs Exact Values')
    plt.xlabel('x (radians)')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sin_taylor, cos_taylor

taylor_sin, taylor_cos = taylor_series_trigonometry()
```


## Why Trigonometric Identities Work

Trigonometric identities are fundamental relationships that arise from the geometric properties of the unit circle and provide powerful tools for simplifying expressions:

```python
def explain_trigonometric_identities():
    """Demonstrate why key trigonometric identities work"""
    
    print("Understanding Trigonometric Identities")
    print("=" * 40)
    
    def verify_pythagorean_identity():
        """Verify sin²(θ) + cos²(θ) = 1 geometrically"""
        
        print("1. Pythagorean Identity: sin²(θ) + cos²(θ) = 1")
        print("   Geometric proof: Point on unit circle has distance 1 from origin")
        
        test_angles = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3]
        
        print(f"{'Angle':>10} {'sin(θ)':>10} {'cos(θ)':>10} {'sin²+cos²':>12}")
        print("-" * 45)
        
        for theta in test_angles:
            sin_val = math.sin(theta)
            cos_val = math.cos(theta)
            identity_result = sin_val**2 + cos_val**2
            
            print(f"{theta:10.3f} {sin_val:10.3f} {cos_val:10.3f} {identity_result:12.6f}")
        
        # Geometric visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Unit circle with right triangle
        theta = math.pi/6  # 30 degrees
        ax1.add_patch(plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2))
        
        # Point on circle
        x, y = math.cos(theta), math.sin(theta)
        ax1.plot(x, y, 'ro', markersize=10, label=f'({x:.3f}, {y:.3f})')
        
        # Right triangle
        ax1.plot([0, x], [0, 0], 'b-', linewidth=2, label=f'cos(θ) = {x:.3f}')
        ax1.plot([x, x], [0, y], 'r-', linewidth=2, label=f'sin(θ) = {y:.3f}')
        ax1.plot([0, x], [0, y], 'k-', linewidth=3, label='radius = 1')
        
        # Angle arc
        arc_angles = np.linspace(0, theta, 50)
        arc_x = 0.3 * np.cos(arc_angles)
        arc_y = 0.3 * np.sin(arc_angles)
        ax1.plot(arc_x, arc_y, 'g-', linewidth=2, label=f'θ = {theta:.3f}')
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Unit Circle: sin²(θ) + cos²(θ) = 1')
        
        # Verification across all angles
        full_angles = np.linspace(0, 2*math.pi, 1000)
        identity_values = np.sin(full_angles)**2 + np.cos(full_angles)**2
        
        ax2.plot(full_angles, identity_values, 'b-', linewidth=2)
        ax2.axhline(y=1, color='r', linestyle='--', linewidth=2, label='y = 1')
        ax2.set_xlabel('Angle (radians)')
        ax2.set_ylabel('sin²(θ) + cos²(θ)')
        ax2.set_title('Pythagorean Identity Verification')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0.99, 1.01)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_angle_addition():
        """Demonstrate angle addition formulas"""
        
        print(f"\n2. Angle Addition Formulas:")
        print("   sin(A + B) = sin(A)cos(B) + cos(A)sin(B)")
        print("   cos(A + B) = cos(A)cos(B) - sin(A)sin(B)")
        
        A, B = math.pi/6, math.pi/4  # 30° and 45°
        
        # Direct calculation
        sin_sum_direct = math.sin(A + B)
        cos_sum_direct = math.cos(A + B)
        
        # Using addition formulas
        sin_A, cos_A = math.sin(A), math.cos(A)
        sin_B, cos_B = math.sin(B), math.cos(B)
        
        sin_sum_formula = sin_A * cos_B + cos_A * sin_B
        cos_sum_formula = cos_A * cos_B - sin_A * sin_B
        
        print(f"\n   A = π/6 = 30°, B = π/4 = 45°")
        print(f"   A + B = 5π/12 = 75°")
        print(f"\n   sin(A + B):")
        print(f"     Direct: {sin_sum_direct:.6f}")
        print(f"     Formula: {sin_sum_formula:.6f}")
        print(f"     Difference: {abs(sin_sum_direct - sin_sum_formula):.2e}")
        
        print(f"\n   cos(A + B):")
        print(f"     Direct: {cos_sum_direct:.6f}")
        print(f"     Formula: {cos_sum_formula:.6f}")
        print(f"     Difference: {abs(cos_sum_direct - cos_sum_formula):.2e}")
        
        # Geometric interpretation
        plt.figure(figsize=(10, 8))
        
        # Unit circle with angle addition
        circle_angles = np.linspace(0, 2*math.pi, 100)
        circle_x = np.cos(circle_angles)
        circle_y = np.sin(circle_angles)
        
        plt.plot(circle_x, circle_y, 'k-', linewidth=2, label='Unit Circle')
        
        # Points for angles A, B, and A+B
        point_A = (math.cos(A), math.sin(A))
        point_B = (math.cos(B), math.sin(B))
        point_sum = (math.cos(A + B), math.sin(A + B))
        
        plt.plot(*point_A, 'ro', markersize=10, label=f'A = {math.degrees(A):.0f}°')
        plt.plot(*point_B, 'go', markersize=10, label=f'B = {math.degrees(B):.0f}°')
        plt.plot(*point_sum, 'bo', markersize=10, label=f'A+B = {math.degrees(A+B):.0f}°')
        
        # Radial lines
        plt.plot([0, point_A[0]], [0, point_A[1]], 'r-', linewidth=2, alpha=0.7)
        plt.plot([0, point_B[0]], [0, point_B[1]], 'g-', linewidth=2, alpha=0.7)
        plt.plot([0, point_sum[0]], [0, point_sum[1]], 'b-', linewidth=3)
        
        # Angle arcs
        arc_A = np.linspace(0, A, 30)
        plt.plot(0.3 * np.cos(arc_A), 0.3 * np.sin(arc_A), 'r-', linewidth=2)
        
        arc_B = np.linspace(A, A + B, 30)
        plt.plot(0.4 * np.cos(arc_B), 0.4 * np.sin(arc_B), 'g-', linewidth=2)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.aspect('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('Angle Addition on Unit Circle')
        plt.show()
    
    def show_double_angle_formulas():
        """Demonstrate double angle formulas"""
        
        print(f"\n3. Double Angle Formulas:")
        print("   sin(2θ) = 2sin(θ)cos(θ)")
        print("   cos(2θ) = cos²(θ) - sin²(θ) = 2cos²(θ) - 1 = 1 - 2sin²(θ)")
        
        test_angles = [math.pi/12, math.pi/6, math.pi/4, math.pi/3, math.pi/2]
        
        print(f"\n   {'θ (deg)':>8} {'sin(2θ)':>12} {'2sin(θ)cos(θ)':>15} {'cos(2θ)':>12} {'cos²-sin²':>12}")
        print("-" * 65)
        
        for theta in test_angles:
            # Direct calculation
            sin_2theta = math.sin(2 * theta)
            cos_2theta = math.cos(2 * theta)
            
            # Using double angle formulas
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            sin_2theta_formula = 2 * sin_theta * cos_theta
            cos_2theta_formula = cos_theta**2 - sin_theta**2
            
            print(f"{math.degrees(theta):8.0f} {sin_2theta:12.6f} {sin_2theta_formula:15.6f} {cos_2theta:12.6f} {cos_2theta_formula:12.6f}")
    
    # Run all demonstrations
    verify_pythagorean_identity()
    demonstrate_angle_addition()
    show_double_angle_formulas()
    
    print(f"\nKey Insights:")
    print(f"• Pythagorean identity comes from distance formula on unit circle")
    print(f"• Angle addition formulas enable calculation of any angle")
    print(f"• Double angle formulas are special cases of addition formulas")
    print(f"• All identities are geometrically motivated by unit circle")

explain_trigonometric_identities()
```

## Common Trigonometric Function Patterns

Standard trigonometric patterns and identities that appear frequently in mathematics and programming:

- **Pythagorean Identity:**\
  \(\sin^2(\theta) + \cos^2(\theta) = 1\)

- **Angle Addition Formulas:**\
  \(\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)\)

- **Double Angle Formulas:**\
  \(\sin(2\theta) = 2\sin(\theta)\cos(\theta)\)

- **Phase Relationships:**\
  \(\cos(\theta) = \sin(\theta + \frac{\pi}{2})\)

Python implementations demonstrating these patterns:

```python
def trigonometric_patterns_library():
    """Collection of common trigonometric patterns and calculations"""
    
    def basic_trigonometric_functions():
        """Demonstrate basic trigonometric function calculations"""
        
        print("Basic Trigonometric Functions in Python")
        print("=" * 45)
        
        # Convert between degrees and radians
        def deg_to_rad(degrees):
            return degrees * math.pi / 180
        
        def rad_to_deg(radians):
            return radians * 180 / math.pi
        
        # Test angles in degrees
        test_angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
        
        print(f"{'Angle (°)':>8} {'Radians':>10} {'sin':>8} {'cos':>8} {'tan':>10}")
        print("-" * 50)
        
        for angle_deg in test_angles_deg:
            angle_rad = deg_to_rad(angle_deg)
            
            sin_val = math.sin(angle_rad)
            cos_val = math.cos(angle_rad)
            
            # Handle tan(90°) and tan(270°) special cases
            if abs(cos_val) < 1e-10:  # cos ≈ 0
                tan_val = "undefined"
            else:
                tan_val = f"{math.tan(angle_rad):8.3f}"
            
            print(f"{angle_deg:>6} {angle_rad:>10.3f} {sin_val:>8.3f} {cos_val:>8.3f} {tan_val:>10}")
        
        return test_angles_deg
    
    def verify_trigonometric_identities():
        """Verify important trigonometric identities"""
        
        print("\nTrigonometric Identity Verification")
        print("=" * 40)
        
        # Test angles
        test_angles = [math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3]
        
        print("1. Pythagorean Identity: sin²(θ) + cos²(θ) = 1")
        print(f"{'Angle':>10} {'sin²+cos²':>12} {'Error':>10}")
        print("-" * 35)
        
        for theta in test_angles:
            sin_val = math.sin(theta)
            cos_val = math.cos(theta)
            identity_result = sin_val**2 + cos_val**2
            error = abs(identity_result - 1)
            
            print(f"{theta:10.3f} {identity_result:12.8f} {error:10.2e}")
        
        print("\n2. Angle Addition Formulas:")
        print("sin(A + B) = sin(A)cos(B) + cos(A)sin(B)")
        print("cos(A + B) = cos(A)cos(B) - sin(A)sin(B)")
        
        A, B = math.pi/6, math.pi/4  # 30° and 45°
        
        # Direct calculation
        sin_sum_direct = math.sin(A + B)
        cos_sum_direct = math.cos(A + B)
        
        # Using addition formulas
        sin_sum_formula = math.sin(A)*math.cos(B) + math.cos(A)*math.sin(B)
        cos_sum_formula = math.cos(A)*math.cos(B) - math.sin(A)*math.sin(B)
        
        print(f"\nA = π/6, B = π/4, A + B = π/6 + π/4 = 5π/12")
        print(f"sin(A + B) direct: {sin_sum_direct:.6f}")
        print(f"sin(A + B) formula: {sin_sum_formula:.6f}")
        print(f"cos(A + B) direct: {cos_sum_direct:.6f}")
        print(f"cos(A + B) formula: {cos_sum_formula:.6f}")
        
        print("\n3. Double Angle Formulas:")
        print("sin(2θ) = 2sin(θ)cos(θ)")
        print("cos(2θ) = cos²(θ) - sin²(θ)")
        
        theta = math.pi/6
        sin_double_direct = math.sin(2 * theta)
        cos_double_direct = math.cos(2 * theta)
        
        sin_double_formula = 2 * math.sin(theta) * math.cos(theta)
        cos_double_formula = math.cos(theta)**2 - math.sin(theta)**2
        
        print(f"\nθ = π/6")
        print(f"sin(2θ) direct: {sin_double_direct:.6f}")
        print(f"sin(2θ) formula: {sin_double_formula:.6f}")
        print(f"cos(2θ) direct: {cos_double_direct:.6f}")
        print(f"cos(2θ) formula: {cos_double_formula:.6f}")
        
        return A, B, theta
    
    def inverse_trigonometric_functions():
        """Explore inverse trigonometric functions and their applications"""
        
        print("\nInverse Trigonometric Functions")
        print("=" * 40)
        
        # Test values
        test_values = [-1, -0.866, -0.707, -0.5, 0, 0.5, 0.707, 0.866, 1]
        
        print(f"{'x':>6} {'arcsin(x)':>12} {'arccos(x)':>12} {'arctan(x)':>12}")
        print(f"{'':>6} {'(radians)':>12} {'(radians)':>12} {'(radians)':>12}")
        print("-" * 50)
        
        for x in test_values:
            # arcsin and arccos only defined for |x| ≤ 1
            if abs(x) <= 1:
                arcsin_val = math.asin(x)
                arccos_val = math.acos(x)
                arcsin_str = f"{arcsin_val:8.3f}"
                arccos_str = f"{arccos_val:8.3f}"
            else:
                arcsin_str = "undefined"
                arccos_str = "undefined"
            
            # arctan defined for all real numbers
            arctan_val = math.atan(x)
            arctan_str = f"{arctan_val:8.3f}"
            
            print(f"{x:6.3f} {arcsin_str:>12} {arccos_str:>12} {arctan_str:>12}")
        
        # Domain and range
        print(f"\nDomain and Range:")
        print(f"• arcsin(x): Domain [-1, 1], Range [-π/2, π/2]")
        print(f"• arccos(x): Domain [-1, 1], Range [0, π]")
        print(f"• arctan(x): Domain (-∞, ∞), Range (-π/2, π/2)")
        
        # Application: Finding angles in triangles
        print(f"\nTriangle Application:")
        print(f"Right triangle with sides: opposite = 3, adjacent = 4, hypotenuse = 5")
        
        opposite = 3
        adjacent = 4
        hypotenuse = 5
        
        # Find angles using inverse trig functions
        angle_A_sin = math.asin(opposite / hypotenuse)
        angle_A_cos = math.acos(adjacent / hypotenuse)
        angle_A_tan = math.atan(opposite / adjacent)
        
        angle_A_deg = math.degrees(angle_A_sin)
        
        print(f"Angle A (opposite side = 3):")
        print(f"  Using arcsin: {angle_A_sin:.4f} rad = {angle_A_deg:.1f}°")
        print(f"  Using arccos: {angle_A_cos:.4f} rad = {math.degrees(angle_A_cos):.1f}°")
        print(f"  Using arctan: {angle_A_tan:.4f} rad = {math.degrees(angle_A_tan):.1f}°")
        
        # Verify: All should give the same angle
        print(f"\nVerification: All methods give same result? {abs(angle_A_sin - angle_A_cos) < 1e-10}")
        
        return test_values
    
    # Run all demonstrations
    test_angles = basic_trigonometric_functions()
    A, B, theta = verify_trigonometric_identities()
    test_values = inverse_trigonometric_functions()
    
    return test_angles, A, B, theta, test_values

trigonometric_patterns_library()
```


## Practical Real-world Applications

Trigonometric functions aren't just abstract mathematics - they're the foundation for modeling waves, rotations, and periodic phenomena across multiple domains:

### Application 1: Wave Analysis and Signal Processing

```python
def wave_analysis():
    """Analyze wave phenomena using trigonometric functions"""
    
    print("Wave Analysis with Trigonometry")
    print("=" * 40)
    
    # Sound wave example
    def sound_wave(t, frequency, amplitude=1, phase=0):
        """Generate sound wave: A·sin(2πft + φ)"""
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Parameters
    duration = 1.0  # seconds
    sample_rate = 1000  # samples per second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Different frequencies
    freq_low = 2  # 2 Hz
    freq_high = 5  # 5 Hz
    
    wave_low = sound_wave(t, freq_low)
    wave_high = sound_wave(t, freq_high)
    wave_combined = 0.5 * (wave_low + wave_high)
    
    print(f"Wave Properties:")
    print(f"• Period of {freq_low} Hz wave: {1/freq_low:.2f} seconds")
    print(f"• Period of {freq_high} Hz wave: {1/freq_high:.2f} seconds")
    print(f"• Wavelength = speed / frequency")
    print(f"• Phase determines starting position of wave")
    
    # Amplitude modulation example
    def am_wave(t, carrier_freq, modulation_freq, modulation_depth=0.5):
        """Amplitude modulated wave"""
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulation = 1 + modulation_depth * np.sin(2 * np.pi * modulation_freq * t)
        return modulation * carrier
    
    # Create AM wave
    am_signal = am_wave(t, 20, 1)  # 20 Hz carrier, 1 Hz modulation
    
    # Frequency modulation example  
    def fm_wave(t, carrier_freq, modulation_freq, frequency_deviation):
        """Frequency modulated wave"""
        modulation = frequency_deviation * np.sin(2 * np.pi * modulation_freq * t)
        instantaneous_phase = 2 * np.pi * carrier_freq * t + modulation
        return np.sin(instantaneous_phase)
    
    # Create FM wave
    fm_signal = fm_wave(t, 20, 1, 5)  # 20 Hz carrier, 1 Hz mod, 5 Hz deviation
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Individual waves
    plt.subplot(3, 2, 1)
    plt.plot(t[:200], wave_low[:200], 'b-', linewidth=2, label=f'{freq_low} Hz')
    plt.plot(t[:200], wave_high[:200], 'r-', linewidth=2, label=f'{freq_high} Hz')
    plt.title('Individual Sine Waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined wave
    plt.subplot(3, 2, 2)
    plt.plot(t[:200], wave_combined[:200], 'g-', linewidth=2, label='Combined Wave')
    plt.title('Superposition of Waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Amplitude modulation
    plt.subplot(3, 2, 3)
    plt.plot(t[:500], am_signal[:500], 'purple', linewidth=1.5, label='AM Signal')
    plt.title('Amplitude Modulated Wave')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency modulation
    plt.subplot(3, 2, 4)
    plt.plot(t[:500], fm_signal[:500], 'orange', linewidth=1.5, label='FM Signal')
    plt.title('Frequency Modulated Wave')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency spectrum (simplified)
    plt.subplot(3, 2, 5)
    frequencies = [1, 2, 3, 4, 5, 6, 7]
    amplitudes = [0, 1, 0, 0, 1, 0, 0]  # Only 2 Hz and 5 Hz present
    
    plt.stem(frequencies, amplitudes, basefmt=' ')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 8)
    
    # Phase relationships
    plt.subplot(3, 2, 6)
    t_phase = np.linspace(0, 2*np.pi, 100)
    sin_wave = np.sin(t_phase)
    cos_wave = np.cos(t_phase)
    sin_shifted = np.sin(t_phase + np.pi/2)
    
    plt.plot(t_phase, sin_wave, 'b-', linewidth=2, label='sin(t)')
    plt.plot(t_phase, cos_wave, 'r-', linewidth=2, label='cos(t)')
    plt.plot(t_phase, sin_shifted, 'g--', linewidth=2, label='sin(t + π/2)')
    plt.title('Phase Relationships')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return wave_low, wave_high, am_signal, fm_signal

wave_analysis()
```

### Application 2: Computer Graphics and Animation

```python
def computer_graphics_applications():
    """Demonstrate trigonometric functions in computer graphics"""
    
    print("Computer Graphics and Animation Applications")
    print("=" * 50)
    
    def circular_motion_animation():
        """Model circular motion for animation"""
        
        # Parameters
        radius = 3  # meters
        angular_velocity = 2  # radians per second
        total_time = 2 * np.pi / angular_velocity  # One complete revolution
        
        t = np.linspace(0, total_time, 100)
        
        # Position as function of time
        def position(t, r, omega):
            """Position on circle: (r·cos(ωt), r·sin(ωt))"""
            x = r * np.cos(omega * t)
            y = r * np.sin(omega * t)
            return x, y
        
        # Velocity (derivative of position)
        def velocity(t, r, omega):
            """Velocity: (-rω·sin(ωt), rω·cos(ωt))"""
            vx = -r * omega * np.sin(omega * t)
            vy = r * omega * np.cos(omega * t)
            return vx, vy
        
        # Calculate positions and velocities
        x, y = position(t, radius, angular_velocity)
        vx, vy = velocity(t, radius, angular_velocity)
        
        print(f"Circular Motion Parameters:")
        print(f"• Radius: {radius} m")
        print(f"• Angular velocity: {angular_velocity} rad/s")
        print(f"• Period: {total_time:.2f} s")
        print(f"• Linear speed: {radius * angular_velocity} m/s")
        
        return x, y, vx, vy, t
    
    def rotation_transformations():
        """2D rotation transformations using trigonometry"""
        
        print(f"\n2D Rotation Transformations:")
        
        # Original points (square)
        original_points = np.array([
            [1, 1],
            [-1, 1], 
            [-1, -1],
            [1, -1],
            [1, 1]  # Close the shape
        ])
        
        def rotate_2d(points, angle):
            """Rotate points by angle using rotation matrix"""
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # Rotation matrix
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            
            # Apply rotation
            rotated = points @ rotation_matrix.T
            return rotated
        
        # Rotate by different angles
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        
        plt.figure(figsize=(15, 10))
        
        # Circular motion visualization
        plt.subplot(2, 3, 1)
        x, y, vx, vy, t = circular_motion_animation()
        
        plt.plot(x, y, 'b-', linewidth=2, label='Path')
        plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
        plt.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
        
        # Add velocity vectors at some points
        for i in range(0, len(t), 20):
            plt.arrow(x[i], y[i], vx[i]*0.1, vy[i]*0.1, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        plt.title('Circular Motion Path')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Position vs time
        plt.subplot(2, 3, 2)
        plt.plot(t, x, 'b-', linewidth=2, label='x(t) = r·cos(ωt)')
        plt.plot(t, y, 'r-', linewidth=2, label='y(t) = r·sin(ωt)')
        plt.title('Position Components vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotation transformations
        plt.subplot(2, 3, 3)
        colors = ['black', 'blue', 'green', 'orange', 'red']
        
        for i, angle in enumerate(angles):
            rotated_points = rotate_2d(original_points, angle)
            plt.plot(rotated_points[:, 0], rotated_points[:, 1], 
                    color=colors[i], linewidth=2, 
                    label=f'{np.degrees(angle):.0f}°')
        
        plt.title('2D Rotation Transformations')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        
        # Sine and cosine for animation timing
        plt.subplot(2, 3, 4)
        animation_t = np.linspace(0, 4*np.pi, 200)
        
        # Different easing functions using trigonometry
        linear = animation_t / (4*np.pi)
        ease_in_out = 0.5 * (1 - np.cos(np.pi * linear))
        bounce = np.abs(np.sin(4 * animation_t))
        
        plt.plot(linear, linear, 'k-', linewidth=2, label='Linear')
        plt.plot(linear, ease_in_out, 'b-', linewidth=2, label='Ease In-Out (cosine)')
        plt.plot(linear, bounce, 'r-', linewidth=2, label='Bounce (sine)')
        
        plt.title('Animation Easing Functions')
        plt.xlabel('Time Progress (0 to 1)')
        plt.ylabel('Animation Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Polar plotting
        plt.subplot(2, 3, 5)
        theta = np.linspace(0, 4*np.pi, 1000)
        
        # Different polar equations
        r1 = 1 + 0.5 * np.cos(3 * theta)  # Rose curve
        r2 = np.exp(-0.1 * theta)         # Spiral
        
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        x2 = r2 * np.cos(theta)
        y2 = r2 * np.sin(theta)
        
        plt.plot(x1, y1, 'purple', linewidth=2, label='Rose: r = 1 + 0.5cos(3θ)')
        plt.plot(x2, y2, 'orange', linewidth=2, label='Spiral: r = e^(-0.1θ)')
        
        plt.title('Polar Coordinate Curves')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Wave interference pattern
        plt.subplot(2, 3, 6)
        x_grid = np.linspace(-5, 5, 100)
        y_grid = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Two wave sources
        source1_x, source1_y = -2, 0
        source2_x, source2_y = 2, 0
        
        # Distance from each source
        r1 = np.sqrt((X - source1_x)**2 + (Y - source1_y)**2)
        r2 = np.sqrt((X - source2_x)**2 + (Y - source2_y)**2)
        
        # Wave interference
        wave_freq = 2
        interference = np.sin(wave_freq * r1) + np.sin(wave_freq * r2)
        
        plt.contour(X, Y, interference, levels=20, cmap='RdBu')
        plt.plot(source1_x, source1_y, 'ro', markersize=10, label='Source 1')
        plt.plot(source2_x, source2_y, 'bo', markersize=10, label='Source 2')
        plt.title('Wave Interference Pattern')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nKey Observations:")
        print(f"• Position components are sinusoidal with 90° phase difference")
        print(f"• Rotation matrices use cos and sin for coordinate transformation")
        print(f"• Trigonometric functions create smooth animation easing")
        print(f"• Wave interference creates complex patterns from simple sine waves")
        
        return original_points, angles
    
    original_points, angles = rotation_transformations()
    
    return original_points, angles

computer_graphics_applications()
```

### Application 3: Physics Simulations and Oscillations

```python
def physics_simulations():
    """Model physical systems using trigonometric functions"""
    
    print("Physics Simulations with Trigonometry")
    print("=" * 45)
    
    def simple_harmonic_motion():
        """Model simple harmonic motion (springs, pendulums)"""
        
        print("Simple Harmonic Motion:")
        print("x(t) = A·cos(ωt + φ)")
        
        # Parameters
        amplitude = 2.0  # meters
        frequency = 0.5  # Hz
        omega = 2 * np.pi * frequency  # angular frequency
        phase = 0  # phase shift
        
        # Time array
        t = np.linspace(0, 4/frequency, 1000)  # 4 periods
        
        # Position, velocity, and acceleration
        position = amplitude * np.cos(omega * t + phase)
        velocity = -amplitude * omega * np.sin(omega * t + phase)
        acceleration = -amplitude * omega**2 * np.cos(omega * t + phase)
        
        print(f"• Amplitude: {amplitude} m")
        print(f"• Frequency: {frequency} Hz")
        print(f"• Period: {1/frequency} s")
        print(f"• Angular frequency: {omega:.3f} rad/s")
        
        return t, position, velocity, acceleration
    
    def damped_harmonic_motion():
        """Model damped harmonic motion with energy loss"""
        
        print(f"\nDamped Harmonic Motion:")
        print("x(t) = A·e^(-γt)·cos(ω't + φ)")
        
        # Parameters
        amplitude = 2.0
        omega0 = 3.0  # natural frequency
        gamma = 0.5   # damping coefficient
        omega_d = np.sqrt(omega0**2 - gamma**2)  # damped frequency
        
        t = np.linspace(0, 10, 1000)
        
        # Different damping regimes
        # Underdamped
        x_underdamped = amplitude * np.exp(-gamma * t) * np.cos(omega_d * t)
        
        # Critically damped
        gamma_critical = omega0
        x_critical = amplitude * (1 + gamma_critical * t) * np.exp(-gamma_critical * t)
        
        # Overdamped
        gamma_over = 2 * omega0
        r1 = -gamma_over + np.sqrt(gamma_over**2 - omega0**2)
        r2 = -gamma_over - np.sqrt(gamma_over**2 - omega0**2)
        x_overdamped = amplitude * (np.exp(r1 * t) + np.exp(r2 * t)) / 2
        
        return t, x_underdamped, x_critical, x_overdamped
    
    def wave_propagation():
        """Model wave propagation in space and time"""
        
        print(f"\nWave Propagation:")
        print("y(x,t) = A·sin(kx - ωt)")
        
        # Parameters
        amplitude = 1.0
        wavelength = 2.0  # meters
        k = 2 * np.pi / wavelength  # wave number
        frequency = 1.0  # Hz
        omega = 2 * np.pi * frequency
        wave_speed = omega / k  # v = ω/k
        
        print(f"• Wavelength: {wavelength} m")
        print(f"• Frequency: {frequency} Hz")
        print(f"• Wave speed: {wave_speed} m/s")
        
        # Spatial and temporal grids
        x = np.linspace(0, 10, 200)
        t_snapshots = np.linspace(0, 2, 5)
        
        # Wave at different times
        waves = []
        for t_val in t_snapshots:
            y = amplitude * np.sin(k * x - omega * t_val)
            waves.append(y)
        
        return x, t_snapshots, waves, wave_speed
    
    def coupled_oscillators():
        """Model coupled harmonic oscillators"""
        
        print(f"\nCoupled Oscillators:")
        print("Two masses connected by springs")
        
        # Parameters
        m1, m2 = 1.0, 1.0  # masses
        k1, k2, k_coupling = 1.0, 1.0, 0.5  # spring constants
        
        # Natural frequencies
        omega1 = np.sqrt(k1 / m1)
        omega2 = np.sqrt(k2 / m2)
        
        # Normal mode frequencies (for equal masses and springs)
        omega_plus = np.sqrt((k1 + k2 + 2*k_coupling) / m1)  # Symmetric mode
        omega_minus = np.sqrt((k1 + k2) / m1)  # Antisymmetric mode
        
        t = np.linspace(0, 20, 1000)
        
        # Initial conditions: mass 1 displaced, mass 2 at rest
        A1, A2 = 1.0, 0.0
        
        # Coupled motion (simplified for equal masses)
        x1 = (A1/2) * (np.cos(omega_minus * t) + np.cos(omega_plus * t))
        x2 = (A1/2) * (np.cos(omega_minus * t) - np.cos(omega_plus * t))
        
        print(f"• Natural frequency 1: {omega1:.3f} rad/s")
        print(f"• Natural frequency 2: {omega2:.3f} rad/s")
        print(f"• Normal mode frequencies: {omega_minus:.3f}, {omega_plus:.3f} rad/s")
        
        return t, x1, x2, omega_minus, omega_plus
    
    # Run all simulations
    t_shm, pos, vel, acc = simple_harmonic_motion()
    t_damp, x_under, x_crit, x_over = damped_harmonic_motion()
    x_wave, t_snaps, waves, v_wave = wave_propagation()
    t_coupled, x1_coupled, x2_coupled, omega_minus, omega_plus = coupled_oscillators()
    
    # Comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Simple harmonic motion
    plt.subplot(3, 3, 1)
    plt.plot(t_shm, pos, 'b-', linewidth=2, label='Position')
    plt.plot(t_shm, vel, 'r-', linewidth=2, label='Velocity')
    plt.plot(t_shm, acc, 'g-', linewidth=2, label='Acceleration')
    plt.title('Simple Harmonic Motion')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase space plot
    plt.subplot(3, 3, 2)
    plt.plot(pos, vel, 'purple', linewidth=2)
    plt.title('Phase Space (Position vs Velocity)')
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Damped harmonic motion
    plt.subplot(3, 3, 3)
    plt.plot(t_damp, x_under, 'b-', linewidth=2, label='Underdamped')
    plt.plot(t_damp, x_crit, 'r-', linewidth=2, label='Critical')
    plt.plot(t_damp, x_over, 'g-', linewidth=2, label='Overdamped')
    plt.title('Damped Harmonic Motion')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wave propagation snapshots
    plt.subplot(3, 3, 4)
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (t_val, wave) in enumerate(zip(t_snaps, waves)):
        plt.plot(x_wave, wave, color=colors[i], linewidth=2, 
                label=f't = {t_val:.1f} s')
    plt.title('Wave Propagation Snapshots')
    plt.xlabel('Position (m)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Coupled oscillators
    plt.subplot(3, 3, 5)
    plt.plot(t_coupled, x1_coupled, 'b-', linewidth=2, label='Mass 1')
    plt.plot(t_coupled, x2_coupled, 'r-', linewidth=2, label='Mass 2')
    plt.title('Coupled Oscillators')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Energy in simple harmonic motion
    plt.subplot(3, 3, 6)
    # Assuming unit mass and spring constant for simplicity
    kinetic_energy = 0.5 * vel**2
    potential_energy = 0.5 * pos**2
    total_energy = kinetic_energy + potential_energy
    
    plt.plot(t_shm, kinetic_energy, 'r-', linewidth=2, label='Kinetic')
    plt.plot(t_shm, potential_energy, 'b-', linewidth=2, label='Potential')
    plt.plot(t_shm, total_energy, 'k-', linewidth=2, label='Total')
    plt.title('Energy in Simple Harmonic Motion')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency spectrum of damped oscillation
    plt.subplot(3, 3, 7)
    # Simple frequency analysis
    fft_under = np.fft.fft(x_under)
    freqs = np.fft.fftfreq(len(t_damp), t_damp[1] - t_damp[0])
    
    # Plot only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = np.abs(fft_under)[:len(freqs)//2]
    
    plt.plot(positive_freqs, magnitude, 'b-', linewidth=2)
    plt.title('Frequency Spectrum (Underdamped)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    
    # Normal modes visualization
    plt.subplot(3, 3, 8)
    t_mode = np.linspace(0, 4*np.pi/omega_minus, 100)
    mode1 = np.cos(omega_minus * t_mode)  # Symmetric mode
    mode2 = np.cos(omega_plus * t_mode)   # Antisymmetric mode
    
    plt.plot(t_mode, mode1, 'b-', linewidth=2, label=f'Mode 1 ({omega_minus:.1f} rad/s)')
    plt.plot(t_mode, mode2, 'r-', linewidth=2, label=f'Mode 2 ({omega_plus:.1f} rad/s)')
    plt.title('Normal Modes of Coupled System')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3D wave surface
    ax = plt.subplot(3, 3, 9, projection='3d')
    X_3d, T_3d = np.meshgrid(x_wave[:50], np.linspace(0, 2, 20))
    Z_3d = np.sin(2*np.pi*X_3d - 2*np.pi*T_3d)
    
    ax.plot_surface(X_3d, T_3d, Z_3d, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Amplitude')
    ax.set_title('3D Wave Propagation')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPhysics Insights:")
    print(f"• Simple harmonic motion: energy oscillates between kinetic and potential")
    print(f"• Damping reduces amplitude while preserving frequency (underdamped case)")
    print(f"• Wave speed relates frequency and wavelength: v = fλ")
    print(f"• Coupled systems have normal modes with characteristic frequencies")
    print(f"• Trigonometric functions model all oscillatory phenomena in physics")
    
    return t_shm, pos, vel, acc

physics_simulations()
```


## Try it Yourself

Ready to master trigonometric functions and the unit circle? Here are some hands-on challenges:

- **Interactive Unit Circle:** Build a dynamic unit circle explorer with angle input and real-time coordinate display.
- **Wave Synthesizer:** Create a digital audio synthesizer using trigonometric functions to generate different waveforms.
- **Animation Framework:** Develop a 2D animation system using trigonometric functions for smooth motion and easing.
- **Physics Simulator:** Build a spring-mass system simulator with real-time visualization of oscillations.
- **Signal Analyzer:** Implement a tool that analyzes periodic signals and identifies their frequency components.
- **Graphics Transformer:** Create a 2D graphics engine that performs rotations, scaling, and transformations using trigonometry.


## Key Takeaways

- The unit circle provides geometric intuition for trigonometric functions, connecting angles to coordinates.
- Trigonometric functions are periodic, making them perfect for modeling waves, rotations, and oscillations.
- sin and cos are fundamental - all other trigonometric functions can be expressed in terms of them.
- Trigonometric identities arise naturally from the geometric properties of the unit circle.
- These functions are essential for computer graphics, signal processing, physics simulations, and animation.
- Phase relationships between sin and cos enable complex transformations and smooth interpolations.
- Understanding both geometric and analytical perspectives deepens comprehension and application ability.


## Next Steps & Further Exploration

Ready to dive deeper into the trigonometric universe?

- Explore **Complex Numbers** and Euler's formula: \(e^{i\theta} = \cos\theta + i\sin\theta\) for advanced connections.
- Study **Fourier Series** to understand how any periodic function can be expressed as sums of sines and cosines.
- Learn **Differential Equations** to see how trigonometric functions naturally arise as solutions to oscillation problems.
- Investigate **Signal Processing** techniques that rely heavily on trigonometric transforms for analysis.
- Apply trigonometry to **3D Graphics** with quaternions and advanced rotation techniques.
- Explore **Wave Mechanics** and **Quantum Physics** where trigonometric functions describe fundamental phenomena.