# Linear Equations

## Mathematical Concept

A **linear equation** in one variable has the form $ax + b = 0$ where $a \neq 0$. The solution is $x = -\frac{b}{a}$.

For systems of linear equations, we have multiple equations with multiple unknowns:
$$\begin{cases}
a_1x + b_1y = c_1 \\
a_2x + b_2y = c_2
\end{cases}$$

## Interactive Linear System Solver

<LinearSystemSolver />

## Python Implementation

### Solving Single Linear Equations

```python
def solve_linear_equation(a, b):
    """
    Solve ax + b = 0
    Returns the solution or a message if no unique solution exists
    """
    if a == 0:
        if b == 0:
            return "Infinite solutions (0 = 0)"
        else:
            return f"No solution ({b} ≠ 0)"
    
    solution = -b / a
    return solution

# Examples
print(solve_linear_equation(3, -12))  # 3x - 12 = 0 → x = 4
print(solve_linear_equation(0, 5))    # 0x + 5 = 0 → No solution
print(solve_linear_equation(0, 0))    # 0x + 0 = 0 → Infinite solutions
```

### Systems of Linear Equations (2x2)

```python
import numpy as np

def solve_2x2_system(a1, b1, c1, a2, b2, c2):
    """
    Solve the system:
    a1*x + b1*y = c1
    a2*x + b2*y = c2
    """
    # Coefficient matrix and constants vector
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([c1, c2])
    
    # Check if system has unique solution
    det = np.linalg.det(A)
    
    if abs(det) < 1e-10:  # Essentially zero
        return "No unique solution (parallel lines or same line)"
    
    # Solve using matrix operations
    solution = np.linalg.solve(A, b)
    return solution

# Example: Solve the system
# 2x + 3y = 7
# x - y = 1
x, y = solve_2x2_system(2, 3, 7, 1, -1, 1)
print(f"Solution: x = {x:.2f}, y = {y:.2f}")

# Verify
print(f"Check equation 1: 2({x}) + 3({y}) = {2*x + 3*y}")
print(f"Check equation 2: {x} - {y} = {x - y}")
```

### Cramer's Rule Implementation

```python
def cramers_rule_2x2(a1, b1, c1, a2, b2, c2):
    """
    Solve 2x2 system using Cramer's rule
    """
    # Main determinant
    det_main = a1 * b2 - a2 * b1
    
    if abs(det_main) < 1e-10:
        return "No unique solution"
    
    # Determinants for x and y
    det_x = c1 * b2 - c2 * b1
    det_y = a1 * c2 - a2 * c1
    
    x = det_x / det_main
    y = det_y / det_main
    
    return x, y

# Example
result = cramers_rule_2x2(3, 2, 12, 1, -1, 1)
print(f"Cramer's rule solution: x = {result[0]}, y = {result[1]}")
```

## Word Problems

### Age Problems

```python
def solve_age_problem():
    """
    Problem: John is twice as old as Mary. In 5 years, the sum of their ages will be 50.
    Find their current ages.
    
    Let j = John's current age, m = Mary's current age
    Equation 1: j = 2m
    Equation 2: (j + 5) + (m + 5) = 50
    """
    # Substitute j = 2m into equation 2
    # (2m + 5) + (m + 5) = 50
    # 3m + 10 = 50
    # 3m = 40
    # m = 40/3
    
    m = 40 / 3
    j = 2 * m
    
    print(f"Mary's current age: {m:.1f} years")
    print(f"John's current age: {j:.1f} years")
    
    # Verify
    print(f"Verification:")
    print(f"John is twice Mary's age: {j} = 2 × {m} ✓")
    print(f"In 5 years: ({j} + 5) + ({m} + 5) = {(j + 5) + (m + 5)} = 50 ✓")

solve_age_problem()
```

### Mixture Problems

```python
def solve_mixture_problem():
    """
    Problem: How many gallons of a 20% acid solution and a 60% acid solution 
    should be mixed to get 40 gallons of a 35% acid solution?
    
    Let x = gallons of 20% solution
    Let y = gallons of 60% solution
    
    Equation 1 (total volume): x + y = 40
    Equation 2 (acid content): 0.20x + 0.60y = 0.35(40)
    """
    # From equation 1: y = 40 - x
    # Substitute into equation 2: 0.20x + 0.60(40 - x) = 14
    # 0.20x + 24 - 0.60x = 14
    # -0.40x = -10
    # x = 25
    
    x = 25  # gallons of 20% solution
    y = 40 - x  # gallons of 60% solution
    
    print(f"20% solution needed: {x} gallons")
    print(f"60% solution needed: {y} gallons")
    
    # Verify
    total_volume = x + y
    acid_from_20 = 0.20 * x
    acid_from_60 = 0.60 * y
    total_acid = acid_from_20 + acid_from_60
    final_concentration = total_acid / total_volume
    
    print(f"Verification:")
    print(f"Total volume: {total_volume} gallons ✓")
    print(f"Total acid: {total_acid} gallons")
    print(f"Final concentration: {final_concentration:.1%} ✓")

solve_mixture_problem()
```

## Matrix Methods for Larger Systems

```python
def solve_linear_system(A, b):
    """
    Solve the linear system Ax = b using numpy
    A: coefficient matrix
    b: constants vector
    """
    try:
        solution = np.linalg.solve(A, b)
        return solution
    except np.linalg.LinAlgError:
        return "System has no unique solution"

# Example: 3x3 system
# x + 2y + z = 6
# 2x + y - z = 1  
# x - y + z = 2

A = np.array([[1, 2, 1],
              [2, 1, -1],
              [1, -1, 1]])
b = np.array([6, 1, 2])

solution = solve_linear_system(A, b)
if isinstance(solution, str):
    print(solution)
else:
    x, y, z = solution
    print(f"Solution: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")
    
    # Verify each equation
    eq1 = x + 2*y + z
    eq2 = 2*x + y - z  
    eq3 = x - y + z
    print(f"Verification:")
    print(f"Equation 1: {x:.2f} + 2({y:.2f}) + {z:.2f} = {eq1:.2f}")
    print(f"Equation 2: 2({x:.2f}) + {y:.2f} - {z:.2f} = {eq2:.2f}")
    print(f"Equation 3: {x:.2f} - {y:.2f} + {z:.2f} = {eq3:.2f}")
```

## Applications in Real Life

### Economics: Supply and Demand

```python
def supply_demand_equilibrium():
    """
    Find market equilibrium where supply equals demand
    Supply: P = 2Q + 10 (price as function of quantity)
    Demand: P = -Q + 40
    """
    # At equilibrium: Supply = Demand
    # 2Q + 10 = -Q + 40
    # 3Q = 30
    # Q = 10
    
    Q_equilibrium = 10
    P_equilibrium = 2 * Q_equilibrium + 10
    
    print(f"Market Equilibrium:")
    print(f"Quantity: {Q_equilibrium} units")
    print(f"Price: ${P_equilibrium}")
    
    # Plot supply and demand curves
    Q = np.linspace(0, 50, 100)
    P_supply = 2 * Q + 10
    P_demand = -Q + 40
    
    plt.figure(figsize=(10, 6))
    plt.plot(Q, P_supply, 'b-', linewidth=2, label='Supply: P = 2Q + 10')
    plt.plot(Q, P_demand, 'r-', linewidth=2, label='Demand: P = -Q + 40')
    plt.plot(Q_equilibrium, P_equilibrium, 'go', markersize=10, label=f'Equilibrium: ({Q_equilibrium}, {P_equilibrium})')
    plt.xlabel('Quantity')
    plt.ylabel('Price ($)')
    plt.title('Supply and Demand Equilibrium')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.show()

supply_demand_equilibrium()
```

## Key Takeaways

1. **Linear equations** have at most one solution
2. **Systems** can have unique solutions, no solutions, or infinite solutions
3. **Matrix methods** efficiently solve larger systems
4. **Graphical interpretation** shows solutions as intersections
5. **Real applications** include economics, physics, and engineering
6. **Python tools** like NumPy make solving systems straightforward

## Next Steps

- Study **quadratic equations** and their solutions
- Learn about **exponential functions** and logarithms
- Explore **matrix operations** in linear algebra
- Apply linear systems to **optimization problems**