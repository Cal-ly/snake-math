# Exponentials and Logarithms

## Mathematical Concept

**Exponential functions** have the form $f(x) = a \cdot b^x$ where $b > 0$ and $b \neq 1$. When $b > 1$, the function shows exponential growth; when $0 < b < 1$, it shows exponential decay.

**Logarithmic functions** are the inverse of exponential functions: if $y = b^x$, then $x = \log_b(y)$.

Key relationships:
- $\log_b(b^x) = x$ and $b^{\log_b(x)} = x$
- Natural logarithm: $\ln(x) = \log_e(x)$ where $e \approx 2.718$
- Change of base: $\log_b(x) = \frac{\ln(x)}{\ln(b)}$

## Interactive Exponential Explorer

<ExponentialCalculator />

## Python Implementation

### Exponential Functions

```python
import math
import numpy as np

def exponential_function(x, base=math.e, coefficient=1):
    """
    Calculate exponential function: f(x) = a * b^x
    """
    return coefficient * (base ** x)

def natural_exponential(x):
    """Natural exponential function: f(x) = e^x"""
    return math.exp(x)

# Examples
print("Exponential function examples:")
print(f"2^3 = {exponential_function(3, 2)}")
print(f"e^2 = {natural_exponential(2):.3f}")
print(f"0.5^4 = {exponential_function(4, 0.5):.3f}")

# Growth and decay
def compound_growth(initial, rate, time):
    """Compound growth: A = P(1 + r)^t"""
    return initial * (1 + rate) ** time

def exponential_decay(initial, decay_rate, time):
    """Exponential decay: A = P * e^(-rt)"""
    return initial * math.exp(-decay_rate * time)

print(f"\nGrowth example: $1000 at 5% for 10 years = ${compound_growth(1000, 0.05, 10):.2f}")
print(f"Decay example: 100g with half-life model after 5 units = {exponential_decay(100, 0.1, 5):.2f}g")
```

### Logarithmic Functions

```python
def logarithm(x, base=math.e):
    """Calculate logarithm with specified base"""
    if x <= 0:
        return "Undefined (x must be positive)"
    if base <= 0 or base == 1:
        return "Invalid base"
    
    if base == math.e:
        return math.log(x)  # Natural log
    elif base == 10:
        return math.log10(x)  # Common log
    else:
        return math.log(x) / math.log(base)  # Change of base formula

def natural_log(x):
    """Natural logarithm: ln(x)"""
    return math.log(x) if x > 0 else "Undefined"

def common_log(x):
    """Common logarithm: log₁₀(x)"""
    return math.log10(x) if x > 0 else "Undefined"

# Examples
print("Logarithm examples:")
print(f"ln(e) = {natural_log(math.e):.3f}")
print(f"log₁₀(100) = {common_log(100)}")
print(f"log₂(8) = {logarithm(8, 2)}")
print(f"log₅(125) = {logarithm(125, 5)}")

# Verify inverse relationship
x = 5
base = 2
exp_result = base ** x
log_result = logarithm(exp_result, base)
print(f"\nInverse verification:")
print(f"2^{x} = {exp_result}")
print(f"log₂({exp_result}) = {log_result}")
```

### Logarithm Properties

```python
def demonstrate_log_properties():
    """Demonstrate key logarithm properties"""
    a, b = 8, 4
    base = 2
    
    print("Logarithm Properties Demonstration:")
    print(f"Using a = {a}, b = {b}, base = {base}")
    
    # Product rule: log_b(xy) = log_b(x) + log_b(y)
    log_a = logarithm(a, base)
    log_b = logarithm(b, base)
    log_product = logarithm(a * b, base)
    sum_logs = log_a + log_b
    
    print(f"\n1. Product Rule:")
    print(f"   log₂({a} × {b}) = log₂({a * b}) = {log_product}")
    print(f"   log₂({a}) + log₂({b}) = {log_a} + {log_b} = {sum_logs}")
    
    # Quotient rule: log_b(x/y) = log_b(x) - log_b(y)
    log_quotient = logarithm(a / b, base)
    diff_logs = log_a - log_b
    
    print(f"\n2. Quotient Rule:")
    print(f"   log₂({a}/{b}) = log₂({a/b}) = {log_quotient}")
    print(f"   log₂({a}) - log₂({b}) = {log_a} - {log_b} = {diff_logs}")
    
    # Power rule: log_b(x^n) = n * log_b(x)
    n = 3
    log_power = logarithm(a ** n, base)
    n_times_log = n * log_a
    
    print(f"\n3. Power Rule:")
    print(f"   log₂({a}³) = log₂({a**n}) = {log_power}")
    print(f"   3 × log₂({a}) = 3 × {log_a} = {n_times_log}")

demonstrate_log_properties()
```

## Interactive Growth and Decay Models

The ExponentialCalculator component provides interactive models for exponential growth and decay, including population growth, radioactive decay, compound interest, and logarithmic functions with real-time visualization.

## Solving Exponential Equations

```python
def solve_exponential_equations():
    """Methods for solving exponential equations"""
    
    print("Solving Exponential Equations")
    print("=" * 30)
    
    # Type 1: Same base
    print("\n1. Same Base: 2^(x+1) = 2^5")
    print("   If a^m = a^n, then m = n")
    print("   x + 1 = 5")
    print("   x = 4")
    
    # Type 2: Taking logarithms
    print("\n2. Taking Logarithms: 3^x = 20")
    x = math.log(20) / math.log(3)
    print(f"   log₃(20) = ln(20)/ln(3) = {x:.3f}")
    print(f"   Verification: 3^{x:.3f} = {3**x:.3f}")
    
    # Type 3: Exponential growth/decay problems
    print("\n3. Growth Problem: How long for $1000 to become $2000 at 5% continuous growth?")
    print("   2000 = 1000 * e^(0.05t)")
    print("   2 = e^(0.05t)")
    print("   ln(2) = 0.05t")
    t = math.log(2) / 0.05
    print(f"   t = ln(2)/0.05 = {t:.1f} years")
    
    # Type 4: Quadratic in exponential form
    print("\n4. Quadratic Form: e^(2x) - 5e^x + 6 = 0")
    print("   Let u = e^x, then u² - 5u + 6 = 0")
    print("   (u - 2)(u - 3) = 0")
    print("   u = 2 or u = 3")
    print("   e^x = 2 → x = ln(2) ≈ 0.693")
    print("   e^x = 3 → x = ln(3) ≈ 1.099")

solve_exponential_equations()
```

## Real-World Applications

### Compound Interest Calculator

```python
def compound_interest_calculator():
    """Comprehensive compound interest analysis"""
    
    def compound_amount(principal, rate, time, compounding_freq=1):
        """A = P(1 + r/n)^(nt)"""
        return principal * (1 + rate/compounding_freq) ** (compounding_freq * time)
    
    def continuous_compound(principal, rate, time):
        """A = Pe^(rt)"""
        return principal * math.exp(rate * time)
    
    # Example calculation
    P = 5000  # Principal
    r = 0.07  # 7% annual rate
    t = 15    # 15 years
    
    print("Compound Interest Comparison")
    print(f"Principal: ${P}")
    print(f"Rate: {r*100}% per year")
    print(f"Time: {t} years")
    print("-" * 40)
    
    # Different compounding frequencies
    frequencies = [
        (1, "Annually"),
        (4, "Quarterly"), 
        (12, "Monthly"),
        (52, "Weekly"),
        (365, "Daily")
    ]
    
    for n, name in frequencies:
        amount = compound_amount(P, r, t, n)
        interest = amount - P
        print(f"{name:10}: ${amount:8.2f} (Interest: ${interest:7.2f})")
    
    # Continuous compounding
    continuous_amount = continuous_compound(P, r, t)
    continuous_interest = continuous_amount - P
    print(f"{'Continuous':10}: ${continuous_amount:8.2f} (Interest: ${continuous_interest:7.2f})")
    
    # Rule of 72 approximation
    doubling_time_exact = math.log(2) / r
    doubling_time_rule72 = 72 / (r * 100)
    print(f"\nDoubling time:")
    print(f"Exact: {doubling_time_exact:.1f} years")
    print(f"Rule of 72: {doubling_time_rule72:.1f} years")

compound_interest_calculator()
```

### Population Growth Models

```python
def population_models():
    """Different population growth models"""
    
    def exponential_growth(t, P0, r):
        """Exponential: P(t) = P₀e^(rt)"""
        return P0 * math.exp(r * t)
    
    def logistic_growth(t, P0, r, K):
        """Logistic: P(t) = K / (1 + ((K-P₀)/P₀)e^(-rt))"""
        return K / (1 + ((K - P0) / P0) * math.exp(-r * t))
    
    # Parameters
    P0 = 1000   # Initial population
    r = 0.1     # Growth rate (10% per year)
    K = 10000   # Carrying capacity (logistic model)
    
    t = np.linspace(0, 50, 200)
    
    # Calculate populations
    P_exp = [exponential_growth(ti, P0, r) for ti in t]
    P_log = [logistic_growth(ti, P0, r, K) for ti in t]
    
    plt.figure(figsize=(12, 8))
    plt.plot(t, P_exp, 'b-', linewidth=2, label='Exponential Growth')
    plt.plot(t, P_log, 'r-', linewidth=2, label='Logistic Growth')
    plt.axhline(y=K, color='r', linestyle='--', alpha=0.7, label=f'Carrying Capacity = {K}')
    plt.xlabel('Time (years)')
    plt.ylabel('Population')
    plt.title('Population Growth Models Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 15000)
    plt.show()
    
    print("Model Comparison:")
    print("• Exponential: Unlimited growth (unrealistic long-term)")
    print("• Logistic: Growth slows as population approaches carrying capacity")
    print(f"• After 30 years:")
    print(f"  Exponential: {exponential_growth(30, P0, r):,.0f}")
    print(f"  Logistic: {logistic_growth(30, P0, r, K):,.0f}")

population_models()
```

### Carbon Dating

```python
def carbon_dating():
    """Carbon-14 dating calculations"""
    
    # Carbon-14 half-life: 5,730 years
    half_life = 5730
    decay_constant = math.log(2) / half_life
    
    def remaining_carbon14(t):
        """N(t) = N₀e^(-λt)"""
        return math.exp(-decay_constant * t)
    
    def age_from_ratio(ratio):
        """Calculate age from remaining C-14 ratio"""
        if ratio <= 0 or ratio > 1:
            return "Invalid ratio"
        return -math.log(ratio) / decay_constant
    
    print("Carbon-14 Dating")
    print("Half-life: 5,730 years")
    print("-" * 30)
    
    # Examples
    test_ages = [1000, 5730, 11460, 17190, 28650]
    
    for age in test_ages:
        ratio = remaining_carbon14(age)
        print(f"After {age:5} years: {ratio:.3f} ({ratio*100:.1f}%) remains")
    
    print("\nReverse calculation examples:")
    test_ratios = [0.5, 0.25, 0.125, 0.1, 0.05]
    
    for ratio in test_ratios:
        age = age_from_ratio(ratio)
        print(f"If {ratio*100:4.1f}% remains: age ≈ {age:,.0f} years")
    
    # Plot decay curve
    t = np.linspace(0, 30000, 1000)
    N_ratio = [remaining_carbon14(ti) for ti in t]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, N_ratio, 'b-', linewidth=2, label='C-14 Remaining')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% (1 half-life)')
    plt.axhline(y=0.25, color='g', linestyle='--', alpha=0.7, label='25% (2 half-lives)')
    plt.axvline(x=5730, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=11460, color='g', linestyle='--', alpha=0.7)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction of C-14 Remaining')
    plt.title('Carbon-14 Radioactive Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30000)
    plt.ylim(0, 1)
    plt.show()

carbon_dating()
```

## Key Takeaways

1. **Exponential functions** model growth and decay processes
2. **Logarithms** are inverses of exponentials and solve exponential equations
3. **e** (≈2.718) is the natural base for continuous growth models
4. **Compound interest** demonstrates the power of exponential growth
5. **Half-life** provides an intuitive measure for decay processes
6. **Real applications** span finance, biology, physics, and archaeology

## Next Steps

- Study **trigonometric functions** and periodic behavior
- Learn about **logarithmic scales** and data visualization
- Explore **differential equations** with exponential solutions
- Apply exponential models to **data fitting** problems