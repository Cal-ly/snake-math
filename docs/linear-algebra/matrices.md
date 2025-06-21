# Matrix Operations

## Mathematical Concept

A **matrix** is a rectangular array of numbers organized in rows and columns. Matrices represent linear transformations and systems of equations. In programmer terms, it is bascially an array of arrays. That means every position of the parent array has an array in it. However, there are some mathematical operation related to matrices. 

Key operations:
- **Addition**: $(A + B)_{ij} = A_{ij} + B_{ij}$
- **Multiplication**: $(AB)_{ij} = \sum_k A_{ik}B_{kj}$
- **Transpose**: $(A^T)_{ij} = A_{ji}$
- **Determinant**: Measures how much a transformation scales area/volume
- **Inverse**: $A^{-1}$ such that $AA^{-1} = I$

## Interactive Matrix Operations

<MatrixTransformations />

## Python Implementation

### Basic Matrix Operations

```python
import numpy as np

def manual_matrix_operations():
    """Implement matrix operations manually to understand the mechanics"""
    
    print("Manual Matrix Operations")
    print("=" * 30)
    
    # Define matrices as lists of lists
    A = [[2, 1, 3],
         [1, 0, 1],
         [1, 1, 1]]
    
    B = [[1, 2],
         [0, 1], 
         [2, 1]]
    
    print("Matrix A (3×3):")
    for row in A:
        print(f"  {row}")
    
    print("\nMatrix B (3×2):")
    for row in B:
        print(f"  {row}")
    
    # Manual matrix multiplication A × B
    def matrix_multiply(A, B):
        """Multiply matrices A and B manually"""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Cannot multiply: incompatible dimensions")
        
        # Initialize result matrix with zeros
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        # Perform multiplication
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    # Calculate A × B
    AB_manual = matrix_multiply(A, B)
    
    print(f"\nMatrix Multiplication A × B (manual):")
    for row in AB_manual:
        print(f"  {row}")
    
    # Compare with NumPy
    A_np = np.array(A)
    B_np = np.array(B)
    AB_numpy = A_np @ B_np
    
    print(f"\nMatrix Multiplication A × B (NumPy):")
    print(AB_numpy)
    
    # Verify they're the same
    print(f"\nManual and NumPy results match: {np.allclose(AB_manual, AB_numpy)}")
    
    return A_np, B_np

manual_matrix_operations()
```

## Solving Linear Systems

### Matrix Equation Form

```python
def solve_linear_systems():
    """Solve systems of linear equations using matrices"""
    
    print("Solving Linear Systems with Matrices")
    print("=" * 40)
    
    # System: 2x + y + z = 8
    #         x + 3y + 2z = 13  
    #         3x + y + 2z = 16
    
    # Coefficient matrix A
    A = np.array([[2, 1, 1],
                  [1, 3, 2],
                  [3, 1, 2]])
    
    # Constants vector b  
    b = np.array([8, 13, 16])
    
    print("System of equations:")
    print("2x + y + z = 8")
    print("x + 3y + 2z = 13")
    print("3x + y + 2z = 16")
    print()
    print("Matrix form: Ax = b")
    print("A =")
    print(A)
    print(f"b = {b}")
    
    # Method 1: Using matrix inverse
    print(f"\nMethod 1: Matrix Inverse")
    det_A = np.linalg.det(A)
    print(f"det(A) = {det_A:.3f}")
    
    if abs(det_A) > 1e-10:
        A_inv = np.linalg.inv(A)
        x_inverse = A_inv @ b
        print(f"x = A⁻¹b = {x_inverse}")
    else:
        print("Matrix is singular - no unique solution")
    
    # Method 2: Direct solve (more numerically stable)
    print(f"\nMethod 2: Direct solve (np.linalg.solve)")
    x_solve = np.linalg.solve(A, b)
    print(f"x = {x_solve}")
    
    # Method 3: LU decomposition
    print(f"\nMethod 3: LU Decomposition")
    from scipy.linalg import lu_solve, lu_factor
    
    lu, piv = lu_factor(A)
    x_lu = lu_solve((lu, piv), b)
    print(f"x = {x_lu}")
    
    # Verify solution
    print(f"\nVerification:")
    verification = A @ x_solve
    print(f"Ax = {verification}")
    print(f"b = {b}")
    print(f"Error: {np.linalg.norm(verification - b):.2e}")
    
    # Condition number (measures numerical stability)
    cond_num = np.linalg.cond(A)
    print(f"\nCondition number: {cond_num:.2f}")
    if cond_num < 100:
        print("Well-conditioned system (numerically stable)")
    elif cond_num < 10000:
        print("Moderately conditioned")
    else:
        print("Ill-conditioned (numerically unstable)")

solve_linear_systems()
```

## Applications

### Principal Component Analysis (PCA)

```python
def pca_demonstration():
    """Demonstrate PCA using eigenvalues and eigenvectors"""
    
    print("Principal Component Analysis (PCA)")
    print("=" * 40)
    
    # Generate sample 2D data with correlation
    np.random.seed(42)
    n_samples = 100
    
    # Create correlated data
    x1 = np.random.normal(0, 2, n_samples)
    x2 = 1.5 * x1 + np.random.normal(0, 1, n_samples)
    
    # Data matrix (each row is a sample)
    X = np.column_stack([x1, x2])
    
    print(f"Data shape: {X.shape}")
    print(f"Sample mean: {np.mean(X, axis=0)}")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)
    print(f"\nCovariance matrix:")
    print(cov_matrix)
    
    # Find eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues (variances along principal components): {eigenvalues}")
    print(f"Explained variance ratio: {eigenvalues / np.sum(eigenvalues)}")
    
    # Project data onto principal components
    X_pca = X_centered @ eigenvectors
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Original Data')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # With principal components
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    
    # Plot principal component directions
    mean_point = np.mean(X, axis=0)
    scale = 3  # Scale for visualization
    
    # First principal component
    pc1_start = mean_point - scale * np.sqrt(eigenvalues[0]) * eigenvectors[:, 0]
    pc1_end = mean_point + scale * np.sqrt(eigenvalues[0]) * eigenvectors[:, 0]
    plt.plot([pc1_start[0], pc1_end[0]], [pc1_start[1], pc1_end[1]], 
             'r-', linewidth=3, label=f'PC1 (var={eigenvalues[0]:.2f})')
    
    # Second principal component
    pc2_start = mean_point - scale * np.sqrt(eigenvalues[1]) * eigenvectors[:, 1]
    pc2_end = mean_point + scale * np.sqrt(eigenvalues[1]) * eigenvectors[:, 1]
    plt.plot([pc2_start[0], pc2_end[0]], [pc2_start[1], pc2_end[1]], 
             'g-', linewidth=3, label=f'PC2 (var={eigenvalues[1]:.2f})')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data with Principal Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Transformed data (PCA coordinates)
    plt.subplot(1, 3, 3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Data in PCA Coordinates')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPCA Results:")
    print(f"• First PC explains {eigenvalues[0]/np.sum(eigenvalues)*100:.1f}% of variance")
    print(f"• Second PC explains {eigenvalues[1]/np.sum(eigenvalues)*100:.1f}% of variance")
    print(f"• Principal components are orthogonal (uncorrelated)")
    print(f"• Data is decorrelated in the new coordinate system")

pca_demonstration()
```

### Markov Chains

```python
def markov_chain_analysis():
    """Analyze Markov chains using matrix operations"""
    
    print("Markov Chain Analysis")
    print("=" * 30)
    
    # Transition matrix for weather example
    # States: [Sunny, Cloudy, Rainy]
    P = np.array([[0.7, 0.2, 0.1],   # From Sunny
                  [0.3, 0.4, 0.3],   # From Cloudy  
                  [0.2, 0.3, 0.5]])  # From Rainy
    
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    print("Weather Transition Matrix:")
    print("     ", "  ".join(f"{s:>6}" for s in states))
    for i, state in enumerate(states):
        print(f"{state:>6}", "  ".join(f"{P[i,j]:6.1f}" for j in range(3)))
    
    # Initial state distribution
    initial = np.array([1, 0, 0])  # Start sunny
    print(f"\nInitial state: {dict(zip(states, initial))}")
    
    # Evolution over time
    print(f"\nState evolution:")
    current = initial.copy()
    
    for day in range(10):
        print(f"Day {day:2d}: {dict(zip(states, np.round(current, 3)))}")
        current = current @ P  # Matrix multiplication for state transition
    
    # Steady-state analysis using eigenvalues
    print(f"\nSteady-state analysis:")
    eigenvalues, eigenvectors = np.linalg.eig(P.T)  # Transpose for left eigenvectors
    
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady_state = np.real(eigenvectors[:, idx])
    steady_state = steady_state / np.sum(steady_state)  # Normalize
    
    print(f"Steady-state probabilities: {dict(zip(states, np.round(steady_state, 3)))}")
    
    # Expected return time to each state
    print(f"\nExpected return times:")
    for i, state in enumerate(states):
        return_time = 1 / steady_state[i]
        print(f"{state}: {return_time:.1f} days")
    
    # Verify steady state
    steady_check = steady_state @ P
    print(f"\nSteady-state verification:")
    print(f"π × P = {np.round(steady_check, 3)}")
    print(f"π     = {np.round(steady_state, 3)}")
    print(f"Difference: {np.linalg.norm(steady_check - steady_state):.2e}")

markov_chain_analysis()
```

## Key Takeaways

1. **Matrices** represent linear transformations and systems
2. **Matrix multiplication** is associative but not commutative
3. **Determinants** measure scaling and orientation changes
4. **Eigenvalues/eigenvectors** reveal fundamental transformation directions
5. **Applications** include computer graphics, data analysis, and modeling
6. **Numerical methods** provide efficient solutions for large systems

## Next Steps

- Study **singular value decomposition (SVD)** for data analysis
- Learn **matrix calculus** for optimization and machine learning
- Explore **graph theory** using adjacency matrices
- Apply matrices to **computer vision** and **image processing**