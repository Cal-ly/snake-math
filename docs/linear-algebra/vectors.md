# Vectors and Operations

## Mathematical Concept

A **vector** is a quantity with both magnitude and direction, represented as an ordered list of numbers. In Python, vectors are typically represented as lists or NumPy arrays.

Key operations:
- **Addition**: $\vec{u} + \vec{v} = (u_1 + v_1, u_2 + v_2, \ldots)$
- **Scalar multiplication**: $c\vec{u} = (cu_1, cu_2, \ldots)$
- **Dot product**: $\vec{u} \cdot \vec{v} = u_1v_1 + u_2v_2 + \ldots$
- **Magnitude**: $|\vec{u}| = \sqrt{u_1^2 + u_2^2 + \ldots}$

## Interactive Vector Visualization

<VectorOperations />

## Python Implementation

### Basic Vector Operations

```python
import numpy as np
import math

def vector_operations():
    """Demonstrate basic vector operations using both manual and NumPy methods"""
    
    print("Vector Operations Comparison")
    print("=" * 40)
    
    # Define vectors as lists and NumPy arrays
    u_list = [3, 4, 1]
    v_list = [2, -1, 3]
    
    u_np = np.array(u_list)
    v_np = np.array(v_list)
    
    print(f"u = {u_list}")
    print(f"v = {v_list}")
    print()
    
    # 1. Vector Addition
    print("1. Vector Addition:")
    add_manual = [u_list[i] + v_list[i] for i in range(len(u_list))]
    add_numpy = u_np + v_np
    print(f"   Manual: {add_manual}")
    print(f"   NumPy:  {add_numpy.tolist()}")
    
    # 2. Scalar Multiplication
    print("\n2. Scalar Multiplication (×3):")
    scalar = 3
    mult_manual = [scalar * u_list[i] for i in range(len(u_list))]
    mult_numpy = scalar * u_np
    print(f"   Manual: {mult_manual}")
    print(f"   NumPy:  {mult_numpy.tolist()}")
    
    # 3. Dot Product
    print("\n3. Dot Product:")
    dot_manual = sum(u_list[i] * v_list[i] for i in range(len(u_list)))
    dot_numpy = np.dot(u_np, v_np)
    print(f"   Manual: {dot_manual}")
    print(f"   NumPy:  {dot_numpy}")
    
    # 4. Magnitude (Length)
    print("\n4. Magnitude:")
    mag_u_manual = math.sqrt(sum(x**2 for x in u_list))
    mag_u_numpy = np.linalg.norm(u_np)
    print(f"   |u| Manual: {mag_u_manual:.3f}")
    print(f"   |u| NumPy:  {mag_u_numpy:.3f}")
    
    # 5. Unit Vector
    print("\n5. Unit Vector (u/|u|):")
    unit_u_manual = [x / mag_u_manual for x in u_list]
    unit_u_numpy = u_np / mag_u_numpy
    print(f"   Manual: {[round(x, 3) for x in unit_u_manual]}")
    print(f"   NumPy:  {np.round(unit_u_numpy, 3).tolist()}")
    
    # Verify unit vector has magnitude 1
    unit_mag = np.linalg.norm(unit_u_numpy)
    print(f"   |unit_u| = {unit_mag:.6f} (should be 1.0)")

vector_operations()
```

### Geometric Applications

```python
def geometric_applications():
    """Apply vectors to solve geometric problems"""
    
    print("Geometric Applications of Vectors")
    print("=" * 40)
    
    # 1. Distance between points
    print("1. Distance Between Points")
    P1 = np.array([1, 2, 3])
    P2 = np.array([4, 6, 8])
    
    distance_vector = P2 - P1
    distance = np.linalg.norm(distance_vector)
    
    print(f"   P1 = {P1}")
    print(f"   P2 = {P2}")
    print(f"   Distance vector = P2 - P1 = {distance_vector}")
    print(f"   Distance = |P2 - P1| = {distance:.3f}")
    
    # 2. Angle between vectors
    print("\n2. Angle Between Vectors")
    a = np.array([1, 0, 0])
    b = np.array([1, 1, 0])
    
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    print(f"   a = {a}")
    print(f"   b = {b}")
    print(f"   cos(θ) = a·b / (|a|×|b|) = {cos_theta:.3f}")
    print(f"   θ = {theta_deg:.1f}°")
    
    # 3. Projection of one vector onto another
    print("\n3. Vector Projection")
    u = np.array([3, 4])
    v = np.array([1, 0])
    
    # Project u onto v: proj_v(u) = (u·v / |v|²) × v
    proj_scalar = np.dot(u, v) / np.dot(v, v)
    proj_vector = proj_scalar * v
    
    print(f"   u = {u}")
    print(f"   v = {v}")
    print(f"   proj_v(u) = {proj_vector}")
    print(f"   Scalar projection = {proj_scalar:.3f}")
    
    # Perpendicular component
    perp_component = u - proj_vector
    print(f"   Perpendicular component = u - proj_v(u) = {perp_component}")
    
    # Verify orthogonality
    dot_product = np.dot(proj_vector, perp_component)
    print(f"   Dot product of components = {dot_product:.10f} (should be 0)")

geometric_applications()
```

## Cross Product and 3D Applications

The VectorOperations component includes 3D vector visualization and cross product demonstrations, showing the geometric meaning and orthogonality properties of the cross product.

## Applications in Physics and Engineering

### Force and Motion

```python
def physics_applications():
    """Vector applications in physics problems"""
    
    print("Physics Applications of Vectors")
    print("=" * 40)
    
    # 1. Force Vectors and Equilibrium
    print("1. Force Vector Analysis")
    print("Three forces acting on an object:")
    
    F1 = np.array([10, 0])    # 10 N eastward
    F2 = np.array([-5, 8.66]) # 10 N at 120° from east
    F3 = np.array([-5, -8.66]) # 10 N at 240° from east
    
    print(f"   F1 = {F1} N (eastward)")
    print(f"   F2 = {F2} N (120° from east)")
    print(f"   F3 = {F3} N (240° from east)")
    
    # Net force
    F_net = F1 + F2 + F3
    F_net_magnitude = np.linalg.norm(F_net)
    
    print(f"   Net force = {F_net} N")
    print(f"   |F_net| = {F_net_magnitude:.3f} N")
    
    if F_net_magnitude < 0.01:
        print("   Object is in equilibrium!")
    
    # 2. Velocity and Displacement
    print("\n2. Kinematics with Vectors")
    
    # Initial velocity and acceleration
    v0 = np.array([5, 10])  # m/s
    a = np.array([0, -9.8]) # m/s² (gravity)
    t = 2.0                 # seconds
    
    # Position and velocity as functions of time
    # x(t) = v0*t + 0.5*a*t²
    # v(t) = v0 + a*t
    
    displacement = v0 * t + 0.5 * a * t**2
    final_velocity = v0 + a * t
    
    print(f"   Initial velocity: {v0} m/s")
    print(f"   Acceleration: {a} m/s²")
    print(f"   Time: {t} s")
    print(f"   Displacement: {displacement} m")
    print(f"   Final velocity: {final_velocity} m/s")
    
    # Speed at final time
    final_speed = np.linalg.norm(final_velocity)
    print(f"   Final speed: {final_speed:.2f} m/s")
    
    # 3. Work and Energy
    print("\n3. Work Calculation")
    
    force = np.array([15, 20])      # Force in N
    displacement_work = np.array([3, 4])  # Displacement in m
    
    # Work = F · d
    work = np.dot(force, displacement_work)
    
    print(f"   Force: {force} N")
    print(f"   Displacement: {displacement_work} m")
    print(f"   Work = F · d = {work} J")
    
    # Angle between force and displacement
    cos_theta = work / (np.linalg.norm(force) * np.linalg.norm(displacement_work))
    theta = np.degrees(np.arccos(cos_theta))
    
    print(f"   Angle between F and d: {theta:.1f}°")

physics_applications()
```

### Computer Graphics Applications

```python
def computer_graphics_applications():
    """Vector applications in computer graphics"""
    
    print("Computer Graphics Vector Applications")
    print("=" * 40)
    
    # 1. 2D Rotation
    print("1. 2D Vector Rotation")
    
    def rotate_2d(vector, angle_degrees):
        """Rotate a 2D vector by given angle"""
        angle_rad = np.radians(angle_degrees)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                   [np.sin(angle_rad), np.cos(angle_rad)]])
        return rotation_matrix @ vector
    
    original = np.array([3, 1])
    rotated_90 = rotate_2d(original, 90)
    rotated_45 = rotate_2d(original, 45)
    
    print(f"   Original vector: {original}")
    print(f"   Rotated 90°: {rotated_90}")
    print(f"   Rotated 45°: {np.round(rotated_45, 3)}")
    
    # 2. Normal Vectors for Lighting
    print("\n2. Surface Normals for Lighting")
    
    # Triangle vertices
    A = np.array([0, 0, 0])
    B = np.array([1, 0, 0])
    C = np.array([0, 1, 0])
    
    # Edge vectors
    AB = B - A
    AC = C - A
    
    # Normal vector (cross product)
    normal = np.cross(AB, AC)
    unit_normal = normal / np.linalg.norm(normal)
    
    print(f"   Triangle vertices: A{A}, B{B}, C{C}")
    print(f"   Edge AB: {AB}")
    print(f"   Edge AC: {AC}")
    print(f"   Normal vector: {normal}")
    print(f"   Unit normal: {unit_normal}")
    
    # Light direction
    light_dir = np.array([1, 1, 1])
    light_unit = light_dir / np.linalg.norm(light_dir)
    
    # Lighting intensity (Lambert's law)
    intensity = max(0, np.dot(unit_normal, light_unit))
    
    print(f"   Light direction: {light_unit}")
    print(f"   Lighting intensity: {intensity:.3f}")
    
    # 3. Camera Viewing
    print("\n3. Camera View Vectors")
    
    camera_pos = np.array([5, 5, 5])
    target_pos = np.array([0, 0, 0])
    up_vector = np.array([0, 0, 1])
    
    # Forward vector (camera to target)
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Right vector (cross product of forward and up)
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)
    
    # Corrected up vector
    up = np.cross(right, forward)
    
    print(f"   Camera position: {camera_pos}")
    print(f"   Target position: {target_pos}")
    print(f"   Forward vector: {np.round(forward, 3)}")
    print(f"   Right vector: {np.round(right, 3)}")
    print(f"   Up vector: {np.round(up, 3)}")
    
    # Verify orthogonality
    print(f"   Forward·Right = {np.dot(forward, right):.6f} (should be ~0)")
    print(f"   Forward·Up = {np.dot(forward, up):.6f} (should be ~0)")
    print(f"   Right·Up = {np.dot(right, up):.6f} (should be ~0)")

computer_graphics_applications()
```

## Key Takeaways

1. **Vectors** represent quantities with magnitude and direction
2. **Dot product** measures similarity and finds angles between vectors
3. **Cross product** finds perpendicular vectors and areas
4. **Unit vectors** provide direction without magnitude
5. **Projections** decompose vectors into components
6. **Applications** span physics, graphics, and engineering

## Next Steps

- Study **matrix operations** and linear transformations
- Learn **eigenvalues and eigenvectors** for advanced applications
- Explore **vector calculus** with gradients and divergence
- Apply vectors to **machine learning** and **data analysis**