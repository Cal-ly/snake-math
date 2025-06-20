# Trigonometric Functions

## Mathematical Concept

**Trigonometric functions** relate angles to the ratios of sides in a right triangle and points on the **unit circle**. The unit circle has radius 1 centered at the origin.

Key relationships:
- $\sin(\theta) = \frac{\text{opposite}}{\text{hypotenuse}} = y\text{-coordinate on unit circle}$
- $\cos(\theta) = \frac{\text{adjacent}}{\text{hypotenuse}} = x\text{-coordinate on unit circle}$  
- $\tan(\theta) = \frac{\sin(\theta)}{\cos(\theta)} = \frac{y}{x}$

Important identities:
- $\sin^2(\theta) + \cos^2(\theta) = 1$ (Pythagorean identity)
- Period: $\sin$ and $\cos$ repeat every $2\pi$ radians (360°)

## Interactive Unit Circle Explorer

<UnitCircleExplorer />

## Python Implementation

### Basic Trigonometric Functions

```python
import math
import numpy as np

def trigonometric_functions():
    """Demonstrate basic trigonometric function calculations"""
    
    print("Trigonometric Functions in Python")
    print("=" * 40)
    
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

trigonometric_functions()
```

### Trigonometric Identities

```python
def verify_trig_identities():
    """Verify important trigonometric identities"""
    
    print("Trigonometric Identity Verification")
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

verify_trig_identities()
```

## Trigonometric Graphs and Transformations

The UnitCircleExplorer component includes comprehensive visualization of trigonometric function graphs and transformations, showing amplitude, frequency, and phase changes.

## Real-World Applications

### Wave Analysis

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
    
    plt.figure(figsize=(15, 10))
    
    # Individual waves
    plt.subplot(3, 1, 1)
    plt.plot(t, wave_low, 'b-', linewidth=2, label=f'{freq_low} Hz')
    plt.plot(t, wave_high, 'r-', linewidth=2, label=f'{freq_high} Hz')
    plt.title('Individual Sine Waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Combined wave
    plt.subplot(3, 1, 2)
    plt.plot(t, wave_combined, 'g-', linewidth=2, label='Combined Wave')
    plt.title('Superposition of Waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Frequency analysis (simplified)
    plt.subplot(3, 1, 3)
    frequencies = [1, 2, 3, 4, 5, 6, 7]
    amplitudes = [0, 1, 0, 0, 1, 0, 0]  # Only 2 Hz and 5 Hz present
    
    plt.stem(frequencies, amplitudes, basefmt=' ')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 8)
    
    plt.tight_layout()
    plt.show()
    
    # Wave properties
    print(f"Wave Properties:")
    print(f"• Period of {freq_low} Hz wave: {1/freq_low:.2f} seconds")
    print(f"• Period of {freq_high} Hz wave: {1/freq_high:.2f} seconds")
    print(f"• Wavelength = speed / frequency")
    print(f"• Phase determines starting position of wave")

wave_analysis()
```

### Circular Motion

```python
def circular_motion_analysis():
    """Model circular motion using trigonometric functions"""
    
    print("Circular Motion Analysis")
    print("=" * 30)
    
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
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Circular path
    ax1.plot(x, y, 'b-', linewidth=2, label='Path')
    ax1.plot(x[0], y[0], 'go', markersize=8, label='Start')
    ax1.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
    
    # Add arrows for direction
    for i in range(0, len(t), 20):
        ax1.arrow(x[i], y[i], vx[i]*0.1, vy[i]*0.1, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    ax1.set_title('Circular Motion Path')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Position vs time
    ax2.plot(t, x, 'b-', linewidth=2, label='x(t) = r·cos(ωt)')
    ax2.plot(t, y, 'r-', linewidth=2, label='y(t) = r·sin(ωt)')
    ax2.set_title('Position Components vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocity vs time
    ax3.plot(t, vx, 'b-', linewidth=2, label='vₓ(t) = -rω·sin(ωt)')
    ax3.plot(t, vy, 'r-', linewidth=2, label='vᵧ(t) = rω·cos(ωt)')
    ax3.set_title('Velocity Components vs Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Speed vs time
    speed = np.sqrt(vx**2 + vy**2)
    ax4.plot(t, speed, 'g-', linewidth=2, label=f'Speed = {radius * angular_velocity} m/s')
    ax4.set_title('Speed vs Time (Constant)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(speed) * 1.1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nKey Observations:")
    print(f"• Position components are sinusoidal with 90° phase difference")
    print(f"• Velocity is always perpendicular to position (90° ahead)")
    print(f"• Speed is constant in uniform circular motion")
    print(f"• Acceleration points toward center (centripetal)")

circular_motion_analysis()
```

## Inverse Trigonometric Functions

```python
def inverse_trig_functions():
    """Explore inverse trigonometric functions and their applications"""
    
    print("Inverse Trigonometric Functions")
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

inverse_trig_functions()
```

## Key Takeaways

1. **Unit circle** connects geometry and trigonometry  
2. **Trigonometric functions** are periodic with applications in waves and cycles
3. **Identities** provide relationships between different trig functions
4. **Transformations** allow modeling of real-world periodic phenomena
5. **Inverse functions** find angles from ratios
6. **Applications** include waves, circular motion, and oscillations

## Next Steps

- Study **complex numbers** and Euler's formula: $e^{i\theta} = \cos\theta + i\sin\theta$
- Learn **Fourier series** for analyzing periodic functions
- Explore **differential equations** with trigonometric solutions
- Apply trigonometry to **signal processing** and **computer graphics**