<!-- docs/.vitepress/theme/components/PyScriptDemo.vue -->
<template>
  <div class="pyscript-demo">
    <div class="py-terminal" ref="outputDiv">
      Loading PyScript...
    </div>
    
    <div class="interactive-section">
      <h3>Interactive Calculator</h3>
      <div class="input-group">
        <label>Enter a value for x:</label>
        <input 
          type="number" 
          v-model="inputX" 
          min="0" 
          max="20" 
          @keypress="handleKeypress"
          class="number-input"
        />
        <button @click="calculateLinear" class="calc-button">
          Calculate y = 2x + 1
        </button>
      </div>
      
      <div ref="calculationResult" class="result-box">
        Result will appear here...
      </div>
    </div>

    <div class="interactive-section">
      <h3>Real-time Slider</h3>
      <div class="slider-group">
        <label>Adjust x with slider:</label>
        <input 
          type="range" 
          v-model="sliderX" 
          min="0" 
          max="10" 
          step="0.5"
          @input="updateCalculation"
          class="slider-input"
        />
        <span>x = {{ sliderX }}</span>
      </div>
      
      <div ref="sliderResult" class="result-box slider-result">
        Move the slider to see real-time calculation...
      </div>
    </div>

    <!-- PyScript container - will be populated via JavaScript -->
    <div ref="pyscriptContainer" style="display: none;"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'

const inputX = ref(5)
const sliderX = ref(5)
const outputDiv = ref(null)
const calculationResult = ref(null)
const sliderResult = ref(null)
const pyscriptContainer = ref(null)

// Python code as a string
const pythonCode = `
import sys
import math
from datetime import datetime
from js import document, console, window

def initialize_pyscript():
    # Find our output div
    output_divs = document.querySelectorAll('.py-terminal')
    if output_divs.length > 0:
        output_div = output_divs[0]  # Use the first one we find
        
        results = []
        results.append("🐍 PyScript 2025.5.1 is working!")
        results.append(f"Python version: {sys.version}")
        results.append(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Test basic math
        result = 2 + 2
        results.append(f"2 + 2 = {result}")
        
        # Test mathematical functions
        results.append("\\n=== Mathematical Functions Test ===")
        results.append(f"√16 = {math.sqrt(16)}")
        results.append(f"π = {math.pi:.6f}")
        results.append(f"e = {math.e:.6f}")
        
        # Trigonometric functions
        angle = math.pi / 4  # 45 degrees
        results.append(f"sin(π/4) = {math.sin(angle):.6f}")
        results.append(f"cos(π/4) = {math.cos(angle):.6f}")
        
        # Powers and logarithms
        results.append(f"2^10 = {2**10}")
        results.append(f"log₂(1024) = {math.log2(1024)}")
        
        # List comprehension
        squares = [x**2 for x in range(1, 6)]
        results.append(f"Squares 1-5: {squares}")
        
        # Update the output div
        output_div.innerHTML = "<br>".join(results)
        console.log("PyScript initialized successfully")

def calculate_linear_func(x_val):
    try:
        x = float(x_val)
        y = 2 * x + 1
        
        # Find result divs by class
        result_divs = document.querySelectorAll('.result-box')
        if result_divs.length > 0:
            result_div = result_divs[0]  # First result box
            result_div.innerHTML = f"""
            <strong>Calculation:</strong><br>
            x = {x}<br>
            y = 2x + 1 = 2({x}) + 1 = <strong style="color: #0066cc;">{y}</strong>
            """
        
        console.log(f"Calculated: x={x}, y={y}")
        return y
    except Exception as e:
        console.log(f"Calculation error: {e}")
        return None

def update_slider_func(x_val):
    try:
        x = float(x_val)
        y = 2 * x + 1
        
        # Find slider result div
        slider_divs = document.querySelectorAll('.slider-result')
        if slider_divs.length > 0:
            result_div = slider_divs[0]
            result_div.innerHTML = f"""
            <strong>Real-time Calculation:</strong><br>
            y = 2x + 1<br>
            y = 2({x}) + 1 = <strong style="color: #0066cc;">{y}</strong>
            """
        
        console.log(f"Slider updated: x={x}, y={y}")
        return y
    except Exception as e:
        console.log(f"Slider error: {e}")
        return None

# Make functions available globally
window.pyCalculateLinear = calculate_linear_func
window.pyUpdateSlider = update_slider_func
window.pyInitialize = initialize_pyscript

# Auto-initialize
initialize_pyscript()
`

const createPyScriptElement = () => {
  // Create a script element with type="py"
  const scriptEl = document.createElement('script')
  scriptEl.type = 'py'
  scriptEl.textContent = pythonCode
  
  // Append to our container
  if (pyscriptContainer.value) {
    pyscriptContainer.value.appendChild(scriptEl)
    console.log('PyScript element created and injected')
  }
}

const calculateLinear = () => {
  if (window.pyCalculateLinear) {
    window.pyCalculateLinear(inputX.value)
  } else {
    console.log('PyScript not ready yet')
  }
}

const updateCalculation = () => {
  if (window.pyUpdateSlider) {
    window.pyUpdateSlider(sliderX.value)
  } else {
    console.log('PyScript not ready yet')
  }
}

const handleKeypress = (e) => {
  if (e.key === 'Enter') {
    calculateLinear()
  }
}

const waitForPyScript = async () => {
  // Wait for PyScript to be available
  let attempts = 0
  const maxAttempts = 50 // 5 seconds max
  
  while (!window.pyscript && attempts < maxAttempts) {
    await new Promise(resolve => setTimeout(resolve, 1000))
    attempts++
  }
  
  if (window.pyscript) {
    console.log('PyScript is ready!')
    // Wait a bit more for functions to be available
    setTimeout(() => {
      calculateLinear()
      updateCalculation()
    }, 500)
  } else {
    console.log('PyScript failed to load')
    if (outputDiv.value) {
      outputDiv.value.innerHTML = '❌ PyScript failed to load. Please refresh the page.'
    }
  }
}

onMounted(async () => {
  console.log('Component mounted, setting up PyScript...')
  
  // Wait for next tick to ensure DOM is ready
  await nextTick()
  
  // Create the PyScript element
  createPyScriptElement()
  
  // Wait for PyScript to initialize
  await waitForPyScript()
})
</script>

<style scoped>
.pyscript-demo {
  margin: 2rem 0;
}

.py-terminal {
  background: #f8f9fa;
  border-left: 4px solid #4CAF50;
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  white-space: pre-line;
  min-height: 100px;
}

.interactive-section {
  margin: 1.5rem 0;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: #fafafa;
}

.input-group, .slider-group {
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.input-group label, .slider-group label {
  font-weight: 500;
  min-width: 140px;
}

.number-input {
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  width: 80px;
}

.slider-input {
  width: 200px;
}

.calc-button {
  background: #4CAF50;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.calc-button:hover {
  background: #45a049;
}

.result-box {
  padding: 10px;
  border-radius: 5px;
  background: #f0f0f0;
  min-height: 50px;
}

.slider-result {
  background: #e8f4fd;
}
</style>