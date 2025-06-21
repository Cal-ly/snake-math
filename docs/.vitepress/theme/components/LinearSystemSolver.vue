<template>
  <div class="interactive-component">
    <div class="component-section">
      <h3 class="section-title">Linear System Solver</h3>
      <p>Solve systems of linear equations in the form:</p>
      <div class="result-highlight">
        <div class="equation">{{ coefficients.a11 }}x + {{ coefficients.a12 }}y = {{ coefficients.b1 }}</div>
        <div class="equation">{{ coefficients.a21 }}x + {{ coefficients.a22 }}y = {{ coefficients.b2 }}</div>
      </div>
      
      <div class="interactive-card">
        <h4 class="input-group-title">First Equation: a₁₁x + a₁₂y = b₁</h4>
        <div class="component-inputs">
          <div class="input-group">
            <label>a₁₁:</label>
            <input type="number" v-model="coefficients.a11" @input="solveSystem" step="0.1">
          </div>
          <div class="input-group">
            <label>a₁₂:</label>
            <input type="number" v-model="coefficients.a12" @input="solveSystem" step="0.1">
          </div>
          <div class="input-group">
            <label>b₁:</label>
            <input type="number" v-model="coefficients.b1" @input="solveSystem" step="0.1">
          </div>
        </div>
        
        <h4 class="input-group-title">Second Equation: a₂₁x + a₂₂y = b₂</h4>
        <div class="component-inputs">
          <div class="input-group">
            <label>a₂₁:</label>
            <input type="number" v-model="coefficients.a21" @input="solveSystem" step="0.1">
          </div>
          <div class="input-group">
            <label>a₂₂:</label>
            <input type="number" v-model="coefficients.a22" @input="solveSystem" step="0.1">
          </div>
          <div class="input-group">
            <label>b₂:</label>
            <input type="number" v-model="coefficients.b2" @input="solveSystem" step="0.1">
          </div>
        </div>
      </div>
      
      <div class="controls-grid">
        <h4 class="input-group-title">Example Systems:</h4>
        <button @click="loadExample(1)" class="preset-btn">Example 1</button>
        <button @click="loadExample(2)" class="preset-btn">Example 2</button>
        <button @click="loadExample(3)" class="preset-btn">No Solution</button>
        <button @click="loadExample(4)" class="preset-btn">Infinite Solutions</button>
      </div>
    </div>
    
    <div class="component-section">
      <h4 class="input-group-title">Solution:</h4>
      <div class="solution-box" :class="solutionType">
        <div v-if="solutionType === 'unique'" class="unique-solution">
          <div class="result-value">x = {{ solution.x }}</div>
          <div class="result-value">y = {{ solution.y }}</div>
          <div class="verification">
            <strong>Verification:</strong><br>
            Equation 1: {{ coefficients.a11 }} × {{ solution.x }} + {{ coefficients.a12 }} × {{ solution.y }} = {{ verification.eq1 }}<br>
            Equation 2: {{ coefficients.a21 }} × {{ solution.x }} + {{ coefficients.a22 }} × {{ solution.y }} = {{ verification.eq2 }}
          </div>
        </div>
        <div v-else-if="solutionType === 'no-solution'" class="no-solution">
          No solution exists (inconsistent system)
        </div>
        <div v-else-if="solutionType === 'infinite'" class="infinite-solution">
          Infinite solutions exist (dependent equations)
        </div>
        <div v-else class="calculating">
          Enter coefficients to solve the system
        </div>
      </div>
      
      <div class="interactive-card">
        <h4 class="input-group-title">Matrix Form: Ax = b</h4>
        <div class="matrix-display">
          <div class="matrix">
            <div class="matrix-bracket">[</div>
            <div class="matrix-content">
              <div class="matrix-row">{{ coefficients.a11 }}&nbsp;&nbsp;{{ coefficients.a12 }}</div>
              <div class="matrix-row">{{ coefficients.a21 }}&nbsp;&nbsp;{{ coefficients.a22 }}</div>
            </div>
            <div class="matrix-bracket">]</div>
          </div>
          <div class="matrix">
            <div class="matrix-bracket">[</div>
            <div class="matrix-content">
              <div class="matrix-row">x</div>
              <div class="matrix-row">y</div>
            </div>
            <div class="matrix-bracket">]</div>
          </div>
          <div class="equals">=</div>
          <div class="matrix">
            <div class="matrix-bracket">[</div>
            <div class="matrix-content">
              <div class="matrix-row">{{ coefficients.b1 }}</div>
              <div class="matrix-row">{{ coefficients.b2 }}</div>
            </div>
            <div class="matrix-bracket">]</div>
          </div>
        </div>
        
        <div class="determinant-info">
          <strong>Determinant:</strong> det(A) = {{ determinant }}
          <div class="det-meaning">
            {{ determinantMeaning }}
          </div>
        </div>
      </div>
    </div>
    
    <div class="visualization-container">
      <h4 class="input-group-title">Graphical Representation:</h4>
      <canvas ref="plotCanvas" width="400" height="400" class="visualization-canvas"></canvas>
      <div class="plot-legend">
        <div class="legend-item">
          <div class="line-sample blue"></div>
          <span>Equation 1: {{ coefficients.a11 }}x + {{ coefficients.a12 }}y = {{ coefficients.b1 }}</span>
        </div>
        <div class="legend-item">
          <div class="line-sample red"></div>
          <span>Equation 2: {{ coefficients.a21 }}x + {{ coefficients.a22 }}y = {{ coefficients.b2 }}</span>
        </div>
        <div v-if="solutionType === 'unique'" class="legend-item">
          <div class="point-sample"></div>
          <span>Solution: ({{ solution.x }}, {{ solution.y }})</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const coefficients = ref({
  a11: 2, a12: 1, b1: 8,
  a21: 1, a22: 3, b2: 13
})

const solution = ref({ x: 0, y: 0 })
const solutionType = ref('calculating')
const plotCanvas = ref(null)

const determinant = computed(() => {
  return (coefficients.value.a11 * coefficients.value.a22 - coefficients.value.a12 * coefficients.value.a21).toFixed(3)
})

const determinantMeaning = computed(() => {
  const det = parseFloat(determinant.value)
  if (Math.abs(det) < 1e-10) {
    return "det = 0: System is singular (no unique solution)"
  } else {
    return "det ≠ 0: System has a unique solution"
  }
})

const verification = computed(() => {
  const { a11, a12, a21, a22, b1, b2 } = coefficients.value
  const { x, y } = solution.value
  return {
    eq1: (a11 * x + a12 * y).toFixed(3),
    eq2: (a21 * x + a22 * y).toFixed(3)
  }
})

const solveSystem = () => {
  const { a11, a12, a21, a22, b1, b2 } = coefficients.value
  
  // Calculate determinant
  const det = a11 * a22 - a12 * a21
  
  if (Math.abs(det) < 1e-10) {
    // Check if system is inconsistent or has infinite solutions
    const ratio1 = Math.abs(a11) > 1e-10 ? a21 / a11 : (Math.abs(a12) > 1e-10 ? a22 / a12 : 0)
    const ratio2 = Math.abs(b1) > 1e-10 ? b2 / b1 : (Math.abs(b2) > 1e-10 ? Infinity : 1)
    
    if (Math.abs(ratio1 - ratio2) < 1e-10) {
      solutionType.value = 'infinite'
    } else {
      solutionType.value = 'no-solution'
    }
    solution.value = { x: 0, y: 0 }
  } else {
    // Unique solution using Cramer's rule
    const x = (b1 * a22 - b2 * a12) / det
    const y = (a11 * b2 - a21 * b1) / det
    
    solution.value = { 
      x: parseFloat(x.toFixed(3)), 
      y: parseFloat(y.toFixed(3)) 
    }
    solutionType.value = 'unique'
  }
  
  updatePlot()
}

const loadExample = (exampleNum) => {
  switch (exampleNum) {
    case 1:
      coefficients.value = { a11: 2, a12: 1, b1: 8, a21: 1, a22: 3, b2: 13 }
      break
    case 2:
      coefficients.value = { a11: 3, a12: -2, b1: 1, a21: 1, a22: 1, b2: 3 }
      break
    case 3: // No solution
      coefficients.value = { a11: 1, a12: 2, b1: 3, a21: 2, a22: 4, b2: 7 }
      break
    case 4: // Infinite solutions
      coefficients.value = { a11: 1, a12: 2, b1: 3, a21: 2, a22: 4, b2: 6 }
      break
  }
  solveSystem()
}

const updatePlot = () => {
  const canvas = plotCanvas.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  const width = canvas.width
  const height = canvas.height
  
  // Clear canvas
  ctx.clearRect(0, 0, width, height)
  
  // Set up coordinate system
  const range = 10
  const centerX = width / 2
  const centerY = height / 2
  const scale = Math.min(width, height) / (2 * range)
  
  // Convert world coordinates to canvas coordinates
  const toCanvasX = (x) => centerX + x * scale
  const toCanvasY = (y) => centerY - y * scale
  
  // Draw grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 1
  
  for (let i = -range; i <= range; i++) {
    // Vertical lines
    ctx.beginPath()
    ctx.moveTo(toCanvasX(i), 0)
    ctx.lineTo(toCanvasX(i), height)
    ctx.stroke()
    
    // Horizontal lines
    ctx.beginPath()
    ctx.moveTo(0, toCanvasY(i))
    ctx.lineTo(width, toCanvasY(i))
    ctx.stroke()
  }
  
  // Draw axes
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2
  
  // X-axis
  ctx.beginPath()
  ctx.moveTo(0, toCanvasY(0))
  ctx.lineTo(width, toCanvasY(0))
  ctx.stroke()
  
  // Y-axis
  ctx.beginPath()
  ctx.moveTo(toCanvasX(0), 0)
  ctx.lineTo(toCanvasX(0), height)
  ctx.stroke()
  
  // Draw equations as lines
  const { a11, a12, a21, a22, b1, b2 } = coefficients.value
  
  // Draw first equation: a11*x + a12*y = b1 => y = (b1 - a11*x) / a12
  if (Math.abs(a12) > 1e-10) {
    ctx.strokeStyle = '#2196F3'
    ctx.lineWidth = 3
    ctx.beginPath()
    
    const x1 = -range
    const y1 = (b1 - a11 * x1) / a12
    const x2 = range
    const y2 = (b1 - a11 * x2) / a12
    
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1))
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2))
    ctx.stroke()
  }
  
  // Draw second equation: a21*x + a22*y = b2 => y = (b2 - a21*x) / a22
  if (Math.abs(a22) > 1e-10) {
    ctx.strokeStyle = '#F44336'
    ctx.lineWidth = 3
    ctx.beginPath()
    
    const x1 = -range
    const y1 = (b2 - a21 * x1) / a22
    const x2 = range
    const y2 = (b2 - a21 * x2) / a22
    
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1))
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2))
    ctx.stroke()
  }
  
  // Draw solution point
  if (solutionType.value === 'unique') {
    ctx.fillStyle = '#FF9800'
    ctx.beginPath()
    ctx.arc(toCanvasX(solution.value.x), toCanvasY(solution.value.y), 6, 0, 2 * Math.PI)
    ctx.fill()
    
    ctx.strokeStyle = '#000000'
    ctx.lineWidth = 2
    ctx.stroke()
  }
}

watch(coefficients, solveSystem, { deep: true })

onMounted(() => {
  solveSystem()
})
</script>

<style scoped>
@import '../styles/components.css';
</style>