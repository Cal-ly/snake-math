<template>
  <div class="quadratic-explorer">
    <div class="controls">
      <h3>Interactive Quadratic Function Explorer</h3>
      <p>Explore quadratic functions of the form: <strong>f(x) = ax² + bx + c</strong></p>
      
      <div class="parameters">
        <div class="slider-group">
          <label>a coefficient:</label>
          <input type="range" v-model="a" min="-3" max="3" step="0.1" @input="updateQuadratic">
          <span class="value">{{ a }}</span>
          <span class="description">({{ a > 0 ? 'opens upward' : a < 0 ? 'opens downward' : 'degenerate' }})</span>
        </div>
        
        <div class="slider-group">
          <label>b coefficient:</label>
          <input type="range" v-model="b" min="-5" max="5" step="0.1" @input="updateQuadratic">
          <span class="value">{{ b }}</span>
        </div>
        
        <div class="slider-group">
          <label>c coefficient:</label>
          <input type="range" v-model="c" min="-10" max="10" step="0.1" @input="updateQuadratic">
          <span class="value">{{ c }}</span>
          <span class="description">(y-intercept)</span>
        </div>
      </div>
      
      <div class="presets">
        <h4>Example Functions:</h4>
        <button @click="loadPreset('standard')" class="preset-btn">Standard Parabola</button>
        <button @click="loadPreset('wide')" class="preset-btn">Wide Parabola</button>
        <button @click="loadPreset('narrow')" class="preset-btn">Narrow Parabola</button>
        <button @click="loadPreset('shifted')" class="preset-btn">Shifted Parabola</button>
      </div>
    </div>
    
    <div class="equation-display">
      <div class="current-equation">
        f(x) = {{ formatCoefficient(a, 'x²') }}{{ formatCoefficient(b, 'x', true) }}{{ formatConstant(c) }}
      </div>
    </div>
    
    <div class="visualization">
      <canvas ref="plotCanvas" width="600" height="400"></canvas>
    </div>
    
    <div class="analysis">
      <div class="analysis-grid">
        <div class="analysis-card">
          <h4>Vertex</h4>
          <div class="vertex-info">
            <div>x = {{ vertex.x }}</div>
            <div>y = {{ vertex.y }}</div>
            <div class="point">({{ vertex.x }}, {{ vertex.y }})</div>
            <div class="vertex-type">{{ vertexType }}</div>
          </div>
        </div>
        
        <div class="analysis-card">
          <h4>Axis of Symmetry</h4>
          <div class="axis-info">
            <div>x = {{ vertex.x }}</div>
            <div class="description">Vertical line through vertex</div>
          </div>
        </div>
        
        <div class="analysis-card">
          <h4>Y-Intercept</h4>
          <div class="intercept-info">
            <div>(0, {{ c }})</div>
            <div class="description">Point where parabola crosses y-axis</div>
          </div>
        </div>
        
        <div class="analysis-card">
          <h4>Discriminant</h4>
          <div class="discriminant-info">
            <div>Δ = b² - 4ac = {{ discriminant }}</div>
            <div class="roots-info">{{ rootsDescription }}</div>
          </div>
        </div>
        
        <div v-if="hasRealRoots" class="analysis-card">
          <h4>X-Intercepts (Roots)</h4>
          <div class="roots-info">
            <div v-if="roots.length === 1">
              x = {{ roots[0] }} (repeated root)
            </div>
            <div v-else>
              <div>x₁ = {{ roots[0] }}</div>
              <div>x₂ = {{ roots[1] }}</div>
            </div>
          </div>
        </div>
        
        <div class="analysis-card">
          <h4>Domain & Range</h4>
          <div class="domain-range">
            <div><strong>Domain:</strong> (-∞, ∞)</div>
            <div><strong>Range:</strong> {{ rangeDescription }}</div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="function-calculator">
      <h4>Function Calculator</h4>
      <div class="calculator-input">
        <label>Evaluate f(x) at x =</label>
        <input type="number" v-model="evalX" @input="evaluateFunction" step="0.1" class="eval-input">
        <span class="calc-result">f({{ evalX }}) = {{ evalResult }}</span>
      </div>
      
      <div class="solver">
        <h4>Equation Solver</h4>
        <div class="solver-input">
          <label>Solve f(x) =</label>
          <input type="number" v-model="solveY" @input="solveEquation" step="0.1" class="solve-input">
          <div class="solve-result">{{ solveResult }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const a = ref(1)
const b = ref(0)
const c = ref(0)
const evalX = ref(0)
const evalResult = ref(0)
const solveY = ref(0)
const solveResult = ref('')
const plotCanvas = ref(null)

const vertex = computed(() => {
  const x = parseFloat((-b.value / (2 * a.value)).toFixed(3))
  const y = parseFloat((a.value * x * x + b.value * x + c.value).toFixed(3))
  return { x, y }
})

const vertexType = computed(() => {
  if (a.value > 0) return 'Minimum point'
  else if (a.value < 0) return 'Maximum point'
  else return 'Not a parabola'
})

const discriminant = computed(() => {
  return parseFloat((b.value * b.value - 4 * a.value * c.value).toFixed(3))
})

const hasRealRoots = computed(() => discriminant.value >= 0)

const roots = computed(() => {
  if (!hasRealRoots.value) return []
  
  const sqrt_discriminant = Math.sqrt(discriminant.value)
  const denominator = 2 * a.value
  
  if (discriminant.value === 0) {
    return [parseFloat((-b.value / denominator).toFixed(3))]
  } else {
    const root1 = parseFloat(((-b.value + sqrt_discriminant) / denominator).toFixed(3))
    const root2 = parseFloat(((-b.value - sqrt_discriminant) / denominator).toFixed(3))
    return [Math.min(root1, root2), Math.max(root1, root2)]
  }
})

const rootsDescription = computed(() => {
  if (discriminant.value > 0) return '2 real roots (intersects x-axis twice)'
  else if (discriminant.value === 0) return '1 real root (touches x-axis once)'
  else return 'No real roots (does not intersect x-axis)'
})

const rangeDescription = computed(() => {
  if (a.value > 0) {
    return `[${vertex.value.y}, ∞)`
  } else if (a.value < 0) {
    return `(-∞, ${vertex.value.y}]`
  } else {
    return 'Not applicable (not a parabola)'
  }
})

const formatCoefficient = (coeff, variable, showPlus = false) => {
  const num = parseFloat(coeff)
  if (num === 0) return ''
  
  let result = ''
  if (showPlus && num > 0) result += ' + '
  else if (showPlus && num < 0) result += ' - '
  else if (num < 0) result += '-'
  
  const absNum = Math.abs(num)
  if (absNum === 1 && variable !== '') {
    result += variable
  } else {
    result += absNum + variable
  }
  
  return result
}

const formatConstant = (coeff) => {
  const num = parseFloat(coeff)
  if (num === 0) return ''
  else if (num > 0) return ' + ' + num
  else return ' - ' + Math.abs(num)
}

const evaluateFunction = () => {
  const x = parseFloat(evalX.value)
  const result = a.value * x * x + b.value * x + c.value
  evalResult.value = parseFloat(result.toFixed(3))
  updatePlot()
}

const solveEquation = () => {
  const targetY = parseFloat(solveY.value)
  const adjustedC = c.value - targetY
  const discriminantSolve = b.value * b.value - 4 * a.value * adjustedC
  
  if (discriminantSolve < 0) {
    solveResult.value = 'No real solutions'
  } else if (discriminantSolve === 0) {
    const x = -b.value / (2 * a.value)
    solveResult.value = `x = ${x.toFixed(3)}`
  } else {
    const sqrt_disc = Math.sqrt(discriminantSolve)
    const x1 = (-b.value + sqrt_disc) / (2 * a.value)
    const x2 = (-b.value - sqrt_disc) / (2 * a.value)
    solveResult.value = `x = ${Math.min(x1, x2).toFixed(3)} or x = ${Math.max(x1, x2).toFixed(3)}`
  }
}

const loadPreset = (preset) => {
  switch (preset) {
    case 'standard':
      a.value = 1; b.value = 0; c.value = 0
      break
    case 'wide':
      a.value = 0.5; b.value = 0; c.value = 0
      break
    case 'narrow':
      a.value = 2; b.value = 0; c.value = 0
      break
    case 'shifted':
      a.value = 1; b.value = -4; c.value = 3
      break
  }
  updateQuadratic()
}

const updateQuadratic = () => {
  evaluateFunction()
  solveEquation()
  updatePlot()
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
  const xMin = -10
  const xMax = 10
  const yMin = -15
  const yMax = 15
  
  const xScale = width / (xMax - xMin)
  const yScale = height / (yMax - yMin)
  
  const toCanvasX = (x) => (x - xMin) * xScale
  const toCanvasY = (y) => height - (y - yMin) * yScale
  
  // Draw grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 1
  
  for (let x = xMin; x <= xMax; x++) {
    ctx.beginPath()
    ctx.moveTo(toCanvasX(x), 0)
    ctx.lineTo(toCanvasX(x), height)
    ctx.stroke()
  }
  
  for (let y = yMin; y <= yMax; y += 5) {
    ctx.beginPath()
    ctx.moveTo(0, toCanvasY(y))
    ctx.lineTo(width, toCanvasY(y))
    ctx.stroke()
  }
  
  // Draw axes
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2
  
  ctx.beginPath()
  ctx.moveTo(0, toCanvasY(0))
  ctx.lineTo(width, toCanvasY(0))
  ctx.stroke()
  
  ctx.beginPath()
  ctx.moveTo(toCanvasX(0), 0)
  ctx.lineTo(toCanvasX(0), height)
  ctx.stroke()
  
  // Draw quadratic function
  ctx.strokeStyle = '#2196F3'
  ctx.lineWidth = 3
  ctx.beginPath()
  
  let firstPoint = true
  for (let x = xMin; x <= xMax; x += 0.1) {
    const y = a.value * x * x + b.value * x + c.value
    
    if (y >= yMin && y <= yMax) {
      const canvasX = toCanvasX(x)
      const canvasY = toCanvasY(y)
      
      if (firstPoint) {
        ctx.moveTo(canvasX, canvasY)
        firstPoint = false
      } else {
        ctx.lineTo(canvasX, canvasY)
      }
    }
  }
  ctx.stroke()
  
  // Draw vertex
  const vertexX = toCanvasX(vertex.value.x)
  const vertexY = toCanvasY(vertex.value.y)
  
  ctx.fillStyle = '#FF5722'
  ctx.beginPath()
  ctx.arc(vertexX, vertexY, 6, 0, 2 * Math.PI)
  ctx.fill()
  
  // Draw axis of symmetry
  ctx.strokeStyle = '#4CAF50'
  ctx.lineWidth = 2
  ctx.setLineDash([5, 5])
  ctx.beginPath()
  ctx.moveTo(vertexX, 0)
  ctx.lineTo(vertexX, height)
  ctx.stroke()
  ctx.setLineDash([])
  
  // Draw roots if they exist
  if (hasRealRoots.value) {
    ctx.fillStyle = '#9C27B0'
    roots.value.forEach(root => {
      const rootX = toCanvasX(root)
      const rootY = toCanvasY(0)
      ctx.beginPath()
      ctx.arc(rootX, rootY, 5, 0, 2 * Math.PI)
      ctx.fill()
    })
  }
  
  // Draw y-intercept
  ctx.fillStyle = '#FF9800'
  ctx.beginPath()
  ctx.arc(toCanvasX(0), toCanvasY(c.value), 5, 0, 2 * Math.PI)
  ctx.fill()
  
  // Draw evaluation point
  if (evalX.value >= xMin && evalX.value <= xMax) {
    const evalCanvasX = toCanvasX(evalX.value)
    const evalCanvasY = toCanvasY(evalResult.value)
    
    ctx.fillStyle = '#E91E63'
    ctx.beginPath()
    ctx.arc(evalCanvasX, evalCanvasY, 5, 0, 2 * Math.PI)
    ctx.fill()
    
    // Draw vertical line to x-axis
    ctx.strokeStyle = '#E91E63'
    ctx.lineWidth = 1
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(evalCanvasX, evalCanvasY)
    ctx.lineTo(evalCanvasX, toCanvasY(0))
    ctx.stroke()
    ctx.setLineDash([])
  }
}

watch([a, b, c], updateQuadratic)

onMounted(() => {
  updateQuadratic()
})
</script>

<style scoped>
.quadratic-explorer {
  margin: 2rem 0;
  padding: 1.5rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: #fafafa;
}

.controls h3 {
  margin-top: 0;
  color: #333;
}

.parameters {
  margin: 1.5rem 0;
}

.slider-group {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0;
}

.slider-group label {
  font-weight: 500;
  min-width: 100px;
}

.slider-group input[type="range"] {
  width: 200px;
}

.value {
  font-weight: bold;
  color: #2196F3;
  min-width: 40px;
}

.description {
  font-style: italic;
  color: #666;
}

.presets {
  margin: 1.5rem 0;
}

.preset-btn {
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
  padding: 0.5rem 1rem;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.preset-btn:hover {
  background: #45a049;
}

.equation-display {
  margin: 1.5rem 0;
  text-align: center;
  padding: 1rem;
  background: #f0f0f0;
  border-radius: 4px;
}

.current-equation {
  font-size: 1.5em;
  font-weight: bold;
  color: #2196F3;
  font-family: 'Times New Roman', serif;
}

.visualization {
  margin: 1.5rem 0;
  text-align: center;
}

.visualization canvas {
  border: 2px solid #ccc;
  border-radius: 4px;
  background: white;
}

.analysis {
  margin: 1.5rem 0;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.analysis-card {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.analysis-card h4 {
  margin: 0 0 0.5rem 0;
  color: #333;
}

.vertex-info, .axis-info, .intercept-info, .discriminant-info, .roots-info, .domain-range {
  font-size: 0.9em;
}

.point {
  font-weight: bold;
  color: #FF5722;
}

.vertex-type {
  font-style: italic;
  color: #666;
  margin-top: 0.5rem;
}

.function-calculator {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 4px;
}

.calculator-input, .solver-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 1rem 0;
}

.eval-input, .solve-input {
  width: 80px;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.calc-result {
  font-weight: bold;
  color: #E91E63;
}

.solve-result {
  font-weight: bold;
  color: #9C27B0;
  margin-left: 0.5rem;
}
</style>