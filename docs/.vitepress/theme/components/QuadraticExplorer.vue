<template>
  <div class="interactive-component">
    <div class="component-section">
      <h3 class="section-title">Interactive Quadratic Function Explorer</h3>
      <p>Explore quadratic functions of the form: <strong>f(x) = ax² + bx + c</strong></p>
      
      <div class="interactive-card">
        <div class="input-group">
          <label>a coefficient:</label>
          <input type="range" v-model="a" min="-3" max="3" step="0.1" @input="updateQuadratic" class="range-input">
          <span class="result-value">{{ a }}</span>
          <span class="description">({{ a > 0 ? 'opens upward' : a < 0 ? 'opens downward' : 'degenerate' }})</span>
        </div>
        
        <div class="input-group">
          <label>b coefficient:</label>
          <input type="range" v-model="b" min="-5" max="5" step="0.1" @input="updateQuadratic" class="range-input">
          <span class="result-value">{{ b }}</span>
        </div>
        
        <div class="input-group">
          <label>c coefficient:</label>
          <input type="range" v-model="c" min="-10" max="10" step="0.1" @input="updateQuadratic" class="range-input">
          <span class="result-value">{{ c }}</span>
          <span class="description">(y-intercept)</span>
        </div>
      </div>
      
      <div class="controls-grid">
        <h4 class="input-group-title">Example Functions:</h4>
        <button @click="loadPreset('standard')" class="preset-btn">Standard Parabola</button>
        <button @click="loadPreset('wide')" class="preset-btn">Wide Parabola</button>
        <button @click="loadPreset('narrow')" class="preset-btn">Narrow Parabola</button>
        <button @click="loadPreset('shifted')" class="preset-btn">Shifted Parabola</button>
      </div>
    </div>
    
    <div class="result-highlight">
      <div class="current-equation">
        f(x) = {{ formatCoefficient(a, 'x²') }}{{ formatCoefficient(b, 'x', true) }}{{ formatConstant(c) }}
      </div>
    </div>
    
    <div class="visualization-container">
      <canvas ref="plotCanvas" width="600" height="400" class="visualization-canvas"></canvas>
    </div>
    
    <div class="component-section">
      <div class="results-grid">
        <div class="result-card">
          <h4 class="result-label">Vertex</h4>
          <div class="vertex-info">
            <div>x = {{ vertex.x }}</div>
            <div>y = {{ vertex.y }}</div>
            <div class="point">({{ vertex.x }}, {{ vertex.y }})</div>
            <div class="vertex-type">{{ vertexType }}</div>
          </div>
        </div>
        
        <div class="result-card">
          <h4 class="result-label">Axis of Symmetry</h4>
          <div class="axis-info">
            <div>x = {{ vertex.x }}</div>
            <div class="description">Vertical line through vertex</div>
          </div>
        </div>
        
        <div class="result-card">
          <h4 class="result-label">Y-Intercept</h4>
          <div class="intercept-info">
            <div>(0, {{ c }})</div>
            <div class="description">Point where parabola crosses y-axis</div>
          </div>
        </div>
        
        <div class="result-card">
          <h4 class="result-label">Discriminant</h4>
          <div class="discriminant-info">
            <div>Δ = b² - 4ac = {{ discriminant }}</div>
            <div class="roots-info">{{ rootsDescription }}</div>
          </div>
        </div>
        
        <div v-if="hasRealRoots" class="result-card">
          <h4 class="result-label">X-Intercepts (Roots)</h4>
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
        
        <div class="result-card">
          <h4 class="result-label">Domain & Range</h4>
          <div class="domain-range">
            <div><strong>Domain:</strong> (-∞, ∞)</div>
            <div><strong>Range:</strong> {{ rangeDescription }}</div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="interactive-card">
      <h4 class="input-group-title">Function Calculator</h4>
      <div class="component-inputs">
        <label>Evaluate f(x) at x =</label>
        <input type="number" v-model="evalX" @input="evaluateFunction" step="0.1" class="eval-input">
        <span class="result-value">f({{ evalX }}) = {{ evalResult }}</span>
      </div>
      
      <div class="solver">
        <h4 class="input-group-title">Equation Solver</h4>
        <div class="component-inputs">
          <label>Solve f(x) =</label>
          <input type="number" v-model="solveY" @input="solveEquation" step="0.1" class="eval-input">
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
@import '../styles/components.css';
</style>