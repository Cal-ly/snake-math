<template>
  <div class="function-plotter">
    <div class="controls">
      <h3>Interactive Function Plotter</h3>
      <div class="input-group">
        <label>Function Type:</label>
        <select v-model="functionType" @change="updatePlot">
          <option value="linear">Linear: f(x) = mx + b</option>
          <option value="quadratic">Quadratic: f(x) = ax² + bx + c</option>
          <option value="exponential">Exponential: f(x) = a·e^(bx)</option>
          <option value="trigonometric">Trigonometric: f(x) = a·sin(bx + c)</option>
        </select>
      </div>
      
      <div v-if="functionType === 'linear'" class="parameter-group">
        <div class="slider-group">
          <label>Slope (m):</label>
          <input type="range" v-model="slope" min="-5" max="5" step="0.1" @input="updatePlot">
          <span>{{ slope }}</span>
        </div>
        <div class="slider-group">
          <label>Y-intercept (b):</label>
          <input type="range" v-model="yIntercept" min="-10" max="10" step="0.1" @input="updatePlot">
          <span>{{ yIntercept }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'quadratic'" class="parameter-group">
        <div class="slider-group">
          <label>a coefficient:</label>
          <input type="range" v-model="quadA" min="-3" max="3" step="0.1" @input="updatePlot">
          <span>{{ quadA }}</span>
        </div>
        <div class="slider-group">
          <label>b coefficient:</label>
          <input type="range" v-model="quadB" min="-5" max="5" step="0.1" @input="updatePlot">
          <span>{{ quadB }}</span>
        </div>
        <div class="slider-group">
          <label>c coefficient:</label>
          <input type="range" v-model="quadC" min="-10" max="10" step="0.1" @input="updatePlot">
          <span>{{ quadC }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'exponential'" class="parameter-group">
        <div class="slider-group">
          <label>Amplitude (a):</label>
          <input type="range" v-model="expA" min="0.1" max="3" step="0.1" @input="updatePlot">
          <span>{{ expA }}</span>
        </div>
        <div class="slider-group">
          <label>Growth rate (b):</label>
          <input type="range" v-model="expB" min="-2" max="2" step="0.1" @input="updatePlot">
          <span>{{ expB }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'trigonometric'" class="parameter-group">
        <div class="slider-group">
          <label>Amplitude (a):</label>
          <input type="range" v-model="trigA" min="0.1" max="3" step="0.1" @input="updatePlot">
          <span>{{ trigA }}</span>
        </div>
        <div class="slider-group">
          <label>Frequency (b):</label>
          <input type="range" v-model="trigB" min="0.1" max="3" step="0.1" @input="updatePlot">
          <span>{{ trigB }}</span>
        </div>
        <div class="slider-group">
          <label>Phase shift (c):</label>
          <input type="range" v-model="trigC" min="-3.14" max="3.14" step="0.1" @input="updatePlot">
          <span>{{ trigC }}</span>
        </div>
      </div>
      
      <div class="point-input">
        <label>Evaluate at x:</label>
        <input type="number" v-model="evalX" step="0.1" @input="evaluateFunction" class="number-input">
        <span class="result">f({{ evalX }}) = {{ evaluatedY }}</span>
      </div>
    </div>
    
    <div class="plot-container">
      <canvas ref="plotCanvas" width="600" height="400"></canvas>
    </div>
    
    <div class="function-info">
      <h4>Current Function:</h4>
      <div class="equation">{{ currentEquation }}</div>
      <div class="properties">
        <strong>Properties:</strong>
        <ul>
          <li v-for="prop in currentProperties" :key="prop">{{ prop }}</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'

const functionType = ref('linear')
const slope = ref(1)
const yIntercept = ref(0)
const quadA = ref(1)
const quadB = ref(0)
const quadC = ref(0)
const expA = ref(1)
const expB = ref(1)
const trigA = ref(1)
const trigB = ref(1)
const trigC = ref(0)
const evalX = ref(0)
const evaluatedY = ref(0)
const plotCanvas = ref(null)

const currentEquation = computed(() => {
  switch (functionType.value) {
    case 'linear':
      return `f(x) = ${slope.value}x + ${yIntercept.value}`
    case 'quadratic':
      return `f(x) = ${quadA.value}x² + ${quadB.value}x + ${quadC.value}`
    case 'exponential':
      return `f(x) = ${expA.value}·e^(${expB.value}x)`
    case 'trigonometric':
      return `f(x) = ${trigA.value}·sin(${trigB.value}x + ${trigC.value})`
    default:
      return ''
  }
})

const currentProperties = computed(() => {
  switch (functionType.value) {
    case 'linear':
      return [
        `Slope: ${slope.value}`,
        slope.value > 0 ? 'Increasing' : slope.value < 0 ? 'Decreasing' : 'Constant',
        `Y-intercept: (0, ${yIntercept.value})`
      ]
    case 'quadratic':
      const discriminant = quadB.value * quadB.value - 4 * quadA.value * quadC.value
      const vertex_x = -quadB.value / (2 * quadA.value)
      const vertex_y = evaluateFunction(vertex_x)
      return [
        quadA.value > 0 ? 'Opens upward' : 'Opens downward',
        `Vertex: (${vertex_x.toFixed(2)}, ${vertex_y.toFixed(2)})`,
        discriminant > 0 ? '2 real roots' : discriminant === 0 ? '1 real root' : 'No real roots'
      ]
    case 'exponential':
      return [
        expB.value > 0 ? 'Growth function' : 'Decay function',
        `Horizontal asymptote: y = 0`,
        `Y-intercept: (0, ${expA.value})`
      ]
    case 'trigonometric':
      return [
        `Amplitude: ${trigA.value}`,
        `Period: ${(2 * Math.PI / trigB.value).toFixed(2)}`,
        `Phase shift: ${trigC.value.toFixed(2)}`
      ]
    default:
      return []
  }
})

const evaluateFunction = (x = evalX.value) => {
  const xVal = parseFloat(x)
  let result
  
  switch (functionType.value) {
    case 'linear':
      result = slope.value * xVal + parseFloat(yIntercept.value)
      break
    case 'quadratic':
      result = quadA.value * xVal * xVal + quadB.value * xVal + parseFloat(quadC.value)
      break
    case 'exponential':
      result = expA.value * Math.exp(expB.value * xVal)
      break
    case 'trigonometric':
      result = trigA.value * Math.sin(trigB.value * xVal + parseFloat(trigC.value))
      break
    default:
      result = 0
  }
  
  if (x === evalX.value) {
    evaluatedY.value = result.toFixed(3)
  }
  return result
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
  const yMin = -10
  const yMax = 10
  
  const xScale = width / (xMax - xMin)
  const yScale = height / (yMax - yMin)
  
  // Convert world coordinates to canvas coordinates
  const toCanvasX = (x) => (x - xMin) * xScale
  const toCanvasY = (y) => height - (y - yMin) * yScale
  
  // Draw grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 1
  
  // Vertical grid lines
  for (let x = xMin; x <= xMax; x++) {
    ctx.beginPath()
    ctx.moveTo(toCanvasX(x), 0)
    ctx.lineTo(toCanvasX(x), height)
    ctx.stroke()
  }
  
  // Horizontal grid lines
  for (let y = yMin; y <= yMax; y++) {
    ctx.beginPath()
    ctx.moveTo(0, toCanvasY(y))
    ctx.lineTo(width, toCanvasY(y))
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
  
  // Draw function
  ctx.strokeStyle = '#2196F3'
  ctx.lineWidth = 3
  ctx.beginPath()
  
  let firstPoint = true
  const step = 0.1
  
  for (let x = xMin; x <= xMax; x += step) {
    const y = evaluateFunction(x)
    
    if (isFinite(y) && y >= yMin && y <= yMax) {
      const canvasX = toCanvasX(x)
      const canvasY = toCanvasY(y)
      
      if (firstPoint) {
        ctx.moveTo(canvasX, canvasY)
        firstPoint = false
      } else {
        ctx.lineTo(canvasX, canvasY)
      }
    } else {
      firstPoint = true
    }
  }
  
  ctx.stroke()
  
  // Draw evaluation point
  if (isFinite(evaluatedY.value)) {
    const pointX = toCanvasX(evalX.value)
    const pointY = toCanvasY(parseFloat(evaluatedY.value))
    
    ctx.fillStyle = '#FF5722'
    ctx.beginPath()
    ctx.arc(pointX, pointY, 5, 0, 2 * Math.PI)
    ctx.fill()
  }
  
  // Update evaluation
  evaluateFunction()
}

onMounted(() => {
  updatePlot()
})
</script>

<style scoped>
.function-plotter {
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

.input-group, .slider-group {
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.input-group label, .slider-group label {
  font-weight: 500;
  min-width: 120px;
}

.parameter-group {
  margin: 1rem 0;
  padding: 1rem;
  background: #f0f0f0;
  border-radius: 4px;
}

.slider-group input[type="range"] {
  width: 200px;
}

.slider-group span {
  min-width: 40px;
  font-weight: bold;
  color: #2196F3;
}

.point-input {
  margin-top: 1rem;
  padding: 1rem;
  background: #e8f4fd;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.number-input {
  width: 80px;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.result {
  font-weight: bold;
  color: #FF5722;
}

.plot-container {
  margin: 1.5rem 0;
  text-align: center;
}

.plot-container canvas {
  border: 2px solid #ccc;
  border-radius: 4px;
  background: white;
}

.function-info {
  margin-top: 1rem;
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 4px;
}

.equation {
  font-size: 1.2em;
  font-weight: bold;
  color: #2196F3;
  margin: 0.5rem 0;
  font-family: 'Times New Roman', serif;
}

.properties ul {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.properties li {
  margin: 0.25rem 0;
}

select {
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
}
</style>