<template>
  <div class="exponential-calculator">
    <div class="controls">
      <h3>Exponential & Logarithmic Functions</h3>
      
      <div class="function-type">
        <label>Function Type:</label>
        <select v-model="functionType" @change="updatePlot">
          <option value="exponential">Exponential: f(x) = a·e^(bx)</option>
          <option value="logarithmic">Logarithmic: f(x) = a·ln(bx)</option>
          <option value="compound">Compound Interest</option>
        </select>
      </div>
      
      <div v-if="functionType === 'exponential'" class="parameters">
        <div class="slider-group">
          <label>Amplitude (a):</label>
          <input type="range" v-model="expA" min="0.1" max="3" step="0.1" @input="updatePlot">
          <span class="value">{{ expA }}</span>
        </div>
        <div class="slider-group">
          <label>Growth rate (b):</label>
          <input type="range" v-model="expB" min="-2" max="2" step="0.1" @input="updatePlot">
          <span class="value">{{ expB }}</span>
          <span class="description">{{ expB > 0 ? '(growth)' : '(decay)' }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'logarithmic'" class="parameters">
        <div class="slider-group">
          <label>Amplitude (a):</label>
          <input type="range" v-model="logA" min="0.1" max="3" step="0.1" @input="updatePlot">
          <span class="value">{{ logA }}</span>
        </div>
        <div class="slider-group">
          <label>Scale (b):</label>
          <input type="range" v-model="logB" min="0.1" max="3" step="0.1" @input="updatePlot">
          <span class="value">{{ logB }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'compound'" class="compound-inputs">
        <div class="input-group">
          <label>Principal (P):</label>
          <input type="number" v-model="principal" min="1" step="100" @input="updateCompound">
          <span>$</span>
        </div>
        <div class="input-group">
          <label>Interest Rate (r):</label>
          <input type="number" v-model="interestRate" min="0" max="0.2" step="0.001" @input="updateCompound">
          <span>{{ (interestRate * 100).toFixed(1) }}%</span>
        </div>
        <div class="input-group">
          <label>Compounds per year (n):</label>
          <select v-model="compoundFreq" @change="updateCompound">
            <option value="1">Annually</option>
            <option value="2">Semi-annually</option>
            <option value="4">Quarterly</option>
            <option value="12">Monthly</option>
            <option value="365">Daily</option>
          </select>
        </div>
        <div class="input-group">
          <label>Time (years):</label>
          <input type="range" v-model="timeYears" min="1" max="30" step="1" @input="updateCompound">
          <span class="value">{{ timeYears }} years</span>
        </div>
      </div>
    </div>
    
    <div class="equation-display">
      <div class="current-equation">{{ currentEquation }}</div>
    </div>
    
    <div class="visualization">
      <canvas ref="plotCanvas" width="600" height="400"></canvas>
    </div>
    
    <div v-if="functionType === 'compound'" class="compound-results">
      <h4>Compound Interest Results</h4>
      <div class="results-grid">
        <div class="result-card">
          <div class="result-label">Final Amount</div>
          <div class="result-value">${{ compoundResult.finalAmount }}</div>
        </div>
        <div class="result-card">
          <div class="result-label">Interest Earned</div>
          <div class="result-value">${{ compoundResult.interestEarned }}</div>
        </div>
        <div class="result-card">
          <div class="result-label">Growth Factor</div>
          <div class="result-value">{{ compoundResult.growthFactor }}x</div>
        </div>
        <div class="result-card">
          <div class="result-label">Effective Annual Rate</div>
          <div class="result-value">{{ compoundResult.effectiveRate }}%</div>
        </div>
      </div>
    </div>
    
    <div class="calculator-section">
      <h4>Function Calculator</h4>
      <div class="calc-input">
        <label>Evaluate at x =</label>
        <input type="number" v-model="evalX" @input="evaluateFunction" step="0.1" class="eval-input">
        <span class="calc-result">f({{ evalX }}) = {{ evalResult }}</span>
      </div>
      
      <div v-if="functionType === 'exponential'" class="exponential-properties">
        <h4>Properties</h4>
        <ul>
          <li>Domain: (-∞, ∞)</li>
          <li>Range: {{ expB > 0 ? '(0, ∞)' : '(0, ∞)' }}</li>
          <li>Horizontal asymptote: y = 0</li>
          <li>Y-intercept: (0, {{ expA }})</li>
          <li>{{ expB > 0 ? 'Increasing' : 'Decreasing' }} function</li>
        </ul>
      </div>
      
      <div v-if="functionType === 'logarithmic'" class="logarithmic-properties">
        <h4>Properties</h4>
        <ul>
          <li>Domain: (0, ∞)</li>
          <li>Range: (-∞, ∞)</li>
          <li>Vertical asymptote: x = 0</li>
          <li>X-intercept: (1, 0)</li>
          <li>Increasing function</li>
        </ul>
      </div>
    </div>
    
    <div class="applications">
      <h4>Real-World Applications</h4>
      <div class="app-grid">
        <div class="app-card" @click="loadApplication('population')" :class="{ active: currentApp === 'population' }">
          <h5>Population Growth</h5>
          <p>Exponential model: P(t) = P₀·e^(rt)</p>
        </div>
        <div class="app-card" @click="loadApplication('radioactive')" :class="{ active: currentApp === 'radioactive' }">
          <h5>Radioactive Decay</h5>
          <p>Decay model: N(t) = N₀·e^(-λt)</p>
        </div>
        <div class="app-card" @click="loadApplication('investment')" :class="{ active: currentApp === 'investment' }">
          <h5>Investment Growth</h5>
          <p>Compound interest: A = P(1 + r/n)^(nt)</p>
        </div>
      </div>
      
      <div v-if="currentApp" class="app-details">
        <h5>{{ applicationDetails.title }}</h5>
        <p>{{ applicationDetails.description }}</p>
        <div class="app-parameters">
          <div v-for="param in applicationDetails.parameters" :key="param.name" class="app-param">
            <strong>{{ param.name }}:</strong> {{ param.value }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const functionType = ref('exponential')
const expA = ref(1)
const expB = ref(1)
const logA = ref(1)
const logB = ref(1)
const principal = ref(1000)
const interestRate = ref(0.05)
const compoundFreq = ref(12)
const timeYears = ref(10)
const evalX = ref(1)
const evalResult = ref(0)
const currentApp = ref(null)
const plotCanvas = ref(null)

const currentEquation = computed(() => {
  switch (functionType.value) {
    case 'exponential':
      return `f(x) = ${expA.value}·e^(${expB.value}x)`
    case 'logarithmic':
      return `f(x) = ${logA.value}·ln(${logB.value}x)`
    case 'compound':
      return `A(t) = ${principal.value}(1 + ${interestRate.value}/${compoundFreq.value})^(${compoundFreq.value}t)`
    default:
      return ''
  }
})

const compoundResult = computed(() => {
  const P = parseFloat(principal.value)
  const r = parseFloat(interestRate.value)
  const n = parseFloat(compoundFreq.value)
  const t = parseFloat(timeYears.value)
  
  const finalAmount = P * Math.pow(1 + r/n, n*t)
  const interestEarned = finalAmount - P
  const growthFactor = finalAmount / P
  const effectiveRate = (Math.pow(1 + r/n, n) - 1) * 100
  
  return {
    finalAmount: finalAmount.toFixed(2),
    interestEarned: interestEarned.toFixed(2),
    growthFactor: growthFactor.toFixed(2),
    effectiveRate: effectiveRate.toFixed(2)
  }
})

const applicationDetails = computed(() => {
  switch (currentApp.value) {
    case 'population':
      return {
        title: 'Population Growth Model',
        description: 'Many populations grow exponentially when resources are abundant. The growth rate determines how quickly the population doubles.',
        parameters: [
          { name: 'Initial Population (P₀)', value: '1000 individuals' },
          { name: 'Growth Rate (r)', value: '0.03 per year (3%)' },
          { name: 'Doubling Time', value: `${(Math.log(2) / 0.03).toFixed(1)} years` }
        ]
      }
    case 'radioactive':
      return {
        title: 'Radioactive Decay Model',
        description: 'Radioactive substances decay exponentially. The half-life is the time it takes for half the material to decay.',
        parameters: [
          { name: 'Initial Amount (N₀)', value: '1000 grams' },
          { name: 'Decay Constant (λ)', value: '0.1 per year' },
          { name: 'Half-life', value: `${(Math.log(2) / 0.1).toFixed(1)} years` }
        ]
      }
    case 'investment':
      return {
        title: 'Investment Growth Model',
        description: 'Compound interest causes investments to grow exponentially. More frequent compounding leads to faster growth.',
        parameters: [
          { name: 'Principal', value: `$${principal.value}` },
          { name: 'Annual Rate', value: `${(interestRate.value * 100).toFixed(1)}%` },
          { name: 'Compounding', value: getCompoundingName(compoundFreq.value) }
        ]
      }
    default:
      return null
  }
})

const getCompoundingName = (freq) => {
  const names = { 1: 'Annually', 2: 'Semi-annually', 4: 'Quarterly', 12: 'Monthly', 365: 'Daily' }
  return names[freq] || `${freq} times per year`
}

const evaluateFunction = () => {
  const x = parseFloat(evalX.value)
  let result
  
  switch (functionType.value) {
    case 'exponential':
      result = expA.value * Math.exp(expB.value * x)
      break
    case 'logarithmic':
      if (logB.value * x > 0) {
        result = logA.value * Math.log(logB.value * x)
      } else {
        result = 'undefined'
      }
      break
    case 'compound':
      const P = parseFloat(principal.value)
      const r = parseFloat(interestRate.value)
      const n = parseFloat(compoundFreq.value)
      result = P * Math.pow(1 + r/n, n*x)
      break
    default:
      result = 0
  }
  
  if (typeof result === 'number') {
    evalResult.value = result.toFixed(3)
  } else {
    evalResult.value = result
  }
  
  updatePlot()
}

const updateCompound = () => {
  evaluateFunction()
}

const loadApplication = (app) => {
  currentApp.value = currentApp.value === app ? null : app
  
  switch (app) {
    case 'population':
      functionType.value = 'exponential'
      expA.value = 1000
      expB.value = 0.03
      break
    case 'radioactive':
      functionType.value = 'exponential'
      expA.value = 1000
      expB.value = -0.1
      break
    case 'investment':
      functionType.value = 'compound'
      break
  }
  
  evaluateFunction()
}

const updatePlot = () => {
  const canvas = plotCanvas.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  const width = canvas.width
  const height = canvas.height
  
  ctx.clearRect(0, 0, width, height)
  
  let xMin, xMax, yMin, yMax
  
  if (functionType.value === 'logarithmic') {
    xMin = 0.1; xMax = 10; yMin = -5; yMax = 5
  } else {
    xMin = 0; xMax = 10; yMin = 0; yMax = 20
  }
  
  const xScale = width / (xMax - xMin)
  const yScale = height / (yMax - yMin)
  
  const toCanvasX = (x) => (x - xMin) * xScale
  const toCanvasY = (y) => height - (y - yMin) * yScale
  
  // Draw grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 1
  
  for (let x = Math.ceil(xMin); x <= xMax; x++) {
    ctx.beginPath()
    ctx.moveTo(toCanvasX(x), 0)
    ctx.lineTo(toCanvasX(x), height)
    ctx.stroke()
  }
  
  for (let y = Math.ceil(yMin); y <= yMax; y += 2) {
    ctx.beginPath()
    ctx.moveTo(0, toCanvasY(y))
    ctx.lineTo(width, toCanvasY(y))
    ctx.stroke()
  }
  
  // Draw axes
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2
  
  if (yMin <= 0 && yMax >= 0) {
    ctx.beginPath()
    ctx.moveTo(0, toCanvasY(0))
    ctx.lineTo(width, toCanvasY(0))
    ctx.stroke()
  }
  
  if (xMin <= 0 && xMax >= 0) {
    ctx.beginPath()
    ctx.moveTo(toCanvasX(0), 0)
    ctx.lineTo(toCanvasX(0), height)
    ctx.stroke()
  }
  
  // Draw function
  ctx.strokeStyle = '#2196F3'
  ctx.lineWidth = 3
  ctx.beginPath()
  
  let firstPoint = true
  const step = (xMax - xMin) / 1000
  
  for (let x = xMin; x <= xMax; x += step) {
    let y
    
    switch (functionType.value) {
      case 'exponential':
        y = expA.value * Math.exp(expB.value * x)
        break
      case 'logarithmic':
        if (logB.value * x > 0) {
          y = logA.value * Math.log(logB.value * x)
        } else {
          continue
        }
        break
      case 'compound':
        const P = parseFloat(principal.value)
        const r = parseFloat(interestRate.value)
        const n = parseFloat(compoundFreq.value)
        y = P * Math.pow(1 + r/n, n*x)
        break
      default:
        continue
    }
    
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
  if (evalX.value >= xMin && evalX.value <= xMax && typeof evalResult.value === 'string' && !isNaN(parseFloat(evalResult.value))) {
    const pointY = parseFloat(evalResult.value)
    if (pointY >= yMin && pointY <= yMax) {
      const pointX = toCanvasX(evalX.value)
      const pointCanvasY = toCanvasY(pointY)
      
      ctx.fillStyle = '#FF5722'
      ctx.beginPath()
      ctx.arc(pointX, pointCanvasY, 5, 0, 2 * Math.PI)
      ctx.fill()
    }
  }
}

watch([functionType, expA, expB, logA, logB], () => {
  evaluateFunction()
})

onMounted(() => {
  evaluateFunction()
})
</script>

<style scoped>
.exponential-calculator {
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

.function-type {
  margin: 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.function-type select {
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.parameters, .compound-inputs {
  margin: 1.5rem 0;
}

.slider-group, .input-group {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0;
}

.slider-group label, .input-group label {
  font-weight: 500;
  min-width: 120px;
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

.input-group input {
  width: 100px;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.equation-display {
  margin: 1.5rem 0;
  text-align: center;
  padding: 1rem;
  background: #f0f0f0;
  border-radius: 4px;
}

.current-equation {
  font-size: 1.3em;
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

.compound-results {
  margin: 1.5rem 0;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.result-card {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  text-align: center;
}

.result-label {
  font-size: 0.9em;
  color: #666;
  margin-bottom: 0.5rem;
}

.result-value {
  font-size: 1.3em;
  font-weight: bold;
  color: #4CAF50;
}

.calculator-section {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 4px;
}

.calc-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 1rem 0;
}

.eval-input {
  width: 80px;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.calc-result {
  font-weight: bold;
  color: #FF5722;
}

.exponential-properties ul, .logarithmic-properties ul {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.applications {
  margin: 1.5rem 0;
}

.app-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.app-card {
  padding: 1rem;
  border: 2px solid #ddd;
  border-radius: 4px;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
}

.app-card:hover {
  border-color: #2196F3;
  background: #f8f9fa;
}

.app-card.active {
  border-color: #4CAF50;
  background: #e8f5e8;
}

.app-card h5 {
  margin: 0 0 0.5rem 0;
  color: #333;
}

.app-card p {
  margin: 0;
  font-size: 0.9em;
  color: #666;
}

.app-details {
  margin: 1rem 0;
  padding: 1rem;
  background: #f0f0f0;
  border-radius: 4px;
}

.app-parameters {
  margin: 1rem 0;
}

.app-param {
  margin: 0.5rem 0;
  font-size: 0.9em;
}
</style>