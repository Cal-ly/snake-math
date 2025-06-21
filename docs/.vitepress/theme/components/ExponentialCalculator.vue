<template>
  <div class="interactive-component">
    <div class="component-section">
      <h3 class="section-title">Exponential & Logarithmic Functions</h3>
      
      <div class="input-group">
        <label>Function Type: </label>
        <select v-model="functionType" @change="updatePlot" class="function-select">
          <option value="exponential">Exponential: f(x) = a·e^(bx)</option>
          <option value="logarithmic">Logarithmic: f(x) = a·ln(bx)</option>
          <option value="compound">Compound Interest</option>
        </select>
      </div>
      
      <div v-if="functionType === 'exponential'" class="input-group">
        <h4 class="input-group-title">Exponential Parameters</h4>
        <div class="component-inputs">
          <label>Amplitude (a):</label>
          <input type="range" v-model="expA" min="0.1" max="3" step="0.1" @input="updatePlot" class="range-input">
          <span class="result-value">{{ expA }}</span>
        </div>
        <div class="component-inputs">
          <label>Growth rate (b):</label>
          <input type="range" v-model="expB" min="-2" max="2" step="0.1" @input="updatePlot" class="range-input">
          <span class="result-value">{{ expB }}</span>
          <span class="result-label">{{ expB > 0 ? '(growth)' : '(decay)' }}</span>
        </div>
        <div class="equation-display">
          <span class="equation-text">{{ currentEquation }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'logarithmic'" class="input-group">
        <h4 class="input-group-title">Logarithmic Parameters</h4>
        <div class="component-inputs">
          <label>Amplitude (a):</label>
          <input type="range" v-model="logA" min="0.1" max="3" step="0.1" @input="updatePlot" class="range-input">
          <span class="result-value">{{ logA }}</span>
        </div>
        <div class="component-inputs">
          <label>Scale (b):</label>
          <input type="range" v-model="logB" min="0.1" max="3" step="0.1" @input="updatePlot" class="range-input">
          <span class="result-value">{{ logB }}</span>
        </div>
        <div class="equation-display">
          <span class="equation-text">{{ currentEquation }}</span>
        </div>
      </div>
      
      <div v-if="functionType === 'compound'" class="input-group">
        <h4 class="input-group-title">Compound Interest Parameters</h4>
        <div class="component-inputs">
          <label>Principal (P):</label>
          <input type="number" v-model="principal" min="1" step="100" @input="updateCompound">
          <span>$</span>
        </div>
        <div class="component-inputs">
          <label>Interest Rate (r):</label>
          <input type="number" v-model="interestRate" min="0" max="0.2" step="0.001" @input="updateCompound">
          <span>{{ (interestRate * 100).toFixed(1) }}%</span>
        </div>
        <div class="component-inputs">
          <label>Compounds per year (n):</label>
          <select v-model="compoundFreq" @change="updateCompound">
            <option value="1">Annually</option>
            <option value="2">Semi-annually</option>
            <option value="4">Quarterly</option>
            <option value="12">Monthly</option>
            <option value="365">Daily</option>
          </select>
        </div>
        <div class="component-inputs">
          <label>Time (years):</label>
          <input type="range" v-model="timeYears" min="1" max="30" step="1" @input="updateCompound" class="range-input">
          <span class="result-value">{{ timeYears }} years</span>
        </div>
        <div class="equation-display">
          <span class="equation-text">{{ currentEquation }}</span>
        </div>
      </div>
    </div>
    
    <div class="visualization-container">
      <canvas
      ref="plotCanvas"
      :width="canvasWidth"
      :height="canvasHeight"
      class="visualization-canvas"
      style="width: 100%; height: auto; max-width: 700px; display: block;"
      ></canvas>
    </div>
    
    <div v-if="functionType === 'compound'" class="component-section">
      <h4 class="input-group-title">Compound Interest Results</h4>
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
    
    <div class="component-section">
      <h4 class="input-group-title">Function Calculator</h4>
      <div class="component-inputs">
        <label>Evaluate at x =</label>
        <input type="number" v-model="evalX" @input="evaluateFunction" step="0.1" class="eval-input">
        <span class="result-value">f({{ evalX }}) = {{ evalResult }}</span>
      </div>
      
      <div v-if="functionType === 'exponential'" class="function-properties">
        <h4 class="input-group-title">Properties</h4>
        <ul>
          <li>Domain: (-∞, ∞)</li>
          <li>Range: {{ expB > 0 ? '(0, ∞)' : '(0, ∞)' }}</li>
          <li>Horizontal asymptote: y = 0</li>
          <li>Y-intercept: (0, {{ expA }})</li>
          <li>{{ expB > 0 ? 'Increasing' : 'Decreasing' }} function</li>
        </ul>
      </div>
      
      <div v-if="functionType === 'logarithmic'" class="function-properties">
        <h4 class="input-group-title">Properties</h4>
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
      <h4 class="input-group-title">Real-World Applications</h4>
      <div class="controls-grid">
        <div class="interactive-card" @click="loadApplication('population')" :class="{ active: currentApp === 'population' }">
          <h5 class="app-title">Population Growth</h5>
          <p class="app-description">Exponential model: P(t) = P₀·e^(rt)</p>
        </div>
        <div class="interactive-card" @click="loadApplication('radioactive')" :class="{ active: currentApp === 'radioactive' }">
          <h5 class="app-title">Radioactive Decay</h5>
          <p class="app-description">Decay model: N(t) = N₀·e^(-λt)</p>
        </div>
        <div class="interactive-card" @click="loadApplication('investment')" :class="{ active: currentApp === 'investment' }">
          <h5 class="app-title">Investment Growth</h5>
          <p class="app-description">Compound interest: A = P(1 + r/n)^(nt)</p>
        </div>
      </div>
      
      <div v-if="currentApp" class="app-details fade-in">
        <h5 class="app-title">{{ applicationDetails.title }}</h5>
        <p class="app-description">{{ applicationDetails.description }}</p>
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
@import '../styles/components.css';

/* Component-specific styles only */
.applications {
  margin: 1.5rem 0;
}

.equation-display {
  margin: 0.2rem;
  text-align: center;
  padding: 0.2rem;
  background: #f8f9fa;
  border-radius: 2px;
  border: 1px solid #e9ecef;
}
</style>