<template>
  <div class="probability-simulator">
    <div class="controls">
      <h3>Probability Distribution Simulator</h3>
      
      <div class="distribution-selector">
        <label>Distribution Type:</label>
        <select v-model="distributionType" @change="updateSimulation">
          <option value="normal">Normal Distribution</option>
          <option value="binomial">Binomial Distribution</option>
          <option value="uniform">Uniform Distribution</option>
          <option value="exponential">Exponential Distribution</option>
        </select>
      </div>
      
      <div v-if="distributionType === 'normal'" class="parameters">
        <div class="input-group">
          <label>Mean (μ):</label>
          <input type="range" v-model="normalMean" min="-5" max="5" step="0.1" @input="updateSimulation">
          <span>{{ normalMean }}</span>
        </div>
        <div class="input-group">
          <label>Standard Deviation (σ):</label>
          <input type="range" v-model="normalStd" min="0.1" max="3" step="0.1" @input="updateSimulation">
          <span>{{ normalStd }}</span>
        </div>
      </div>
      
      <div v-if="distributionType === 'binomial'" class="parameters">
        <div class="input-group">
          <label>Number of trials (n):</label>
          <input type="range" v-model="binomialN" min="1" max="50" step="1" @input="updateSimulation">
          <span>{{ binomialN }}</span>
        </div>
        <div class="input-group">
          <label>Probability of success (p):</label>
          <input type="range" v-model="binomialP" min="0" max="1" step="0.01" @input="updateSimulation">
          <span>{{ binomialP }}</span>
        </div>
      </div>
      
      <div class="simulation-controls">
        <div class="input-group">
          <label>Sample Size:</label>
          <input type="range" v-model="sampleSize" min="100" max="10000" step="100" @input="updateSimulation">
          <span>{{ sampleSize }}</span>
        </div>
        
        <button @click="runSimulation" class="simulate-btn">Generate New Sample</button>
      </div>
    </div>
    
    <div class="visualization">
      <canvas ref="histogramCanvas" width="600" height="400"></canvas>
    </div>
    
    <div class="statistics">
      <h4>Sample Statistics:</h4>
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-label">Sample Mean</div>
          <div class="stat-value">{{ sampleStats.mean }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Sample Std Dev</div>
          <div class="stat-value">{{ sampleStats.std }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Sample Min</div>
          <div class="stat-value">{{ sampleStats.min }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Sample Max</div>
          <div class="stat-value">{{ sampleStats.max }}</div>
        </div>
      </div>
      
      <div class="theoretical-comparison">
        <h4>Theoretical vs Sample:</h4>
        <div class="comparison-grid">
          <div class="comparison-item">
            <div class="comparison-label">Mean</div>
            <div class="comparison-values">
              <div>Theoretical: {{ theoreticalStats.mean }}</div>
              <div>Sample: {{ sampleStats.mean }}</div>
            </div>
          </div>
          <div class="comparison-item">
            <div class="comparison-label">Standard Deviation</div>
            <div class="comparison-values">
              <div>Theoretical: {{ theoreticalStats.std }}</div>
              <div>Sample: {{ sampleStats.std }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="probability-calculator">
      <h4>Probability Calculator:</h4>
      <div class="calc-controls">
        <div class="input-group">
          <label>P(X ≤ value):</label>
          <input type="number" v-model="probValue" @input="calculateProbability" step="0.1">
          <span class="prob-result">≈ {{ calculatedProbability }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const distributionType = ref('normal')
const normalMean = ref(0)
const normalStd = ref(1)
const binomialN = ref(20)
const binomialP = ref(0.5)
const sampleSize = ref(1000)
const probValue = ref(0)
const calculatedProbability = ref(50)
const samples = ref([])
const histogramCanvas = ref(null)

const theoreticalStats = computed(() => {
  switch (distributionType.value) {
    case 'normal':
      return {
        mean: parseFloat(normalMean.value.toFixed(3)),
        std: parseFloat(normalStd.value.toFixed(3))
      }
    case 'binomial':
      return {
        mean: parseFloat((binomialN.value * binomialP.value).toFixed(3)),
        std: parseFloat(Math.sqrt(binomialN.value * binomialP.value * (1 - binomialP.value)).toFixed(3))
      }
    case 'uniform':
      return {
        mean: 0.5,
        std: parseFloat((1 / Math.sqrt(12)).toFixed(3))
      }
    case 'exponential':
      return {
        mean: 1.0,
        std: 1.0
      }
    default:
      return { mean: 0, std: 1 }
  }
})

const sampleStats = computed(() => {
  if (samples.value.length === 0) return { mean: 0, std: 0, min: 0, max: 0 }
  
  const mean = samples.value.reduce((sum, val) => sum + val, 0) / samples.value.length
  const variance = samples.value.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / samples.value.length
  const std = Math.sqrt(variance)
  const min = Math.min(...samples.value)
  const max = Math.max(...samples.value)
  
  return {
    mean: parseFloat(mean.toFixed(3)),
    std: parseFloat(std.toFixed(3)),
    min: parseFloat(min.toFixed(3)),
    max: parseFloat(max.toFixed(3))
  }
})

const generateSamples = () => {
  const newSamples = []
  
  switch (distributionType.value) {
    case 'normal':
      for (let i = 0; i < sampleSize.value; i++) {
        newSamples.push(normalRandom(normalMean.value, normalStd.value))
      }
      break
    case 'binomial':
      for (let i = 0; i < sampleSize.value; i++) {
        newSamples.push(binomialRandom(binomialN.value, binomialP.value))
      }
      break
    case 'uniform':
      for (let i = 0; i < sampleSize.value; i++) {
        newSamples.push(Math.random())
      }
      break
    case 'exponential':
      for (let i = 0; i < sampleSize.value; i++) {
        newSamples.push(-Math.log(1 - Math.random()))
      }
      break
  }
  
  samples.value = newSamples
}

const normalRandom = (mean, std) => {
  // Box-Muller transform
  const u1 = Math.random()
  const u2 = Math.random()
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  return mean + std * z
}

const binomialRandom = (n, p) => {
  let successes = 0
  for (let i = 0; i < n; i++) {
    if (Math.random() < p) successes++
  }
  return successes
}

const calculateProbability = () => {
  if (samples.value.length === 0) return
  
  const count = samples.value.filter(val => val <= probValue.value).length
  calculatedProbability.value = parseFloat(((count / samples.value.length) * 100).toFixed(1))
}

const drawHistogram = () => {
  const canvas = histogramCanvas.value
  if (!canvas || samples.value.length === 0) return
  
  const ctx = canvas.getContext('2d')
  const width = canvas.width
  const height = canvas.height
  
  ctx.clearRect(0, 0, width, height)
  
  // Create bins
  const binCount = 30
  const min = Math.min(...samples.value)
  const max = Math.max(...samples.value)
  const binWidth = (max - min) / binCount
  
  const bins = new Array(binCount).fill(0)
  samples.value.forEach(value => {
    const binIndex = Math.min(Math.floor((value - min) / binWidth), binCount - 1)
    bins[binIndex]++
  })
  
  const maxFreq = Math.max(...bins)
  const margin = 40
  const plotWidth = width - 2 * margin
  const plotHeight = height - 2 * margin
  
  // Draw bars
  const barWidth = plotWidth / binCount
  
  bins.forEach((freq, i) => {
    const barHeight = (freq / maxFreq) * plotHeight
    const x = margin + i * barWidth
    const y = height - margin - barHeight
    
    ctx.fillStyle = '#2196F3'
    ctx.fillRect(x, y, barWidth - 2, barHeight)
  })
  
  // Draw theoretical curve overlay for normal distribution
  if (distributionType.value === 'normal') {
    ctx.strokeStyle = '#F44336'
    ctx.lineWidth = 3
    ctx.beginPath()
    
    let first = true
    for (let x = min; x <= max; x += (max - min) / 200) {
      const density = normalPDF(x, normalMean.value, normalStd.value)
      const scaledDensity = (density * samples.value.length * binWidth / maxFreq) * plotHeight
      const canvasX = margin + ((x - min) / (max - min)) * plotWidth
      const canvasY = height - margin - scaledDensity
      
      if (first) {
        ctx.moveTo(canvasX, canvasY)
        first = false
      } else {
        ctx.lineTo(canvasX, canvasY)
      }
    }
    ctx.stroke()
  }
  
  // Draw axes
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(margin, height - margin)
  ctx.lineTo(width - margin, height - margin)
  ctx.moveTo(margin, margin)
  ctx.lineTo(margin, height - margin)
  ctx.stroke()
  
  // Labels
  ctx.fillStyle = '#000000'
  ctx.font = '12px Arial'
  ctx.textAlign = 'center'
  
  // X-axis labels
  for (let i = 0; i <= 5; i++) {
    const value = min + (i / 5) * (max - min)
    const x = margin + (i / 5) * plotWidth
    ctx.fillText(value.toFixed(2), x, height - 5)
  }
  
  // Y-axis labels
  ctx.textAlign = 'right'
  for (let i = 0; i <= 5; i++) {
    const value = (i / 5) * maxFreq
    const y = height - margin - (i / 5) * plotHeight
    ctx.fillText(Math.round(value), margin - 5, y + 3)
  }
}

const normalPDF = (x, mean, std) => {
  return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2))
}

const runSimulation = () => {
  generateSamples()
  drawHistogram()
  calculateProbability()
}

const updateSimulation = () => {
  setTimeout(() => {
    runSimulation()
  }, 100)
}

watch([distributionType, sampleSize], updateSimulation)

onMounted(() => {
  runSimulation()
})
</script>

<style scoped>
.probability-simulator {
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

.distribution-selector {
  margin: 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.distribution-selector select {
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.parameters {
  margin: 1rem 0;
  padding: 1rem;
  background: white;
  border-radius: 4px;
}

.input-group {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 0.5rem 0;
}

.input-group label {
  min-width: 150px;
}

.input-group input[type="range"] {
  width: 150px;
}

.input-group span {
  font-weight: bold;
  color: #2196F3;
  min-width: 60px;
}

.simulation-controls {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f0f0f0;
  border-radius: 4px;
}

.simulate-btn {
  padding: 0.5rem 1rem;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  margin-left: 1rem;
}

.simulate-btn:hover {
  background: #45a049;
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

.statistics {
  margin: 1.5rem 0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.stat-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  text-align: center;
}

.stat-label {
  font-size: 0.9em;
  color: #666;
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 1.3em;
  font-weight: bold;
  color: #2196F3;
}

.theoretical-comparison {
  margin: 1.5rem 0;
}

.comparison-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.comparison-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.comparison-label {
  font-weight: bold;
  margin-bottom: 0.5rem;
  color: #333;
}

.comparison-values div {
  margin: 0.25rem 0;
  font-size: 0.9em;
}

.probability-calculator {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 4px;
}

.calc-controls {
  margin: 1rem 0;
}

.calc-controls input[type="number"] {
  width: 80px;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.prob-result {
  font-weight: bold;
  color: #FF5722;
  margin-left: 1rem;
}
</style>