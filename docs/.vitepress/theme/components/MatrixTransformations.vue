<template>
  <div class="matrix-transformations">
    <div class="controls">
      <h3>Interactive Matrix Transformations</h3>
      
      <div class="transformation-selector">
        <label>Transformation Type:</label>
        <select v-model="transformationType" @change="updateMatrix">
          <option value="identity">Identity</option>
          <option value="scale">Scale</option>
          <option value="rotation">Rotation</option>
          <option value="shear">Shear</option>
          <option value="reflection">Reflection</option>
          <option value="custom">Custom</option>
        </select>
      </div>
      
      <div v-if="transformationType === 'scale'" class="parameters">
        <div class="input-group">
          <label>Scale X:</label>
          <input type="range" v-model="scaleX" min="0.1" max="3" step="0.1" @input="updateMatrix">
          <span>{{ scaleX }}</span>
        </div>
        <div class="input-group">
          <label>Scale Y:</label>
          <input type="range" v-model="scaleY" min="0.1" max="3" step="0.1" @input="updateMatrix">
          <span>{{ scaleY }}</span>
        </div>
      </div>
      
      <div v-if="transformationType === 'rotation'" class="parameters">
        <div class="input-group">
          <label>Angle (degrees):</label>
          <input type="range" v-model="rotationAngle" min="0" max="360" step="1" @input="updateMatrix">
          <span>{{ rotationAngle }}°</span>
        </div>
      </div>
      
      <div v-if="transformationType === 'shear'" class="parameters">
        <div class="input-group">
          <label>Shear X:</label>
          <input type="range" v-model="shearX" min="-2" max="2" step="0.1" @input="updateMatrix">
          <span>{{ shearX }}</span>
        </div>
        <div class="input-group">
          <label>Shear Y:</label>
          <input type="range" v-model="shearY" min="-2" max="2" step="0.1" @input="updateMatrix">
          <span>{{ shearY }}</span>
        </div>
      </div>
      
      <div v-if="transformationType === 'custom'" class="matrix-input">
        <h4>Custom 2×2 Matrix:</h4>
        <div class="matrix-grid">
          <input type="number" v-model="customMatrix.a" @input="updateTransformation" step="0.1">
          <input type="number" v-model="customMatrix.b" @input="updateTransformation" step="0.1">
          <input type="number" v-model="customMatrix.c" @input="updateTransformation" step="0.1">
          <input type="number" v-model="customMatrix.d" @input="updateTransformation" step="0.1">
        </div>
      </div>
    </div>
    
    <div class="matrix-display">
      <h4>Transformation Matrix:</h4>
      <div class="matrix">
        <div class="matrix-bracket">[</div>
        <div class="matrix-content">
          <div class="matrix-row">{{ matrix.a }} {{ matrix.b }}</div>
          <div class="matrix-row">{{ matrix.c }} {{ matrix.d }}</div>
        </div>
        <div class="matrix-bracket">]</div>
      </div>
      
      <div class="matrix-properties">
        <div class="property">
          <strong>Determinant:</strong> {{ determinant }}
        </div>
        <div class="property">
          <strong>Effect:</strong> {{ matrixEffect }}
        </div>
      </div>
    </div>
    
    <div class="visualization">
      <canvas ref="transformCanvas" width="600" height="400"></canvas>
    </div>
    
    <div class="transformation-info">
      <h4>How it works:</h4>
      <div class="explanation">
        Matrix multiplication transforms each point (x, y) to (x', y') where:<br>
        x' = {{ matrix.a }}x + {{ matrix.b }}y<br>
        y' = {{ matrix.c }}x + {{ matrix.d }}y
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const transformationType = ref('identity')
const scaleX = ref(1)
const scaleY = ref(1)
const rotationAngle = ref(45)
const shearX = ref(0.5)
const shearY = ref(0)
const customMatrix = ref({ a: 1, b: 0, c: 0, d: 1 })
const transformCanvas = ref(null)

const matrix = ref({ a: 1, b: 0, c: 0, d: 1 })

const determinant = computed(() => {
  return parseFloat((matrix.value.a * matrix.value.d - matrix.value.b * matrix.value.c).toFixed(3))
})

const matrixEffect = computed(() => {
  const det = determinant.value
  if (Math.abs(det) < 0.001) return 'Collapses to lower dimension'
  if (det > 0) return `Preserves orientation, scales area by ${Math.abs(det)}`
  return `Reverses orientation, scales area by ${Math.abs(det)}`
})

const updateMatrix = () => {
  switch (transformationType.value) {
    case 'identity':
      matrix.value = { a: 1, b: 0, c: 0, d: 1 }
      break
    case 'scale':
      matrix.value = { a: scaleX.value, b: 0, c: 0, d: scaleY.value }
      break
    case 'rotation':
      const rad = rotationAngle.value * Math.PI / 180
      matrix.value = {
        a: parseFloat(Math.cos(rad).toFixed(3)),
        b: parseFloat(-Math.sin(rad).toFixed(3)),
        c: parseFloat(Math.sin(rad).toFixed(3)),
        d: parseFloat(Math.cos(rad).toFixed(3))
      }
      break
    case 'shear':
      matrix.value = { a: 1, b: shearX.value, c: shearY.value, d: 1 }
      break
    case 'reflection':
      matrix.value = { a: 1, b: 0, c: 0, d: -1 } // Reflection across x-axis
      break
    case 'custom':
      matrix.value = { ...customMatrix.value }
      break
  }
  drawTransformation()
}

const updateTransformation = () => {
  if (transformationType.value === 'custom') {
    matrix.value = { ...customMatrix.value }
    drawTransformation()
  }
}

const drawTransformation = () => {
  const canvas = transformCanvas.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  const width = canvas.width
  const height = canvas.height
  const centerX = width / 2
  const centerY = height / 2
  const scale = 50
  
  ctx.clearRect(0, 0, width, height)
  
  // Draw grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 1
  for (let i = -5; i <= 5; i++) {
    ctx.beginPath()
    ctx.moveTo(centerX + i * scale, 0)
    ctx.lineTo(centerX + i * scale, height)
    ctx.stroke()
    
    ctx.beginPath()
    ctx.moveTo(0, centerY + i * scale)
    ctx.lineTo(width, centerY + i * scale)
    ctx.stroke()
  }
  
  // Draw axes
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(0, centerY)
  ctx.lineTo(width, centerY)
  ctx.stroke()
  
  ctx.beginPath()
  ctx.moveTo(centerX, 0)
  ctx.lineTo(centerX, height)
  ctx.stroke()
  
  // Original unit square
  const square = [
    { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }, { x: 0, y: 0 }
  ]
  
  // Draw original square
  ctx.strokeStyle = '#2196F3'
  ctx.lineWidth = 2
  ctx.beginPath()
  square.forEach((point, i) => {
    const x = centerX + point.x * scale
    const y = centerY - point.y * scale
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
  
  // Transform and draw transformed square
  const transformed = square.map(point => ({
    x: matrix.value.a * point.x + matrix.value.b * point.y,
    y: matrix.value.c * point.x + matrix.value.d * point.y
  }))
  
  ctx.strokeStyle = '#F44336'
  ctx.lineWidth = 3
  ctx.beginPath()
  transformed.forEach((point, i) => {
    const x = centerX + point.x * scale
    const y = centerY - point.y * scale
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
  
  // Draw unit vectors
  const unitI = { x: 1, y: 0 }
  const unitJ = { x: 0, y: 1 }
  
  const transformedI = {
    x: matrix.value.a * unitI.x + matrix.value.b * unitI.y,
    y: matrix.value.c * unitI.x + matrix.value.d * unitI.y
  }
  
  const transformedJ = {
    x: matrix.value.a * unitJ.x + matrix.value.b * unitJ.y,
    y: matrix.value.c * unitJ.x + matrix.value.d * unitJ.y
  }
  
  // Draw original unit vectors
  ctx.strokeStyle = '#4CAF50'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(centerX, centerY)
  ctx.lineTo(centerX + scale, centerY)
  ctx.stroke()
  
  ctx.beginPath()
  ctx.moveTo(centerX, centerY)
  ctx.lineTo(centerX, centerY - scale)
  ctx.stroke()
  
  // Draw transformed unit vectors
  ctx.strokeStyle = '#FF9800'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(centerX, centerY)
  ctx.lineTo(centerX + transformedI.x * scale, centerY - transformedI.y * scale)
  ctx.stroke()
  
  ctx.beginPath()
  ctx.moveTo(centerX, centerY)
  ctx.lineTo(centerX + transformedJ.x * scale, centerY - transformedJ.y * scale)
  ctx.stroke()
  
  // Labels
  ctx.fillStyle = '#000000'
  ctx.font = '12px Arial'
  ctx.fillText('Original (blue)', 10, 20)
  ctx.fillStyle = '#2196F3'
  ctx.fillRect(120, 12, 20, 3)
  
  ctx.fillStyle = '#000000'
  ctx.fillText('Transformed (red)', 10, 40)
  ctx.fillStyle = '#F44336'
  ctx.fillRect(140, 32, 20, 3)
}

watch([transformationType, scaleX, scaleY, rotationAngle, shearX, shearY], updateMatrix)
watch(customMatrix, updateTransformation, { deep: true })

onMounted(() => {
  updateMatrix()
})
</script>

<style scoped>
.matrix-transformations {
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

.transformation-selector {
  margin: 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.transformation-selector select {
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
  min-width: 100px;
}

.input-group input[type="range"] {
  width: 150px;
}

.matrix-input {
  margin: 1rem 0;
  padding: 1rem;
  background: white;
  border-radius: 4px;
}

.matrix-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  width: 150px;
  margin: 1rem 0;
}

.matrix-grid input {
  width: 60px;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  text-align: center;
}

.matrix-display {
  margin: 1.5rem 0;
  text-align: center;
}

.matrix {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  margin: 1rem 0;
  font-family: 'Times New Roman', serif;
  font-size: 1.2em;
}

.matrix-bracket {
  font-size: 2em;
  font-weight: bold;
}

.matrix-content {
  text-align: center;
}

.matrix-row {
  margin: 0.25rem 0;
}

.matrix-properties {
  margin: 1rem 0;
}

.property {
  margin: 0.5rem 0;
  font-size: 0.9em;
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

.transformation-info {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f0f0f0;
  border-radius: 4px;
}

.explanation {
  font-family: 'Times New Roman', serif;
  font-size: 1.1em;
  color: #333;
  line-height: 1.5;
}
</style>