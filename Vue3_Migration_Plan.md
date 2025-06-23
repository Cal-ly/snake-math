# Snake Math: Vue 3 + Vite Migration Plan

## Why Vue 3 + Vite is Perfect for Snake Math

### Current Pain Points with VitePress
- ❌ Not designed for complex interactive applications
- ❌ Bundle optimization struggles with 19 components
- ❌ Limited routing flexibility for mathematical content
- ❌ SSG limitations for dynamic mathematical visualizations
- ❌ Poor performance with heavy canvas/mathematical computations

### Vue 3 + Vite Advantages
- ✅ **Composition API**: Perfect for mathematical state management
- ✅ **Vite Performance**: Sub-second dev builds, optimized production bundles
- ✅ **Component Chunking**: Automatic code splitting for your 19 interactive components
- ✅ **Mathematical Libraries**: Easy integration with NumJs, ML-Matrix, D3.js
- ✅ **Canvas Performance**: Native support for high-performance visualizations
- ✅ **PWA Ready**: Can work offline for better educational experience

## Migration Strategy

### Phase 1: Foundation Setup (Week 1)
1. **New Vue 3 + Vite Project Structure**
```
snake-math-vue3/
├── src/
│   ├── main.js                 # Vue 3 app entry
│   ├── App.vue                 # Root component
│   ├── router/                 # Vue Router for navigation
│   ├── components/             # Interactive math components
│   │   ├── common/             # Utility components (CodeFold, etc.)
│   │   ├── statistics/         # StatisticsCalculator, ProbabilitySimulator
│   │   ├── algebra/            # QuadraticExplorer, LinearSystemSolver
│   │   ├── trigonometry/       # UnitCircleExplorer
│   │   ├── linear-algebra/     # VectorOperations, MatrixTransformations
│   │   └── calculus/           # LimitsExplorer
│   ├── views/                  # Page components for content areas
│   ├── composables/            # Reusable mathematical logic
│   ├── utils/                  # Mathematical computation engine
│   └── assets/                 # Static assets
├── public/                     # Static files
├── content/                    # Markdown content (processed by custom loader)
└── dist/                       # Build output
```

2. **Enhanced Mathematical Architecture**
```javascript
// src/utils/MathEngine.js - Framework-agnostic mathematical core
export class MathematicalEngine {
  // High-precision calculations
  static calculateWithPrecision(value, decimals = 3) {
    return parseFloat(value.toFixed(decimals))
  }
  
  // Quadratic calculations with full analysis
  static analyzeQuadratic(a, b, c) {
    const discriminant = b * b - 4 * a * c
    const vertex = {
      x: -b / (2 * a),
      y: (4 * a * c - b * b) / (4 * a)
    }
    
    return {
      discriminant,
      vertex,
      roots: this.calculateQuadraticRoots(a, b, c),
      axisOfSymmetry: -b / (2 * a),
      yIntercept: c,
      hasRealRoots: discriminant >= 0
    }
  }
  
  // Statistics calculations
  static analyzeDataset(data) {
    const sorted = [...data].sort((a, b) => a - b)
    const n = data.length
    
    return {
      mean: data.reduce((sum, val) => sum + val, 0) / n,
      median: this.calculateMedian(sorted),
      mode: this.calculateMode(data),
      standardDeviation: this.calculateStandardDeviation(data),
      quartiles: this.calculateQuartiles(sorted),
      outliers: this.identifyOutliers(data)
    }
  }
}
```

### Phase 2: Component Migration (Week 2-3)

#### Convert Vue 2 Components to Vue 3 Composition API

**Example: StatisticsCalculator Migration**
```vue
<!-- src/components/statistics/StatisticsCalculator.vue -->
<template>
  <div class="statistics-calculator">
    <div class="input-section">
      <textarea 
        v-model="dataInput" 
        @input="updateDataset"
        placeholder="Enter numbers separated by commas or spaces"
        class="data-input"
      />
      <div class="preset-datasets">
        <button 
          v-for="preset in presetDatasets" 
          :key="preset.name"
          @click="loadPreset(preset)"
          class="preset-btn"
        >
          {{ preset.name }}
        </button>
      </div>
    </div>
    
    <div class="results-section" v-if="statistics">
      <div class="stats-grid">
        <div class="stat-item">
          <label>Mean:</label>
          <span>{{ statistics.mean }}</span>
        </div>
        <div class="stat-item">
          <label>Median:</label>
          <span>{{ statistics.median }}</span>
        </div>
        <div class="stat-item">
          <label>Standard Deviation:</label>
          <span>{{ statistics.standardDeviation }}</span>
        </div>
      </div>
      
      <!-- Canvas visualization -->
      <canvas 
        ref="visualizationCanvas"
        width="600" 
        height="400"
        class="visualization-canvas"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { MathematicalEngine } from '@/utils/MathEngine'
import { CanvasRenderer } from '@/utils/CanvasRenderer'

// Props with defaults
const props = defineProps({
  initialData: {
    type: Array,
    default: () => []
  },
  precision: {
    type: Number,
    default: 3
  },
  showVisualization: {
    type: Boolean,
    default: true
  }
})

// Reactive state
const dataInput = ref('')
const dataset = ref([...props.initialData])
const visualizationCanvas = ref(null)

// Computed statistics
const statistics = computed(() => {
  if (dataset.value.length === 0) return null
  return MathematicalEngine.analyzeDataset(dataset.value)
})

// Preset datasets for quick testing
const presetDatasets = [
  { name: 'Normal Distribution', data: [2, 4, 4, 4, 5, 5, 7, 9] },
  { name: 'Skewed Data', data: [1, 1, 2, 2, 2, 3, 8, 10, 15] },
  { name: 'Large Dataset', data: Array.from({length: 50}, () => Math.random() * 100) }
]

// Methods
const updateDataset = () => {
  const numbers = dataInput.value
    .split(/[,\s]+/)
    .map(str => parseFloat(str.trim()))
    .filter(num => !isNaN(num))
  
  dataset.value = numbers
}

const loadPreset = (preset) => {
  dataset.value = [...preset.data]
  dataInput.value = preset.data.join(', ')
}

const updateVisualization = async () => {
  if (!props.showVisualization || !visualizationCanvas.value || !statistics.value) return
  
  await nextTick()
  const canvas = visualizationCanvas.value
  const ctx = canvas.getContext('2d')
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // Render histogram and box plot
  CanvasRenderer.renderHistogram(ctx, dataset.value, statistics.value)
  CanvasRenderer.renderBoxPlot(ctx, statistics.value)
}

// Watchers
watch(statistics, updateVisualization, { deep: true })

// Lifecycle
onMounted(() => {
  if (props.initialData.length > 0) {
    dataInput.value = props.initialData.join(', ')
  }
  updateVisualization()
})

// Expose methods for parent components
defineExpose({
  reset: () => {
    dataset.value = []
    dataInput.value = ''
  },
  exportData: () => ({
    dataset: dataset.value,
    statistics: statistics.value
  })
})
</script>

<style scoped>
.statistics-calculator {
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1rem 0;
}

.data-input {
  width: 100%;
  min-height: 100px;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  resize: vertical;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 4px;
}

.visualization-canvas {
  border: 1px solid #ddd;
  border-radius: 4px;
  max-width: 100%;
  height: auto;
}

.preset-btn {
  margin: 0.25rem;
  padding: 0.5rem 1rem;
  border: 1px solid #007acc;
  background: transparent;
  color: #007acc;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.preset-btn:hover {
  background: #007acc;
  color: white;
}
</style>
```

### Phase 3: Content System (Week 4)

#### Custom Markdown Processing with Mathematical Content
```javascript
// src/utils/ContentLoader.js
import { createMarkdownRenderer } from './MarkdownRenderer'

export class ContentLoader {
  static async loadMathematicalContent(contentPath) {
    const response = await fetch(`/content/${contentPath}.md`)
    const rawMarkdown = await response.text()
    
    // Parse frontmatter for metadata
    const { frontmatter, content } = this.parseFrontmatter(rawMarkdown)
    
    // Render markdown with mathematical extensions
    const renderer = createMarkdownRenderer({
      mathjax: true,
      highlightCode: true,
      interactiveComponents: true
    })
    
    const htmlContent = renderer.render(content)
    
    return {
      metadata: frontmatter,
      content: htmlContent,
      interactiveComponents: this.extractComponents(content)
    }
  }
  
  static extractComponents(content) {
    // Extract component usage from markdown
    const componentRegex = /<([A-Z][a-zA-Z]*)\s*([^>]*)?\/>/g
    const components = []
    let match
    
    while ((match = componentRegex.exec(content)) !== null) {
      components.push({
        name: match[1],
        props: this.parseProps(match[2] || '')
      })
    }
    
    return components
  }
}
```

### Phase 4: Routing and Navigation (Week 5)

#### Vue Router Setup for Mathematical Content
```javascript
// src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import { ContentLoader } from '@/utils/ContentLoader'

// Lazy-loaded page components
const HomePage = () => import('@/views/HomePage.vue')
const ConceptPage = () => import('@/views/ConceptPage.vue')
const CategoryIndex = () => import('@/views/CategoryIndex.vue')

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomePage
  },
  {
    path: '/basics',
    name: 'BasicsIndex',
    component: CategoryIndex,
    props: { category: 'basics' }
  },
  {
    path: '/basics/:concept',
    name: 'BasicsConcept',
    component: ConceptPage,
    props: route => ({ 
      category: 'basics', 
      concept: route.params.concept 
    })
  },
  {
    path: '/algebra',
    name: 'AlgebraIndex', 
    component: CategoryIndex,
    props: { category: 'algebra' }
  },
  {
    path: '/algebra/:area/:concept?',
    name: 'AlgebraConcept',
    component: ConceptPage,
    props: route => ({
      category: 'algebra',
      area: route.params.area,
      concept: route.params.concept || 'index'
    })
  },
  // Repeat pattern for other mathematical domains...
]

export default createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else {
      return { top: 0 }
    }
  }
})
```

## Performance Optimizations

### 1. **Component Lazy Loading**
```javascript
// Automatic component chunking
const LazyStatisticsCalculator = defineAsyncComponent({
  loader: () => import('@/components/statistics/StatisticsCalculator.vue'),
  loadingComponent: LoadingSpinner,
  errorComponent: ErrorComponent,
  delay: 200,
  timeout: 3000
})
```

### 2. **Mathematical Computation Caching**
```javascript
// src/composables/useMathematicalCache.js
import { ref, computed } from 'vue'

export function useMathematicalCache() {
  const cache = ref(new Map())
  
  const getCachedResult = (key, computeFn) => {
    return computed(() => {
      if (cache.value.has(key)) {
        return cache.value.get(key)
      }
      
      const result = computeFn()
      cache.value.set(key, result)
      return result
    })
  }
  
  const clearCache = () => {
    cache.value.clear()
  }
  
  return {
    getCachedResult,
    clearCache
  }
}
```

### 3. **Bundle Optimization**
```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Mathematical computation engine
          'math-engine': ['./src/utils/MathEngine.js', './src/utils/CanvasRenderer.js'],
          
          // Statistics components
          'statistics': [
            './src/components/statistics/StatisticsCalculator.vue',
            './src/components/statistics/ProbabilitySimulator.vue'
          ],
          
          // Algebra components
          'algebra': [
            './src/components/algebra/QuadraticExplorer.vue',
            './src/components/algebra/LinearSystemSolver.vue',
            './src/components/algebra/SummationDemo.vue'
          ],
          
          // Visualization components
          'visualization': [
            './src/components/trigonometry/UnitCircleExplorer.vue',
            './src/components/linear-algebra/VectorOperations.vue',
            './src/components/calculus/LimitsExplorer.vue'
          ]
        }
      }
    },
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  }
}
```

## Expected Performance Improvements

### Before (VitePress):
- ❌ Build time: 60+ seconds
- ❌ Bundle size: ~2MB+ total
- ❌ Component load time: 200-500ms
- ❌ Math calculation response: 100-200ms
- ❌ Canvas rendering: 15-20fps

### After (Vue 3 + Vite):
- ✅ Build time: 10-15 seconds
- ✅ Bundle size: ~800KB total (chunked)
- ✅ Component load time: 50-100ms
- ✅ Math calculation response: 10-50ms
- ✅ Canvas rendering: 60fps

## Migration Timeline

**Week 1**: Project setup, mathematical engine, basic routing
**Week 2**: Convert 5 critical components (CodeFold, StatisticsCalculator, QuadraticExplorer, UnitCircleExplorer, VectorOperations)
**Week 3**: Convert remaining 14 components
**Week 4**: Content loading system and markdown processing
**Week 5**: Navigation, optimization, testing
**Week 6**: Deployment setup and performance validation

This migration will transform Snake Math from a struggling VitePress documentation site into a high-performance, scalable mathematical learning platform that can easily handle your current content and grow with future additions.
