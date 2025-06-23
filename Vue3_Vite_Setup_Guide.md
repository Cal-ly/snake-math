# Snake Math: Vue 3 + Vite Project Setup Guide

## Prerequisites

### Required Software
- **Node.js**: Version 18+ (LTS recommended)
- **npm**: Version 9+ (comes with Node.js)
- **Git**: Latest version
- **VSCode**: Latest version

### Verify Prerequisites
```bash
node --version    # Should be 18+
npm --version     # Should be 9+
git --version     # Any recent version
code --version    # VSCode version
```

## Part 1: VSCode Extensions Setup

### Essential Extensions for Vue 3 + Vite Development

1. **Vue Language Features (Volar)** - `Vue.volar`
   - Official Vue 3 language support
   - Replaces Vetur for Vue 3 projects

2. **TypeScript Vue Plugin (Volar)** - `Vue.vscode-typescript-vue-plugin`
   - Enhanced TypeScript support for Vue SFCs

3. **ESLint** - `dbaeumer.vscode-eslint`
   - Code linting and formatting

4. **Prettier - Code formatter** - `esbenp.prettier-vscode`
   - Code formatting

5. **Auto Rename Tag** - `formulahendry.auto-rename-tag`
   - Automatically rename paired HTML/XML tags

6. **Bracket Pair Colorizer 2** - `CoenraadS.bracket-pair-colorizer-2`
   - Colorize matching brackets

7. **GitLens** - `eamodio.gitlens`
   - Enhanced Git capabilities

8. **Live Server** - `ritwickdey.LiveServer`
   - Local development server for testing

### Mathematical/Scientific Extensions
9. **Markdown All in One** - `yzhang.markdown-all-in-one`
   - Enhanced markdown support for mathematical content

10. **Markdown Preview Enhanced** - `shd101wyy.markdown-preview-enhanced`
    - Rich markdown preview with LaTeX support

### Install Extensions via Command Palette
```
Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (Mac)
> Extensions: Install Extensions
```

Or install via command line:
```bash
code --install-extension Vue.volar
code --install-extension Vue.vscode-typescript-vue-plugin
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
code --install-extension formulahendry.auto-rename-tag
code --install-extension CoenraadS.bracket-pair-colorizer-2
code --install-extension eamodio.gitlens
code --install-extension ritwickdey.LiveServer
code --install-extension yzhang.markdown-all-in-one
code --install-extension shd101wyy.markdown-preview-enhanced
```

## Part 2: Project Initialization

### Step 1: Create New Vue 3 + Vite Project

```bash
# Navigate to your projects directory
cd /path/to/your/projects

# Create new Vue 3 project with Vite
npm create vue@latest snake-math-vue3

# You'll be prompted with options - choose these:
# ✔ Project name: snake-math-vue3
# ✔ Add TypeScript? › No (for simplicity, can add later)
# ✔ Add JSX Support? › No
# ✔ Add Vue Router for Single Page Application development? › Yes
# ✔ Add Pinia for state management? › Yes (for complex mathematical state)
# ✔ Add Vitest for Unit Testing? › Yes
# ✔ Add an End-to-End Testing Solution? › No (can add later)
# ✔ Add ESLint for code quality? › Yes
# ✔ Add Prettier for code formatting? › Yes

# Navigate to project directory
cd snake-math-vue3

# Install dependencies
npm install
```

### Step 2: Install Additional Mathematical Libraries

```bash
# Mathematical computation libraries
npm install --save \
  ml-matrix \
  simple-statistics \
  d3-scale \
  d3-shape \
  mathjs

# Mathematical notation rendering (LaTeX support)
npm install --save \
  katex \
  vue-katex

# Code syntax highlighting for embedded examples
npm install --save \
  highlight.js \
  @highlightjs/vue-plugin

# Development dependencies
npm install --save-dev \
  @types/d3-scale \
  @types/d3-shape \
  @types/katex
```

### Step 3: Open Project in VSCode

```bash
# Open project in VSCode
code .
```

## Part 3: VSCode Workspace Configuration

### Step 1: Create VSCode Settings

Create `.vscode/settings.json`:

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "emmet.includeLanguages": {
    "vue-html": "html"
  },
  "files.associations": {
    "*.vue": "vue"
  },
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "vue"
  ],
  "vetur.validation.template": false,
  "vetur.validation.script": false,
  "vetur.validation.style": false,
  "[vue]": {
    "editor.defaultFormatter": "Vue.volar"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "typescript.preferences.includePackageJsonAutoImports": "auto",
  "editor.quickSuggestions": {
    "strings": true
  },
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "files.eol": "\n"
}
```

### Step 2: Create VSCode Extensions Recommendations

Create `.vscode/extensions.json`:

```json
{
  "recommendations": [
    "Vue.volar",
    "Vue.vscode-typescript-vue-plugin",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "formulahendry.auto-rename-tag",
    "CoenraadS.bracket-pair-colorizer-2",
    "eamodio.gitlens",
    "ritwickdey.LiveServer",
    "yzhang.markdown-all-in-one",
    "shd101wyy.markdown-preview-enhanced"
  ]
}
```

### Step 3: Create Launch Configuration for Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Launch Chrome",
      "request": "launch",
      "type": "chrome",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/src",
      "breakOnLoad": true,
      "sourceMapPathOverrides": {
        "webpack:///src/*": "${webRoot}/*"
      }
    },
    {
      "name": "Launch Firefox",
      "request": "launch",
      "type": "firefox",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/src"
    }
  ]
}
```

### Step 4: Create Tasks Configuration

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "dev",
      "type": "shell",
      "command": "npm",
      "args": ["run", "dev"],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      },
      "isBackground": true,
      "problemMatcher": {
        "owner": "vite",
        "pattern": {
          "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)$",
          "file": 1,
          "line": 2,
          "column": 3,
          "severity": 4,
          "message": 5
        },
        "background": {
          "activeOnStart": true,
          "beginsPattern": "^.*Local:.*$",
          "endsPattern": "^.*ready in.*$"
        }
      }
    },
    {
      "label": "build",
      "type": "shell",
      "command": "npm",
      "args": ["run", "build"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "test",
      "type": "shell",
      "command": "npm",
      "args": ["run", "test:unit"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "lint",
      "type": "shell",
      "command": "npm",
      "args": ["run", "lint"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
```

## Part 4: Project Structure Setup

### Step 1: Create Mathematical Content Directory Structure

```bash
# Create content view directories (instead of markdown content)
mkdir -p src/views/basics
mkdir -p src/views/algebra/summation-notation
mkdir -p src/views/algebra/product-notation
mkdir -p src/views/algebra/linear-equations
mkdir -p src/views/algebra/quadratics
mkdir -p src/views/algebra/exponentials-logarithms
mkdir -p src/views/statistics/descriptive-stats
mkdir -p src/views/statistics/probability
mkdir -p src/views/trigonometry/unit-circle
mkdir -p src/views/linear-algebra/vectors
mkdir -p src/views/linear-algebra/matrices
mkdir -p src/views/calculus/limits

# Create component directories
mkdir -p src/components/common
mkdir -p src/components/statistics
mkdir -p src/components/algebra
mkdir -p src/components/trigonometry
mkdir -p src/components/linear-algebra
mkdir -p src/components/calculus

# Create utility directories
mkdir -p src/utils
mkdir -p src/composables

# Create shared content components
mkdir -p src/components/content

# Create assets directories
mkdir -p src/assets/images
mkdir -p src/assets/styles
```

### Step 2: Configure Vite for Mathematical Components

Update `vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@content': resolve(__dirname, 'src/content'),
      '@assets': resolve(__dirname, 'src/assets')
    }
  },
  server: {
    port: 5173,
    open: true,
    hmr: {
      overlay: true
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Mathematical computation engine
          'math-engine': [
            './src/utils/MathEngine.js',
            './src/utils/CanvasRenderer.js'
          ],
          
          // Statistics components and views
          'statistics': [
            './src/components/statistics/StatisticsCalculator.vue',
            './src/components/statistics/ProbabilitySimulator.vue',
            './src/views/statistics/descriptive-stats/BasicsView.vue',
            './src/views/statistics/probability/BasicsView.vue'
          ],
          
          // Algebra components and views
          'algebra': [
            './src/components/algebra/QuadraticExplorer.vue',
            './src/components/algebra/LinearSystemSolver.vue',
            './src/components/algebra/SummationDemo.vue',
            './src/views/algebra/quadratics/BasicsView.vue',
            './src/views/algebra/summation-notation/BasicsView.vue'
          ],
          
          // Visualization components and views
          'visualization': [
            './src/components/trigonometry/UnitCircleExplorer.vue',
            './src/components/linear-algebra/VectorOperations.vue',
            './src/components/calculus/LimitsExplorer.vue',
            './src/views/trigonometry/unit-circle/BasicsView.vue',
            './src/views/linear-algebra/vectors/BasicsView.vue'
          ],
          
          // Common utilities and content components
          'utils': [
            './src/components/common/CodeFold.vue',
            './src/components/common/InteractiveSlider.vue',
            './src/components/common/MathDisplay.vue',
            './src/components/content/ConceptLayout.vue',
            './src/components/content/SectionHeader.vue'
          ]
        }
      }
    },
    target: 'es2020',
    minify: 'terser',
    sourcemap: true
  },
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/assets/styles/variables.scss";`
      }
    }
  },
  optimizeDeps: {
    include: ['vue', 'vue-router', 'pinia', 'ml-matrix', 'simple-statistics', 'katex']
  }
})
```

### Step 3: Create Mathematical Engine Foundation

Create `src/utils/MathEngine.js`:

```javascript
/**
 * Mathematical computation engine for Snake Math
 * Framework-agnostic mathematical calculations with high precision
 */
export class MathematicalEngine {
  /**
   * Calculate with specified precision
   * @param {number} value - The value to round
   * @param {number} decimals - Number of decimal places
   * @returns {number} Rounded value
   */
  static calculateWithPrecision(value, decimals = 3) {
    if (typeof value !== 'number' || isNaN(value)) {
      throw new Error('Value must be a valid number')
    }
    return parseFloat(value.toFixed(decimals))
  }

  /**
   * Comprehensive quadratic analysis
   * @param {number} a - Coefficient of x²
   * @param {number} b - Coefficient of x
   * @param {number} c - Constant term
   * @returns {object} Complete quadratic analysis
   */
  static analyzeQuadratic(a, b, c) {
    if (a === 0) {
      throw new Error('Coefficient "a" cannot be zero for quadratic function')
    }

    const discriminant = b * b - 4 * a * c
    const vertex = {
      x: -b / (2 * a),
      y: (4 * a * c - b * b) / (4 * a)
    }

    let roots = []
    if (discriminant > 0) {
      const sqrtDiscriminant = Math.sqrt(discriminant)
      roots = [
        (-b + sqrtDiscriminant) / (2 * a),
        (-b - sqrtDiscriminant) / (2 * a)
      ]
    } else if (discriminant === 0) {
      roots = [-b / (2 * a)]
    }

    return {
      discriminant: this.calculateWithPrecision(discriminant),
      vertex: {
        x: this.calculateWithPrecision(vertex.x),
        y: this.calculateWithPrecision(vertex.y)
      },
      roots: roots.map(root => this.calculateWithPrecision(root)),
      axisOfSymmetry: this.calculateWithPrecision(-b / (2 * a)),
      yIntercept: c,
      hasRealRoots: discriminant >= 0,
      opensUpward: a > 0,
      vertexType: a > 0 ? 'minimum' : 'maximum'
    }
  }

  /**
   * Comprehensive dataset statistical analysis
   * @param {number[]} data - Array of numerical data
   * @returns {object} Statistical analysis results
   */
  static analyzeDataset(data) {
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('Data must be a non-empty array')
    }

    const validData = data.filter(val => typeof val === 'number' && !isNaN(val))
    if (validData.length === 0) {
      throw new Error('Data must contain at least one valid number')
    }

    const sorted = [...validData].sort((a, b) => a - b)
    const n = validData.length

    // Basic statistics
    const sum = validData.reduce((acc, val) => acc + val, 0)
    const mean = sum / n

    // Median calculation
    const median = n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)]

    // Mode calculation
    const frequency = {}
    validData.forEach(val => {
      frequency[val] = (frequency[val] || 0) + 1
    })
    const maxFreq = Math.max(...Object.values(frequency))
    const modes = Object.keys(frequency)
      .filter(val => frequency[val] === maxFreq)
      .map(Number)

    // Variance and standard deviation
    const variance = validData.reduce((acc, val) => 
      acc + Math.pow(val - mean, 2), 0) / n
    const standardDeviation = Math.sqrt(variance)

    // Quartiles
    const q1Index = Math.floor((n + 1) / 4) - 1
    const q3Index = Math.floor(3 * (n + 1) / 4) - 1
    const q1 = sorted[Math.max(0, q1Index)]
    const q3 = sorted[Math.min(n - 1, q3Index)]
    const iqr = q3 - q1

    // Outliers (using IQR method)
    const lowerBound = q1 - 1.5 * iqr
    const upperBound = q3 + 1.5 * iqr
    const outliers = validData.filter(val => val < lowerBound || val > upperBound)

    return {
      count: n,
      sum: this.calculateWithPrecision(sum),
      mean: this.calculateWithPrecision(mean),
      median: this.calculateWithPrecision(median),
      mode: modes.length === validData.length ? 'No mode' : modes,
      range: this.calculateWithPrecision(sorted[n - 1] - sorted[0]),
      variance: this.calculateWithPrecision(variance),
      standardDeviation: this.calculateWithPrecision(standardDeviation),
      quartiles: {
        q1: this.calculateWithPrecision(q1),
        q2: this.calculateWithPrecision(median),
        q3: this.calculateWithPrecision(q3),
        iqr: this.calculateWithPrecision(iqr)
      },
      outliers: outliers.map(val => this.calculateWithPrecision(val)),
      min: sorted[0],
      max: sorted[n - 1]
    }
  }

  /**
   * Vector operations for linear algebra
   * @param {number[]} vector1 - First vector
   * @param {number[]} vector2 - Second vector
   * @returns {object} Vector operation results
   */
  static vectorOperations(vector1, vector2) {
    if (!Array.isArray(vector1) || !Array.isArray(vector2)) {
      throw new Error('Vectors must be arrays')
    }
    if (vector1.length !== vector2.length) {
      throw new Error('Vectors must have the same length')
    }

    // Dot product
    const dotProduct = vector1.reduce((sum, val, index) => 
      sum + val * vector2[index], 0)

    // Magnitudes
    const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0))
    const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0))

    // Angle between vectors (in radians)
    const angle = Math.acos(dotProduct / (magnitude1 * magnitude2))

    return {
      dotProduct: this.calculateWithPrecision(dotProduct),
      magnitude1: this.calculateWithPrecision(magnitude1),
      magnitude2: this.calculateWithPrecision(magnitude2),
      angle: this.calculateWithPrecision(angle),
      angleDegrees: this.calculateWithPrecision(angle * 180 / Math.PI),
      isOrthogonal: Math.abs(dotProduct) < 1e-10
    }
  }
}
```

### Step 4: Create Canvas Rendering Utility

Create `src/utils/CanvasRenderer.js`:

```javascript
/**
 * Canvas rendering utilities for mathematical visualizations
 */
export class CanvasRenderer {
  /**
   * Render histogram on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {number[]} data - Dataset to visualize
   * @param {object} statistics - Statistical analysis
   * @param {object} options - Rendering options
   */
  static renderHistogram(ctx, data, statistics, options = {}) {
    const {
      bins = 10,
      color = '#4CAF50',
      width = ctx.canvas.width,
      height = ctx.canvas.height,
      padding = 40
    } = options

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    if (!data || data.length === 0) {
      this.renderNoDataMessage(ctx, width, height)
      return
    }

    // Calculate histogram data
    const min = Math.min(...data)
    const max = Math.max(...data)
    const binWidth = (max - min) / bins
    const binCounts = new Array(bins).fill(0)

    data.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1)
      binCounts[binIndex]++
    })

    const maxCount = Math.max(...binCounts)
    const chartWidth = width - 2 * padding
    const chartHeight = height - 2 * padding
    const barWidth = chartWidth / bins

    // Draw bars
    ctx.fillStyle = color
    binCounts.forEach((count, index) => {
      const barHeight = (count / maxCount) * chartHeight
      const x = padding + index * barWidth
      const y = padding + chartHeight - barHeight

      ctx.fillRect(x, y, barWidth * 0.8, barHeight)
    })

    // Draw axes and labels
    this.drawAxes(ctx, padding, chartWidth, chartHeight, min, max, maxCount)
  }

  /**
   * Render box plot on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {object} statistics - Statistical analysis
   * @param {object} options - Rendering options
   */
  static renderBoxPlot(ctx, statistics, options = {}) {
    const {
      color = '#2196F3',
      width = ctx.canvas.width,
      height = ctx.canvas.height,
      padding = 40
    } = options

    const { quartiles, min, max, outliers } = statistics
    const chartWidth = width - 2 * padding
    const chartHeight = height - 2 * padding
    const boxTop = padding + chartHeight * 0.3
    const boxHeight = chartHeight * 0.4

    // Scale for x-axis
    const scale = (value) => padding + ((value - min) / (max - min)) * chartWidth

    // Draw box
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    const boxLeft = scale(quartiles.q1)
    const boxRight = scale(quartiles.q3)
    
    ctx.strokeRect(boxLeft, boxTop, boxRight - boxLeft, boxHeight)

    // Draw median line
    const medianX = scale(quartiles.q2)
    ctx.beginPath()
    ctx.moveTo(medianX, boxTop)
    ctx.lineTo(medianX, boxTop + boxHeight)
    ctx.stroke()

    // Draw whiskers
    const whiskerY = boxTop + boxHeight / 2
    ctx.beginPath()
    // Left whisker
    ctx.moveTo(scale(min), whiskerY)
    ctx.lineTo(boxLeft, whiskerY)
    // Right whisker
    ctx.moveTo(boxRight, whiskerY)
    ctx.lineTo(scale(max), whiskerY)
    ctx.stroke()

    // Draw outliers
    ctx.fillStyle = color
    outliers.forEach(outlier => {
      const x = scale(outlier)
      ctx.beginPath()
      ctx.arc(x, whiskerY, 3, 0, 2 * Math.PI)
      ctx.fill()
    })
  }

  /**
   * Render quadratic function on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {number} a - Coefficient of x²
   * @param {number} b - Coefficient of x
   * @param {number} c - Constant term
   * @param {object} options - Rendering options
   */
  static renderQuadratic(ctx, a, b, c, options = {}) {
    const {
      color = '#FF5722',
      width = ctx.canvas.width,
      height = ctx.canvas.height,
      xMin = -10,
      xMax = 10,
      yMin = -10,
      yMax = 10,
      showVertex = true,
      showRoots = true
    } = options

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Coordinate transformation
    const toCanvas = (x, y) => ({
      x: ((x - xMin) / (xMax - xMin)) * width,
      y: height - ((y - yMin) / (yMax - yMin)) * height
    })

    // Draw grid
    this.drawGrid(ctx, width, height, xMin, xMax, yMin, yMax)

    // Draw parabola
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()

    const step = (xMax - xMin) / 200
    let firstPoint = true

    for (let x = xMin; x <= xMax; x += step) {
      const y = a * x * x + b * x + c
      const point = toCanvas(x, y)

      if (firstPoint) {
        ctx.moveTo(point.x, point.y)
        firstPoint = false
      } else {
        ctx.lineTo(point.x, point.y)
      }
    }

    ctx.stroke()

    // Draw vertex if requested
    if (showVertex) {
      const vertex = { x: -b / (2 * a), y: (4 * a * c - b * b) / (4 * a) }
      const vertexPoint = toCanvas(vertex.x, vertex.y)
      
      ctx.fillStyle = '#4CAF50'
      ctx.beginPath()
      ctx.arc(vertexPoint.x, vertexPoint.y, 5, 0, 2 * Math.PI)
      ctx.fill()
    }

    // Draw roots if requested and they exist
    if (showRoots) {
      const discriminant = b * b - 4 * a * c
      if (discriminant >= 0) {
        const sqrtD = Math.sqrt(discriminant)
        const root1 = (-b + sqrtD) / (2 * a)
        const root2 = (-b - sqrtD) / (2 * a)

        [root1, root2].forEach(root => {
          const rootPoint = toCanvas(root, 0)
          ctx.fillStyle = '#F44336'
          ctx.beginPath()
          ctx.arc(rootPoint.x, rootPoint.y, 4, 0, 2 * Math.PI)
          ctx.fill()
        })
      }
    }
  }

  /**
   * Draw coordinate grid
   */
  static drawGrid(ctx, width, height, xMin, xMax, yMin, yMax) {
    ctx.strokeStyle = '#E0E0E0'
    ctx.lineWidth = 1

    // Vertical grid lines
    const xStep = (xMax - xMin) / 10
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }

    // Horizontal grid lines
    const yStep = (yMax - yMin) / 10
    for (let i = 0; i <= 10; i++) {
      const y = (i / 10) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw axes
    ctx.strokeStyle = '#666'
    ctx.lineWidth = 2

    // X-axis
    const yZero = height - ((0 - yMin) / (yMax - yMin)) * height
    if (yZero >= 0 && yZero <= height) {
      ctx.beginPath()
      ctx.moveTo(0, yZero)
      ctx.lineTo(width, yZero)
      ctx.stroke()
    }

    // Y-axis
    const xZero = ((0 - xMin) / (xMax - xMin)) * width
    if (xZero >= 0 && xZero <= width) {
      ctx.beginPath()
      ctx.moveTo(xZero, 0)
      ctx.lineTo(xZero, height)
      ctx.stroke()
    }
  }

  /**
   * Draw axes for histogram
   */
  static drawAxes(ctx, padding, chartWidth, chartHeight, min, max, maxCount) {
    ctx.strokeStyle = '#666'
    ctx.lineWidth = 1

    // X-axis
    ctx.beginPath()
    ctx.moveTo(padding, padding + chartHeight)
    ctx.lineTo(padding + chartWidth, padding + chartHeight)
    ctx.stroke()

    // Y-axis
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, padding + chartHeight)
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#333'
    ctx.font = '12px Arial'
    ctx.textAlign = 'center'

    // X-axis labels
    for (let i = 0; i <= 5; i++) {
      const value = min + (max - min) * (i / 5)
      const x = padding + (chartWidth * i) / 5
      ctx.fillText(value.toFixed(1), x, padding + chartHeight + 20)
    }

    // Y-axis labels
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const value = (maxCount * i) / 5
      const y = padding + chartHeight - (chartHeight * i) / 5
      ctx.fillText(Math.round(value), padding - 10, y + 4)
    }
  }

  /**
   * Render "no data" message
   */
  static renderNoDataMessage(ctx, width, height) {
    ctx.fillStyle = '#666'
    ctx.font = '16px Arial'
    ctx.textAlign = 'center'
    ctx.fillText('No data to display', width / 2, height / 2)
  }
}
```

## Part 5: Native Vue Content System

### Step 1: Create Content Layout Components

Create `src/components/content/ConceptLayout.vue`:

```vue
<template>
  <div class="concept-layout">
    <!-- Page Header -->
    <header class="concept-header">
      <div class="header-content">
        <nav class="breadcrumb" v-if="breadcrumbs">
          <router-link 
            v-for="(crumb, index) in breadcrumbs" 
            :key="index"
            :to="crumb.path"
            class="breadcrumb-item"
          >
            {{ crumb.name }}
          </router-link>
        </nav>
        
        <h1 class="concept-title">
          <span v-if="symbol" class="concept-symbol">{{ symbol }}</span>
          {{ title }}
        </h1>
        
        <div class="concept-meta">
          <span class="difficulty-badge" :class="difficulty">{{ difficulty }}</span>
          <span class="category-badge">{{ category }}</span>
        </div>
        
        <p class="concept-description">{{ description }}</p>
        
        <div class="concept-tags" v-if="tags.length">
          <span v-for="tag in tags" :key="tag" class="tag">{{ tag }}</span>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="concept-content">
      <slot></slot>
    </main>

    <!-- Navigation Footer -->
    <footer class="concept-navigation" v-if="navigation">
      <router-link 
        v-if="navigation.previous" 
        :to="navigation.previous.path"
        class="nav-link nav-previous"
      >
        ← {{ navigation.previous.title }}
      </router-link>
      
      <router-link 
        v-if="navigation.next" 
        :to="navigation.next.path"
        class="nav-link nav-next"
      >
        {{ navigation.next.title }} →
      </router-link>
    </footer>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  symbol: {
    type: String,
    default: ''
  },
  difficulty: {
    type: String,
    default: 'beginner',
    validator: value => ['beginner', 'intermediate', 'advanced'].includes(value)
  },
  category: {
    type: String,
    required: true
  },
  tags: {
    type: Array,
    default: () => []
  },
  breadcrumbs: {
    type: Array,
    default: () => []
  },
  navigation: {
    type: Object,
    default: null
  }
})
</script>

<style scoped>
.concept-layout {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.concept-header {
  padding: 2rem 0;
  border-bottom: 1px solid #e1e5e9;
  margin-bottom: 2rem;
}

.breadcrumb {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  font-size: 0.875rem;
}

.breadcrumb-item {
  color: #6c757d;
  text-decoration: none;
  position: relative;
}

.breadcrumb-item:not(:last-child)::after {
  content: ' / ';
  margin-left: 0.5rem;
  color: #adb5bd;
}

.breadcrumb-item:hover {
  color: #007acc;
}

.concept-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: #212529;
  margin: 0.5rem 0;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.concept-symbol {
  font-size: 3rem;
  color: #007acc;
  font-family: 'Times New Roman', serif;
}

.concept-meta {
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
}

.difficulty-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.difficulty-badge.beginner {
  background: #d4edda;
  color: #155724;
}

.difficulty-badge.intermediate {
  background: #fff3cd;
  color: #856404;
}

.difficulty-badge.advanced {
  background: #f8d7da;
  color: #721c24;
}

.category-badge {
  padding: 0.25rem 0.75rem;
  background: #e9ecef;
  color: #495057;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}

.concept-description {
  font-size: 1.125rem;
  color: #6c757d;
  line-height: 1.6;
  margin: 1rem 0;
}

.concept-tags {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.tag {
  background: #f8f9fa;
  color: #495057;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  border: 1px solid #e9ecef;
}

.concept-content {
  margin-bottom: 3rem;
}

.concept-navigation {
  display: flex;
  justify-content: space-between;
  padding: 2rem 0;
  border-top: 1px solid #e1e5e9;
}

.nav-link {
  padding: 1rem 1.5rem;
  background: #f8f9fa;
  color: #495057;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.2s ease;
  border: 1px solid #e9ecef;
}

.nav-link:hover {
  background: #007acc;
  color: white;
  border-color: #007acc;
}

.nav-previous {
  margin-right: auto;
}

.nav-next {
  margin-left: auto;
}
</style>
```

### Step 2: Create Section Components

Create `src/components/content/SectionHeader.vue`:

```vue
<template>
  <section class="content-section">
    <header class="section-header">
      <h2 class="section-title">
        <span v-if="icon" class="section-icon">{{ icon }}</span>
        {{ title }}
      </h2>
      <p v-if="description" class="section-description">{{ description }}</p>
    </header>
    
    <div class="section-content">
      <slot></slot>
    </div>
  </section>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  },
  icon: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.content-section {
  margin: 2rem 0;
}

.section-header {
  margin-bottom: 1.5rem;
}

.section-title {
  font-size: 1.75rem;
  font-weight: 600;
  color: #212529;
  margin: 0 0 0.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.section-icon {
  font-size: 1.5rem;
  color: #007acc;
}

.section-description {
  font-size: 1rem;
  color: #6c757d;
  line-height: 1.6;
  margin: 0;
}

.section-content {
  /* Content styling is handled by child components */
}
</style>
```

### Step 3: Create Mathematical Display Component

Create `src/components/content/MathDisplay.vue`:

```vue
<template>
  <div class="math-display" :class="{ inline: display === 'inline', block: display === 'block' }">
    <div ref="mathContainer" class="math-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const props = defineProps({
  expression: {
    type: String,
    required: true
  },
  display: {
    type: String,
    default: 'block',
    validator: value => ['inline', 'block'].includes(value)
  }
})

const mathContainer = ref(null)

const renderMath = () => {
  if (!mathContainer.value) return
  
  try {
    katex.render(props.expression, mathContainer.value, {
      displayMode: props.display === 'block',
      throwOnError: false,
      strict: false
    })
  } catch (error) {
    console.warn('KaTeX rendering error:', error)
    mathContainer.value.textContent = props.expression
  }
}

watch(() => props.expression, renderMath)
watch(() => props.display, renderMath)

onMounted(renderMath)
</script>

<style scoped>
.math-display.inline {
  display: inline;
}

.math-display.block {
  display: block;
  margin: 1rem 0;
  text-align: center;
}

.math-container {
  /* KaTeX styles are applied automatically */
}
</style>
```

### Step 4: Create Python Code Display Component

Create `src/components/content/PythonCode.vue`:

```vue
<template>
  <div class="python-code">
    <div class="code-header" v-if="title || showRunButton">
      <h4 v-if="title" class="code-title">{{ title }}</h4>
      <div class="code-actions">
        <button v-if="enableCopy" @click="copyCode" class="action-btn copy-btn">
          {{ copied ? 'Copied!' : 'Copy' }}
        </button>
        <button v-if="showRunButton" @click="runCode" class="action-btn run-btn">
          Run Code
        </button>
      </div>
    </div>
    
    <pre class="code-block"><code ref="codeElement" class="language-python">{{ code }}</code></pre>
    
    <div v-if="output && showOutput" class="code-output">
      <div class="output-header">Output:</div>
      <pre class="output-content">{{ output }}</pre>
    </div>
    
    <div v-if="explanation" class="code-explanation">
      <h5>Explanation:</h5>
      <p>{{ explanation }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import 'highlight.js/styles/github.css'

// Register Python language
hljs.registerLanguage('python', python)

const props = defineProps({
  code: {
    type: String,
    required: true
  },
  title: {
    type: String,
    default: ''
  },
  explanation: {
    type: String,
    default: ''
  },
  showRunButton: {
    type: Boolean,
    default: false
  },
  showOutput: {
    type: Boolean,
    default: true
  },
  enableCopy: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['run'])

const codeElement = ref(null)
const copied = ref(false)
const output = ref('')

const highlightCode = async () => {
  await nextTick()
  if (codeElement.value) {
    hljs.highlightElement(codeElement.value)
  }
}

const copyCode = async () => {
  try {
    await navigator.clipboard.writeText(props.code)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  } catch (error) {
    console.warn('Failed to copy code:', error)
  }
}

const runCode = () => {
  // Emit event for parent to handle Python execution
  emit('run', props.code)
  
  // For now, simulate output
  output.value = '# Code execution would run here in PyScript environment'
}

onMounted(highlightCode)
</script>

<style scoped>
.python-code {
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  margin: 1rem 0;
  background: #ffffff;
  overflow: hidden;
}

.code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: #f8f9fa;
  border-bottom: 1px solid #e1e5e9;
}

.code-title {
  margin: 0;
  font-size: 0.9rem;
  font-weight: 600;
  color: #495057;
}

.code-actions {
  display: flex;
  gap: 0.5rem;
}

.action-btn {
  padding: 0.25rem 0.75rem;
  border: none;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.copy-btn {
  background: #6c757d;
  color: white;
}

.copy-btn:hover {
  background: #5a6268;
}

.run-btn {
  background: #28a745;
  color: white;
}

.run-btn:hover {
  background: #218838;
}

.code-block {
  margin: 0;
  padding: 1rem;
  background: #f8f9fa;
  overflow-x: auto;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.875rem;
  line-height: 1.5;
}

.code-block code {
  background: none;
  padding: 0;
  font-family: inherit;
  font-size: inherit;
}

.code-output {
  border-top: 1px solid #e1e5e9;
}

.output-header {
  padding: 0.5rem 1rem;
  background: #e8f5e8;
  font-size: 0.8rem;
  font-weight: 600;
  color: #155724;
}

.output-content {
  margin: 0;
  padding: 1rem;
  background: #f8fff8;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.875rem;
  color: #155724;
  border: none;
}

.code-explanation {
  padding: 1rem;
  background: #fff3cd;
  border-top: 1px solid #e1e5e9;
}

.code-explanation h5 {
  margin: 0 0 0.5rem 0;
  font-size: 0.9rem;
  color: #856404;
}

.code-explanation p {
  margin: 0;
  font-size: 0.875rem;
  color: #856404;
  line-height: 1.5;
}
</style>
```

### Step 5: Create Example Content View

Create `src/views/algebra/quadratics/BasicsView.vue`:

```vue
<template>
  <ConceptLayout
    title="Quadratic Functions"
    description="Understanding quadratic functions, their properties, and real-world applications"
    symbol="ax² + bx + c"
    difficulty="intermediate"
    category="Algebra"
    :tags="['functions', 'parabolas', 'polynomials', 'algebra']"
    :breadcrumbs="breadcrumbs"
    :navigation="navigation"
  >
    <!-- Understanding Section -->
    <SectionHeader 
      title="Understanding Quadratic Functions"
      description="A quadratic function is a polynomial function of degree 2"
      icon="📐"
    >
      <p>
        A quadratic function is any function that can be written in the form:
      </p>
      
      <MathDisplay expression="f(x) = ax^2 + bx + c" />
      
      <p>
        where <em>a</em>, <em>b</em>, and <em>c</em> are constants and <em>a ≠ 0</em>. 
        The graph of a quadratic function is a parabola that opens upward if <em>a > 0</em> 
        or downward if <em>a < 0</em>.
      </p>

      <div class="key-concepts">
        <div class="concept-item">
          <h4>Key Components:</h4>
          <ul>
            <li><strong>a</strong>: Controls the width and direction of opening</li>
            <li><strong>b</strong>: Affects the position of the vertex</li>
            <li><strong>c</strong>: The y-intercept of the parabola</li>
          </ul>
        </div>
        
        <div class="concept-item">
          <h4>Important Properties:</h4>
          <ul>
            <li><strong>Vertex</strong>: The maximum or minimum point</li>
            <li><strong>Axis of Symmetry</strong>: <MathDisplay expression="x = -\frac{b}{2a}" display="inline" /></li>
            <li><strong>Discriminant</strong>: <MathDisplay expression="\Delta = b^2 - 4ac" display="inline" /></li>
          </ul>
        </div>
      </div>
    </SectionHeader>

    <!-- Interactive Exploration -->
    <SectionHeader 
      title="Interactive Exploration"
      description="Experiment with quadratic functions and see how they change"
      icon="🔧"
    >
      <QuadraticExplorer />
    </SectionHeader>

    <!-- Mathematical Forms -->
    <SectionHeader 
      title="Different Forms of Quadratic Functions"
      description="Quadratic functions can be expressed in multiple equivalent forms"
      icon="📝"
    >
      <div class="forms-grid">
        <div class="form-card">
          <h4>Standard Form</h4>
          <MathDisplay expression="f(x) = ax^2 + bx + c" />
          <p>Best for identifying the y-intercept and applying the quadratic formula.</p>
        </div>
        
        <div class="form-card">
          <h4>Vertex Form</h4>
          <MathDisplay expression="f(x) = a(x - h)^2 + k" />
          <p>Best for identifying the vertex (h, k) and understanding transformations.</p>
        </div>
        
        <div class="form-card">
          <h4>Factored Form</h4>
          <MathDisplay expression="f(x) = a(x - r_1)(x - r_2)" />
          <p>Best for identifying the roots (x-intercepts) r₁ and r₂.</p>
        </div>
      </div>
    </SectionHeader>

    <!-- Programming Implementation -->
    <SectionHeader 
      title="Python Implementation"
      description="How to work with quadratic functions in Python"
      icon="🐍"
    >
      <PythonCode 
        title="Basic Quadratic Function"
        :code="basicQuadraticCode"
        explanation="This function evaluates a quadratic expression for given coefficients and x value."
        show-run-button
        @run="runPythonCode"
      />

      <PythonCode 
        title="Finding Roots Using Quadratic Formula"
        :code="quadraticFormulaCode"
        explanation="Implementation of the quadratic formula to find roots, including handling of complex roots."
      />

      <PythonCode 
        title="Plotting Quadratic Functions"
        :code="plottingCode"
        explanation="Using matplotlib to visualize quadratic functions and their key features."
      />
    </SectionHeader>

    <!-- Real-World Applications -->
    <SectionHeader 
      title="Real-World Applications"
      description="Where quadratic functions appear in practical scenarios"
      icon="🌍"
    >
      <div class="applications-grid">
        <div class="application-card">
          <h4>Projectile Motion</h4>
          <MathDisplay expression="h(t) = -\frac{1}{2}gt^2 + v_0t + h_0" />
          <p>Height of a projectile over time, where g is gravity, v₀ is initial velocity, and h₀ is initial height.</p>
        </div>
        
        <div class="application-card">
          <h4>Profit Optimization</h4>
          <MathDisplay expression="P(x) = -ax^2 + bx - c" />
          <p>Profit as a function of production quantity, helping businesses find optimal production levels.</p>
        </div>
        
        <div class="application-card">
          <h4>Parabolic Reflectors</h4>
          <MathDisplay expression="y = \frac{1}{4p}x^2" />
          <p>Shape of satellite dishes and solar collectors that focus signals or energy to a single point.</p>
        </div>
      </div>
    </SectionHeader>

    <!-- Practice Problems -->
    <SectionHeader 
      title="Practice & Exploration"
      description="Strengthen your understanding with these exercises"
      icon="💪"
    >
      <div class="practice-section">
        <h4>Try These Exercises:</h4>
        <ol>
          <li>Find the vertex of <MathDisplay expression="f(x) = 2x^2 - 8x + 6" display="inline" /></li>
          <li>Determine the roots of <MathDisplay expression="x^2 - 5x + 6 = 0" display="inline" /></li>
          <li>Convert <MathDisplay expression="f(x) = x^2 + 4x + 1" display="inline" /> to vertex form</li>
          <li>Model a ball thrown upward with initial velocity 20 m/s from height 2m</li>
        </ol>
        
        <div class="exploration-ideas">
          <h4>Further Exploration:</h4>
          <ul>
            <li>Investigate how the discriminant relates to the number of real roots</li>
            <li>Explore the relationship between quadratic functions and conic sections</li>
            <li>Research applications in computer graphics and game physics</li>
            <li>Study optimization problems in economics and engineering</li>
          </ul>
        </div>
      </div>
    </SectionHeader>
  </ConceptLayout>
</template>

<script setup>
import { computed } from 'vue'
import ConceptLayout from '@/components/content/ConceptLayout.vue'
import SectionHeader from '@/components/content/SectionHeader.vue'
import MathDisplay from '@/components/content/MathDisplay.vue'
import PythonCode from '@/components/content/PythonCode.vue'
import QuadraticExplorer from '@/components/algebra/QuadraticExplorer.vue'

// Navigation setup
const breadcrumbs = [
  { name: 'Home', path: '/' },
  { name: 'Algebra', path: '/algebra' },
  { name: 'Quadratics', path: '/algebra/quadratics' }
]

const navigation = {
  previous: {
    title: 'Linear Equations',
    path: '/algebra/linear-equations/basics'
  },
  next: {
    title: 'Solving Quadratics',
    path: '/algebra/quadratics/solving'
  }
}

// Python code examples
const basicQuadraticCode = `def quadratic_function(x, a, b, c):
    """
    Evaluate a quadratic function f(x) = ax² + bx + c
    
    Args:
        x: The input value
        a, b, c: Coefficients of the quadratic function
    
    Returns:
        The value of f(x)
    """
    return a * x**2 + b * x + c

# Example usage
a, b, c = 1, -2, 1  # f(x) = x² - 2x + 1
x_values = [-1, 0, 1, 2, 3]

for x in x_values:
    y = quadratic_function(x, a, b, c)
    print(f"f({x}) = {y}")

# Output:
# f(-1) = 4
# f(0) = 1
# f(1) = 0
# f(2) = 1
# f(3) = 4`

const quadraticFormulaCode = `import math

def solve_quadratic(a, b, c):
    """
    Solve quadratic equation ax² + bx + c = 0 using the quadratic formula
    
    Returns:
        tuple: (root1, root2) or None if no real solutions
    """
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        print("No real solutions")
        return None
    elif discriminant == 0:
        root = -b / (2*a)
        print(f"One solution: x = {root}")
        return (root, root)
    else:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        print(f"Two solutions: x = {root1}, x = {root2}")
        return (root1, root2)

# Example: Solve x² - 5x + 6 = 0
roots = solve_quadratic(1, -5, 6)
# Output: Two solutions: x = 3.0, x = 2.0`

const plottingCode = `import numpy as np
import matplotlib.pyplot as plt

def plot_quadratic(a, b, c, x_range=(-10, 10)):
    """
    Plot a quadratic function with its key features
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = a * x**2 + b * x + c
    
    # Calculate vertex
    vertex_x = -b / (2*a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c
    
    # Calculate roots if they exist
    discriminant = b**2 - 4*a*c
    roots = []
    if discriminant >= 0:
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)
        roots = [root1, root2]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {a}x² + {b}x + {c}')
    
    # Mark vertex
    plt.plot(vertex_x, vertex_y, 'ro', markersize=8, label=f'Vertex ({vertex_x:.2f}, {vertex_y:.2f})')
    
    # Mark roots
    for root in roots:
        plt.plot(root, 0, 'go', markersize=8)
    
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quadratic Function Graph')
    plt.legend()
    plt.show()

# Example usage
plot_quadratic(1, -4, 3)  # f(x) = x² - 4x + 3`

// Methods
const runPythonCode = (code) => {
  console.log('Running Python code:', code)
  // In a real implementation, this would interface with PyScript
}
</script>

<style scoped>
.key-concepts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin: 2rem 0;
}

.concept-item {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 4px solid #007acc;
}

.concept-item h4 {
  margin: 0 0 1rem 0;
  color: #007acc;
}

.concept-item ul {
  margin: 0;
  padding-left: 1.5rem;
}

.concept-item li {
  margin: 0.5rem 0;
  line-height: 1.5;
}

.forms-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.form-card {
  background: #ffffff;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.form-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.form-card h4 {
  margin: 0 0 1rem 0;
  color: #007acc;
}

.form-card p {
  margin: 1rem 0 0 0;
  font-size: 0.9rem;
  color: #6c757d;
  line-height: 1.5;
}

.applications-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.application-card {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  padding: 1.5rem;
}

.application-card h4 {
  margin: 0 0 1rem 0;
  color: #856404;
}

.application-card p {
  margin: 1rem 0 0 0;
  color: #856404;
  line-height: 1.5;
}

.practice-section {
  background: #e8f5e8;
  padding: 2rem;
  border-radius: 8px;
  border-left: 4px solid #28a745;
}

.practice-section h4 {
  margin: 0 0 1rem 0;
  color: #155724;
}

.practice-section ol {
  margin: 1rem 0;
  padding-left: 2rem;
}

.practice-section li {
  margin: 0.75rem 0;
  line-height: 1.6;
}

.exploration-ideas {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #c3e6cb;
}

.exploration-ideas h4 {
  margin: 0 0 1rem 0;
  color: #155724;
}

.exploration-ideas ul {
  margin: 0;
  padding-left: 2rem;
}

.exploration-ideas li {
  margin: 0.5rem 0;
  line-height: 1.5;
}
</style>
```

## Part 6: Basic Component Setup

### Step 1: Create CodeFold Component (Essential Utility)

Create `src/components/common/CodeFold.vue`:

```vue
<template>
  <div class="code-fold">
    <button 
      @click="toggleExpanded" 
      class="code-fold-header"
      :aria-expanded="isExpanded"
      :aria-controls="`code-content-${componentId}`"
    >
      <span class="fold-icon" :class="{ expanded: isExpanded }">▶</span>
      <span class="fold-title">{{ title }}</span>
      <span class="fold-language" v-if="language">{{ language }}</span>
    </button>
    
    <div 
      :id="`code-content-${componentId}`"
      class="code-fold-content" 
      :class="{ expanded: isExpanded }"
      ref="contentRef"
    >
      <div class="code-wrapper">
        <button 
          v-if="enableCopy" 
          @click="copyCode" 
          class="copy-button"
          :class="{ copied: showCopied }"
        >
          {{ showCopied ? 'Copied!' : 'Copy' }}
        </button>
        <pre v-if="language" :class="`language-${language}`"><code><slot></slot></code></pre>
        <div v-else><slot></slot></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'

// Props
const props = defineProps({
  title: {
    type: String,
    default: 'Show Code'
  },
  language: {
    type: String,
    default: ''
  },
  initialState: {
    type: String,
    default: 'collapsed',
    validator: (value) => ['expanded', 'collapsed'].includes(value)
  },
  enableCopy: {
    type: Boolean,
    default: true
  }
})

// Reactive state
const isExpanded = ref(props.initialState === 'expanded')
const showCopied = ref(false)
const contentRef = ref(null)
const componentId = ref(`cf-${Math.random().toString(36).substr(2, 9)}`)

// Methods
const toggleExpanded = () => {
  isExpanded.value = !isExpanded.value
}

const copyCode = async () => {
  if (!contentRef.value) return
  
  const codeElement = contentRef.value.querySelector('code') || contentRef.value
  const text = codeElement.textContent || codeElement.innerText || ''
  
  try {
    await navigator.clipboard.writeText(text)
    showCopied.value = true
    setTimeout(() => {
      showCopied.value = false
    }, 2000)
  } catch (err) {
    console.warn('Failed to copy code:', err)
  }
}

// Expose methods for parent components
defineExpose({
  toggle: toggleExpanded,
  expand: () => { isExpanded.value = true },
  collapse: () => { isExpanded.value = false }
})
</script>

<style scoped>
.code-fold {
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  margin: 1rem 0;
  overflow: hidden;
  background: #ffffff;
}

.code-fold-header {
  width: 100%;
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  background: #f8f9fa;
  border: none;
  cursor: pointer;
  font-size: 0.9rem;
  color: #495057;
  transition: background-color 0.2s ease;
}

.code-fold-header:hover {
  background: #e9ecef;
}

.code-fold-header:focus {
  outline: 2px solid #007acc;
  outline-offset: -2px;
}

.fold-icon {
  margin-right: 0.5rem;
  transition: transform 0.2s ease;
  font-size: 0.8rem;
}

.fold-icon.expanded {
  transform: rotate(90deg);
}

.fold-title {
  flex: 1;
  text-align: left;
  font-weight: 500;
}

.fold-language {
  background: #007acc;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.code-fold-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.code-fold-content.expanded {
  max-height: 1000px; /* Adjust based on your needs */
}

.code-wrapper {
  position: relative;
  padding: 1rem;
}

.copy-button {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  padding: 0.25rem 0.5rem;
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 0.75rem;
  cursor: pointer;
  z-index: 10;
  transition: all 0.2s ease;
}

.copy-button:hover {
  background: #495057;
}

.copy-button.copied {
  background: #28a745;
}

pre {
  margin: 0;
  padding: 0;
  background: #f8f9fa;
  border-radius: 4px;
  padding: 1rem;
  overflow-x: auto;
  font-family: 'Courier New', monospace;
  font-size: 0.875rem;
  line-height: 1.5;
}

code {
  font-family: inherit;
  font-size: inherit;
}

/* Syntax highlighting - basic styles */
.language-python .hljs-keyword { color: #007020; font-weight: bold; }
.language-python .hljs-string { color: #4070a0; }
.language-python .hljs-comment { color: #60a0b0; font-style: italic; }
.language-python .hljs-number { color: #40a070; }

.language-javascript .hljs-keyword { color: #a626a4; }
.language-javascript .hljs-string { color: #50a14f; }
.language-javascript .hljs-comment { color: #a0a1a7; font-style: italic; }
.language-javascript .hljs-number { color: #986801; }
</style>
```

## Part 7: Router Configuration for Native Vue Content

### Step 1: Configure Vue Router for Content Views

Update `src/router/index.js`:

```javascript
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    
    // Basics routes
    {
      path: '/basics',
      name: 'basics-index',
      component: () => import('@/views/basics/IndexView.vue')
    },
    {
      path: '/basics/:concept',
      name: 'basics-concept',
      component: () => import('@/views/basics/ConceptView.vue'),
      props: true
    },
    
    // Algebra routes
    {
      path: '/algebra',
      name: 'algebra-index',
      component: () => import('@/views/algebra/IndexView.vue')
    },
    {
      path: '/algebra/quadratics',
      name: 'algebra-quadratics-index',
      component: () => import('@/views/algebra/quadratics/IndexView.vue')
    },
    {
      path: '/algebra/quadratics/basics',
      name: 'algebra-quadratics-basics',
      component: () => import('@/views/algebra/quadratics/BasicsView.vue')
    },
    {
      path: '/algebra/quadratics/solving',
      name: 'algebra-quadratics-solving',
      component: () => import('@/views/algebra/quadratics/SolvingView.vue')
    },
    {
      path: '/algebra/quadratics/applications',
      name: 'algebra-quadratics-applications',
      component: () => import('@/views/algebra/quadratics/ApplicationsView.vue')
    },
    {
      path: '/algebra/summation-notation',
      name: 'algebra-summation-index',
      component: () => import('@/views/algebra/summation-notation/IndexView.vue')
    },
    {
      path: '/algebra/summation-notation/basics',
      name: 'algebra-summation-basics',
      component: () => import('@/views/algebra/summation-notation/BasicsView.vue')
    },
    
    // Statistics routes
    {
      path: '/statistics',
      name: 'statistics-index',
      component: () => import('@/views/statistics/IndexView.vue')
    },
    {
      path: '/statistics/descriptive-stats/basics',
      name: 'statistics-descriptive-basics',
      component: () => import('@/views/statistics/descriptive-stats/BasicsView.vue')
    },
    {
      path: '/statistics/probability/basics',
      name: 'statistics-probability-basics',
      component: () => import('@/views/statistics/probability/BasicsView.vue')
    },
    
    // Trigonometry routes
    {
      path: '/trigonometry',
      name: 'trigonometry-index',
      component: () => import('@/views/trigonometry/IndexView.vue')
    },
    {
      path: '/trigonometry/unit-circle/basics',
      name: 'trigonometry-unit-circle-basics',
      component: () => import('@/views/trigonometry/unit-circle/BasicsView.vue')
    },
    
    // Linear Algebra routes
    {
      path: '/linear-algebra',
      name: 'linear-algebra-index',
      component: () => import('@/views/linear-algebra/IndexView.vue')
    },
    {
      path: '/linear-algebra/vectors/basics',
      name: 'linear-algebra-vectors-basics',
      component: () => import('@/views/linear-algebra/vectors/BasicsView.vue')
    },
    {
      path: '/linear-algebra/matrices/basics',
      name: 'linear-algebra-matrices-basics',
      component: () => import('@/views/linear-algebra/matrices/BasicsView.vue')
    },
    
    // Calculus routes
    {
      path: '/calculus',
      name: 'calculus-index',
      component: () => import('@/views/calculus/IndexView.vue')
    },
    {
      path: '/calculus/limits/basics',
      name: 'calculus-limits-basics',
      component: () => import('@/views/calculus/limits/BasicsView.vue')
    }
  ],
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else {
      return { top: 0 }
    }
  }
})

export default router
```

### Step 2: Create Domain Index Views

Create `src/views/algebra/IndexView.vue`:

```vue
<template>
  <ConceptLayout
    title="Algebra"
    description="Master algebraic concepts from basic operations to advanced functions"
    category="Mathematics"
    :tags="['algebra', 'functions', 'equations', 'polynomials']"
    :breadcrumbs="breadcrumbs"
  >
    <div class="domain-overview">
      <p class="intro-text">
        Algebra is the branch of mathematics that uses symbols to represent numbers and express mathematical relationships. 
        In this section, we'll explore key algebraic concepts and their applications in programming and data science.
      </p>
    </div>

    <div class="concepts-grid">
      <ConceptCard
        title="Summation Notation"
        description="Learn the mathematical notation for sums and its programming equivalents"
        symbol="Σ"
        difficulty="beginner"
        path="/algebra/summation-notation"
        :topics="['for loops', 'series', 'mathematical notation']"
      />
      
      <ConceptCard
        title="Product Notation"
        description="Understand mathematical products and their computational implementations"
        symbol="Π"
        difficulty="beginner"
        path="/algebra/product-notation"
        :topics="['loops', 'factorials', 'compound calculations']"
      />
      
      <ConceptCard
        title="Linear Equations"
        description="Solve linear equations and systems using algebraic and computational methods"
        symbol="ax + b = c"
        difficulty="beginner"
        path="/algebra/linear-equations"
        :topics="['solving equations', 'systems', 'matrix methods']"
      />
      
      <ConceptCard
        title="Quadratic Functions"
        description="Explore parabolas, roots, and optimization in quadratic functions"
        symbol="ax² + bx + c"
        difficulty="intermediate"
        path="/algebra/quadratics"
        :topics="['parabolas', 'optimization', 'projectile motion']"
      />
      
      <ConceptCard
        title="Exponentials & Logarithms"
        description="Master exponential growth and logarithmic relationships"
        symbol="e^x, log(x)"
        difficulty="intermediate"
        path="/algebra/exponentials-logarithms"
        :topics="['growth models', 'data scaling', 'complexity analysis']"
      />
    </div>
  </ConceptLayout>
</template>

<script setup>
import ConceptLayout from '@/components/content/ConceptLayout.vue'
import ConceptCard from '@/components/content/ConceptCard.vue'

const breadcrumbs = [
  { name: 'Home', path: '/' }
]
</script>

<style scoped>
.domain-overview {
  margin: 2rem 0 3rem 0;
}

.intro-text {
  font-size: 1.125rem;
  line-height: 1.7;
  color: #495057;
  text-align: center;
  max-width: 800px;
  margin: 0 auto;
}

.concepts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}
</style>
```

### Step 3: Create Concept Card Component

Create `src/components/content/ConceptCard.vue`:

```vue
<template>
  <router-link :to="path" class="concept-card">
    <div class="card-header">
      <div class="concept-symbol">{{ symbol }}</div>
      <div class="difficulty-indicator" :class="difficulty">
        {{ difficulty }}
      </div>
    </div>
    
    <div class="card-content">
      <h3 class="concept-title">{{ title }}</h3>
      <p class="concept-description">{{ description }}</p>
      
      <div class="topics-list">
        <span v-for="topic in topics" :key="topic" class="topic-tag">
          {{ topic }}
        </span>
      </div>
    </div>
    
    <div class="card-footer">
      <span class="explore-text">Explore →</span>
    </div>
  </router-link>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  symbol: {
    type: String,
    required: true
  },
  difficulty: {
    type: String,
    default: 'beginner',
    validator: value => ['beginner', 'intermediate', 'advanced'].includes(value)
  },
  path: {
    type: String,
    required: true
  },
  topics: {
    type: Array,
    default: () => []
  }
})
</script>

<style scoped>
.concept-card {
  display: block;
  background: #ffffff;
  border: 1px solid #e1e5e9;
  border-radius: 12px;
  padding: 1.5rem;
  text-decoration: none;
  color: inherit;
  transition: all 0.3s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.concept-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(0, 122, 204, 0.15);
  border-color: #007acc;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.concept-symbol {
  font-size: 2rem;
  font-weight: bold;
  color: #007acc;
  font-family: 'Times New Roman', serif;
}

.difficulty-indicator {
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.difficulty-indicator.beginner {
  background: #d4edda;
  color: #155724;
}

.difficulty-indicator.intermediate {
  background: #fff3cd;
  color: #856404;
}

.difficulty-indicator.advanced {
  background: #f8d7da;
  color: #721c24;
}

.card-content {
  flex: 1;
}

.concept-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #212529;
  margin: 0 0 0.75rem 0;
}

.concept-description {
  color: #6c757d;
  line-height: 1.6;
  margin: 0 0 1rem 0;
}

.topics-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.topic-tag {
  background: #f8f9fa;
  color: #495057;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  border: 1px solid #e9ecef;
}

.card-footer {
  text-align: right;
  margin-top: auto;
}

.explore-text {
  color: #007acc;
  font-weight: 500;
  font-size: 0.9rem;
}
</style>
```

## Part 8: Development Workflow

### Step 1: Start Development Server

```bash
# Start the development server (runs on http://localhost:5173)
npm run dev

# Or use VSCode task: Ctrl+Shift+P > Tasks: Run Task > dev
```

### Step 2: Test the Setup

Create a simple test component in `src/components/algebra/QuadraticExplorer.vue`:

```vue
<template>
  <div class="quadratic-explorer">
    <h3>Quadratic Function Explorer</h3>
    
    <div class="controls">
      <label>
        a: <input v-model.number="a" type="number" step="0.1">
      </label>
      <label>
        b: <input v-model.number="b" type="number" step="0.1">
      </label>
      <label>
        c: <input v-model.number="c" type="number" step="0.1">
      </label>
    </div>
    
    <div v-if="analysis" class="analysis">
      <p>Function: y = {{ a }}x² + {{ b }}x + {{ c }}</p>
      <p>Vertex: ({{ analysis.vertex.x }}, {{ analysis.vertex.y }})</p>
      <p>Discriminant: {{ analysis.discriminant }}</p>
      <p>Roots: {{ analysis.roots.length ? analysis.roots.join(', ') : 'No real roots' }}</p>
    </div>
    
    <canvas 
      ref="canvas" 
      width="600" 
      height="400"
      class="graph-canvas"
    ></canvas>
    
    <CodeFold title="Python Implementation" language="python">
import numpy as np
import matplotlib.pyplot as plt

def quadratic_function(x, a={{ a }}, b={{ b }}, c={{ c }}):
    return a * x**2 + b * x + c

# Generate x values
x = np.linspace(-10, 10, 200)
y = quadratic_function(x)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'y = {a}x² + {b}x + {c}')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Function')
plt.legend()
plt.show()
    </CodeFold>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { MathematicalEngine } from '@/utils/MathEngine'
import { CanvasRenderer } from '@/utils/CanvasRenderer'
import CodeFold from '@/components/common/CodeFold.vue'

// Reactive coefficients
const a = ref(1)
const b = ref(0)
const c = ref(0)
const canvas = ref(null)

// Computed analysis
const analysis = computed(() => {
  try {
    return MathematicalEngine.analyzeQuadratic(a.value, b.value, c.value)
  } catch (error) {
    return null
  }
})

// Watch for changes and update graph
watch([a, b, c], () => {
  updateGraph()
}, { immediate: true })

const updateGraph = () => {
  if (!canvas.value || !analysis.value) return
  
  const ctx = canvas.value.getContext('2d')
  CanvasRenderer.renderQuadratic(ctx, a.value, b.value, c.value, {
    xMin: -10,
    xMax: 10,
    yMin: -10,
    yMax: 10,
    showVertex: true,
    showRoots: true
  })
}

onMounted(() => {
  updateGraph()
})
</script>

<style scoped>
.quadratic-explorer {
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1rem 0;
}

.controls {
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
}

.controls label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.controls input {
  width: 80px;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.analysis {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 4px;
  margin: 1rem 0;
}

.graph-canvas {
  border: 1px solid #ddd;
  border-radius: 4px;
  max-width: 100%;
  height: auto;
}
</style>
```

### Step 3: Update Router and Test

Update `src/router/index.js` to include your test route:

```javascript
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/test-quadratic',
      name: 'test-quadratic',
      component: () => import('@/components/algebra/QuadraticExplorer.vue')
    }
  ]
})

export default router
```

### Step 4: Essential VSCode Shortcuts for Vue + Vite Development

- **Start dev server**: `Ctrl+Shift+P` > "Tasks: Run Task" > "dev"
- **Build project**: `Ctrl+Shift+P` > "Tasks: Run Task" > "build"
- **Format code**: `Shift+Alt+F`
- **Auto-fix ESLint issues**: `Ctrl+Shift+P` > "ESLint: Fix all auto-fixable Problems"
- **Go to definition**: `F12`
- **Find all references**: `Shift+F12`
- **Rename symbol**: `F2`
- **Quick file search**: `Ctrl+P`
- **Command palette**: `Ctrl+Shift+P`

## Part 9: Next Steps for Native Vue Content

### 1. Content Creation Strategy
Instead of converting markdown to Vue, create native Vue views for each concept:

**Advantages of Native Vue Content:**
- ✅ **Better Performance**: No markdown parsing overhead
- ✅ **Rich Interactivity**: Seamless integration of interactive components
- ✅ **Type Safety**: Better IDE support and error catching
- ✅ **Component Reuse**: Shared components across all content
- ✅ **Dynamic Content**: Real-time updates and calculations
- ✅ **Better SEO**: Pre-rendered static content with dynamic features

### 2. Content Structure Pattern
Follow this pattern for each concept:

```
src/views/[domain]/[concept]/
├── IndexView.vue          # Concept overview and navigation
├── BasicsView.vue         # Fundamental concepts
├── MethodsView.vue        # Techniques and algorithms  
├── ApplicationsView.vue   # Real-world applications
└── TheoryView.vue         # Advanced mathematical theory
```

### 3. Shared Component Library
Build these reusable content components:

- `ConceptLayout.vue` - Standard page layout
- `SectionHeader.vue` - Section titles and descriptions  
- `MathDisplay.vue` - LaTeX mathematical notation
- `PythonCode.vue` - Syntax-highlighted Python code
- `ConceptCard.vue` - Navigation cards
- `DefinitionBox.vue` - Mathematical definitions
- `ExampleBox.vue` - Worked examples
- `PracticeBox.vue` - Exercise sections

### 4. Interactive Component Integration
Embed your 19 interactive components directly:

```vue
<template>
  <SectionHeader title="Interactive Exploration">
    <!-- Direct component usage -->
    <StatisticsCalculator :initial-data="[1,2,3,4,5]" />
    <QuadraticExplorer :initial-coefficients="{a: 1, b: 2, c: 1}" />
  </SectionHeader>
</template>
```

### 5. Mathematical Notation System
Use KaTeX for mathematical expressions:

```vue
<template>
  <!-- Block math -->
  <MathDisplay expression="\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n" />
  
  <!-- Inline math -->
  <p>The formula <MathDisplay expression="f(x) = ax^2 + bx + c" display="inline" /> represents...</p>
</template>
```

### 6. Content Development Workflow

**Step 1: Create View Structure**
```bash
# For each new concept
mkdir -p src/views/[domain]/[concept]
touch src/views/[domain]/[concept]/BasicsView.vue
```

**Step 2: Implement Content Systematically**
1. Start with `BasicsView.vue` using the provided template
2. Add interactive components where appropriate  
3. Include Python code examples with explanations
4. Add real-world applications and practice sections

**Step 3: Update Router**
Add routes for each new view in `src/router/index.js`

**Step 4: Test Interactivity**
Ensure all mathematical components work correctly and calculations are accurate

### 7. Performance Optimization
The native Vue approach provides:

- **Code Splitting**: Each view loads only when needed
- **Component Caching**: Shared components cached across views  
- **Bundle Optimization**: Automatic tree-shaking of unused code
- **Fast Rendering**: No markdown parsing at runtime
- **SEO Benefits**: Pre-rendered HTML with dynamic features

This native Vue approach will give you a much more maintainable, performant, and interactive mathematical learning platform compared to markdown-based content.

This setup provides you with a solid foundation for migrating Snake Math to Vue 3 + Vite. The project structure is scalable, the development environment is optimized for mathematical content, and you have the essential tools and configurations needed for efficient development in VSCode.

You now have:
- ✅ Complete Vue 3 + Vite project setup
- ✅ VSCode workspace configuration with all essential extensions
- ✅ Mathematical computation engine foundation
- ✅ Canvas rendering utilities
- ✅ Essential CodeFold component
- ✅ Development workflow and debugging setup
- ✅ Performance optimization configuration

You're ready to start the migration process!
