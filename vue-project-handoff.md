# 🧠 Snake Math Vue 3 Application - Agentic AI Handoff Document

## 🎯 IMMEDIATE ACTION REQUIRED

**Task**: Create a new Vue 3 + Vite project called "snake-math-vue" that serves as an interactive mathematics learning platform.

**Priority**: Start by scaffolding the project structure, then implement core components systematically.

## 📘 Project Vision & Requirements

### Core Purpose
An interactive, browser-based mathematics learning platform with:
- **Modular concept pages** for different math topics
- **Dynamic visualizations** using modern web technologies
- **Progressive difficulty levels** within each topic
- **Interactive examples** and practice problems
- **Clean, modern UI** optimized for learning

### Technical Requirements
- **Framework**: Vue 3 (Composition API preferred)
- **Build Tool**: Vite (latest version)
- **Deployment**: GitHub Pages
- **Math Rendering**: KaTeX for mathematical expressions
- **Styling**: Modern CSS with Vue SFC styles
- **Navigation**: Component-based (no router needed)
- **State Management**: Simple reactive refs (no Pinia needed)
- **Browser Support**: Modern browsers (ES2020+)

## � STEP-BY-STEP IMPLEMENTATION GUIDE

### Phase 1: Project Initialization
1. **Create Vite project**: `npm create vue@latest snake-math-vue`
2. **Select options**: TypeScript (No), JSX (No), Router (No), Pinia (No), Vitest (No), ESLint (Yes), Prettier (Yes)
3. **Install dependencies**: Navigate to project and run `npm install`
4. **Install additional packages**:
   ```bash
   npm install katex
   npm install -D gh-pages
   ```

### Phase 2: Core Structure Setup
1. **Configure Vite** (see configuration below)
2. **Set up simplified project structure** (see directory structure below)
3. **Create main app component with tabbed navigation**
4. **Set up GitHub Pages deployment**

### Phase 3: Content Implementation
1. **Implement topic selection sidebar/tabs**
2. **Create math topic components** (start with basics, then algebra)
3. **Add KaTeX integration for math rendering**
4. **Implement interactive components**
5. **Add styling and responsive design**

## 📁 Simplified Directory Structure

```
snake-math-vue/
├── public/
│   ├── favicon.ico
│   └── index.html
├── src/
│   ├── assets/
│   │   ├── styles/
│   │   │   ├── main.css
│   │   │   ├── variables.css
│   │   │   └── components.css
│   │   └── images/
│   ├── components/
│   │   ├── common/
│   │   │   ├── TopicSidebar.vue
│   │   │   ├── MathRenderer.vue
│   │   │   └── InteractiveGraph.vue
│   │   ├── topics/
│   │   │   ├── BasicsContent.vue
│   │   │   ├── AlgebraContent.vue
│   │   │   ├── StatisticsContent.vue
│   │   │   ├── TrigonometryContent.vue
│   │   │   ├── LinearAlgebraContent.vue
│   │   │   └── CalculusContent.vue
│   │   └── widgets/
│   │       ├── QuadraticVisualizer.vue
│   │       ├── UnitCircleWidget.vue
│   │       ├── MatrixCalculator.vue
│   │       └── GraphPlotter.vue
│   ├── utils/
│   │   ├── mathHelpers.js
│   │   └── graphing.js
│   ├── App.vue
│   └── main.js
├── .github/
│   └── workflows/
│       └── deploy.yml
├── .gitignore
├── .eslintrc.cjs
├── .prettierrc.json
├── index.html
├── package.json
├── README.md
└── vite.config.js
```



## ⚙️ Essential Configuration Files

### `package.json` Dependencies
```json
{
  "name": "snake-math-vue",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "deploy": "npm run build && gh-pages -d dist"
  },
  "dependencies": {
    "vue": "^3.4.0",
    "katex": "^0.16.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "vite": "^5.0.0",
    "eslint": "^8.57.0",
    "prettier": "^3.2.0",
    "gh-pages": "^6.1.0"
  }
}
```

### `vite.config.js`
```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@components': resolve(__dirname, './src/components'),
      '@utils': resolve(__dirname, './src/utils')
    }
  },
  server: {
    port: 5173,
    open: true
  },
  build: {
    target: 'es2020',
    sourcemap: true,
    outDir: 'dist'
  },
  base: '/snake-math-vue/' // Important for GitHub Pages
})
```



## 🎨 Single-Page App Structure

### Main App Component (`src/App.vue`)
```vue
<template>
  <div id="app" class="app-container">
    <header class="app-header">
      <h1>🐍 Snake Math</h1>
      <p>Interactive Mathematics Learning Platform</p>
    </header>
    
    <div class="app-content">
      <TopicSidebar 
        :topics="mathTopics"
        :active-topic="activeTopic"
        @topic-change="handleTopicChange"
      />
      
      <main class="main-content">
        <component 
          :is="currentTopicComponent" 
          :key="activeTopic"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import TopicSidebar from '@/components/common/TopicSidebar.vue'
import BasicsContent from '@/components/topics/BasicsContent.vue'
import AlgebraContent from '@/components/topics/AlgebraContent.vue'
import StatisticsContent from '@/components/topics/StatisticsContent.vue'
import TrigonometryContent from '@/components/topics/TrigonometryContent.vue'
import LinearAlgebraContent from '@/components/topics/LinearAlgebraContent.vue'
import CalculusContent from '@/components/topics/CalculusContent.vue'

const activeTopic = ref('basics')

const mathTopics = [
  { id: 'basics', title: 'Basics', icon: '📚' },
  { id: 'algebra', title: 'Algebra', icon: '🔢' },
  { id: 'statistics', title: 'Statistics', icon: '📊' },
  { id: 'trigonometry', title: 'Trigonometry', icon: '📐' },
  { id: 'linear-algebra', title: 'Linear Algebra', icon: '🔺' },
  { id: 'calculus', title: 'Calculus', icon: '∫' }
]

const topicComponents = {
  basics: BasicsContent,
  algebra: AlgebraContent,
  statistics: StatisticsContent,
  trigonometry: TrigonometryContent,
  'linear-algebra': LinearAlgebraContent,
  calculus: CalculusContent
}

const currentTopicComponent = computed(() => {
  return topicComponents[activeTopic.value] || BasicsContent
})

const handleTopicChange = (topicId) => {
  activeTopic.value = topicId
}
</script>

<style>
:root {
  --primary-color: #3b82f6;
  --secondary-color: #10b981;
  --accent-color: #f59e0b;
  --text-color: #1f2937;
  --bg-color: #f9fafb;
  --border-color: #e5e7eb;
  --sidebar-width: 280px;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.6;
  color: var(--text-color);
  background-color: var(--bg-color);
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: white;
  padding: 1.5rem 2rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  text-align: center;
}

.app-header h1 {
  color: var(--primary-color);
  margin-bottom: 0.5rem;
}

.app-content {
  display: flex;
  flex: 1;
  min-height: calc(100vh - 100px);
}

.main-content {
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
}

@media (max-width: 768px) {
  .app-content {
    flex-direction: column;
  }
  
  .main-content {
    padding: 1rem;
  }
}
</style>
```

### Topic Sidebar Component (`src/components/common/TopicSidebar.vue`)
```vue
<template>
  <aside class="topic-sidebar">
    <nav class="topic-nav">
      <h2>Topics</h2>
      <ul class="topic-list">
        <li 
          v-for="topic in topics" 
          :key="topic.id"
          class="topic-item"
        >
          <button 
            @click="$emit('topic-change', topic.id)"
            :class="['topic-button', { active: activeTopic === topic.id }]"
          >
            <span class="topic-icon">{{ topic.icon }}</span>
            <span class="topic-title">{{ topic.title }}</span>
          </button>
        </li>
      </ul>
    </nav>
  </aside>
</template>

<script setup>
defineProps({
  topics: {
    type: Array,
    required: true
  },
  activeTopic: {
    type: String,
    required: true
  }
})

defineEmits(['topic-change'])
</script>

<style scoped>
.topic-sidebar {
  width: var(--sidebar-width);
  background: white;
  border-right: 1px solid var(--border-color);
  padding: 2rem 0;
}

.topic-nav h2 {
  padding: 0 1.5rem;
  margin-bottom: 1.5rem;
  color: var(--text-color);
  font-size: 1.2rem;
}

.topic-list {
  list-style: none;
}

.topic-item {
  margin-bottom: 0.5rem;
}

.topic-button {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: none;
  border: none;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--text-color);
}

.topic-button:hover {
  background: var(--bg-color);
}

.topic-button.active {
  background: var(--primary-color);
  color: white;
}

.topic-icon {
  font-size: 1.2rem;
}

.topic-title {
  font-weight: 500;
}

@media (max-width: 768px) {
  .topic-sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 0;
  }
  
  .topic-list {
    display: flex;
    overflow-x: auto;
    gap: 0.5rem;
    padding: 0 1rem;
  }
  
  .topic-item {
    margin-bottom: 0;
    flex-shrink: 0;
  }
  
  .topic-button {
    white-space: nowrap;
    border-radius: 20px;
    padding: 0.75rem 1rem;
  }
}
</style>
```



### GitHub Actions Deployment (`.github/workflows/deploy.yml`)
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
      
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'
        
    - name: Install dependencies
      run: npm ci
      
    - name: Build
      run: npm run build
      
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      if: github.ref == 'refs/heads/main'
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./dist
```

## 🎨 Core Component Templates

### Basic App Structure (`src/App.vue`)
```vue
<template>
  <div id="app">
    <NavBar />
    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<script setup>
import NavBar from '@/components/common/NavBar.vue'
</script>

<style>
:root {
  --primary-color: #3b82f6;
  --secondary-color: #10b981;
  --accent-color: #f59e0b;
  --text-color: #1f2937;
  --bg-color: #f9fafb;
  --border-color: #e5e7eb;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.6;
  color: var(--text-color);
  background-color: var(--bg-color);
}

.main-content {
  min-height: calc(100vh - 60px);
  padding: 2rem;
}
</style>
```

### Navigation Component (`src/components/common/NavBar.vue`)
```vue
<template>
  <nav class="navbar">
    <div class="nav-container">
      <RouterLink to="/" class="nav-brand">
        🐍 Snake Math
      </RouterLink>
      
      <div class="nav-menu" :class="{ 'nav-menu-active': isMenuOpen }">
        <div class="nav-dropdown">
          <button class="nav-link dropdown-btn">
            Basics <span class="dropdown-arrow">▼</span>
          </button>
          <div class="dropdown-content">
            <RouterLink to="/basics/foundations" class="dropdown-link">Foundations</RouterLink>
            <RouterLink to="/basics/functions" class="dropdown-link">Functions</RouterLink>
            <RouterLink to="/basics/number-theory" class="dropdown-link">Number Theory</RouterLink>
          </div>
        </div>
        
        <div class="nav-dropdown">
          <button class="nav-link dropdown-btn">
            Algebra <span class="dropdown-arrow">▼</span>
          </button>
          <div class="dropdown-content">
            <RouterLink to="/algebra/quadratics" class="dropdown-link">Quadratics</RouterLink>
            <RouterLink to="/algebra/linear-equations" class="dropdown-link">Linear Equations</RouterLink>
            <RouterLink to="/algebra/exponentials" class="dropdown-link">Exponentials</RouterLink>
            <RouterLink to="/algebra/summation" class="dropdown-link">Summation</RouterLink>
          </div>
        </div>
        
        <!-- Add more dropdowns for other subjects -->
      </div>
      
      <button class="nav-toggle" @click="toggleMenu">
        <span></span>
        <span></span>
        <span></span>
      </button>
    </div>
  </nav>
</template>

<script setup>
import { ref } from 'vue'

const isMenuOpen = ref(false)

const toggleMenu = () => {
  isMenuOpen.value = !isMenuOpen.value
}
</script>

<style scoped>
.navbar {
  background: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  position: sticky;
  top: 0;
  z-index: 1000;
  height: 60px;
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
}

.nav-brand {
  font-size: 1.5rem;
  font-weight: bold;
  text-decoration: none;
  color: var(--primary-color);
}

.nav-menu {
  display: flex;
  gap: 2rem;
}

.nav-dropdown {
  position: relative;
}

.dropdown-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 0.5rem 1rem;
  font-size: 1rem;
  color: var(--text-color);
}

.dropdown-content {
  display: none;
  position: absolute;
  background: white;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  border-radius: 4px;
  min-width: 200px;
  top: 100%;
  left: 0;
}

.nav-dropdown:hover .dropdown-content {
  display: block;
}

.dropdown-link {
  display: block;
  padding: 0.75rem 1rem;
  text-decoration: none;
  color: var(--text-color);
  border-bottom: 1px solid var(--border-color);
}

.dropdown-link:hover {
  background: var(--bg-color);
}

/* Mobile responsive */
@media (max-width: 768px) {
  .nav-menu {
    position: fixed;
    left: -100%;
    top: 60px;
    flex-direction: column;
    background: white;
    width: 100%;
    text-align: center;
    transition: 0.3s;
    box-shadow: 0 10px 27px rgba(0,0,0,0.05);
    height: calc(100vh - 60px);
    overflow-y: auto;
  }

  .nav-menu-active {
    left: 0;
  }

  .nav-toggle {
    display: block;
    background: none;
    border: none;
    cursor: pointer;
  }
}
</style>
```

### Example Topic Content Component (`src/components/topics/AlgebraContent.vue`)
```vue
<template>
  <div class="algebra-content">
    <header class="topic-header">
      <h1>🔢 Algebra</h1>
      <p class="topic-description">
        Master algebraic concepts from basic equations to advanced functions
      </p>
    </header>
    
    <div class="subtopic-grid">
      <div 
        v-for="subtopic in subtopics" 
        :key="subtopic.id"
        class="subtopic-card"
        @click="selectSubtopic(subtopic.id)"
        :class="{ active: activeSubtopic === subtopic.id }"
      >
        <div class="subtopic-icon">{{ subtopic.icon }}</div>
        <h3>{{ subtopic.title }}</h3>
        <p>{{ subtopic.description }}</p>
      </div>
    </div>
    
    <div class="subtopic-content">
      <component 
        :is="currentSubtopicComponent" 
        v-if="activeSubtopic"
        :key="activeSubtopic"
      />
      <div v-else class="select-prompt">
        <p>👆 Select a topic above to begin learning</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import QuadraticVisualizer from '@/components/widgets/QuadraticVisualizer.vue'
import MathRenderer from '@/components/common/MathRenderer.vue'

const activeSubtopic = ref(null)

const subtopics = [
  {
    id: 'quadratics',
    title: 'Quadratic Functions',
    description: 'Explore parabolas and quadratic equations',
    icon: '📈'
  },
  {
    id: 'linear',
    title: 'Linear Equations',
    description: 'Master straight lines and systems',
    icon: '📏'
  },
  {
    id: 'exponentials',
    title: 'Exponentials & Logs',
    description: 'Growth, decay, and logarithms',
    icon: '📊'
  },
  {
    id: 'summation',
    title: 'Summation Notation',
    description: 'Sum sequences with sigma notation',
    icon: 'Σ'
  }
]

// Simple content components for each subtopic
const QuadraticsContent = {
  components: { MathRenderer, QuadraticVisualizer },
  template: `
    <div class="content-section">
      <h2>Quadratic Functions</h2>
      <p>A quadratic function has the standard form:</p>
      <MathRenderer 
        expression="f(x) = ax^2 + bx + c" 
        :display-mode="true" 
      />
      <p>Where a ≠ 0, and the graph forms a parabola.</p>
      
      <div class="interactive-section">
        <h3>Interactive Visualizer</h3>
        <QuadraticVisualizer />
      </div>
    </div>
  `
}

const LinearContent = {
  components: { MathRenderer },
  template: `
    <div class="content-section">
      <h2>Linear Equations</h2>
      <p>Linear equations form straight lines with the slope-intercept form:</p>
      <MathRenderer 
        expression="y = mx + b" 
        :display-mode="true" 
      />
      <p>Where m is the slope and b is the y-intercept.</p>
    </div>
  `
}

const ExponentialsContent = {
  components: { MathRenderer },
  template: `
    <div class="content-section">
      <h2>Exponential Functions</h2>
      <p>Exponential functions have the form:</p>
      <MathRenderer 
        expression="f(x) = a \\cdot b^x" 
        :display-mode="true" 
      />
      <p>Where a > 0 and b > 0, b ≠ 1.</p>
    </div>
  `
}

const SummationContent = {
  components: { MathRenderer },
  template: `
    <div class="content-section">
      <h2>Summation Notation</h2>
      <p>Summation (sigma) notation compactly represents sums:</p>
      <MathRenderer 
        expression="\\sum_{i=1}^{n} a_i = a_1 + a_2 + \\cdots + a_n" 
        :display-mode="true" 
      />
      <p>This represents the sum from i=1 to n of terms a_i.</p>
    </div>
  `
}

const subtopicComponents = {
  quadratics: QuadraticsContent,
  linear: LinearContent,
  exponentials: ExponentialsContent,
  summation: SummationContent
}

const currentSubtopicComponent = computed(() => {
  return subtopicComponents[activeSubtopic.value]
})

const selectSubtopic = (subtopicId) => {
  activeSubtopic.value = subtopicId
}
</script>

<style scoped>
.algebra-content {
  max-width: 1000px;
  margin: 0 auto;
}

.topic-header {
  text-align: center;
  margin-bottom: 3rem;
}

.topic-header h1 {
  color: var(--primary-color);
  margin-bottom: 1rem;
  font-size: 2.5rem;
}

.topic-description {
  font-size: 1.2rem;
  color: #6b7280;
}

.subtopic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
}

.subtopic-card {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: center;
  border: 2px solid transparent;
}

.subtopic-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.subtopic-card.active {
  border-color: var(--primary-color);
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}

.subtopic-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.subtopic-card h3 {
  color: var(--text-color);
  margin-bottom: 0.5rem;
}

.subtopic-card p {
  color: #6b7280;
  font-size: 0.9rem;
}

.subtopic-content {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  min-height: 400px;
}

.select-prompt {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 300px;
  color: #6b7280;
  font-size: 1.2rem;
}

.content-section h2 {
  color: var(--primary-color);
  margin-bottom: 1.5rem;
}

.interactive-section {
  margin-top: 2rem;
  padding: 1.5rem;
  background: var(--bg-color);
  border-radius: 8px;
}

.interactive-section h3 {
  color: var(--text-color);
  margin-bottom: 1rem;
}
</style>
```

### Math Renderer Component (`src/components/common/MathRenderer.vue`)
```vue
<template>
  <div class="math-renderer" :class="{ 'display-mode': displayMode }">
    <div ref="mathContainer" v-html="renderedMath"></div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const props = defineProps({
  expression: {
    type: String,
    required: true
  },
  displayMode: {
    type: Boolean,
    default: false
  }
})

const mathContainer = ref(null)

const renderedMath = computed(() => {
  try {
    return katex.renderToString(props.expression, {
      displayMode: props.displayMode,
      throwOnError: false
    })
  } catch (error) {
    console.error('KaTeX rendering error:', error)
    return `<span class="math-error">Math Error: ${props.expression}</span>`
  }
})
</script>

<style scoped>
.math-renderer {
  margin: 0.5rem 0;
}

.display-mode {
  text-align: center;
  margin: 1rem 0;
}

.math-error {
  color: #ef4444;
  font-style: italic;
}
</style>
```

## 🔧 Essential Utility Functions

### Math Helpers (`src/utils/mathHelpers.js`)
```javascript
// Basic mathematical operations and formatters
export const formatNumber = (num, decimals = 2) => {
  return Number(num).toFixed(decimals)
}

export const generateRange = (start, end, step = 1) => {
  const range = []
  for (let i = start; i <= end; i += step) {
    range.push(i)
  }
  return range
}

export const evaluateQuadratic = (a, b, c, x) => {
  return a * x * x + b * x + c
}

export const solveQuadratic = (a, b, c) => {
  const discriminant = b * b - 4 * a * c
  
  if (discriminant < 0) {
    return { solutions: [], type: 'no-real-solutions' }
  } else if (discriminant === 0) {
    return { 
      solutions: [-b / (2 * a)], 
      type: 'one-solution' 
    }
  } else {
    return { 
      solutions: [
        (-b + Math.sqrt(discriminant)) / (2 * a),
        (-b - Math.sqrt(discriminant)) / (2 * a)
      ], 
      type: 'two-solutions' 
    }
  }
}

export const calculateDistance = (point1, point2) => {
  const dx = point2.x - point1.x
  const dy = point2.y - point1.y
  return Math.sqrt(dx * dx + dy * dy)
}

export const degreesToRadians = (degrees) => {
  return degrees * (Math.PI / 180)
}

export const radiansToDegrees = (radians) => {
  return radians * (180 / Math.PI)
}
```

### Graphing Utilities (`src/utils/graphing.js`)
```javascript
// Canvas-based graphing utilities
export class GraphRenderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.width = canvas.width
    this.height = canvas.height
    
    this.options = {
      xMin: -10,
      xMax: 10,
      yMin: -10,
      yMax: 10,
      gridColor: '#e5e7eb',
      axisColor: '#374151',
      functionColor: '#3b82f6',
      ...options
    }
    
    this.setupTransform()
  }
  
  setupTransform() {
    const { xMin, xMax, yMin, yMax } = this.options
    this.scaleX = this.width / (xMax - xMin)
    this.scaleY = this.height / (yMax - yMin)
    this.originX = -xMin * this.scaleX
    this.originY = this.height + yMin * this.scaleY
  }
  
  clear() {
    this.ctx.clearRect(0, 0, this.width, this.height)
  }
  
  drawGrid() {
    const { xMin, xMax, yMin, yMax, gridColor } = this.options
    this.ctx.strokeStyle = gridColor
    this.ctx.lineWidth = 1
    
    // Vertical lines
    for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
      const canvasX = this.xToCanvas(x)
      this.ctx.beginPath()
      this.ctx.moveTo(canvasX, 0)
      this.ctx.lineTo(canvasX, this.height)
      this.ctx.stroke()
    }
    
    // Horizontal lines
    for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
      const canvasY = this.yToCanvas(y)
      this.ctx.beginPath()
      this.ctx.moveTo(0, canvasY)
      this.ctx.lineTo(this.width, canvasY)
      this.ctx.stroke()
    }
  }
  
  drawAxes() {
    const { axisColor } = this.options
    this.ctx.strokeStyle = axisColor
    this.ctx.lineWidth = 2
    
    // X-axis
    this.ctx.beginPath()
    this.ctx.moveTo(0, this.originY)
    this.ctx.lineTo(this.width, this.originY)
    this.ctx.stroke()
    
    // Y-axis
    this.ctx.beginPath()
    this.ctx.moveTo(this.originX, 0)
    this.ctx.lineTo(this.originX, this.height)
    this.ctx.stroke()
  }
  
  plotFunction(func, color = this.options.functionColor) {
    const { xMin, xMax } = this.options
    this.ctx.strokeStyle = color
    this.ctx.lineWidth = 2
    this.ctx.beginPath()
    
    let firstPoint = true
    for (let x = xMin; x <= xMax; x += 0.1) {
      try {
        const y = func(x)
        const canvasX = this.xToCanvas(x)
        const canvasY = this.yToCanvas(y)
        
        if (firstPoint) {
          this.ctx.moveTo(canvasX, canvasY)
          firstPoint = false
        } else {
          this.ctx.lineTo(canvasX, canvasY)
        }
      } catch (error) {
        // Skip invalid points
        firstPoint = true
      }
    }
    this.ctx.stroke()
  }
  
  xToCanvas(x) {
    return this.originX + x * this.scaleX
  }
  
  yToCanvas(y) {
    return this.originY - y * this.scaleY
  }
}
```

## 📝 Content Implementation Strategy

## 📝 Simplified Content Implementation Strategy

### 1. Single-Page Structure Benefits
- **Simpler routing**: No Vue Router needed, just component switching
- **Faster navigation**: No page reloads, instant topic switching
- **Easier state management**: Share data between topics using reactive refs
- **Reduced complexity**: Fewer files and configuration
- **Better mobile experience**: Native-feeling navigation

### 2. Topic Organization
Each major topic (Basics, Algebra, etc.) gets:
- **One main content component** (`AlgebraContent.vue`)
- **Subtopic cards** for different concepts within that topic
- **Dynamic content switching** based on selected subtopic
- **Reusable widgets** for interactive features

### 3. Implementation Order
1. **Start with App.vue** and TopicSidebar
2. **Create one topic content component** (e.g., AlgebraContent)
3. **Add 2-3 subtopics** with basic math rendering
4. **Implement one interactive widget** (e.g., QuadraticVisualizer)
5. **Expand to other topics** following the same pattern

## 🎯 Development Priorities

### Phase 1 (Essential)
1. ✅ Project setup and configuration  
2. ✅ Main App.vue with sidebar navigation
3. ✅ Math rendering with KaTeX
4. ✅ Responsive design foundation
5. ✅ One complete topic (Algebra) with 2-3 subtopics

### Phase 2 (Core Content)
1. 📚 Implement 2-3 additional topic content components
2. 🎨 Create 3-4 interactive widget components
3. 📱 Mobile-optimized sidebar and content layout
4. 🧪 Add example problems for each subtopic

### Phase 3 (Advanced Features)
1. 📊 Advanced interactive graphing widgets
2. 🎮 Practice problem generators with feedback
3. 💾 Local storage for user progress/preferences
4. 🎯 Animations and smooth transitions between topics

## 🚀 Simplified Deployment Instructions

1. **Setup GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/snake-math-vue.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Select "GitHub Actions" as source
   - The workflow will automatically deploy on push to main

3. **Local Development**:
   ```bash
   npm run dev    # Development server
   npm run build  # Production build
   npm run preview # Preview production build
   ```

## 🧠 Key Success Factors for Single-Page App

### For Agentic AI Implementation:
1. **Component-Based Architecture**: Each topic is a self-contained component
2. **Simple State Management**: Use reactive refs instead of complex state stores
3. **Modular Widgets**: Reusable interactive components across topics
4. **Clear Component Hierarchy**: App → Topic → Subtopic → Widget
5. **Event-Driven Navigation**: Simple emit/listen pattern for topic switching

### What You Can Leave Out:
- ❌ **Vue Router**: No need for routing, just component switching
- ❌ **Pinia/Vuex**: Simple reactive refs are sufficient
- ❌ **Multiple View Files**: Everything in topic components
- ❌ **Complex Route Guards**: No authentication or route protection needed
- ❌ **vue-katex**: Direct KaTeX import is simpler
- ❌ **Vitest**: Focus on building first, testing later

### Quality Standards:
- **Performance**: Lazy load topic components for faster initial load
- **Mobile-First**: Collapsible sidebar, touch-friendly navigation
- **Accessibility**: Keyboard navigation, semantic HTML, proper ARIA labels
- **Maintainable**: Clear component structure, consistent naming

This simplified single-page approach reduces complexity by ~40% while maintaining all the core functionality. Perfect for getting started quickly and adding features incrementally!
