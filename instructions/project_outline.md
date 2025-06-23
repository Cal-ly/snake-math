# Snake Math Project: Complete Technical Handoff Specification

## Executive Summary

**Snake Math** is an interactive mathematical learning platform designed to bridge mathematical concepts with Python programming through in-browser execution and dynamic visualizations. This document provides comprehensive technical specifications for rebuilding the project with modern technologies (e.g., Vue 3, React, Angular, or alternative frameworks).

### Core Objectives
1. **Educational Bridge**: Connect abstract mathematical concepts to concrete programming implementations
2. **Zero Friction Access**: Browser-based Python execution without installation requirements
3. **Interactive Learning**: Real-time mathematical exploration through dynamic components
4. **Progressive Complexity**: Structured learning paths from basic concepts to advanced applications
5. **Production Quality**: Deployment-ready static site with optimal performance

### Current Technology Stack (Reference Implementation - Has Scale Limitations)
- **VitePress 1.x** - ⚠️ Static site generator struggling with 55+ pages and 19 interactive components
- **Vue.js 2.x** - ⚠️ Component framework requiring upgrade for better performance
- **PyScript 2025.5.1** - ✅ In-browser Python execution engine (works well)
- **MathJax** - ✅ Mathematical notation rendering (LaTeX support)
- **GitHub Pages** - ✅ Static hosting with CI/CD via GitHub Actions
- **Node.js** - ✅ Build toolchain and dependency management

### **Recommended Migration Path: Vue 3 + Vite**
**Why this is the optimal choice:**
- **Minimal Migration Effort**: 4-6 weeks vs 10+ weeks for other frameworks
- **Performance Gains**: 3-5x faster builds, 50% smaller bundles
- **Mathematical Focus**: Composition API perfect for complex calculations
- **Component Architecture**: Natural chunking for 19 interactive components
- **Proven Stability**: Mature ecosystem with excellent mathematical library support

### Alternative Migration Targets (Analysis Available)
- **Astro + Vue** (for maximum performance and minimal bundle sizes)
- **Nuxt 3** (for enhanced content management and SEO)
- **SvelteKit** (for compile-time optimizations and smallest bundles)
- **React + Next.js** (for React ecosystem, requires complete rewrite)

---

## Technical Architecture Requirements

### 1. Mathematical Computation Engine

**Core Requirement**: All mathematical calculations must be performed with high precision and accuracy. Components must handle:

- **Floating Point Precision**: Calculations displayed to 3 decimal places, internal precision to 10+ decimals
- **Edge Case Handling**: Division by zero, complex number results, infinite values
- **Mathematical Domain Validation**: Input validation for domain restrictions (e.g., log(x) where x > 0)
- **Real-time Computation**: Sub-50ms response time for parameter changes
- **Mathematical Formula Parsing**: Support for LaTeX rendering and mathematical expression evaluation

**Implementation Requirements**:
```javascript
// Example precision handling
const calculateWithPrecision = (value, decimals = 3) => {
  return parseFloat(value.toFixed(decimals))
}

// Example edge case handling
const safeLog = (x, base = Math.E) => {
  if (x <= 0) throw new Error('Logarithm undefined for non-positive values')
  return Math.log(x) / Math.log(base)
}
```

### 2. Interactive Component API Specification

All interactive components must implement a standardized interface:

**Required Props**:
- `initialValues?: object` - Default parameter values
- `constrains?: object` - Min/max bounds for inputs
- `precision?: number` - Decimal precision for displays (default: 3)
- `realTime?: boolean` - Enable real-time updates (default: true)
- `showSteps?: boolean` - Display step-by-step calculations
- `theme?: 'light' | 'dark'` - Theme support

**Required Events**:
- `@parameterChange` - Emitted when any parameter changes
- `@calculationComplete` - Emitted when computation finishes
- `@error` - Emitted on calculation errors
- `@export` - Emitted when user requests data export

**Required Methods**:
- `reset()` - Reset to initial state
- `updateParameters(params)` - Programmatically update parameters
- `exportData()` - Export current state/results
- `validateInput(value, parameter)` - Validate parameter input

### 3. Python Code Execution Requirements

**In-Browser Python Environment**:
- **Library Support**: NumPy, SciPy equivalents for mathematical operations
- **Error Handling**: Graceful error display with educational feedback
- **Performance**: Code execution within 500ms for typical examples
- **Security**: Sandboxed execution environment
- **Persistence**: Session-based variable storage

**Code Example Format**:
```python
# Standard code block format
def example_function(parameter):
    """
    Clear docstring explaining purpose
    Args: parameter descriptions
    Returns: return value description
    """
    # Step-by-step implementation with comments
    result = mathematical_operation(parameter)
    return result

# Example usage with expected output
print(f"Result: {example_function(5)}")  # Output: Result: 25
```

### 4. Canvas Visualization Requirements

**Rendering Specifications**:
- **Resolution**: 600x400px default, responsive scaling
- **Performance**: 60fps for real-time updates
- **Accessibility**: Keyboard navigation support
- **Export**: PNG/SVG export capability
- **Responsiveness**: Mobile-friendly touch interactions

**Standard Coordinate System**:
```javascript
// Canvas coordinate transformation
const toCanvasCoords = (x, y, bounds, canvasSize) => ({
  x: (x - bounds.xMin) / (bounds.xMax - bounds.xMin) * canvasSize.width,
  y: canvasSize.height - (y - bounds.yMin) / (bounds.yMax - bounds.yMin) * canvasSize.height
})
```

### 5. Content Management System

**Page Template Structure** (Must be strictly followed):

```yaml
---
title: "Concept Name"
description: "Brief description (50-100 chars)"
tags: ["mathematics", "programming", "specific-topic"]
difficulty: "beginner" | "intermediate" | "advanced"
category: "concept" | "tutorial" | "reference"
symbol: "Mathematical Symbol (if applicable)"
prerequisites: ["concept-1", "concept-2"]
related_concepts: ["related-1", "related-2"]
applications: ["programming", "data-science", "engineering"]
interactive: true | false
code_examples: true | false
complexity_analysis: true | false
real_world_examples: true | false
layout: "concept-page"
date_created: "YYYY-MM-DD"
last_updated: "YYYY-MM-DD"
author: "Author Name"
reviewers: ["reviewer-1", "reviewer-2"]
version: "X.Y"
---
```

**Content Section Requirements**:
1. **Understanding Section** (required)
   - Mathematical definition with LaTeX notation
   - Programming analogy or connection
   - Simple example with clear explanation

2. **Interactive Exploration** (if interactive: true)
   - Component implementation with descriptive comment
   - Clear instructions for user interaction
   - Expected learning outcomes

3. **Techniques & Efficiency** (required)
   - Multiple implementation approaches
   - Big O complexity analysis where applicable
   - Performance comparisons with code examples

4. **Common Patterns** (required)
   - Standard mathematical formulas
   - Frequent use cases with code implementations
   - Best practices

5. **Real-world Applications** (if real_world_examples: true)
   - Industry-specific use cases
   - Practical programming scenarios
   - Complete working examples

6. **Practice Section** (required)
   - Suggested exercises
   - Extended exploration ideas
   - Related concept recommendations

### 6. Quality Assurance Requirements

**Mathematical Accuracy**:
- All formulas verified against authoritative mathematical sources
- Edge cases thoroughly tested
- Numerical stability validated for all parameter ranges

**Code Quality**:
- All Python code executable and tested
- Clear, educational comments
- Proper error handling and input validation

**Performance Benchmarks**:
- Page load time < 2 seconds
- Component render time < 100ms
- Mathematical calculation time < 50ms
- Canvas update rate ≥ 30fps

**Accessibility Standards**:
- WCAG 2.1 AA compliance
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support

---

## Detailed Component Specifications

### Core Interactive Components (19 Required Components)

#### 1. StatisticsCalculator Component

**Purpose**: Interactive descriptive statistics exploration with real-time calculations and visualizations

**Required Features**:
- **Data Input**: Manual entry, preset datasets, random generation
- **Statistical Measures**: Mean, median, mode, range, standard deviation, variance, quartiles
- **Visualizations**: Box plot, histogram with adjustable bins, distribution analysis
- **Analysis Tools**: Skewness detection, outlier identification, probability calculations
- **Export Functionality**: Statistical summaries, visualization downloads

**Technical Specifications**:
```typescript
interface StatisticsCalculatorProps {
  initialData?: number[]
  presetDatasets?: PresetDataset[]
  maxDataPoints?: number
  precision?: number
  showVisualization?: boolean
  enableExport?: boolean
}

interface StatisticsResult {
  mean: number
  median: number
  mode: string | number[]
  range: number
  standardDeviation: number
  variance: number
  quartiles: {q1: number, q3: number}
  outliers: number[]
  skewness: SkewnessAnalysis
}
```

**Canvas Requirements**:
- Box plot: 500x150px, interactive quartile highlighting
- Histogram: 500x300px, dynamic binning (5-20 bins)
- Real-time updates on data changes

#### 2. QuadraticExplorer Component

**Purpose**: Interactive quadratic function exploration with multiple forms and real-world applications

**Required Features**:
- **Multiple Forms**: Standard (ax²+bx+c), Vertex (a(x-h)²+k), Factored (a(x-r₁)(x-r₂))
- **Real-time Visualization**: Parabola plotting with vertex, roots, axis of symmetry
- **Analysis Tools**: Discriminant calculation, domain/range display
- **Form Conversion**: Automatic conversion between forms
- **Scenario Modeling**: Projectile motion, profit optimization, parabolic reflectors
- **Problem Solving**: Step-by-step quadratic formula derivation

**Technical Specifications**:
```typescript
interface QuadraticExplorerProps {
  initialCoefficients?: {a: number, b: number, c: number}
  enableMultipleForms?: boolean
  showStepByStep?: boolean
  enableScenarios?: boolean
  graphBounds?: {xMin: number, xMax: number, yMin: number, yMax: number}
}

interface QuadraticAnalysis {
  vertex: {x: number, y: number}
  discriminant: number
  roots: number[]
  axisOfSymmetry: number
  yIntercept: number
  hasRealRoots: boolean
  vertexType: 'minimum' | 'maximum'
}
```

**Canvas Requirements**:
- Graph canvas: 600x500px with responsive scaling
- Coordinate system: -10 to 10 (x), -15 to 15 (y) default
- Interactive elements: draggable points, hover tooltips

#### 3. CodeFold Component

**Purpose**: Collapsible code block container for improved content organization

**Required Features**:
- **Collapsible Interface**: Show/hide toggle with smooth animation
- **Syntax Highlighting**: Support for Python, JavaScript, mathematical notation
- **Accessibility**: Keyboard navigation, screen reader support
- **Theme Support**: Light/dark mode compatibility

**Technical Specifications**:
```typescript
interface CodeFoldProps {
  initialState?: 'expanded' | 'collapsed'
  title?: string
  language?: string
  showLineNumbers?: boolean
  enableCopy?: boolean
}
```

**Styling Requirements**:
- Consistent border radius (8px)
- Smooth transitions (0.2s ease)
- Accessible color contrast ratios
- Mobile-responsive behavior

### Mathematical Visualization Components

#### 4. UnitCircleExplorer Component
- **Canvas Size**: 500x500px
- **Interactive Elements**: Draggable angle marker, coordinate display
- **Trigonometric Functions**: sin, cos, tan with real-time calculation
- **Quadrant Analysis**: Angle classification and sign analysis

#### 5. FunctionPlotter Component
- **Multi-function Support**: Overlay multiple functions with color coding
- **Zoom/Pan**: Interactive graph navigation
- **Function Types**: Polynomial, trigonometric, exponential, logarithmic
- **Export**: PNG/SVG graph export

#### 6. VectorOperations Component
- **2D/3D Visualization**: Vector representation and operations
- **Operations**: Addition, subtraction, dot product, cross product
- **Interactive Manipulation**: Drag-to-modify vectors
- **Magnitude/Direction**: Real-time calculation display

### Advanced Mathematical Components

#### 7. MatrixTransformations Component
- **Matrix Operations**: Addition, multiplication, determinant, inverse
- **Geometric Transformations**: Rotation, scaling, reflection visualization
- **Interactive Input**: Click-to-edit matrix elements
- **Step-by-step**: Matrix operation breakdown

#### 8. LimitsExplorer Component
- **Numerical Approximation**: Table of values approaching limit
- **Graphical Visualization**: Function behavior near limit point
- **Multiple Approaches**: Left/right limits, infinity limits
- **L'Hôpital's Rule**: Automatic application for indeterminate forms

#### 9. ProbabilitySimulator Component
- **Distribution Types**: Normal, binomial, Poisson, uniform
- **Interactive Parameters**: Real-time distribution shape updates
- **Simulation Engine**: Monte Carlo sampling with visualization
- **Statistical Testing**: Hypothesis testing tools

---

## Content Organization System

### Learning Path Structure

The content is organized into 6 major mathematical domains with progressive complexity:

```
Learning Progression:
Basics → Algebra → Statistics → Trigonometry → Linear Algebra → Calculus

Each domain contains:
├── index.md (overview & navigation)
├── concept-1/ (modular sub-topics)
│   ├── index.md (concept overview)
│   ├── basics.md (fundamentals)
│   ├── methods.md (techniques & algorithms)
│   ├── applications.md (real-world usage)
│   └── [additional-topics].md
```

### Content Areas Specification

#### 1. Basics (Mathematical Foundations)
**Target Audience**: Complete beginners, novice programmers
**Learning Objectives**: 
- Understand fundamental mathematical concepts
- Connect mathematical operations to programming constructs
- Build foundation for advanced topics

**Required Content**:
- **foundations.md**: Number systems, operations, mathematical reasoning (foundational concepts)
- **functions.md**: Function definitions, types, operations, domain/range
- **variables-expressions.md**: Variables, algebraic expressions, substitution
- **number-theory.md**: Prime numbers, divisibility, number properties
- **order-of-operations.md**: PEMDAS/BODMAS, expression evaluation

**Interactive Components Required**: VariableExpressionExplorer, FunctionsVisualization, NumberTypeExplorer

#### 2. Algebra (Symbolic Mathematics)
**Target Audience**: Developers familiar with basic mathematics
**Learning Objectives**:
- Master algebraic notation and manipulation
- Understand mathematical series and sequences
- Apply algebraic concepts to programming problems

**Required Content Structure**:
```
algebra/
├── summation-notation/ (5 sub-pages, ~250 lines each)
│   ├── basics.md → Σ syntax, simple examples, for-loop analogies
│   ├── properties.md → Algebraic properties, manipulation rules
│   ├── advanced.md → Double summations, infinite series, induction
│   └── applications.md → Real-world statistical/CS applications
├── product-notation/ (5 sub-pages)
│   ├── basics.md → Π syntax, mathematical definition
│   ├── properties.md → Empty products, logarithmic relationships
│   ├── advanced.md → Infinite products, optimization
│   └── applications.md → ML/Naive Bayes, compound interest
├── linear-equations/ (4 sub-pages)
│   ├── basics.md → Single variable equations, solving techniques
│   ├── systems.md → Multi-variable systems, matrix methods
│   └── applications.md → Data science, engineering applications
├── quadratics/ (5 sub-pages)
│   ├── basics.md → Quadratic functions, standard form
│   ├── solving.md → Solution methods, quadratic formula
│   ├── theory.md → Mathematical foundations, derivations
│   └── applications.md → Physics, optimization, graphics
└── exponentials-logarithms/ (4 sub-pages)
    ├── exponentials.md → Exponential functions, growth patterns
    ├── logarithms.md → Logarithmic functions, properties
    └── applications.md → Computer science, data science applications
```

**Interactive Components Required**: SummationDemo, ProductNotationVisualizer, QuadraticExplorer, ExponentialCalculator, LinearSystemSolver

#### 3. Statistics (Data Analysis)
**Target Audience**: Data analysts, data scientists, researchers
**Learning Objectives**:
- Understand descriptive and inferential statistics
- Apply statistical concepts to data analysis
- Implement statistical algorithms in Python

**Required Content Structure**:
```
statistics/
├── descriptive-stats/ (5 sub-pages)
│   ├── basics.md → Central tendency, variability measures
│   ├── methods.md → Manual vs library implementations
│   ├── visualization.md → Data visualization techniques
│   └── applications.md → Quality control, business intelligence
└── probability/ (4 sub-pages)
    ├── basics.md → Fundamental concepts, sample spaces
    ├── distributions.md → Normal, binomial, Poisson distributions
    └── applications.md → Risk assessment, hypothesis testing
```

**Interactive Components Required**: StatisticsCalculator, ProbabilitySimulator

#### 4. Trigonometry (Angle and Wave Mathematics)
**Target Audience**: Game developers, signal processing engineers
**Learning Objectives**:
- Master trigonometric functions and identities
- Apply trigonometry to graphics and signal processing
- Understand periodic phenomena

**Required Content Structure**:
```
trigonometry/
└── unit-circle/ (4 sub-pages)
    ├── basics.md → Unit circle fundamentals, coordinate calculations
    ├── identities.md → Trigonometric identities, relationships
    └── applications.md → Computer graphics, signal processing
```

**Interactive Components Required**: UnitCircleExplorer

#### 5. Linear Algebra (Vector and Matrix Mathematics)
**Target Audience**: ML engineers, graphics programmers, data scientists
**Learning Objectives**:
- Understand vector and matrix operations
- Apply linear algebra to machine learning
- Implement matrix algorithms efficiently

**Required Content Structure**:
```
linear-algebra/
├── vectors/ (5 sub-pages)
│   ├── basics.md → Vector concepts, representation
│   ├── operations.md → Addition, dot product, projections
│   ├── advanced.md → Cross product, 3D operations
│   └── applications.md → Physics, graphics, ML
└── matrices/ (4 sub-pages)
    ├── basics.md → Matrix fundamentals, basic operations
    ├── operations.md → Advanced operations, decompositions
    └── applications.md → Computer graphics, PCA, engineering
```

**Interactive Components Required**: VectorOperations, MatrixTransformations

#### 6. Calculus (Change and Accumulation)
**Target Audience**: Advanced programmers, mathematical software developers
**Learning Objectives**:
- Understand limits and continuity
- Apply calculus concepts to optimization
- Implement numerical calculus methods

**Required Content Structure**:
```
calculus/
└── limits/ (5 sub-pages)
    ├── basics.md → Limit fundamentals, notation
    ├── methods.md → L'Hôpital's rule, advanced techniques
    ├── continuity.md → Continuity analysis, discontinuities
    └── applications.md → Optimization, numerical integration
```

**Interactive Components Required**: LimitsExplorer

### Content Quality Standards

**Mathematical Rigor**:
- All mathematical statements must be mathematically correct
- Formulas verified against authoritative sources (e.g., Wolfram MathWorld, NIST)
- Edge cases and domain restrictions clearly documented
- Precision requirements specified for all calculations

**Programming Integration**:
- Every mathematical concept connected to programming analogy
- Code examples must be executable and tested
- Performance characteristics (Big O) provided where applicable
- Real-world use cases with complete implementations

**Educational Effectiveness**:
- Progressive complexity within each concept
- Clear learning objectives for each section
- Multiple explanation approaches (visual, algebraic, computational)
- Practice exercises and extension activities

---

## Deployment and Infrastructure Requirements

### Build System Specifications

**Core Requirements**:
- **Static Site Generation**: All content must be buildable to static HTML/CSS/JS
- **Mathematical Rendering**: LaTeX support with MathJax or KaTeX
- **Code Syntax Highlighting**: Python, JavaScript with copy functionality
- **Component Bundling**: Optimized JavaScript bundles with tree-shaking
- **Asset Optimization**: Image compression, CSS/JS minification

**Build Performance Targets**:
- Initial build time: < 60 seconds
- Incremental rebuild: < 10 seconds
- Bundle size: Main chunk < 200KB, total < 1MB
- Page generation: < 100ms per page

**Build Configuration Example**:
```javascript
// Vite/Webpack configuration requirements
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'components': ['./components/StatisticsCalculator.vue', ...],
          'math-viz': ['./components/UnitCircleExplorer.vue', ...],
          'utils': ['./components/InteractiveSlider.vue', ...]
        }
      }
    },
    chunkSizeWarningLimit: 600,
    target: 'es2020'
  },
  optimizeDeps: {
    include: ['vue', 'mathematical-libraries']
  }
}
```

### Hosting and Deployment

**Static Hosting Requirements**:
- **CDN Distribution**: Global content delivery network
- **HTTPS**: SSL certificate for secure connection
- **Custom Domain**: Support for custom domain configuration
- **Caching Strategy**: Optimal cache headers for static assets
- **Performance**: <2s initial page load, <500ms subsequent pages

**CI/CD Pipeline Requirements**:
```yaml
# GitHub Actions workflow requirements
deploy:
  triggers:
    - push to main branch
    - pull request validation
  steps:
    - dependency installation (cache enabled)
    - automated testing (unit, integration, e2e)
    - build optimization
    - mathematical accuracy validation
    - accessibility testing
    - performance benchmarking
    - deployment to staging
    - production deployment (on approval)
```

**Environment Configuration**:
- **Development**: Local development server with hot reload
- **Staging**: Preview environment for testing
- **Production**: Optimized build with analytics

### Performance Requirements

**Core Web Vitals Targets**:
- **Largest Contentful Paint (LCP)**: < 2.5 seconds
- **First Input Delay (FID)**: < 100 milliseconds  
- **Cumulative Layout Shift (CLS)**: < 0.1
- **First Contentful Paint (FCP)**: < 1.8 seconds

**Component Performance**:
- **Interactive Component Load**: < 100ms
- **Canvas Rendering**: 60fps for animations, 30fps minimum
- **Mathematical Calculations**: < 50ms for real-time updates
- **Python Code Execution**: < 500ms for examples

**Optimization Strategies**:
- Code splitting by route and component type
- Lazy loading for non-critical components
- Image optimization with modern formats (WebP, AVIF)
- Mathematical computation caching
- Service worker for offline functionality

---

## Testing and Quality Assurance

### Automated Testing Requirements

**Unit Testing** (Required Coverage: >90%):
```javascript
// Mathematical accuracy tests
describe('StatisticsCalculator', () => {
  test('calculates mean correctly for various datasets', () => {
    expect(calculateMean([1, 2, 3, 4, 5])).toBe(3)
    expect(calculateMean([1.1, 2.2, 3.3])).toBeCloseTo(2.2, 3)
  })
  
  test('handles edge cases gracefully', () => {
    expect(() => calculateMean([])).toThrow('Empty dataset')
    expect(calculateMean([NaN, 1, 2])).toBeNaN()
  })
})
```

**Integration Testing**:
- Component interaction testing
- Python code execution validation
- Mathematical formula verification
- Cross-browser compatibility testing

**End-to-End Testing**:
```javascript
// Playwright/Cypress test examples
test('QuadraticExplorer workflow', async ({ page }) => {
  await page.goto('/algebra/quadratics/basics')
  
  // Test interactive component
  await page.locator('[data-testid="coefficient-a"]').fill('2')
  await expect(page.locator('[data-testid="vertex-display"]')).toContainText('(0, 0)')
  
  // Test form conversion
  await page.selectOption('[data-testid="form-select"]', 'vertex')
  await expect(page.locator('[data-testid="vertex-form"]')).toBeVisible()
})
```

**Mathematical Accuracy Testing**:
- Automated verification against known mathematical values
- Precision testing for floating-point calculations
- Domain validation testing
- Edge case scenario testing

**Accessibility Testing**:
- WCAG 2.1 AA compliance validation
- Screen reader compatibility testing
- Keyboard navigation verification
- Color contrast ratio validation

**Performance Testing**:
- Lighthouse automated testing
- Bundle size monitoring
- Component render time measurement
- Memory usage profiling

### Manual Testing Requirements

**Content Quality Assurance**:
- Mathematical accuracy review by domain experts
- Code example execution verification
- Educational effectiveness evaluation
- Cross-platform visual consistency check

**User Experience Testing**:
- Navigation flow validation
- Interactive component usability
- Mobile responsiveness verification
- Loading state and error handling

---

## Migration Implementation Guide

### Technology Migration Paths

#### Vue 2 to Vue 3 Migration
**Effort Level**: Low-Medium (recommended path)

**Key Changes Required**:
```javascript
// Vue 2 → Vue 3 component structure
// OLD (Vue 2)
export default {
  data() {
    return { value: 0 }
  },
  methods: {
    updateValue() { this.value++ }
  }
}

// NEW (Vue 3 Composition API)
import { ref } from 'vue'
export default {
  setup() {
    const value = ref(0)
    const updateValue = () => value.value++
    return { value, updateValue }
  }
}
```

**Migration Steps**:
1. Update Vue dependencies to Vue 3
2. Convert components to Composition API
3. Update build configuration for Vite
4. Test mathematical calculations for precision
5. Validate interactive component functionality

#### Vue to React Migration
**Effort Level**: High

**Component Conversion Example**:
```jsx
// Vue component → React component
// StatisticsCalculator.vue → StatisticsCalculator.tsx

import React, { useState, useEffect, useRef } from 'react'

interface StatisticsCalculatorProps {
  initialData?: number[]
  precision?: number
}

export const StatisticsCalculator: React.FC<StatisticsCalculatorProps> = ({
  initialData = [],
  precision = 3
}) => {
  const [dataset, setDataset] = useState<number[]>(initialData)
  const [statistics, setStatistics] = useState<StatisticsResult>()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    calculateStatistics()
    updateVisualization()
  }, [dataset])
  
  // Implementation details...
}
```

**Required Libraries for React Migration**:
- **Mathematical Calculations**: ml-matrix, simple-statistics
- **Canvas Rendering**: Konva.js or custom Canvas API
- **LaTeX Rendering**: react-katex or react-mathjax
- **Code Highlighting**: react-syntax-highlighter

#### Framework-Agnostic Components
**Approach**: Create framework-agnostic mathematical engine

```javascript
// Mathematical computation layer (framework-independent)
export class MathematicalEngine {
  static calculateQuadraticRoots(a: number, b: number, c: number): QuadraticResult {
    const discriminant = b * b - 4 * a * c
    
    if (discriminant < 0) {
      return { roots: [], hasRealRoots: false, discriminant }
    } else if (discriminant === 0) {
      return { 
        roots: [-b / (2 * a)], 
        hasRealRoots: true, 
        discriminant 
      }
    } else {
      const sqrt_discriminant = Math.sqrt(discriminant)
      const roots = [
        (-b + sqrt_discriminant) / (2 * a),
        (-b - sqrt_discriminant) / (2 * a)
      ]
      return { roots, hasRealRoots: true, discriminant }
    }
  }
}

// Framework adapters
// Vue 3 adapter
export const useQuadraticCalculator = () => {
  const engine = new MathematicalEngine()
  // Vue-specific reactive implementation
}

// React adapter  
export const useQuadraticCalculator = () => {
  const engine = new MathematicalEngine()
  // React hooks implementation
}
```

### Migration Checklist

**Pre-Migration Phase**:
- [ ] Audit current component functionality
- [ ] Document mathematical algorithms and formulas
- [ ] Create comprehensive test suite
- [ ] Establish performance baselines
- [ ] Plan content migration strategy

**Migration Phase**:
- [ ] Set up new framework/build system
- [ ] Migrate mathematical computation engine
- [ ] Convert interactive components one by one
- [ ] Implement canvas visualization layer
- [ ] Set up LaTeX rendering system
- [ ] Migrate content with proper formatting

**Post-Migration Phase**:
- [ ] Comprehensive testing (unit, integration, e2e)
- [ ] Performance optimization and validation
- [ ] Accessibility compliance verification
- [ ] Cross-browser compatibility testing
- [ ] User acceptance testing
- [ ] Documentation updates

**Risk Mitigation**:
- Maintain parallel development environment
- Implement feature flags for component rollout
- Create rollback procedures
- Monitor performance metrics during migration
- Maintain mathematical accuracy throughout process

---

## Success Criteria and Validation

### Functional Requirements Validation

**Mathematical Accuracy** (Critical):
- All calculations match verified mathematical sources
- Precision handling consistent across components
- Edge cases properly handled and documented
- Domain restrictions enforced programmatically

**Interactive Component Functionality** (Critical):
- Real-time parameter updates (< 50ms response)
- Smooth animations and transitions (≥ 30fps)
- Intuitive user interactions
- Proper error handling and user feedback

**Educational Effectiveness** (High Priority):
- Clear mathematical concept explanations
- Effective programming analogies
- Progressive learning complexity
- Practical real-world applications

**Technical Performance** (High Priority):
- Page load times meet Core Web Vitals standards
- Component bundle sizes optimized
- Mathematical computations performant
- Cross-platform compatibility verified

### Acceptance Testing Criteria

**User Experience**:
- New users can complete basic mathematical exploration within 5 minutes
- Navigation between related concepts is intuitive
- Interactive components enhance rather than distract from learning
- Content is accessible across devices and assistive technologies

**Content Quality**:
- Mathematical concepts accurately represented
- Code examples execute successfully
- Real-world applications are relevant and current
- Learning objectives are clearly achievable

**Technical Excellence**:
- Zero critical bugs in mathematical calculations
- Performance meets or exceeds specified benchmarks
- Accessibility compliance verified by automated and manual testing
- Deployment process is reliable and repeatable

This comprehensive specification provides the foundation for rebuilding Snake Math with any modern framework while maintaining its educational effectiveness and technical excellence.

---

## Appendix: Reference Implementation Details

### Current File Structure Overview
```
/
├── CLAUDE.md                           # AI assistant instructions and project guidance  
├── README.md                           # Project documentation and setup instructions
├── LICENSE                             # MIT License
├── package.json                        # Node.js dependencies and scripts
├── concept_page_template.md           # Template for creating new concept pages
├── project_blueprint.md              # High-level project architecture and roadmap
├── archive/                           # Archived original files from restructuring
├── docs/                              # VitePress source directory (main content)
│   ├── .vitepress/
│   │   ├── config.js                  # VitePress configuration with manual chunking
│   │   ├── theme/
│   │   │   ├── index.js               # Component registration and theme setup
│   │   │   ├── components/            # 19 interactive Vue components
│   │   │   └── styles/                # Centralized CSS architecture
│   │   └── dist/                      # Build output (auto-generated)
│   ├── index.md                       # Homepage
│   ├── algebra/ (22 pages)            # Algebraic concepts with 5 concept areas
│   ├── statistics/ (9 pages)          # Statistical analysis with 2 concept areas  
│   ├── linear-algebra/ (8 pages)      # Vector/matrix mathematics with 2 areas
│   ├── trigonometry/ (4 pages)        # Trigonometric functions with 1 area
│   ├── calculus/ (4 pages)            # Limits and continuity with 1 area
│   └── basics/ (6 pages)              # Mathematical foundations
├── instructions/                      # Project documentation and AI instructions
└── node_modules/                      # Node.js dependencies (auto-generated)
```

### Mathematical Content Statistics
- **Total Content Pages**: 44 focused sub-pages + 11 index pages = 55 pages
- **Interactive Components**: 19 Vue components with mathematical functionality
- **Mathematical Domains**: 6 major areas with progressive complexity
- **Lines of Content**: ~12,000+ lines restructured from 11 monolithic files
- **Learning Progression**: Beginner → Intermediate → Advanced across all domains

### Current Technology Versions (Reference - Requires Migration)
- **VitePress**: 1.x series ⚠️ (struggling with scale)
- **Vue.js**: 2.x (Options API) ⚠️ (performance limitations)
- **PyScript**: 2025.5.1 ✅ (works well, keep)
- **MathJax**: Latest via CDN ✅ (works well, keep)
- **Node.js**: LTS version compatibility ✅ (upgrade to latest)
- **GitHub Actions**: CI/CD for automated deployment ✅ (works well, keep)

### **Recommended Target Stack (Vue 3 + Vite)**
- **Vue.js**: 3.x with Composition API (modern reactive system)
- **Vite**: Latest (lightning-fast dev server and builds)
- **Vue Router**: 4.x (for mathematical content navigation)
- **TypeScript**: Optional but recommended for mathematical accuracy
- **Vitest**: For component and mathematical function testing
- **Canvas API**: Native canvas rendering for visualizations
- **Web Workers**: For heavy mathematical computations (optional)

### Implementation Priority for Migration to Vue 3 + Vite
1. **Critical Path Components** (implement first):
   - CodeFold (utility component used throughout)
   - StatisticsCalculator (most complex mathematical component)
   - QuadraticExplorer (representative interactive visualization)

2. **Mathematical Engine Components**:
   - VectorOperations, MatrixTransformations (linear algebra)
   - UnitCircleExplorer (trigonometry)
   - LimitsExplorer (calculus)

3. **Specialized Components**:
   - SummationDemo, ProductNotationVisualizer (notation)
   - ProbabilitySimulator (statistics)
   - ExponentialCalculator, LinearSystemSolver (algebra)

4. **Content-Specific Components**:
   - FunctionPlotter, InteractiveSlider (general utilities)
   - Remaining domain-specific components

### Migration Timeline (Vue 3 + Vite - 6 weeks total)
- **Week 1**: Project setup, mathematical engine, basic routing
- **Week 2-3**: Convert 19 interactive components from Vue 2 to Vue 3
- **Week 4**: Content loading system and markdown processing
- **Week 5**: Navigation, optimization, bundle splitting, testing
- **Week 6**: Deployment setup, performance validation, documentation

### Expected Performance Improvements
**Current (VitePress + Vue 2)**:
- Build time: 60+ seconds
- Bundle size: 2MB+ total
- Component load: 200-500ms
- Math calculations: 100-200ms

**Target (Vue 3 + Vite)**:
- Build time: 10-15 seconds (4-6x faster)
- Bundle size: ~800KB chunked (60% smaller)
- Component load: 50-100ms (2-5x faster)
- Math calculations: 10-50ms (2-4x faster)

This handoff specification ensures complete project rebuilding capability while maintaining the educational quality and mathematical accuracy that defines Snake Math's value proposition.