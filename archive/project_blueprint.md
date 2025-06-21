# Project Overview

**Introduction and Intent**:
 This project aims to make math accessible for programmers by serving as a concise look-up reference rather than a full course. Each section aims to bridge a mathematical concept with a programming analogue—for example, showing for example Σ notation as a simple for‑loop or that mapping number types is analogous to programming data types. The tone is intentionally informal and playful: math doesn’t have to be dry. 
 
 Math can often be easier to grasp when visualized, so there is a focus on making interactive and exploratory componets where it helps to support understanding of the mathematical concept in a dynamic manner - these are often made with JavaScript using Vue components, so they can run directly in the browser.
 
 All examples are copy‑and‑paste–ready Python code snippets, with simple illustrations, comments and real‑world applications where appropriate. Where it makes sense, as there often is a performance-hit, PyScript is integrated as a playground for math concepts, that couldn't be illustrated in another way.

 It is made with **VitePress**, **Vue** and **PyScript** with the goal to eliminate server infrastructure and runtime environments.

**Why VitePress?**
 Quick start-up and updates in real-time, and is **markdown** based with the addition of **Vue components**, it is the natural succesor to VuePress. Have in mind, that it is mainly for docs, so thing like complex navigation or custom themes take a bit more effort. I have write most of the interactive, eplorative parts as Vue components and only leverage **PyScript** where it was neccesary.

**Why PyScript?**:
 The intent on leveraging **PyScript**, is that users can run Python code directly in the browser. This approach bridges math notation and executable code in real-time, allowing learners to tweak variables and immediately visualize results without installing Python or waiting for remote computation. It lowers the barrier to entry and fosters an interactive, exploratory learning experience.

**Why GitHub Pages?**: 
 Free, reliable, and globally distributed static hosting. Integration with GitHub Actions automates deployments on every push, giving **continuous integration** and **continuous delivery** with minimal configuration.

**Architecture**:
 Is it set up with the standard VitePress setup for file and folder structure. Most of the interactive components - where the goal is just to illustrate the math - are made with JavaScript. However,where it makes sense to illustrate it in Python code, they are made with PyScript. This division is to ensure both simplicity and a fast load-time, where the point is to illustrate and the user wouldn't know the difference anyway.

## 1. Repository Structure

```
/<repo-root>
│
├─ docs/                  # VitePress source directory
│   ├─ .vitepress/
│   │   ├─ dist/          # Build output (GitHub Pages source)
│   │   ├─ cache/
│   │   ├─ theme/         # Custom theme
│   │   │   ├─ components/
│   │   │   │   ├─ ExponentialCalculator.vue
│   │   │   │   ├─ FunctionPlotter.vue
│   │   │   │   ├─ InteractiveSlider.vue
│   │   │   │   ├─ LimitsExplorer.vue
│   │   │   │   ├─ LinearSystemSolver.vue
│   │   │   │   ├─ MathDisplay.vue
│   │   │   │   ├─ MatrixTransformations.vue
│   │   │   │   ├─ ProbabilitySimulator.vue
│   │   │   │   ├─ ProductDemo.vue
│   │   │   │   ├─ QuadraticExplorer.vue
│   │   │   │   ├─ StatisticsCalculator.vue
│   │   │   │   ├─ SummationDemo.vue
│   │   │   │   ├─ UnitCircleExplorer.vue
│   │   │   │   ├─ VariablesDemo.vue
│   │   │   │   └─ VectorOperations.vue
│   │   │   │
│   │   │   └─ index.js   # Component registration
│   │   │
│   │   └─ config.js      # VitePress configuration
│   │
│   ├─ basics/
│   │   ├─ variables-expressions.md
│   │   ├─ functions.md
│   │   ├─ number-theory.md
│   │   └─ order-of-operations.md
│   │
│   ├─ algebra/
│   │   ├─ summation-notation.md
│   │   ├─ product-notation.md
│   │   ├─ linear-equations.md
│   │   ├─ quadratics.md
│   │   └─ exponentials-logarithms.md
│   │
│   ├─ statistics/
│   │   ├─ descriptive-stats.md
│   │   └─ probability.md
│   │
│   ├─ trigonometry/
│   │   └─ unit-circle.md
│   │
│   ├─ linear-algebra/
│   │   ├─ vectors.md
│   │   └─ matrices.md
│   │
│   ├─ calculus/
│   │   └─ limits.md
│   │
│   ├─ public/            # Static assets (images, etc.)
│   │
│   └─ index.md           # Homepage with PyScript setup
│
├─ .github/
│   └─ workflows/
│       └─ deploy.yml     # GitHub Actions: build & deploy
│
├─ CLAUDE.md              # Claude Code instructions
├─ project_blueprint.md   # This file
├─ package.json           # VitePress + dependencies
├─ package-lock.json
└─ .gitignore             # Ignore node_modules, dist, etc.
```


## 2. Content Format

- **Markdown** with embedded LaTeX for formulas (using VitePress built-in math support).
- **Accordion panels** via native HTML:
  ```html
  <details>
    <summary>Theory: Σ notation</summary>
    <p>…explanation…</p>
  </details>
  ```
- **Interactive code** in .vue `<Components />` or where only Python can solve the problem, `<py-script>` blocks

- **Sliders/inputs** with real-time interaction:
  ```html
  <label for="coefficient">Coefficient a: <span id="a-value">1</span></label>
  <input id="a" type="range" min="-5" max="5" value="1" 
         oninput="document.getElementById('a-value').textContent = this.value"/>
  
  <py-script>
  from js import document
  
  def update_calculation():
      a = float(document.getElementById("a").value)
      result = a * 2 + 1
      print(f"When a = {a}, result = {result}")
  
  # Auto-update on slider change
  document.getElementById("a").addEventListener("input", lambda e: update_calculation())
  update_calculation()  # Initial calculation
  </py-script>
  ```
- **Vue components** for the most basic or complex interactions (when PyScript isn't suitable):
  ```vue
  <MathDisplay :equation="'x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}'" />
  ```

## 3. Deployment Steps

1. **VitePress Setup**
   - Install VitePress: `npm install -D vitepress`
   - Configure `docs/.vitepress/config.js` with site metadata and navigation
   - Set build output to `docs/.vitepress/dist/`

2. **GitHub Actions** (`.github/workflows/deploy.yml`)
   - Checkout repository
   - Setup Node.js and install dependencies
   - Run `npm run docs:build` to generate static files
   - Deploy `docs/.vitepress/dist/` to `gh-pages` branch

3. **Enable GitHub Pages**
   - Settings → Pages → Source: `gh-pages` branch, `/ (root)` folder
   - Custom domain setup (optional)

4. **Development Workflow**
   - Local development: `npm run docs:dev`
   - Content authoring in `/docs` → `git push` → CI builds → live site
   - Hot reload for instant preview during development

## 4. Ideas Section

Concrete examples showing how we distill math concepts into Python code. Each entry pairs the math idea with a stripped-down snippet.

### 4.1 Σ-Notation ↔ For-Loop

$$
\sum_{i=1}^{n} i = 1 + 2 + \dots + n
$$

```html
<py-script>
n = 10
# direct Python loop
total = 0
for i in range(1, n+1):
    total += i
print(f"Sum 1..{n} =", total)
</py-script>
```


**Math**:

$$
x = -\frac{b}{a}
$$

```html
<py-script>
a = float(Element("a").value)  # from an <input id="a">
b = float(Element("b").value)
x = -b/a
print(f"x = {x:.2f}")
</py-script>
```


### 4.3 Limits via Numerical Approximation

**Math**:

$$
\lim_{x \to c} f(x)
$$

```html
<py-script>
import numpy as np
def f(x): return (x**2 - 1)/(x-1)
c = 1.0
eps = 1e-3
vals = [f(c + delta) for delta in (eps, -eps)]
print("Approx limit:", np.mean(vals))
</py-script>
```


### 4.4 Quadratic Plot & Roots

**Math**: Plot \(y = ax^2 + bx + c\) and show roots.

```html
<input id="a" type="range" min="-5" max="5" value="1"/>
<input id="b" type="range" min="-5" max="5" value="0"/>
<input id="c" type="range" min="-5" max="5" value="0"/>

<py-script>
from js import document
import matplotlib.pyplot as plt
import cmath

def draw(*_):
    a = float(document.getElementById("a").value)
    b = float(document.getElementById("b").value)
    c = float(document.getElementById("c").value)
    # compute roots
    disc = b**2 - 4*a*c
    roots = [(-b + cmath.sqrt(disc))/(2*a), (-b - cmath.sqrt(disc))/(2*a)]
    # plot
    xs = [i/10 for i in range(-50,51)]
    ys = [a*x**2 + b*x + c for x in xs]
    plt.clf(); plt.plot(xs, ys); plt.scatter([r.real for r in roots], [0,0]); plt.show()
    print("Roots:", roots)

for id in ("a","b","c"):
    document.getElementById(id).addEventListener("input", draw)
draw()
</py-script>
```


### 4.5 Matrix Multiplication (Linear Algebra)

**Math**:

$$
C = A \times B,\quad C_{ij} = \sum_k A_{ik}B_{kj}
$$

```html
<py-script>
import numpy as np
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = A.dot(B)
print("C =\n", C)
</py-script>
```


### 4.6 Differential Equation: Euler’s Method

**Math**:

$$
\frac{dy}{dt} = f(t,y),\quad y_{n+1} = y_n + h\,f(t_n,y_n)
$$

```html
<py-script>
import matplotlib.pyplot as plt

def f(t,y): return -y
t, y, h = 0, 1, 0.1
ts, ys = [t], [y]
for _ in range(50):
    y += h * f(t,y)
    t += h
    ts.append(t); ys.append(y)
plt.plot(ts, ys); plt.show()
</py-script>
```


## 5. To Do List

### 🚀 Phase 1: Foundation
- [x] **Project Setup**
  - [x] Initialize npm project with VitePress
  - [x] Create basic folder structure
  - [x] Set up GitHub repository
  - [x] Configure GitHub Actions for deployment

- [ ] **Core Configuration**
  - [x] Configure `docs/.vitepress/config.js` with navigation structure
  - [x] Set up PyScript integration in default layout
  - [x] Enable VitePress math support (markdown-it-mathjax3)
  - [x] Create custom CSS for PyScript output styling
  - [ ] Add favicon and basic branding

### 📝 Phase 2: Content Creation
- [x] **Homepage (`docs/index.md`)**
  - [x] Create hero section with project tagline
  - [x] Add "Try it now" interactive demo
  - [x] Include navigation to main sections
  - [x] Add getting started guide

- [x] **Basics Section**
  - [x] `variables-expressions.md` - Python variables ↔ algebra
  - [x] `functions.md` - Function definition and plotting
  - [x] `number-theory.md` - Number theory fundamentals
  - [x] `order-of-operations.md` - Mathematical operation precedence
  - [x] Add interactive number line component
  - [x] Include "Python vs Math notation" comparison tables

- [x] **Algebra Section**
  - [x] `summation-notation.md` - Σ notation with for-loops
  - [x] `product-notation.md` - Π notation with for-loops
  - [x] `linear-equations.md` - Solving ax + b = 0 interactively
  - [x] `quadratics.md` - Real-time quadratic graphing
  - [x] `exponentials-logarithms.md` - Exponential and logarithmic functions
  - [x] Add equation solver components

- [x] **Statistics Section**
  - [x] `descriptive-stats.md` - Mean, median, mode, standard deviation
  - [x] `probability.md` - Probability distributions and simulations

- [x] **Trigonometry Section**
  - [x] `unit-circle.md` - Unit circle and trigonometric functions

- [x] **Linear Algebra Section**
  - [x] `vectors.md` - Vector operations and visualization
  - [x] `matrices.md` - Matrix operations and transformations

- [x] **Calculus Section**
  - [x] `limits.md` - Limits and continuity concepts

### 🎨 Phase 3: Enhancement
- [x] **Interactive Components**
  - [x] `MathDisplay.vue` - LaTeX renderer with copy button
  - [x] `InteractiveSlider.vue` - Labeled range input with live updates
  - [x] `FunctionPlotter.vue` - General-purpose function plotting
  - [x] `ExponentialCalculator.vue` - Exponential function explorer
  - [x] `QuadraticExplorer.vue` - Quadratic function visualization
  - [x] `LinearSystemSolver.vue` - Linear equation system solver
  - [x] `StatisticsCalculator.vue` - Statistical calculations
  - [x] `ProbabilitySimulator.vue` - Probability distribution simulator
  - [x] `UnitCircleExplorer.vue` - Trigonometric unit circle
  - [x] `VectorOperations.vue` - Vector arithmetic visualization
  - [x] `MatrixTransformations.vue` - Matrix operation visualization
  - [x] `LimitsExplorer.vue` - Limits calculation and visualization
  - [x] `SummationDemo.vue` - Summation notation demonstrations
  - [x] `ProductDemo.vue` - Product notation demonstrations
  - [x] `VariablesDemo.vue` - Variable manipulation demonstrations

- [ ] **User Experience**
  - [ ] Configure VitePress search (local search)
  - [ ] Implement custom dark theme for code blocks
  - [ ] Add mobile-responsive navigation
  - [ ] Create loading spinners for PyScript initialization
  - [ ] Add "Edit on GitHub" links to pages

### 🚀 Phase 4: Advanced Features
- [ ] **Content Expansion**
  - [ ] `calculus/limits.md` - Numerical limit approximation
  - [ ] `calculus/derivatives.md` - Symbolic and numerical derivatives
  - [ ] `statistics/` section with probability distributions
  - [ ] `linear-algebra/` section with matrix operations

- [ ] **Performance & Polish**
  - [ ] Lazy-load PyScript on first interaction
  - [ ] Add service worker for offline functionality
  - [ ] Implement client-side analytics (privacy-focused)
  - [ ] Create contribution guidelines and templates
  - [ ] Add automated testing for code examples

### 🔧 Phase 5: Maintenance
- [ ] **Documentation**
  - [ ] `CONTRIBUTING.md` with content guidelines
  - [ ] Component documentation in `/docs/.vitepress/theme/`
  - [ ] Deployment troubleshooting guide
  - [ ] Performance optimization checklist

- [ ] **Quality Assurance**
  - [ ] GitHub Actions for link checking
  - [ ] Cross-browser testing automation
  - [ ] Mathematical accuracy review process
  - [ ] Lighthouse performance auditing
  - [ ] Accessibility compliance (WCAG 2.1)

### 🎯 MVP Milestone
**Target: First deployable version** ✅ **COMPLETED**
- [x] Homepage with working PyScript demo
- [x] Multiple complete sections with interactive components
- [x] GitHub Pages deployment ready
- [x] Full navigation and styling
- [x] Mobile responsiveness
- [x] Comprehensive component library (15+ interactive components)
- [x] Complete content coverage across 6 mathematical domains

## Current Status & Next Steps

### ✅ **Project Status: Production Ready**
The Snake Math project has successfully achieved its MVP goals and is currently a fully functional mathematical learning platform with:

**Core Infrastructure:**
- ✅ VitePress-based static site generator
- ✅ PyScript 2025.5.1 integration for in-browser Python execution
- ✅ MathJax/LaTeX rendering for mathematical notation
- ✅ GitHub Pages deployment pipeline
- ✅ Responsive design and mobile compatibility

**Content Completion:**
- ✅ 6 major mathematical domains covered
- ✅ 15+ interactive Vue.js components
- ✅ Progressive learning path from basics to advanced topics
- ✅ Real-time code execution and visualization

**Component Architecture:**
- ✅ Modular Vue component system
- ✅ Global component registration
- ✅ Consistent styling and UX patterns
- ✅ Error handling and graceful degradation

### 🚀 **Immediate Action Items**
- [ ] Add comprehensive input validation to prevent calculation errors
- [ ] Implement loading states for PyScript initialization
- [ ] Add error boundaries for component failures
- [ ] Create automated testing suite for mathematical accuracy
- [ ] Enhance accessibility features (ARIA labels, keyboard navigation)

### 🔮 **Future Enhancement Opportunities**
- [ ] Advanced calculus topics (derivatives, integrals)
- [ ] 3D plotting capabilities
- [ ] Interactive theorem provers
- [ ] Collaborative features (shared notebooks)
- [ ] Performance optimization (lazy loading, caching)
- [ ] Multi-language support for international users

---