# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Snake Math is an interactive mathematical learning platform that bridges mathematical concepts with Python programming. It's a zero-maintenance, purely in-browser Python math playground built with VitePress and PyScript, hosted on GitHub Pages.

## Development Commands

```bash
# Start local development server with hot reload
npm run docs:dev

# Build for production (outputs to docs/.vitepress/dist/)
npm run docs:build

# Preview production build locally
npm run docs:serve
```

## Architecture

### Core Stack
- **VitePress** - Static site generator and documentation framework
- **Vue.js** - Component framework for interactive elements
- **PyScript** - In-browser Python execution engine
- **MathJax** - Mathematical notation rendering

### Key Directories
- `docs/` - VitePress source directory
- `docs/.vitepress/config.js` - Main VitePress configuration
- `docs/.vitepress/theme/components/` - Vue components for interactive features
- `docs/.vitepress/dist/` - Build output (auto-generated)

### Interactive Components
- **ExponentialCalculator.vue** - Interactive exponential and logarithmic function explorer
- **FunctionPlotter.vue** - General-purpose function plotting component with customizable parameters
- **InteractiveSlider.vue** - Reusable slider component for real-time parameter manipulation
- **LimitsExplorer.vue** - Interactive limits calculation and visualization
- **LinearSystemSolver.vue** - Matrix-based linear equation system solver
- **MathDisplay.vue** - LaTeX math expression renderer integrated with calculations
- **MatrixTransformations.vue** - Visual matrix operations and transformations
- **ProbabilitySimulator.vue** - Statistical probability distribution simulator
- **ProductDemo.vue** - Product notation (Π) interactive demonstrations
- **QuadraticExplorer.vue** - Quadratic function graphing and root finding
- **StatisticsCalculator.vue** - Descriptive statistics calculation tool
- **SummationDemo.vue** - Summation notation (Σ) interactive demonstrations  
- **UnitCircleExplorer.vue** - Trigonometric unit circle visualization
- **VariablesDemo.vue** - Variable manipulation and expression evaluation
- **VectorOperations.vue** - Vector arithmetic and geometric operations

## Content Structure

Content follows a progressive learning path:
- `docs/basics/` - Variables, expressions, functions, number theory, order of operations
- `docs/algebra/` - Summation notation, product notation, linear equations, quadratics, exponentials & logarithms
- `docs/statistics/` - Descriptive statistics, probability distributions
- `docs/trigonometry/` - Unit circle, trigonometric functions
- `docs/linear-algebra/` - Vectors, matrices, transformations
- `docs/calculus/` - Limits, continuity, derivatives (planned)

Each content file combines:
1. Mathematical theory with LaTeX notation. 
    - This should be in-depth, informal and relate to programming, e.g. "Summation is just a for-loop"
2. Executable Python code examples (only if applicable and it adds understanding)
3. Interactive components for hands-on learning
4. Python Code examples with explanation

## Development Patterns

### Component Integration
- Interactive components are registered globally in `docs/.vitepress/theme/index.js`
- PyScript is loaded via CDN in VitePress head configuration
- Math rendering uses both built-in VitePress math support and MathJax

### PyScript Integration
- PyScript 2025.5.1 is configured for browser Python execution
- Components handle PyScript initialization and error states
- Terminal output is styled with custom CSS

### Content Format
- Markdown files with frontmatter for VitePress
- LaTeX math expressions using `$...$` (inline) and `$$...$$` (block)
- Vue components embedded directly in markdown
- Progressive complexity from basic concepts to advanced topics

## Deployment

- Automatic deployment via GitHub Actions on push to `main` branch
- Deploys to GitHub Pages at `/snake-math/` base path
- Build artifacts are generated in `docs/.vitepress/dist/`

## Claude Code Permissions

The following permissions have been configured for Claude Code:
- **Bash** - Full filesystem access (ls commands and all operations)
- **WebFetch** - Web content retrieval for documentation and research

## Project Blueprint

Refer to `project_blueprint.md` for comprehensive project documentation including:
- Architecture decisions and rationale
- Complete development roadmap (MVP through Phase 5)
- Content format specifications
- Future feature planning