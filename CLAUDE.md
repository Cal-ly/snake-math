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

Each content file follows the `concept_page_template.md` structure:
1. **Mathematical Theory** - In-depth, informal explanations relating to programming (e.g., "Summation is just a for-loop")
2. **Interactive Components** - Vue-based exploration tools with comment block descriptions
3. **Code Examples** - Python snippets wrapped in `<CodeFold>` components
4. **Efficiency Analysis** - Big O complexity comparisons between different methods
5. **Real-world Applications** - Practical programming scenarios and use cases
6. **Progressive Learning** - From basic understanding to advanced patterns and applications

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
Content pages follow the standardized template in `concept_page_template.md`:

**Required Structure:**
- YAML frontmatter with title, description, tags, difficulty, prerequisites, related_concepts, applications
- Progressive sections: Understanding → Interactive Exploration → Techniques & Efficiency → Common Patterns → Real-world Applications → Practice → Key Takeaways
- LaTeX math expressions using `$...$` (inline) and `$$...$$` (block)
- All code blocks wrapped in `<CodeFold>` components
- Vue components with descriptive comment blocks explaining functionality
- Light, witty tone with minimal emojis
- Complexity analysis (Big O notation) for algorithms and methods
- Real-world Python examples for practical applications

## Deployment

- Automatic deployment via GitHub Actions on push to `main` branch
- Deploys to GitHub Pages at `/snake-math/` base path
- Build artifacts are generated in `docs/.vitepress/dist/`

## Claude Code Permissions

The following permissions have been configured for Claude Code:
- **Bash** - Full filesystem access (ls commands and all operations)
- **WebFetch** - Web content retrieval for documentation and research

## Recent Major Achievements (June 2025)

### ✅ COMPLETED: Major Content Restructuring Initiative
**Date Completed**: June 23, 2025

**Major Achievement**: Successfully refactored the entire Snake Math project by splitting 11 large monolithic concept pages (500-1500+ lines each) into 44 focused, modular sub-pages while preserving 100% of interactive components and mathematical content.

**Quantitative Results**:
- **Files Split**: 11 large concept pages → 44 focused topic pages  
- **Total Lines Refactored**: ~12,000+ lines of content
- **Interactive Components**: 100% preserved and functional
- **Build Status**: ✅ VitePress builds successfully with no dead link errors

**Enhanced Navigation & Architecture**:
- **6 New Index Pages**: Created comprehensive overview pages for all major sections
- **Fixed Dead Links**: Resolved all 14 VitePress build errors for deployment readiness
- **Modular Structure**: Each concept area now has focused sub-topics (basics → methods → applications)
- **Archive Management**: All original files safely preserved with date stamps in `/archive/`

**Performance Optimizations**:
- Manual chunking strategy grouping components by functionality
- Modular content structure reduces page load times and improves SEO
- Responsive canvas elements with proper scaling

## Project Blueprint

Refer to `instructions/project_blueprint.md` for comprehensive project documentation including:
- Architecture decisions and rationale
- Complete development roadmap and recent refactoring achievements
- Content format specifications and template compliance
- Future feature planning

Additional documentation in `instructions/` folder:
- `instructions/task-log.md` - Detailed progress log of the restructuring initiative
- `instructions/concept_page_template.md` - Standardized template for all content pages
- `instructions/claude-inst.md` - AI assistant instructions for project work