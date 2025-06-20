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
- **PyScriptDemo.vue** - Main component for Python code execution with interactive calculators
- **InteractiveSlider.vue** - Reusable slider component for real-time parameter manipulation
- **MathDisplay.vue** - LaTeX math expression renderer integrated with calculations
- **SummationDemo.vue** & **ProductDemo.vue** - Subject-specific interactive demonstrations

## Content Structure

Content follows a progressive learning path:
- `docs/basics/` - Variables, expressions, fundamental concepts
- `docs/algebra/` - Summation notation, product notation, algebraic concepts
- `docs/calculus/` - Planned advanced topics

Each content file combines:
1. Mathematical theory with LaTeX notation
2. Executable Python code examples
3. Interactive components for hands-on learning

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

## Project Blueprint

Refer to `project_blueprint.md` for comprehensive project documentation including:
- Architecture decisions and rationale
- Complete development roadmap (MVP through Phase 5)
- Content format specifications
- Future feature planning