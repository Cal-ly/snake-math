// docs/.vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import PyScriptDemo from './components/PyScriptDemo.vue'
import InteractiveSlider from './components/InteractiveSlider.vue'
import MathDisplay from './components/MathDisplay.vue'
import SummationDemo from './components/SummationDemo.vue'
import ProductDemo from './components/ProductDemo.vue'
import VariablesDemo from './components/VariablesDemo.vue'
import FunctionPlotter from './components/FunctionPlotter.vue'
import LinearSystemSolver from './components/LinearSystemSolver.vue'
import QuadraticExplorer from './components/QuadraticExplorer.vue'
import ExponentialCalculator from './components/ExponentialCalculator.vue'
import StatisticsCalculator from './components/StatisticsCalculator.vue'
import ProbabilitySimulator from './components/ProbabilitySimulator.vue'
import UnitCircleExplorer from './components/UnitCircleExplorer.vue'
import VectorOperations from './components/VectorOperations.vue'
import MatrixTransformations from './components/MatrixTransformations.vue'
import LimitsExplorer from './components/LimitsExplorer.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register all components globally
    app.component('PyScriptDemo', PyScriptDemo)
    app.component('InteractiveSlider', InteractiveSlider)
    app.component('MathDisplay', MathDisplay)
    app.component('SummationDemo', SummationDemo)
    app.component('ProductDemo', ProductDemo)
    app.component('VariablesDemo', VariablesDemo)
    app.component('FunctionPlotter', FunctionPlotter)
    app.component('LinearSystemSolver', LinearSystemSolver)
    app.component('QuadraticExplorer', QuadraticExplorer)
    app.component('ExponentialCalculator', ExponentialCalculator)
    app.component('StatisticsCalculator', StatisticsCalculator)
    app.component('ProbabilitySimulator', ProbabilitySimulator)
    app.component('UnitCircleExplorer', UnitCircleExplorer)
    app.component('VectorOperations', VectorOperations)
    app.component('MatrixTransformations', MatrixTransformations)
    app.component('LimitsExplorer', LimitsExplorer)
  }
}