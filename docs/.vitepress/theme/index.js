// docs/.vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import PyScriptDemo from './components/PyScriptDemo.vue'
import InteractiveSlider from './components/InteractiveSlider.vue'
import MathDisplay from './components/MathDisplay.vue'
import SummationDemo from './components/SummationDemo.vue'
import ProductDemo from './components/ProductDemo.vue'
import VariablesDemo from './components/VariablesDemo.vue'

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
  }
}