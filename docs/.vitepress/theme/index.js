// docs/.vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import PyScriptDemo from './components/PyScriptDemo.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register the PyScript component globally
    app.component('PyScriptDemo', PyScriptDemo)
  }
}