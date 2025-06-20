<!-- docs/.vitepress/theme/components/MathDisplay.vue -->
<template>
  <div class="math-display-container">
    <!-- Summation notation display -->
    <div v-if="isSummation" class="math-notation">
      <span class="sigma">Σ</span>
      <div class="limits">
        <div class="upper">{{ upperLimit }}</div>
        <div class="lower">i={{ lowerLimit }}</div>
      </div>
      <span class="expression">{{ expression }}</span>
      <span class="equals">=</span>
      <span class="expansion">{{ expansionText }}</span>
      <span class="equals">=</span>
      <span class="result">{{ result }}</span>
    </div>
    
    <!-- General mathematical expression display -->
    <div v-else class="math-expression" v-html="renderedMath"></div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  // For summation notation
  upperLimit: { type: Number, default: 5 },
  lowerLimit: { type: Number, default: 1 },
  expression: { type: String, default: 'i' },
  result: { type: Number, default: 0 },
  
  // For general math expressions
  mathExpression: { type: String, default: '' },
  block: { type: Boolean, default: false }
})

const isSummation = computed(() => {
  return props.upperLimit !== undefined && props.lowerLimit !== undefined && !props.mathExpression
})

const expansionText = computed(() => {
  if (!isSummation.value) return ''
  
  const terms = []
  const limit = Math.min(props.upperLimit, 6)
  
  for (let i = props.lowerLimit; i <= limit; i++) {
    if (props.expression === 'i') {
      terms.push(i.toString())
    } else if (props.expression === 'i²') {
      terms.push(`${i}²`)
    } else {
      terms.push(`f(${i})`)
    }
  }
  
  if (props.upperLimit > 6) {
    terms.push('...')
    if (props.expression === 'i') {
      terms.push(props.upperLimit.toString())
    } else if (props.expression === 'i²') {
      terms.push(`${props.upperLimit}²`)
    } else {
      terms.push(`f(${props.upperLimit})`)
    }
  }
  
  return terms.join(' + ')
})

const renderedMath = computed(() => {
  if (isSummation.value || !props.mathExpression) return ''
  
  // Simple math rendering without KaTeX dependency
  // This is a basic implementation - you could enhance with a proper math renderer
  return props.mathExpression
    .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '<span class="fraction"><span class="numerator">$1</span><span class="denominator">$2</span></span>')
    .replace(/\\sum_\{([^}]+)\}\^\{([^}]+)\}/g, '<span class="summation">Σ<sub>$1</sub><sup>$2</sup></span>')
    .replace(/\\ldots/g, '...')
    .replace(/\\\\/g, '<br>')
})
</script>

<style scoped>
.math-display-container {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 12px;
  border-left: 4px solid #007bff;
}

.math-notation {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.sigma {
  font-size: 3rem;
  font-weight: bold;
  color: #007bff;
  margin-right: 0.2rem;
}

.limits {
  display: flex;
  flex-direction: column;
  font-size: 0.9rem;
  margin-right: 0.5rem;
  line-height: 1;
}

.upper, .lower {
  text-align: center;
}

.expression {
  font-style: italic;
  margin-right: 0.5rem;
  color: #333;
}

.equals {
  margin: 0 0.5rem;
  font-weight: bold;
  color: #666;
}

.expansion {
  color: #28a745;
  font-family: monospace;
  font-size: 0.9em;
}

.result {
  color: #dc3545;
  font-weight: bold;
  font-size: 1.2em;
}

.math-expression {
  text-align: center;
  font-size: 1.2rem;
  font-family: 'Times New Roman', serif;
  padding: 1rem;
}

.fraction {
  display: inline-block;
  text-align: center;
  vertical-align: middle;
}

.numerator {
  display: block;
  border-bottom: 1px solid #333;
  padding-bottom: 2px;
}

.denominator {
  display: block;
  padding-top: 2px;
}

.summation {
  font-size: 1.5em;
}

@media (max-width: 768px) {
  .math-notation {
    font-size: 1.2rem;
  }
  
  .sigma {
    font-size: 2.5rem;
  }
}
</style>