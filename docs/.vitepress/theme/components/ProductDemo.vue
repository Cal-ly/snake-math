<!-- filepath: /home/cally/Code/GitHub/snake-math/docs/.vitepress/theme/components/ProductDemo.vue -->
<template>
  <div class="interactive-component">
    <div class="component-section">
      <div class="input-group">
        <label for="n-input">Calculate product up to n:</label>
        <input 
          id="n-input"
          v-model.number="n" 
          type="range" 
          min="1" 
          max="10" 
          class="range-input"
        />
        <span class="result-value">{{ n }}</span>
      </div>
      
      <div class="input-group">
        <label for="product-type">Product type:</label>
        <select id="product-type" v-model="productType" class="function-select">
          <option value="factorial">Factorial (1 × 2 × 3 × ... × n)</option>
          <option value="double">Double factorial (n × (n-2) × (n-4) × ...)</option>
          <option value="evens">Even numbers (2 × 4 × 6 × ... × 2n)</option>
          <option value="odds">Odd numbers (1 × 3 × 5 × ... × (2n-1))</option>
        </select>
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <h4 class="input-group-title">Calculation:</h4>
        <div class="result-highlight">{{ mathExpression }}</div>
      </div>
      
      <div class="result-card">
        <h4 class="input-group-title">Result:</h4>
        <div class="result-value" :class="{ 'large-number': result > 1000000 }">
          {{ formattedResult }}
        </div>
        <div class="result-info">
          <small>{{ resultInfo }}</small>
        </div>
      </div>
    </div>

    <div class="component-section">
      <h4 class="input-group-title">Growth Comparison:</h4>
      <div class="growth-chart">
        <div class="chart-labels">
          <span>Sum (Σ)</span>
          <span>Product (Π)</span>
        </div>
        <div class="bars">
          <div class="bar sum-bar" :style="{ height: sumBarHeight + '%' }">
            <span class="bar-value">{{ sumValue }}</span>
          </div>
          <div class="bar product-bar" :style="{ height: productBarHeight + '%' }">
            <span class="bar-value">{{ Math.min(result, 999999) }}{{ result > 999999 ? '+' : '' }}</span>
          </div>
        </div>
      </div>
      <p class="growth-note">
        Notice how products grow <strong>much faster</strong> than sums!
      </p>
    </div>

    <div class="interactive-card">
      <h4 class="input-group-title">Python Code:</h4>
      <pre><code>{{ pythonCode }}</code></pre>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ProductDemo',
  data() {
    return {
      n: 5,
      productType: 'factorial'
    }
  },
  computed: {
    result() {
      switch (this.productType) {
        case 'factorial':
          return this.factorial(this.n)
        case 'double':
          return this.doubleFactorial(this.n)
        case 'evens':
          return this.productEvens(this.n)
        case 'odds':
          return this.productOdds(this.n)
        default:
          return 0
      }
    },
    
    formattedResult() {
      if (this.result > 1000000) {
        return this.result.toExponential(2)
      }
      return this.result.toLocaleString()
    },
    
    resultInfo() {
      const digits = this.result.toString().length
      if (this.result > 1000000) {
        return `${digits} digits - Getting very large!`
      } else if (this.result > 1000) {
        return `${digits} digits`
      }
      return ''
    },
    
    mathExpression() {
      switch (this.productType) {
        case 'factorial':
          return this.getFactorialExpression()
        case 'double':
          return this.getDoubleFactorialExpression()
        case 'evens':
          return this.getEvensExpression()
        case 'odds':
          return this.getOddsExpression()
        default:
          return ''
      }
    },
    
    sumValue() {
      return this.n * (this.n + 1) / 2
    },
    
    sumBarHeight() {
      const maxValue = Math.max(this.sumValue, Math.min(this.result, 1000))
      return (this.sumValue / maxValue) * 100
    },
    
    productBarHeight() {
      const maxValue = Math.max(this.sumValue, Math.min(this.result, 1000))
      const productValue = Math.min(this.result, 1000)
      return (productValue / maxValue) * 100
    },
    
    pythonCode() {
      switch (this.productType) {
        case 'factorial':
          return `import math

def factorial(n):
    return math.prod(range(1, n + 1))

result = factorial(${this.n})
print(f"${this.n}! = {result}")  # ${this.result}`
        
        case 'double':
          return `import math

def double_factorial(n):
    return math.prod(range(n, 0, -2))

result = double_factorial(${this.n})
print(f"${this.n}!! = {result}")  # ${this.result}`
        
        case 'evens':
          return `import math

def product_evens(n):
    return math.prod(range(2, 2*n + 1, 2))

result = product_evens(${this.n})
print(f"Product of first {n} evens = {result}")  # ${this.result}`
        
        case 'odds':
          return `import math

def product_odds(n):
    return math.prod(range(1, 2*n, 2))

result = product_odds(${this.n})
print(f"Product of first {n} odds = {result}")  # ${this.result}`
        
        default:
          return ''
      }
    }
  },
  
  methods: {
    factorial(n) {
      if (n <= 1) return 1
      let result = 1
      for (let i = 2; i <= n; i++) {
        result *= i
      }
      return result
    },
    
    doubleFactorial(n) {
      let result = 1
      for (let i = n; i > 0; i -= 2) {
        result *= i
      }
      return result
    },
    
    productEvens(n) {
      let result = 1
      for (let i = 2; i <= 2 * n; i += 2) {
        result *= i
      }
      return result
    },
    
    productOdds(n) {
      let result = 1
      for (let i = 1; i < 2 * n; i += 2) {
        result *= i
      }
      return result
    },
    
    getFactorialExpression() {
      const terms = []
      for (let i = 1; i <= this.n; i++) {
        terms.push(i.toString())
      }
      return terms.join(' × ')
    },
    
    getDoubleFactorialExpression() {
      const terms = []
      for (let i = this.n; i > 0; i -= 2) {
        terms.push(i.toString())
      }
      return terms.join(' × ')
    },
    
    getEvensExpression() {
      const terms = []
      for (let i = 2; i <= 2 * this.n; i += 2) {
        terms.push(i.toString())
      }
      return terms.join(' × ')
    },
    
    getOddsExpression() {
      const terms = []
      for (let i = 1; i < 2 * this.n; i += 2) {
        terms.push(i.toString())
      }
      return terms.join(' × ')
    }
  }
}
</script>

<style scoped>
@import '../styles/components.css';

/* Code block styling for ProductDemo */
pre {
  background: #f8f9fa;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 15px;
  overflow-x: auto;
  margin: 0;
}

code {
  font-family: 'Courier New', monospace;
  font-size: 14px;
}
</style>