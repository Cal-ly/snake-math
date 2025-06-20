<!-- filepath: /home/cally/Code/GitHub/snake-math/docs/.vitepress/theme/components/ProductDemo.vue -->
<template>
  <div class="product-demo">
    <div class="controls">
      <div class="control-group">
        <label for="n-input">Calculate product up to n:</label>
        <input 
          id="n-input"
          v-model.number="n" 
          type="range" 
          min="1" 
          max="10" 
          class="slider"
        />
        <span class="value">{{ n }}</span>
      </div>
      
      <div class="control-group">
        <label for="product-type">Product type:</label>
        <select id="product-type" v-model="productType" class="select">
          <option value="factorial">Factorial (1 × 2 × 3 × ... × n)</option>
          <option value="double">Double factorial (n × (n-2) × (n-4) × ...)</option>
          <option value="evens">Even numbers (2 × 4 × 6 × ... × 2n)</option>
          <option value="odds">Odd numbers (1 × 3 × 5 × ... × (2n-1))</option>
        </select>
      </div>
    </div>

    <div class="results">
      <div class="calculation">
        <h4>Calculation:</h4>
        <div class="math-expression">{{ mathExpression }}</div>
      </div>
      
      <div class="result">
        <h4>Result:</h4>
        <div class="result-value" :class="{ 'large-number': result > 1000000 }">
          {{ formattedResult }}
        </div>
        <div class="result-info">
          <small>{{ resultInfo }}</small>
        </div>
      </div>
    </div>

    <div class="visualization">
      <h4>Growth Comparison:</h4>
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

    <div class="code-example">
      <h4>Python Code:</h4>
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
.product-demo {
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
  background: var(--vp-c-bg-soft);
}

.controls {
  display: grid;
  gap: 15px;
  margin-bottom: 20px;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.control-group label {
  font-weight: 500;
  min-width: 180px;
}

.slider {
  flex: 1;
  min-width: 150px;
}

.select {
  flex: 1;
  padding: 5px;
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  background: var(--vp-c-bg);
  min-width: 200px;
}

.value {
  font-weight: bold;
  color: var(--vp-c-brand);
  min-width: 30px;
}

.results {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.calculation h4,
.result h4 {
  margin: 0 0 10px 0;
  color: var(--vp-c-text-1);
}

.math-expression {
  font-family: 'Courier New', monospace;
  font-size: 16px;
  padding: 10px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  word-break: break-all;
}

.result-value {
  font-size: 24px;
  font-weight: bold;
  color: var(--vp-c-brand);
  padding: 10px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  text-align: center;
}

.result-value.large-number {
  color: var(--vp-c-warning);
  animation: pulse 2s infinite;
}

.result-info {
  margin-top: 5px;
  text-align: center;
}

.visualization {
  margin-bottom: 20px;
}

.visualization h4 {
  margin: 0 0 15px 0;
}

.growth-chart {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 400px;
}

.chart-labels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  font-weight: 500;
  text-align: center;
}

.bars {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  height: 100px;
  align-items: end;
}

.bar {
  position: relative;
  border-radius: 4px 4px 0 0;
  transition: height 0.3s ease;
  min-height: 10px;
  display: flex;
  align-items: end;
  justify-content: center;
}

.sum-bar {
  background: linear-gradient(to top, #3b82f6, #60a5fa);
}

.product-bar {
  background: linear-gradient(to top, #ef4444, #f87171);
}

.bar-value {
  position: absolute;
  top: -25px;
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.growth-note {
  margin-top: 10px;
  font-style: italic;
  text-align: center;
}

.code-example {
  border-top: 1px solid var(--vp-c-border);
  padding-top: 20px;
}

.code-example h4 {
  margin: 0 0 10px 0;
}

.code-example pre {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  padding: 15px;
  overflow-x: auto;
  margin: 0;
}

.code-example code {
  font-family: 'Courier New', monospace;
  font-size: 14px;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

@media (max-width: 768px) {
  .results {
    grid-template-columns: 1fr;
  }
  
  .control-group {
    flex-direction: column;
    align-items: stretch;
  }
  
  .control-group label {
    min-width: auto;
  }
}
</style>