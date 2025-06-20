# Mathematical Foundations

## Variables and Expressions

In mathematics, we write: *y = 2x + 1*

In Python, this becomes:

```python
x = 5
y = 2 * x + 1
print(f"When x = {x}, y = {y}")
```

This would output: `When x = 5, y = 11`

## Interactive Example

<div id="basics-demo">
  <label>Enter a value for x: </label>
  <input type="range" id="x-slider" min="0" max="10" value="5" />
  <span id="x-value">x = 5</span>
  <br><br>
  <span id="y-result">y = 2(5) + 1 = 11</span>
</div>

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const slider = document.getElementById('x-slider');
  const xValue = document.getElementById('x-value');
  const yResult = document.getElementById('y-result');
  
  function updateCalculation() {
    const x = parseInt(slider.value);
    const y = 2 * x + 1;
    xValue.textContent = `x = ${x}`;
    yResult.textContent = `y = 2(${x}) + 1 = ${y}`;
  }
  
  if (slider && xValue && yResult) {
    slider.addEventListener('input', updateCalculation);
    updateCalculation();
  }
});
</script>
