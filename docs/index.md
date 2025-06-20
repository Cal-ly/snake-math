---
layout: home
hero:
  name: "Snake Math"
  text: "Interactive Python Mathematics"
  tagline: "Mathematical concepts powered by Python in your browser"
  actions:
    - theme: brand
      text: Mathematical Foundations
      link: /basics/
    - theme: alt
      text: View on GitHub
      link: https://github.com/cally/snake-math

features:
  - title: Zero Installation
    details: Run Python math directly in your browser without any setup
  - title: Interactive Examples
    details: Adjust parameters and see immediate results
  - title: Progressive Learning
    details: From basic concepts to advanced mathematics
---

## Quick Example

<div id="demo-container">
  <label>Enter a value: </label>
  <input type="range" id="slider" min="0" max="10" value="5" />
  <span id="output">5² = 25</span>
</div>

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const slider = document.getElementById('slider');
  const output = document.getElementById('output');
  
  function updateOutput() {
    const value = slider.value;
    const result = value * value;
    output.textContent = value + '² = ' + result;
  }
  
  if (slider && output) {
    slider.addEventListener('input', updateOutput);
    updateOutput();
  }
});
</script>
