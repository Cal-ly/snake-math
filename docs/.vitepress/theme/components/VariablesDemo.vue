<!-- docs/.vitepress/theme/components/VariablesDemo.vue -->
<template>
  <div class="variables-demo">
    <InteractiveSlider 
      label="Adjust x value" 
      :min="0" 
      :max="10" 
      :initial-value="5"
      :show-calculation="true"
      :calculation-function="calculateLinearVariable"
    />
  </div>
</template>

<script setup>
import { onMounted } from 'vue'

const calculateLinearVariable = (x_val) => {
  const x = parseInt(x_val)
  const y = 2 * x + 1
  return `y = 2(${x}) + 1 = ${y}`
}

onMounted(() => {
  // Make the function globally available for the InteractiveSlider component
  window.calculateLinearVariable = calculateLinearVariable
  console.log("Variables JavaScript functions loaded")
})
</script>

<style scoped>
.variables-demo {
  margin: 2rem 0;
}
</style>