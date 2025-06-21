<!-- docs/.vitepress/theme/components/SummationDemo.vue -->
<template>
  <div class="interactive-component">
    <MathDisplay 
      :upper-limit="currentN" 
      :lower-limit="1" 
      expression="i" 
      :result="currentResult" 
    />
    
    <InteractiveSlider 
      label="Adjust n (upper limit)" 
      :min="1" 
      :max="20" 
      :initial-value="5"
      :show-calculation="true"
      :calculation-function="calculateSummation"
      @value-changed="updateValues"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const currentN = ref(5)
const currentResult = ref(15)

const calculateSummation = (n_val) => {
  const n = parseInt(n_val)
  const total = Array.from({length: n}, (_, i) => i + 1).reduce((sum, val) => sum + val, 0)
  return `Sum of 1 to ${n} = ${total}`
}

const updateValues = (newN) => {
  currentN.value = newN
  currentResult.value = newN * (newN + 1) / 2
}

onMounted(() => {
  // Make the function globally available for the InteractiveSlider component
  window.calculateSummation = calculateSummation
  console.log("Summation JavaScript functions loaded")
})
</script>

<style>
@import '../styles/components.css';
</style>