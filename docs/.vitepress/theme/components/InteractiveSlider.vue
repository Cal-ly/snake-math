<!-- docs/.vitepress/theme/components/InteractiveSlider.vue -->
<template>
  <div class="interactive-slider">
    <div class="slider-controls">
      <label>{{ label }}:</label>
      <input 
        type="range" 
        :min="min" 
        :max="max" 
        :step="step"
        v-model="currentValue"
        @input="handleInput"
        class="slider"
      />
      <span class="value-display">{{ label.split(' ')[0] }} = {{ currentValue }}</span>
    </div>
    
    <div v-if="showCalculation" class="calculation-display" ref="calculationOutput">
      {{ calculationText }}
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'

const props = defineProps({
  label: { type: String, default: 'Value' },
  min: { type: Number, default: 1 },
  max: { type: Number, default: 10 },
  step: { type: Number, default: 1 },
  initialValue: { type: Number, default: 5 },
  showCalculation: { type: Boolean, default: false },
  calculationFunction: { type: [String, Function], default: null } // Function name or direct function
})

const emit = defineEmits(['valueChanged'])

const currentValue = ref(props.initialValue)
const calculationOutput = ref(null)
const calculationText = ref('')

const handleInput = () => {
  emit('valueChanged', parseInt(currentValue.value))
  
  // Call calculation function if provided
  if (props.calculationFunction) {
    let result
    
    if (typeof props.calculationFunction === 'function') {
      // Direct function reference
      result = props.calculationFunction(currentValue.value)
    } else if (typeof props.calculationFunction === 'string' && window[props.calculationFunction]) {
      // Function name as string (for global functions)
      result = window[props.calculationFunction](currentValue.value)
    }
    
    if (result !== undefined) {
      calculationText.value = result
    }
  }
}

// Watch for external value changes
watch(() => props.initialValue, (newVal) => {
  currentValue.value = newVal
  handleInput()
})

onMounted(() => {
  // Initial calculation
  setTimeout(() => {
    handleInput()
  }, 100)
})
</script>

<style scoped>
.interactive-slider {
  margin: 1rem 0;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: #f9f9f9;
}

.slider-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.slider-controls label {
  font-weight: 500;
  min-width: 120px;
}

.slider {
  flex: 1;
  min-width: 150px;
  max-width: 300px;
}

.value-display {
  font-weight: bold;
  color: #007bff;
  min-width: 80px;
}

.calculation-display {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: #e8f4f8;
  border-radius: 4px;
  font-family: monospace;
  color: #0066cc;
}

@media (max-width: 768px) {
  .slider-controls {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .slider {
    width: 100%;
    max-width: none;
  }
}
</style>