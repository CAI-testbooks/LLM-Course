<template>
  <div class="border-t bg-white p-4">
    <div class="max-w-4xl mx-auto">
      <div class="flex gap-3">
        <div class="flex-1 relative">
          <input
            type="text"
            :value="modelValue"
            @input="$emit('update:modelValue', $event.target.value)"
            @keypress.enter.prevent="$emit('send')"
            placeholder="输入您的医学健康问题..."
            :disabled="isLoading"
            class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-50 disabled:cursor-not-allowed"
          />
          <button
            @click="$emit('send')"
            :disabled="isLoading || !modelValue?.trim()"
            class="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
      <div class="mt-2 text-xs text-gray-400 text-center">
        提示：按 Enter 发送消息 · 所有回答均基于知识库，仅供参考
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  modelValue: {
    type: String,
    default: ''
  },
  isLoading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['update:modelValue', 'send'])
</script>
