<template>
  <div class="bg-white border-b px-6 py-3">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-2xl">🏥</span>
        <div>
          <h1 class="text-lg font-semibold text-gray-800">RAG 医学问答系统</h1>
          <p class="text-xs text-gray-500">基于知识库的智能问答 · 支持多轮对话 · 来源可追溯</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <span v-if="conversationId" class="text-xs text-gray-400 font-mono">
          会话: {{ conversationId }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  conversationId: {
    type: String,
    default: null
  }
})
</script>
