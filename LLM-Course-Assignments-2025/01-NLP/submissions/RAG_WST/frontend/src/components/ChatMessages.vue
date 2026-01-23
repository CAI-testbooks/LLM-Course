<template>
  <div ref="containerRef" class="flex-1 overflow-y-auto p-6 space-y-4">
    <!-- 欢迎消息 -->
    <div v-if="messages.length === 0" class="text-center py-12">
      <div class="text-6xl mb-4">👋</div>
      <h2 class="text-xl font-semibold text-gray-700 mb-2">你好！我是医学问答助手</h2>
      <p class="text-gray-500 mb-6">请输入您的医学健康问题，我会基于知识库为您解答</p>
      <div class="flex flex-wrap justify-center gap-2">
        <button
          v-for="q in quickQuestions"
          :key="q"
          @click="$emit('ask-question', q)"
          class="px-3 py-1.5 bg-primary-50 text-primary-600 rounded-full text-sm hover:bg-primary-100 transition"
        >
          {{ q }}
        </button>
      </div>
    </div>

    <!-- 消息列表 -->
    <template v-for="(msg, index) in messages" :key="index">
      <!-- 用户消息 -->
      <div v-if="msg.role === 'user'" class="flex justify-end">
        <div class="message-user max-w-[70%] text-white rounded-2xl rounded-tr-sm px-4 py-3">
          {{ msg.content }}
        </div>
      </div>

      <!-- 助手消息 -->
      <div v-else class="flex justify-start">
        <div class="message-assistant max-w-[70%] rounded-2xl rounded-tl-sm px-4 py-3">
          <!-- 加载中 -->
          <div v-if="msg.isStreaming && !msg.content" class="typing-indicator flex gap-1.5">
            <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
            <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
            <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
          </div>
          <!-- 内容 -->
          <div v-else class="markdown-content" v-html="renderMarkdown(msg.content)"></div>
        </div>
      </div>
    </template>

    <!-- 加载指示器 -->
    <div v-if="isLoading && messages.length > 0 && !messages[messages.length - 1]?.isStreaming" class="flex justify-start">
      <div class="message-assistant rounded-2xl rounded-tl-sm px-4 py-3">
        <div class="typing-indicator flex gap-1.5">
          <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
          <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
          <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  },
  isLoading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['ask-question'])

const containerRef = ref(null)

const quickQuestions = [
  '高血压有什么症状？',
  '糖尿病如何预防？',
  '感冒发烧怎么办？'
]

function renderMarkdown(content) {
  if (!content) return ''
  return marked(content)
}

function scrollToBottom() {
  nextTick(() => {
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  })
}

defineExpose({
  scrollToBottom
})
</script>
