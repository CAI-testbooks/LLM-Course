<template>
  <div class="w-64 bg-white border-r flex flex-col">
    <!-- 新建对话按钮 -->
    <div class="p-4 border-b">
      <button
        @click="$emit('new-chat')"
        class="w-full py-2.5 px-4 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition flex items-center justify-center gap-2"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
        </svg>
        新建对话
      </button>
    </div>

    <!-- 会话列表 -->
    <div class="flex-1 overflow-y-auto p-2">
      <div class="text-xs text-gray-500 px-2 py-1 mb-1">历史会话</div>
      
      <div v-if="conversations.length === 0" class="text-center text-gray-400 text-sm py-4">
        暂无历史会话
      </div>
      
      <div v-else class="space-y-1">
        <div
          v-for="conv in conversations"
          :key="conv.conversation_id"
          class="group px-3 py-2 rounded-lg cursor-pointer transition"
          :class="{
            'bg-primary-50 border-l-3 border-primary-500': conv.conversation_id === currentConvId,
            'hover:bg-gray-50': conv.conversation_id !== currentConvId
          }"
          @click="$emit('select-conversation', conv.conversation_id)"
        >
          <div class="flex items-center justify-between">
            <div class="flex-1 min-w-0">
              <div class="text-sm text-gray-700 truncate">
                {{ conv.preview || '新对话' }}
              </div>
              <div class="text-xs text-gray-400 mt-0.5">
                {{ formatTime(conv.last_message_at) }}
              </div>
            </div>
            <button
              @click.stop="$emit('delete-conversation', conv.conversation_id)"
              class="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部统计 -->
    <div class="p-3 border-t text-xs text-gray-400">
      <div>📚 知识库: {{ stats.total_chunks?.toLocaleString() || 0 }} 条</div>
      <div>💬 活跃会话: {{ stats.active_conversations || 0 }} 个</div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  conversations: {
    type: Array,
    default: () => []
  },
  currentConvId: {
    type: String,
    default: null
  },
  stats: {
    type: Object,
    default: () => ({})
  }
})

defineEmits(['new-chat', 'select-conversation', 'delete-conversation'])

function formatTime(isoString) {
  if (!isoString) return ''
  const date = new Date(isoString)
  const now = new Date()
  const diff = now - date

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return Math.floor(diff / 60000) + ' 分钟前'
  if (diff < 86400000) return Math.floor(diff / 3600000) + ' 小时前'
  if (diff < 604800000) return Math.floor(diff / 86400000) + ' 天前'

  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}
</script>
