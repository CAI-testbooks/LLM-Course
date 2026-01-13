<template>
  <div class="conversation-list">
    <div 
      v-for="conv in store.conversations" 
      :key="conv.id"
      class="conversation-item"
      :class="{ active: conv.id === store.currentConversationId }"
      @click="handleSelect(conv.id)"
    >
      <div class="conv-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <div class="conv-content">
        <div class="conv-title">{{ conv.title || '新对话' }}</div>
        <div class="conv-meta">
          <span class="conv-count">{{ conv.message_count }} 条消息</span>
          <span class="conv-time">{{ formatTime(conv.updated_at) }}</span>
        </div>
      </div>
    </div>
    
    <div v-if="store.conversations.length === 0" class="empty-list">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <span>暂无对话记录</span>
      <span class="hint">开始新的医疗咨询吧</span>
    </div>
  </div>
</template>

<script setup>
import { useChatStore } from '../stores/chat'

const store = useChatStore()

function handleSelect(convId) {
  store.loadConversation(convId)
}

function formatTime(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date
  
  // 今天
  if (diff < 24 * 60 * 60 * 1000 && date.getDate() === now.getDate()) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  
  // 昨天
  const yesterday = new Date(now)
  yesterday.setDate(yesterday.getDate() - 1)
  if (date.getDate() === yesterday.getDate()) {
    return '昨天'
  }
  
  // 本周
  if (diff < 7 * 24 * 60 * 60 * 1000) {
    const days = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
    return days[date.getDay()]
  }
  
  // 更早
  return date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}
</script>

<style scoped>
.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.conversation-item {
  display: flex;
  gap: 12px;
  padding: 12px;
  border-radius: var(--border-radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.conversation-item:hover {
  background: var(--gray-100);
}

.conversation-item.active {
  background: var(--primary-50);
}

.conv-icon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--gray-100);
  border-radius: var(--border-radius-sm);
  color: var(--text-tertiary);
  flex-shrink: 0;
}

.conversation-item.active .conv-icon {
  background: var(--primary-100);
  color: var(--primary-600);
}

.conv-content {
  flex: 1;
  min-width: 0;
}

.conv-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 4px;
}

.conv-meta {
  display: flex;
  gap: 8px;
  font-size: 0.75rem;
  color: var(--text-tertiary);
}

.empty-list {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
  color: var(--text-tertiary);
}

.empty-list svg {
  margin-bottom: 12px;
  opacity: 0.5;
}

.empty-list span {
  font-size: 0.875rem;
}

.empty-list .hint {
  font-size: 0.75rem;
  margin-top: 4px;
}
</style>
