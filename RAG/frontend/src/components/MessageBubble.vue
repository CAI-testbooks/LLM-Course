<template>
  <div class="message" :class="[message.role, { 'is-last': isLast }]">
    <!-- Avatar -->
    <div class="message-avatar" v-if="message.role === 'assistant'">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    </div>
    
    <!-- Content -->
    <div class="message-body">
      <div class="message-content" :class="{ 'has-sources': hasSources }">
        <div 
          v-if="message.role === 'assistant'" 
          class="markdown-content"
          v-html="renderedContent"
        ></div>
        <div v-else class="user-content">{{ message.content }}</div>
      </div>
      
      <!-- Sources -->
      <div v-if="hasSources" class="sources">
        <div class="sources-header" @click="showSources = !showSources">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <span>参考来源 ({{ message.sources.length }})</span>
          <svg 
            class="chevron" 
            :class="{ expanded: showSources }"
            width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
          >
            <path d="m6 9 6 6 6-6"/>
          </svg>
        </div>
        
        <div v-if="showSources" class="sources-list">
          <div 
            v-for="(source, idx) in message.sources" 
            :key="idx"
            class="source-item"
          >
            <div class="source-header">
              <span class="source-index">[{{ idx + 1 }}]</span>
              <span class="source-name">{{ source.doc_name }}</span>
              <span class="source-score">{{ (source.score * 100).toFixed(0) }}%</span>
            </div>
            <div class="source-content">{{ truncate(source.content, 200) }}</div>
          </div>
        </div>
      </div>
      
      <!-- Timestamp -->
      <div class="message-meta">
        <span class="timestamp">{{ formatTime(message.timestamp) }}</span>
      </div>
    </div>
    
    <!-- User Avatar -->
    <div class="message-avatar user-avatar" v-if="message.role === 'user'">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  message: {
    type: Object,
    required: true
  },
  isLast: {
    type: Boolean,
    default: false
  }
})

const showSources = ref(false)

const hasSources = computed(() => {
  return props.message.sources && props.message.sources.length > 0
})

const renderedContent = computed(() => {
  if (!props.message.content) return ''
  return marked.parse(props.message.content)
})

function formatTime(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function truncate(text, length) {
  if (!text) return ''
  if (text.length <= length) return text
  return text.slice(0, length) + '...'
}
</script>

<style scoped>
.message {
  display: flex;
  gap: 12px;
  animation: slideUp 0.3s ease-out;
}

.message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  flex-shrink: 0;
}

.message.assistant .message-avatar {
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  color: white;
}

.message.user .message-avatar {
  background: var(--gray-200);
  color: var(--gray-600);
}

.message-body {
  max-width: 75%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.message.user .message-body {
  align-items: flex-end;
}

.message-content {
  padding: 14px 18px;
  border-radius: var(--border-radius-lg);
  line-height: 1.6;
}

.message.assistant .message-content {
  background: var(--bg-primary);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-color);
}

.message.user .message-content {
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  color: white;
}

.message.user .message-content.has-sources {
  border-bottom-right-radius: var(--border-radius-sm);
}

.user-content {
  white-space: pre-wrap;
  word-break: break-word;
}

/* Sources */
.sources {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  overflow: hidden;
}

.sources-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  font-size: 0.8125rem;
  font-weight: 500;
  color: var(--text-secondary);
  cursor: pointer;
  transition: background var(--transition-fast);
}

.sources-header:hover {
  background: var(--gray-50);
}

.sources-header .chevron {
  margin-left: auto;
  transition: transform var(--transition-fast);
}

.sources-header .chevron.expanded {
  transform: rotate(180deg);
}

.sources-list {
  border-top: 1px solid var(--border-color);
}

.source-item {
  padding: 12px 14px;
  border-bottom: 1px solid var(--border-color);
}

.source-item:last-child {
  border-bottom: none;
}

.source-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.source-index {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--primary-600);
}

.source-name {
  font-size: 0.8125rem;
  font-weight: 500;
  color: var(--text-primary);
  flex: 1;
}

.source-score {
  font-size: 0.75rem;
  padding: 2px 8px;
  background: var(--primary-100);
  color: var(--primary-700);
  border-radius: 100px;
}

.source-content {
  font-size: 0.8125rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

/* Meta */
.message-meta {
  padding: 0 4px;
}

.timestamp {
  font-size: 0.6875rem;
  color: var(--text-tertiary);
}

/* Animation for last message */
.message.is-last .message-content {
  animation: pulse 0.5s ease-out;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.01); }
}
</style>
