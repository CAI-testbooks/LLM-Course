<template>
  <div class="chat-view">
    <!-- Messages Area -->
    <div class="messages-container" ref="messagesContainer">
      <!-- Empty State -->
      <div v-if="!store.hasMessages" class="empty-state">
        <div class="empty-icon">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25"/>
          </svg>
        </div>
        <h2>医疗问答助手</h2>
        <p>基于 RAG 技术的智能医学知识问答系统</p>
        <div class="quick-actions">
          <button 
            v-for="q in quickQuestions" 
            :key="q"
            class="quick-btn"
            @click="handleQuickQuestion(q)"
          >
            {{ q }}
          </button>
        </div>
        <div class="features">
          <div class="feature">
            <span class="feature-icon">📚</span>
            <span>专业医学知识库</span>
          </div>
          <div class="feature">
            <span class="feature-icon">🔍</span>
            <span>智能检索引用</span>
          </div>
          <div class="feature">
            <span class="feature-icon">💬</span>
            <span>多轮对话理解</span>
          </div>
        </div>
      </div>
      
      <!-- Messages -->
      <div v-else class="messages">
        <MessageBubble 
          v-for="(msg, idx) in store.messages" 
          :key="idx"
          :message="msg"
          :isLast="idx === store.messages.length - 1"
        />
        
        <!-- Loading -->
        <div v-if="store.isLoading" class="message assistant loading">
          <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div class="message-content">
            <div class="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Error Banner -->
    <div v-if="store.error" class="error-banner slide-up">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 8v4M12 16h.01"/>
      </svg>
      <span>{{ store.error }}</span>
      <button @click="store.clearError()" class="btn btn-ghost btn-sm">关闭</button>
    </div>
    
    <!-- Input Area -->
    <div class="input-area">
      <div class="input-container">
        <textarea
          ref="inputRef"
          v-model="inputText"
          class="chat-input"
          placeholder="输入您的医疗健康问题..."
          rows="1"
          @keydown.enter.exact="handleSend"
          @input="autoResize"
        ></textarea>
        <button 
          class="send-btn" 
          :disabled="!inputText.trim() || store.isLoading"
          @click="handleSend"
        >
          <svg v-if="!store.isLoading" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
          </svg>
          <div v-else class="spinner"></div>
        </button>
      </div>
      <div class="input-hint">
        <span>按 Enter 发送，Shift+Enter 换行</span>
        <span v-if="store.config.enableRewrite" class="badge badge-primary">
          {{ rewriteModeLabel }}
        </span>
        <span v-if="store.config.enableRerank" class="badge badge-info">
          重排序
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useChatStore } from '../stores/chat'
import MessageBubble from '../components/MessageBubble.vue'

const store = useChatStore()
const inputText = ref('')
const inputRef = ref(null)
const messagesContainer = ref(null)

const quickQuestions = [
  '感冒了应该怎么办？',
  '高血压有什么症状？',
  '如何预防糖尿病？',
  '头痛的常见原因是什么？'
]

const rewriteModeLabel = computed(() => {
  const labels = {
    single: '单查询重写',
    multi: '多查询扩展',
    context: '上下文感知',
    auto: '自动模式',
    hyde: 'HyDE'
  }
  return labels[store.config.rewriteMode] || store.config.rewriteMode
})

watch(() => store.messages.length, () => {
  nextTick(() => {
    scrollToBottom()
  })
})

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

function autoResize() {
  const el = inputRef.value
  if (el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 150) + 'px'
  }
}

function handleSend(e) {
  if (e.shiftKey) return
  e.preventDefault()
  
  if (!inputText.value.trim() || store.isLoading) return
  
  store.sendMessage(inputText.value)
  inputText.value = ''
  
  nextTick(() => {
    if (inputRef.value) {
      inputRef.value.style.height = 'auto'
    }
  })
}

function handleQuickQuestion(q) {
  inputText.value = q
  handleSend({ preventDefault: () => {} })
}
</script>

<style scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

/* Empty State */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  max-width: 600px;
  margin: 0 auto;
  animation: fadeIn 0.5s ease-out;
}

.empty-icon {
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--primary-100), var(--primary-50));
  border-radius: 50%;
  color: var(--primary-500);
  margin-bottom: 24px;
}

.empty-state h2 {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.empty-state p {
  color: var(--text-secondary);
  margin-bottom: 32px;
}

.quick-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  margin-bottom: 40px;
}

.quick-btn {
  padding: 10px 18px;
  font-size: 0.9375rem;
  color: var(--text-secondary);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 100px;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.quick-btn:hover {
  color: var(--primary-600);
  border-color: var(--primary-400);
  background: var(--primary-50);
}

.features {
  display: flex;
  gap: 32px;
  color: var(--text-tertiary);
  font-size: 0.875rem;
}

.feature {
  display: flex;
  align-items: center;
  gap: 8px;
}

.feature-icon {
  font-size: 1.25rem;
}

/* Messages */
.messages {
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
}

.message.loading {
  display: flex;
  gap: 12px;
}

.message-avatar {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  border-radius: 50%;
  color: white;
  flex-shrink: 0;
}

.message-content {
  padding: 16px 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-sm);
}

/* Error Banner */
.error-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  margin: 0 24px 12px;
  background: rgb(239 68 68 / 0.1);
  border: 1px solid rgb(239 68 68 / 0.2);
  border-radius: var(--border-radius-md);
  color: var(--error);
  font-size: 0.875rem;
}

.error-banner svg {
  flex-shrink: 0;
}

.error-banner span {
  flex: 1;
}

/* Input Area */
.input-area {
  padding: 16px 24px 24px;
  background: linear-gradient(to top, var(--bg-primary) 60%, transparent);
}

.input-container {
  display: flex;
  gap: 12px;
  max-width: 900px;
  margin: 0 auto;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-xl);
  padding: 8px 8px 8px 20px;
  box-shadow: var(--shadow-md);
  transition: all var(--transition-fast);
}

.input-container:focus-within {
  border-color: var(--primary-400);
  box-shadow: var(--shadow-lg), 0 0 0 3px rgb(13 148 136 / 0.1);
}

.chat-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-family: inherit;
  font-size: 1rem;
  color: var(--text-primary);
  resize: none;
  line-height: 1.5;
  padding: 8px 0;
  max-height: 150px;
}

.chat-input::placeholder {
  color: var(--text-tertiary);
}

.send-btn {
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  border: none;
  border-radius: 50%;
  color: white;
  cursor: pointer;
  transition: all var(--transition-fast);
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
  transform: scale(1.05);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.input-hint {
  display: flex;
  align-items: center;
  gap: 12px;
  max-width: 900px;
  margin: 12px auto 0;
  padding: 0 12px;
  font-size: 0.75rem;
  color: var(--text-tertiary);
}
</style>
