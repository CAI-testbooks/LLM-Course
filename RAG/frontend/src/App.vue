<template>
  <div class="app">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="logo">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5z" fill="currentColor" opacity="0.8"/>
            <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <div class="logo-text">
            <span class="logo-title">医疗问答</span>
            <span class="logo-subtitle">RAG System v2.1</span>
          </div>
        </div>
        <button class="btn btn-primary btn-sm" @click="handleNewChat">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 5v14M5 12h14"/>
          </svg>
          新对话
        </button>
      </div>
      
      <div class="sidebar-content">
        <ConversationList />
      </div>
      
      <div class="sidebar-footer">
        <div class="stats-mini" v-if="store.stats">
          <div class="stat-item">
            <span class="stat-value">{{ store.stats.total_chunks }}</span>
            <span class="stat-label">知识块</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ store.stats.total_docs }}</span>
            <span class="stat-label">文档</span>
          </div>
        </div>
      </div>
    </aside>
    
    <!-- Main Content -->
    <main class="main">
      <ChatView />
    </main>
    
    <!-- Config Panel -->
    <aside class="config-panel" :class="{ collapsed: !showConfig }">
      <button class="config-toggle" @click="showConfig = !showConfig">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
          <circle cx="12" cy="12" r="3"/>
        </svg>
      </button>
      <ConfigPanel v-if="showConfig" />
    </aside>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useChatStore } from './stores/chat'
import ConversationList from './components/ConversationList.vue'
import ChatView from './views/ChatView.vue'
import ConfigPanel from './components/ConfigPanel.vue'

const store = useChatStore()
const showConfig = ref(true)

onMounted(async () => {
  await store.loadStats()
  await store.loadConversations()
  await store.loadCacheStats()
})

function handleNewChat() {
  store.newConversation()
}
</script>

<style scoped>
.app {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* Sidebar */
.sidebar {
  width: var(--sidebar-width);
  background: var(--bg-primary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid var(--border-color);
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  color: var(--primary-600);
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.logo-title {
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--text-primary);
}

.logo-subtitle {
  font-size: 0.75rem;
  color: var(--text-tertiary);
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.sidebar-footer {
  padding: 16px 20px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-secondary);
}

.stats-mini {
  display: flex;
  gap: 24px;
}

.stat-item {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--primary-600);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-tertiary);
}

/* Main */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--bg-chat);
}

/* Config Panel */
.config-panel {
  width: var(--config-width);
  background: var(--bg-primary);
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  position: relative;
  transition: width var(--transition-normal), margin var(--transition-normal);
}

.config-panel.collapsed {
  width: 0;
  margin-left: -1px;
  overflow: hidden;
}

.config-toggle {
  position: absolute;
  left: -44px;
  top: 16px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  color: var(--text-secondary);
  transition: all var(--transition-fast);
  z-index: 10;
}

.config-toggle:hover {
  color: var(--primary-600);
  border-color: var(--primary-400);
}

.config-panel.collapsed .config-toggle {
  left: -44px;
}

/* Responsive */
@media (max-width: 1200px) {
  .config-panel {
    position: fixed;
    right: 0;
    top: 0;
    bottom: 0;
    z-index: 100;
    box-shadow: var(--shadow-xl);
  }
  
  .config-panel.collapsed {
    transform: translateX(100%);
  }
}

@media (max-width: 768px) {
  .sidebar {
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    z-index: 100;
    transform: translateX(-100%);
    box-shadow: var(--shadow-xl);
  }
  
  .sidebar.open {
    transform: translateX(0);
  }
}
</style>
