<template>
  <div class="config-panel-content">
    <div class="config-header">
      <h3>检索配置</h3>
    </div>
    
    <div class="config-body">
      <!-- 基础配置 -->
      <section class="config-section">
        <h4 class="section-title">基础设置</h4>
        
        <div class="config-item">
          <div class="config-label">
            <span>检索数量 (Top-K)</span>
          </div>
          <select 
            class="input select"
            :value="store.config.topK"
            @change="updateConfig('topK', parseInt($event.target.value))"
          >
            <option :value="3">3 条</option>
            <option :value="5">5 条</option>
            <option :value="10">10 条</option>
            <option :value="15">15 条</option>
          </select>
        </div>
        
        <div class="config-item">
          <div class="config-label">
            <span>流式输出</span>
            <span class="config-hint">实时显示回答内容</span>
          </div>
          <div 
            class="toggle" 
            :class="{ active: store.config.stream }"
            @click="updateConfig('stream', !store.config.stream)"
          ></div>
        </div>
      </section>
      
      <!-- 查询重写 -->
      <section class="config-section">
        <h4 class="section-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
          查询重写
        </h4>
        
        <div class="config-item">
          <div class="config-label">
            <span>启用重写</span>
            <span class="config-hint">优化查询词以提高检索效果</span>
          </div>
          <div 
            class="toggle" 
            :class="{ active: store.config.enableRewrite }"
            @click="updateConfig('enableRewrite', !store.config.enableRewrite)"
          ></div>
        </div>
        
        <div v-if="store.config.enableRewrite" class="config-item fade-in">
          <div class="config-label">
            <span>重写模式</span>
          </div>
          <select 
            class="input select"
            :value="store.config.rewriteMode"
            @change="updateConfig('rewriteMode', $event.target.value)"
          >
            <option value="single">单查询重写</option>
            <option value="multi">多查询扩展</option>
            <option value="context">上下文感知</option>
            <option value="auto">自动选择</option>
            <option value="hyde">HyDE 假设文档</option>
          </select>
        </div>
        
        <div v-if="store.config.enableRewrite" class="mode-info fade-in">
          <div class="info-card" :class="store.config.rewriteMode">
            <div class="info-icon">
              {{ modeInfo[store.config.rewriteMode]?.icon }}
            </div>
            <div class="info-text">
              <strong>{{ modeInfo[store.config.rewriteMode]?.title }}</strong>
              <p>{{ modeInfo[store.config.rewriteMode]?.desc }}</p>
            </div>
          </div>
        </div>
      </section>
      
      <!-- 重排序 -->
      <section class="config-section">
        <h4 class="section-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 6h18M7 12h10M11 18h2"/>
          </svg>
          重排序
        </h4>
        
        <div class="config-item">
          <div class="config-label">
            <span>启用重排序</span>
            <span class="config-hint">使用 Cross-Encoder 精排</span>
          </div>
          <div 
            class="toggle" 
            :class="{ active: store.config.enableRerank }"
            @click="updateConfig('enableRerank', !store.config.enableRerank)"
          ></div>
        </div>
      </section>
      
      <!-- 缓存统计 -->
      <section class="config-section" v-if="store.cacheStats">
        <h4 class="section-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          查询缓存
        </h4>
        
        <div class="cache-stats" v-if="store.cacheStats.enabled">
          <div class="stat-row">
            <span class="stat-label">缓存条目</span>
            <span class="stat-value">{{ store.cacheStats.size }} / {{ store.cacheStats.max_size }}</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">命中次数</span>
            <span class="stat-value text-success">{{ store.cacheStats.hits }}</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">未命中</span>
            <span class="stat-value text-warning">{{ store.cacheStats.misses }}</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">命中率</span>
            <span class="stat-value">
              <span class="progress-bar">
                <span 
                  class="progress-fill"
                  :style="{ width: (store.cacheStats.hit_rate * 100) + '%' }"
                ></span>
              </span>
              {{ (store.cacheStats.hit_rate * 100).toFixed(1) }}%
            </span>
          </div>
          <button class="btn btn-secondary btn-sm" @click="handleClearCache">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            清空缓存
          </button>
        </div>
        <div v-else class="cache-disabled">
          <span>缓存未启用</span>
        </div>
      </section>
      
      <!-- 检索详情 -->
      <section class="config-section" v-if="store.lastRetrievalDetails">
        <h4 class="section-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.35-4.35"/>
          </svg>
          最近检索
        </h4>
        
        <div class="retrieval-details">
          <div v-if="store.lastRetrievalDetails.rewritten_queries" class="detail-item">
            <span class="detail-label">重写后查询</span>
            <div class="query-list">
              <span 
                v-for="(q, idx) in store.lastRetrievalDetails.rewritten_queries" 
                :key="idx"
                class="query-tag"
              >
                {{ q }}
              </span>
            </div>
          </div>
          
          <div v-if="store.lastRetrievalDetails.hyde_document" class="detail-item">
            <span class="detail-label">HyDE 文档</span>
            <div class="hyde-doc">
              {{ truncate(store.lastRetrievalDetails.hyde_document, 200) }}
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { useChatStore } from '../stores/chat'

const store = useChatStore()

const modeInfo = {
  single: {
    icon: '✏️',
    title: '单查询重写',
    desc: '将口语化查询优化为专业检索词'
  },
  multi: {
    icon: '🔀',
    title: '多查询扩展',
    desc: '生成多个查询变体，提高召回率'
  },
  context: {
    icon: '💬',
    title: '上下文感知',
    desc: '结合对话历史补全代词和省略'
  },
  auto: {
    icon: '🤖',
    title: '自动选择',
    desc: '根据查询特征自动选择最佳模式'
  },
  hyde: {
    icon: '📄',
    title: 'HyDE 假设文档',
    desc: '生成假设性理想文档进行语义匹配'
  }
}

function updateConfig(key, value) {
  store.updateConfig({ [key]: value })
}

async function handleClearCache() {
  await store.clearCache()
}

function truncate(text, length) {
  if (!text) return ''
  if (text.length <= length) return text
  return text.slice(0, length) + '...'
}
</script>

<style scoped>
.config-panel-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.config-header {
  padding: 20px;
  border-bottom: 1px solid var(--border-color);
}

.config-header h3 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.config-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.config-section {
  margin-bottom: 24px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 16px;
}

.config-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.config-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.config-label span:first-child {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
}

.config-hint {
  font-size: 0.75rem;
  color: var(--text-tertiary);
}

.config-item .input {
  width: auto;
  min-width: 120px;
}

/* Mode Info */
.mode-info {
  margin-top: 12px;
}

.info-card {
  display: flex;
  gap: 12px;
  padding: 12px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
}

.info-card.hyde {
  background: linear-gradient(135deg, rgb(139 92 246 / 0.05), rgb(59 130 246 / 0.05));
  border-color: rgb(139 92 246 / 0.2);
}

.info-icon {
  font-size: 1.25rem;
}

.info-text {
  flex: 1;
}

.info-text strong {
  display: block;
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
}

.info-text p {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin: 0;
}

/* Cache Stats */
.cache-stats {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.stat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.8125rem;
}

.stat-row .stat-label {
  color: var(--text-secondary);
}

.stat-row .stat-value {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  color: var(--text-primary);
}

.text-success { color: var(--success); }
.text-warning { color: var(--warning); }

.progress-bar {
  width: 60px;
  height: 6px;
  background: var(--gray-200);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary-500);
  border-radius: 3px;
  transition: width var(--transition-normal);
}

.cache-disabled {
  padding: 12px;
  text-align: center;
  font-size: 0.8125rem;
  color: var(--text-tertiary);
  background: var(--bg-secondary);
  border-radius: var(--border-radius-md);
}

/* Retrieval Details */
.retrieval-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.detail-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-tertiary);
}

.query-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.query-tag {
  padding: 4px 10px;
  font-size: 0.75rem;
  background: var(--primary-100);
  color: var(--primary-700);
  border-radius: 100px;
}

.hyde-doc {
  padding: 10px 12px;
  font-size: 0.8125rem;
  color: var(--text-secondary);
  background: var(--bg-secondary);
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
  line-height: 1.5;
}
</style>
