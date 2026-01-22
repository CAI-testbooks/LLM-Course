import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api'

export const useChatStore = defineStore('chat', () => {
  // State
  const conversations = ref([])
  const currentConversationId = ref(null)
  const messages = ref([])
  const isLoading = ref(false)
  const error = ref(null)
  
  // 配置
  const config = ref({
    enableRewrite: false,
    rewriteMode: 'single',
    enableRerank: false,
    topK: 5,
    stream: true
  })
  
  // 统计信息
  const stats = ref(null)
  const cacheStats = ref(null)
  
  // 最后一次检索详情
  const lastRetrievalDetails = ref(null)

  // Computed
  const currentConversation = computed(() => {
    return conversations.value.find(c => c.id === currentConversationId.value)
  })

  const hasMessages = computed(() => messages.value.length > 0)

  // Actions
  async function loadStats() {
    try {
      stats.value = await api.getStats()
    } catch (e) {
      console.error('Failed to load stats:', e)
    }
  }

  async function loadCacheStats() {
    try {
      cacheStats.value = await api.getCacheStats()
    } catch (e) {
      console.error('Failed to load cache stats:', e)
    }
  }

  async function clearCache() {
    try {
      await api.clearCache()
      await loadCacheStats()
      return true
    } catch (e) {
      error.value = e.message
      return false
    }
  }

  async function loadConversations() {
    try {
      const data = await api.getConversations()
      conversations.value = data
    } catch (e) {
      console.error('Failed to load conversations:', e)
    }
  }

  async function loadConversation(convId) {
    try {
      const data = await api.getConversation(convId)
      currentConversationId.value = convId
      messages.value = data.messages || []
    } catch (e) {
      error.value = e.message
    }
  }

  function newConversation() {
    currentConversationId.value = null
    messages.value = []
    lastRetrievalDetails.value = null
  }

  async function sendMessage(content) {
    if (!content.trim() || isLoading.value) return

    // 添加用户消息
    messages.value.push({
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString()
    })

    isLoading.value = true
    error.value = null

    const params = {
      message: content.trim(),
      conversation_id: currentConversationId.value,
      top_k: config.value.topK,
      enable_rewrite: config.value.enableRewrite,
      rewrite_mode: config.value.rewriteMode,
      enable_rerank: config.value.enableRerank
    }

    try {
      if (config.value.stream) {
        // 流式响应
        const assistantMessage = {
          role: 'assistant',
          content: '',
          sources: [],
          timestamp: new Date().toISOString()
        }
        messages.value.push(assistantMessage)

        for await (const chunk of api.chatStream(params)) {
          if (chunk.type === 'content') {
            assistantMessage.content += chunk.content
          } else if (chunk.type === 'meta') {
            if (chunk.sources) {
              assistantMessage.sources = chunk.sources
            }
            if (!currentConversationId.value && chunk.conversation_id) {
              currentConversationId.value = chunk.conversation_id
            }
            if (chunk.rewritten_queries) {
              lastRetrievalDetails.value = {
                rewritten_queries: chunk.rewritten_queries,
                rewrite_mode: chunk.rewrite_mode
              }
            }
          }
        }
      } else {
        // 非流式响应
        const response = await api.chat(params)
        
        messages.value.push({
          role: 'assistant',
          content: response.message,
          sources: response.sources || [],
          timestamp: new Date().toISOString()
        })

        if (!currentConversationId.value) {
          currentConversationId.value = response.conversation_id
        }

        if (response.rewritten_queries) {
          lastRetrievalDetails.value = {
            rewritten_queries: response.rewritten_queries,
            rewrite_mode: response.rewrite_mode
          }
        }
      }

      // 刷新缓存统计
      if (config.value.enableRewrite) {
        loadCacheStats()
      }
    } catch (e) {
      error.value = e.message
      // 移除失败的助手消息（如果存在）
      if (messages.value.length > 0 && messages.value[messages.value.length - 1].role === 'assistant') {
        const lastMsg = messages.value[messages.value.length - 1]
        if (!lastMsg.content) {
          messages.value.pop()
        }
      }
    } finally {
      isLoading.value = false
    }
  }

  async function analyzeQuery(query) {
    try {
      return await api.analyze(query)
    } catch (e) {
      error.value = e.message
      return null
    }
  }

  async function generateHyDE(query, short = false) {
    try {
      return await api.generateHyDE(query, short)
    } catch (e) {
      error.value = e.message
      return null
    }
  }

  async function retrieveWithDetails(query) {
    try {
      const params = {
        message: query,
        top_k: config.value.topK,
        enable_rewrite: config.value.enableRewrite,
        rewrite_mode: config.value.rewriteMode,
        enable_rerank: config.value.enableRerank
      }
      lastRetrievalDetails.value = await api.retrieveDetails(params)
      return lastRetrievalDetails.value
    } catch (e) {
      error.value = e.message
      return null
    }
  }

  function clearError() {
    error.value = null
  }

  function updateConfig(newConfig) {
    config.value = { ...config.value, ...newConfig }
  }

  return {
    // State
    conversations,
    currentConversationId,
    messages,
    isLoading,
    error,
    config,
    stats,
    cacheStats,
    lastRetrievalDetails,
    
    // Computed
    currentConversation,
    hasMessages,
    
    // Actions
    loadStats,
    loadCacheStats,
    clearCache,
    loadConversations,
    loadConversation,
    newConversation,
    sendMessage,
    analyzeQuery,
    generateHyDE,
    retrieveWithDetails,
    clearError,
    updateConfig
  }
})