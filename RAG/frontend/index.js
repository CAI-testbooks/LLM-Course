import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 响应拦截器
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    console.error('API Error:', message)
    return Promise.reject(new Error(message))
  }
)

export default {
  // 健康检查
  health() {
    return api.get('/health')
  },

  // 获取统计信息
  getStats() {
    return api.get('/stats')
  },

  // 聊天
  chat(params) {
    return api.post('/chat', params)
  },

  // 流式聊天
  async *chatStream(params) {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') return
          try {
            yield JSON.parse(data)
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }
  },

  // 检索
  retrieve(params) {
    return api.post('/retrieve', params)
  },

  // 检索详情
  retrieveDetails(params) {
    return api.post('/retrieve/details', params)
  },

  // 查询重写
  rewrite(params) {
    return api.post('/rewrite', params)
  },

  // 查询分析
  analyze(query) {
    return api.post('/analyze', { query })
  },

  // HyDE 生成
  generateHyDE(query, short = false) {
    return api.post('/hyde', { query, short })
  },

  // 缓存统计
  getCacheStats() {
    return api.get('/cache/stats')
  },

  // 清空缓存
  clearCache() {
    return api.post('/cache/clear')
  },

  // 获取会话列表
  getConversations() {
    return api.get('/conversations')
  },

  // 获取会话详情
  getConversation(convId) {
    return api.get(`/conversations/${convId}`)
  },

  // 清空所有会话
  clearConversations() {
    return api.post('/conversations/clear')
  }
}
