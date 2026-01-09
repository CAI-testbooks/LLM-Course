import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000
})

/**
 * 获取系统统计信息
 */
export async function getStats() {
  const response = await api.get('/api/stats')
  return response.data
}

/**
 * 健康检查
 */
export async function healthCheck() {
  const response = await api.get('/api/health')
  return response.data
}

/**
 * 发送聊天消息（非流式）
 */
export async function sendMessage(message, conversationId = null, topK = 5) {
  const response = await api.post('/api/chat', {
    message,
    conversation_id: conversationId,
    top_k: topK
  })
  return response.data
}

/**
 * 发送聊天消息（流式）
 * @returns {Promise<Response>} 原始 fetch response 用于流式处理
 */
export async function sendMessageStream(message, conversationId = null, topK = 5) {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      top_k: topK
    })
  })
  return response
}

/**
 * 获取会话列表
 */
export async function getConversations() {
  const response = await api.get('/api/conversations')
  return response.data
}

/**
 * 获取会话详情
 */
export async function getConversation(convId) {
  const response = await api.get(`/api/conversations/${convId}`)
  return response.data
}

/**
 * 删除会话
 */
export async function deleteConversation(convId) {
  const response = await api.delete(`/api/conversations/${convId}`)
  return response.data
}

/**
 * 清空所有会话
 */
export async function clearConversations() {
  const response = await api.post('/api/conversations/clear')
  return response.data
}

/**
 * 仅检索
 */
export async function retrieve(query, topK = 5) {
  const response = await api.post('/api/retrieve', {
    message: query,
    top_k: topK
  })
  return response.data
}

export default api
