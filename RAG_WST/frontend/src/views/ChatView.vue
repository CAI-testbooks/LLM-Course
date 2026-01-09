<template>
  <div class="flex h-screen">
    <!-- 左侧边栏 -->
    <Sidebar
      :conversations="conversations"
      :currentConvId="conversationId"
      :stats="stats"
      @new-chat="handleNewChat"
      @select-conversation="handleSelectConversation"
      @delete-conversation="handleDeleteConversation"
    />

    <!-- 主内容区 -->
    <div class="flex-1 flex flex-col">
      <!-- 头部 -->
      <Header :conversationId="conversationId" />

      <!-- 聊天区域 -->
      <div class="flex-1 flex overflow-hidden">
        <!-- 消息列表 -->
        <ChatMessages
          ref="chatMessagesRef"
          :messages="messages"
          :isLoading="isLoading"
          @ask-question="handleAskQuestion"
        />

        <!-- 来源面板 -->
        <SourcesPanel :sources="currentSources" :refused="currentRefused" />
      </div>

      <!-- 输入区 -->
      <ChatInput
        v-model="inputMessage"
        :isLoading="isLoading"
        @send="handleSendMessage"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { marked } from 'marked'
import Sidebar from '@/components/Sidebar.vue'
import Header from '@/components/Header.vue'
import ChatMessages from '@/components/ChatMessages.vue'
import ChatInput from '@/components/ChatInput.vue'
import SourcesPanel from '@/components/SourcesPanel.vue'
import { getStats, getConversations, getConversation, deleteConversation, sendMessageStream } from '@/api'

// 状态
const conversationId = ref(null)
const messages = ref([])
const inputMessage = ref('')
const isLoading = ref(false)
const conversations = ref([])
const stats = ref({ total_chunks: 0, active_conversations: 0 })
const currentSources = ref([])
const currentRefused = ref(false)
const chatMessagesRef = ref(null)

// 加载统计信息
async function loadStats() {
  try {
    stats.value = await getStats()
  } catch (error) {
    console.error('Failed to load stats:', error)
  }
}

// 加载会话列表
async function loadConversations() {
  try {
    conversations.value = await getConversations()
  } catch (error) {
    console.error('Failed to load conversations:', error)
  }
}

// 新建对话
function handleNewChat() {
  conversationId.value = null
  messages.value = []
  currentSources.value = []
  currentRefused.value = false
  inputMessage.value = ''
}

// 选择会话
async function handleSelectConversation(convId) {
  try {
    const data = await getConversation(convId)
    conversationId.value = convId
    messages.value = data.messages.map(m => ({
      role: m.role,
      content: m.content
    }))
    currentSources.value = []
    currentRefused.value = false
  } catch (error) {
    console.error('Failed to load conversation:', error)
  }
}

// 删除会话
async function handleDeleteConversation(convId) {
  try {
    await deleteConversation(convId)
    if (conversationId.value === convId) {
      handleNewChat()
    }
    await loadConversations()
  } catch (error) {
    console.error('Failed to delete conversation:', error)
  }
}

// 快捷提问
function handleAskQuestion(question) {
  inputMessage.value = question
  handleSendMessage()
}

// 发送消息
async function handleSendMessage() {
  const message = inputMessage.value.trim()
  if (!message || isLoading.value) return

  inputMessage.value = ''
  isLoading.value = true

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: message
  })

  // 添加助手消息占位
  const assistantIndex = messages.value.length
  messages.value.push({
    role: 'assistant',
    content: '',
    isStreaming: true
  })

  await nextTick()
  chatMessagesRef.value?.scrollToBottom()

  try {
    const response = await sendMessageStream(message, conversationId.value)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    let fullResponse = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = decoder.decode(value)
      const lines = text.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === 'meta') {
              conversationId.value = data.conversation_id
              currentSources.value = data.sources || []
              currentRefused.value = data.refused || false
            } else if (data.type === 'content') {
              fullResponse += data.data
              messages.value[assistantIndex].content = fullResponse
              await nextTick()
              chatMessagesRef.value?.scrollToBottom()
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }

    messages.value[assistantIndex].isStreaming = false
    await loadConversations()

  } catch (error) {
    console.error('Send message error:', error)
    messages.value[assistantIndex].content = '⚠️ 发生错误，请稍后重试'
    messages.value[assistantIndex].isStreaming = false
  }

  isLoading.value = false
}

// 初始化
onMounted(() => {
  loadStats()
  loadConversations()
})
</script>
