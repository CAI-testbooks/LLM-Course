<template>
  <div class="w-80 bg-gray-50 border-l flex flex-col">
    <!-- 标题 -->
    <div class="p-4 border-b bg-white">
      <h3 class="font-semibold text-gray-700 flex items-center gap-2">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
        参考来源
      </h3>
    </div>

    <!-- 来源列表 -->
    <div class="flex-1 overflow-y-auto p-4 space-y-3">
      <!-- 空状态 -->
      <div v-if="!sources || sources.length === 0" class="text-center text-gray-400 py-8">
        <svg class="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p class="text-sm">发送问题后</p>
        <p class="text-sm">这里会显示相关来源</p>
      </div>

      <!-- 来源卡片 -->
      <div
        v-for="source in sources"
        :key="source.index"
        class="source-card bg-white rounded-lg shadow-sm overflow-hidden"
      >
        <div
          class="p-3 cursor-pointer"
          @click="toggleSource(source.index)"
        >
          <div class="flex items-start justify-between gap-2">
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-1.5">
                <span class="flex-shrink-0 w-5 h-5 bg-primary-100 text-primary-600 rounded text-xs flex items-center justify-center font-medium">
                  {{ source.index }}
                </span>
                <span class="font-medium text-sm text-gray-700 truncate">
                  {{ source.doc_name }}
                </span>
              </div>
            </div>
            <span class="flex-shrink-0 text-xs font-medium" :class="getScoreClass(source.score)">
              {{ (source.score * 100).toFixed(1) }}%
            </span>
          </div>
          <!-- 分数条 -->
          <div class="mt-2 h-1 bg-gray-100 rounded-full overflow-hidden">
            <div
              class="h-full transition-all"
              :class="getScoreBarClass(source.score)"
              :style="{ width: Math.min(100, source.score * 100) + '%' }"
            ></div>
          </div>
        </div>
        <!-- 展开的内容 -->
        <div
          v-show="expandedSources.includes(source.index)"
          class="border-t bg-gray-50 p-3"
        >
          <p class="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap">
            {{ source.content }}
          </p>
        </div>
      </div>

      <!-- 拒绝提示 -->
      <div v-if="refused" class="mt-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
        <div class="flex items-center gap-2 text-orange-700 text-sm">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>检索结果相关度较低，已谨慎回答</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  sources: {
    type: Array,
    default: () => []
  },
  refused: {
    type: Boolean,
    default: false
  }
})

const expandedSources = ref([])

function toggleSource(index) {
  const idx = expandedSources.value.indexOf(index)
  if (idx === -1) {
    expandedSources.value.push(index)
  } else {
    expandedSources.value.splice(idx, 1)
  }
}

function getScoreClass(score) {
  if (score > 0.7) return 'score-high'
  if (score > 0.5) return 'score-medium'
  return 'score-low'
}

function getScoreBarClass(score) {
  if (score > 0.7) return 'bg-green-500'
  if (score > 0.5) return 'bg-yellow-500'
  return 'bg-red-500'
}
</script>
