import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'

const routes = [
  {
    path: '/',
    name: 'chat',
    component: ChatView
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
