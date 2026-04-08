import { createRouter, createWebHistory } from 'vue-router'

import EvaluationView from './views/EvaluationView.vue'
import OverviewView from './views/OverviewView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: OverviewView },
    { path: '/evaluation', name: 'evaluation', component: EvaluationView },
  ],
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }
    if (to.hash && to.path === from.path) {
      return { el: to.hash, top: 96, behavior: 'smooth' }
    }
    return { top: 0 }
  },
})

export default router
