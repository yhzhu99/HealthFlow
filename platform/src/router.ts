import { createRouter, createWebHistory } from 'vue-router'

import BenchmarksView from './views/BenchmarksView.vue'
import EvaluationView from './views/EvaluationView.vue'
import OverviewView from './views/OverviewView.vue'
import ResourcesView from './views/ResourcesView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'overview', component: OverviewView },
    { path: '/benchmarks', name: 'benchmarks', component: BenchmarksView },
    { path: '/evaluation', name: 'evaluation', component: EvaluationView },
    { path: '/resources', name: 'resources', component: ResourcesView },
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
