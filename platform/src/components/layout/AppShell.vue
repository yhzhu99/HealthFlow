<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const props = withDefaults(
  defineProps<{
    contentWidth?: 'default' | 'wide'
  }>(),
  {
    contentWidth: 'default',
  },
)

const route = useRoute()

const navItems = computed(() => [
  { to: '/', label: 'Home' },
  { to: '/evaluation', label: 'Evaluation' },
])

const isActive = (path: string) => route.path === path

const contentWidthClass = computed(() =>
  props.contentWidth === 'wide' ? 'max-w-[1580px]' : 'max-w-7xl',
)
</script>

<template>
  <div class="min-h-screen bg-[var(--hf-bg)] text-[var(--hf-ink)]">
    <div class="pointer-events-none fixed inset-0 overflow-hidden">
      <div class="absolute inset-x-0 top-[-12rem] h-[36rem] bg-[radial-gradient(circle_at_top,rgba(71,123,227,0.24),transparent_52%)]" />
      <div class="absolute bottom-[-14rem] right-[-6rem] h-[28rem] w-[28rem] rounded-full bg-[radial-gradient(circle,rgba(43,181,160,0.16),transparent_62%)] blur-3xl" />
    </div>

    <header class="sticky top-0 z-50 px-4 py-4 sm:px-6">
      <div
        class="mx-auto flex w-full items-center justify-between gap-4 rounded-full border border-white/70 bg-white/78 px-4 py-3 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:px-5"
        :class="contentWidthClass"
      >
        <RouterLink to="/" class="flex items-center gap-3 text-sm font-semibold tracking-[0.18em] text-slate-900">
          <img src="/branding/healthflow-icon.svg" alt="" class="h-9 w-9 rounded-full bg-slate-950/5 p-1.5" />
          <span>HealthFlow</span>
        </RouterLink>

        <nav class="flex items-center gap-1 rounded-full border border-slate-200/80 bg-slate-50/80 p-1">
          <RouterLink
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="rounded-full px-4 py-2 text-sm font-medium transition"
            :class="isActive(item.to) ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-950'"
          >
            {{ item.label }}
          </RouterLink>
        </nav>
      </div>
    </header>

    <main class="relative">
      <div class="mx-auto w-full px-4 pb-16 sm:px-6 sm:pb-24" :class="contentWidthClass">
        <slot />
      </div>
    </main>
  </div>
</template>
