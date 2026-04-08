<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const navItems = computed(() => [
  { to: '/', label: 'Overview' },
  { to: '/benchmarks', label: 'Benchmarks' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/resources', label: 'Resources' },
])

const isActive = (path: string) => route.path === path
</script>

<template>
  <div class="min-h-screen bg-[var(--hf-bg)] text-[var(--hf-ink)]">
    <div class="pointer-events-none fixed inset-0 overflow-hidden">
      <div class="absolute inset-x-0 top-[-12rem] h-[36rem] bg-[radial-gradient(circle_at_top,rgba(71,123,227,0.24),transparent_52%)]" />
      <div class="absolute bottom-[-14rem] right-[-6rem] h-[28rem] w-[28rem] rounded-full bg-[radial-gradient(circle,rgba(43,181,160,0.16),transparent_62%)] blur-3xl" />
    </div>

    <header class="sticky top-0 z-50 px-4 py-4 sm:px-6">
      <div class="mx-auto flex w-full max-w-7xl items-center justify-between rounded-full border border-white/60 bg-white/72 px-5 py-3 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-xl">
        <RouterLink to="/" class="flex items-center gap-3 text-sm font-semibold tracking-[0.24em] text-slate-900 uppercase">
          <img src="/branding/healthflow-icon.svg" alt="" class="h-9 w-9 rounded-full bg-slate-950/5 p-1.5" />
          <span>HealthFlow</span>
        </RouterLink>

        <nav class="mx-4 hidden items-center gap-1 rounded-full bg-slate-950/[0.03] p-1 lg:flex">
          <RouterLink
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="rounded-full px-4 py-2 text-sm font-medium transition"
            :class="isActive(item.to) ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-600 hover:text-slate-950'"
          >
            {{ item.label }}
          </RouterLink>
        </nav>

        <a
          href="https://arxiv.org/abs/2508.02621"
          target="_blank"
          rel="noreferrer"
          class="hidden rounded-full border border-slate-950/10 bg-slate-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 md:inline-flex"
        >
          Read Paper
        </a>
      </div>

      <div class="mx-auto mt-3 w-full max-w-7xl overflow-x-auto lg:hidden">
        <nav class="inline-flex min-w-full gap-2 rounded-full border border-white/60 bg-white/72 p-1 shadow-[0_14px_36px_rgba(15,23,42,0.05)] backdrop-blur-xl">
          <RouterLink
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="rounded-full px-4 py-2 text-sm font-medium whitespace-nowrap transition"
            :class="isActive(item.to) ? 'bg-slate-950 text-white' : 'text-slate-600'"
          >
            {{ item.label }}
          </RouterLink>
        </nav>
      </div>
    </header>

    <main class="relative">
      <div class="mx-auto w-full max-w-7xl px-4 pb-16 sm:px-6 sm:pb-24">
        <slot />
      </div>
    </main>

    <footer class="px-4 pb-8 sm:px-6">
      <div class="mx-auto flex max-w-7xl flex-col gap-4 rounded-[2rem] border border-white/70 bg-white/75 px-6 py-5 text-sm text-slate-500 shadow-[0_20px_60px_rgba(15,23,42,0.06)] backdrop-blur xl:flex-row xl:items-center xl:justify-between">
        <div>HealthFlow Platform unifies paper-facing storytelling, benchmark definitions, and blind evaluation review.</div>
        <div class="flex gap-4 text-slate-900">
          <a href="https://arxiv.org/abs/2508.02621" target="_blank" rel="noreferrer">arXiv</a>
          <RouterLink to="/evaluation">Evaluation</RouterLink>
        </div>
      </div>
    </footer>
  </div>
</template>
