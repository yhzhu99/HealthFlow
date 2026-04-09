<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { toBasePath } from '../../lib/assets'

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
  { path: '/', href: toBasePath('/'), label: 'Home', external: false },
  { path: '/evaluation', href: toBasePath('/evaluation'), label: 'Evaluation', external: false },
  { path: null, href: 'https://healthflow.medx-pku.com/app/', label: 'App', external: true },
])

const isActive = (path: string) => route.path === path
const navItemClass = (path: string | null, external: boolean) =>
  !external && path && isActive(path)
    ? 'bg-sky-50 text-sky-900 shadow-[0_12px_28px_rgba(56,189,248,0.14)] ring-1 ring-sky-200'
    : 'text-slate-500 hover:bg-white hover:text-slate-950'

const contentWidthClass = computed(() =>
  props.contentWidth === 'wide' ? 'max-w-[1680px]' : 'max-w-7xl',
)

const brandIconUrl = toBasePath('branding/healthflow-icon.svg')
</script>

<template>
  <div class="min-h-screen bg-[var(--hf-bg)] text-[var(--hf-ink)]">
    <div class="pointer-events-none fixed inset-0 overflow-hidden">
      <div class="absolute inset-x-0 top-[-12rem] h-[36rem] bg-[radial-gradient(circle_at_top,rgba(71,123,227,0.24),transparent_52%)]" />
      <div class="absolute bottom-[-14rem] right-[-6rem] h-[28rem] w-[28rem] rounded-full bg-[radial-gradient(circle,rgba(43,181,160,0.16),transparent_62%)] blur-3xl" />
    </div>

    <header class="sticky top-0 z-50 px-2.5 py-3 sm:px-3 lg:px-4">
      <div
        class="mx-auto flex w-full items-center justify-between gap-4 rounded-full border border-white/70 bg-white/78 px-4 py-3 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:px-5"
        :class="contentWidthClass"
      >
        <RouterLink to="/" class="flex items-center gap-3 text-sm font-semibold tracking-[0.12em] text-slate-900">
          <img :src="brandIconUrl" alt="" class="h-9 w-9 rounded-full bg-slate-950/5 p-1.5" />
          <span>HealthFlow</span>
        </RouterLink>

        <nav class="flex items-center gap-1 rounded-full border border-slate-200/80 bg-slate-50/80 p-1">
          <a
            v-for="item in navItems"
            :key="item.href"
            :href="item.href"
            target="_blank"
            rel="noreferrer"
            class="rounded-full px-4 py-2 text-sm font-semibold tracking-[-0.01em] transition"
            :aria-current="!item.external && item.path && isActive(item.path) ? 'page' : undefined"
            :class="navItemClass(item.path, item.external)"
          >
            {{ item.label }}
          </a>
        </nav>
      </div>
    </header>

    <main class="relative">
      <div class="mx-auto w-full px-2.5 pb-14 sm:px-3 sm:pb-20 lg:px-4 xl:px-[1.125rem]" :class="contentWidthClass">
        <slot />
      </div>
    </main>
  </div>
</template>
