<script setup lang="ts">
import { citation, resourceCards } from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppCard from '../components/ui/AppCard.vue'
import SectionHeader from '../components/ui/SectionHeader.vue'
</script>

<template>
  <AppShell>
    <section class="py-12 sm:py-16">
      <SectionHeader
        eyebrow="Resources"
        title="Paper, runtime, and benchmark build assets."
        description="The frontend is intentionally static. The canonical sources of truth stay in the HealthFlow repo: README, processed benchmark files, and `benchmark_results` workspaces."
      />
    </section>

    <section class="grid gap-5 lg:grid-cols-3">
      <a
        v-for="item in resourceCards"
        :key="item.title"
        :href="item.href"
        :target="item.href.startsWith('http') ? '_blank' : undefined"
        :rel="item.href.startsWith('http') ? 'noreferrer' : undefined"
      >
        <AppCard class="h-full transition hover:translate-y-[-2px] hover:shadow-[0_32px_80px_rgba(15,23,42,0.12)]">
          <div class="space-y-4">
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.title }}</div>
            <div class="text-2xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.label }}</div>
            <p class="text-base leading-8 text-slate-600">{{ item.body }}</p>
          </div>
        </AppCard>
      </a>
    </section>

    <section class="grid gap-5 py-16 lg:grid-cols-[0.9fr_1.1fr]">
      <AppCard>
        <div class="space-y-4">
          <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Commands</div>
          <pre class="overflow-x-auto rounded-[1.5rem] bg-slate-950 px-4 py-5 text-sm leading-7 text-slate-100"><code>cd platform
npm install
npm run snapshot
npm run dev</code></pre>
        </div>
      </AppCard>
      <AppCard>
        <div class="space-y-4">
          <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Citation</div>
          <pre class="overflow-x-auto rounded-[1.5rem] bg-slate-950 px-4 py-5 text-sm leading-7 text-slate-100"><code>{{ citation }}</code></pre>
        </div>
      </AppCard>
    </section>
  </AppShell>
</template>
