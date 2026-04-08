<script setup lang="ts">
import AppShell from '../components/layout/AppShell.vue'
</script>

<template>
  <AppShell>
    <section class="py-12 sm:py-16">
      <div class="max-w-3xl space-y-4">
        <p class="text-sm font-semibold tracking-[0.24em] text-slate-500 uppercase">Evaluation</p>
        <h1 class="text-4xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl">
          Blind review workspace for 20 benchmark tasks.
        </h1>
        <p class="text-lg leading-8 text-slate-600">
          This route will host reviewer identity, question navigation, candidate artifact rendering, and local export.
        </p>
      </div>
    </section>
  </AppShell>
</template>
