<script setup lang="ts">
import AppShell from '../components/layout/AppShell.vue'
</script>

<template>
  <AppShell>
    <section class="grid min-h-[calc(100vh-8rem)] items-center gap-12 py-10 lg:grid-cols-[1.15fr_0.85fr] lg:py-16">
      <div class="space-y-7">
        <div class="inline-flex rounded-full border border-slate-950/10 bg-white/75 px-4 py-1.5 text-xs font-semibold tracking-[0.3em] text-slate-600 uppercase backdrop-blur">
          HealthFlow Platform
        </div>
        <div class="space-y-5">
          <p class="max-w-2xl text-[clamp(3.5rem,8vw,7rem)] font-semibold leading-[0.95] tracking-[-0.06em] text-slate-950">
            Self-evolving agents for healthcare research.
          </p>
          <p class="max-w-2xl text-lg leading-8 text-slate-600 sm:text-xl">
            One platform for the HealthFlow paper narrative, benchmark definitions, and blind human evaluation across
            MedAgentBoard and EHRFlowBench.
          </p>
        </div>
        <div class="flex flex-wrap gap-3">
          <RouterLink to="/evaluation" class="rounded-full bg-slate-950 px-6 py-3 text-sm font-medium text-white transition hover:bg-slate-800">
            Open Evaluation
          </RouterLink>
          <RouterLink to="/benchmarks" class="rounded-full border border-slate-950/10 bg-white/80 px-6 py-3 text-sm font-medium text-slate-900 transition hover:border-slate-950/20">
            Explore Benchmarks
          </RouterLink>
        </div>
      </div>

      <div class="rounded-[2rem] border border-white/70 bg-white/80 p-5 shadow-[0_30px_70px_rgba(15,23,42,0.1)] backdrop-blur-xl">
        <img
          src="/media/healthflow-framework.png"
          alt="HealthFlow framework"
          class="aspect-[4/3] w-full rounded-[1.5rem] object-cover"
        />
      </div>
    </section>
  </AppShell>
</template>
