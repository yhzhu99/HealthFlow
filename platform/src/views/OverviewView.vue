<script setup lang="ts">
import { frameworkStages, overviewStats, resultHighlights } from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppCard from '../components/ui/AppCard.vue'
import AppButton from '../components/ui/AppButton.vue'
import SectionHeader from '../components/ui/SectionHeader.vue'
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
          <RouterLink to="/evaluation"><AppButton>Open Evaluation</AppButton></RouterLink>
          <RouterLink to="/benchmarks"><AppButton variant="secondary">Explore Benchmarks</AppButton></RouterLink>
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

    <section class="grid gap-4 py-10 sm:grid-cols-2 xl:grid-cols-4">
      <AppCard v-for="item in overviewStats" :key="item.label">
        <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.label }}</div>
        <div class="mt-4 text-xl font-semibold tracking-[-0.03em] text-slate-950">{{ item.value }}</div>
      </AppCard>
    </section>

    <section class="space-y-10 py-14 sm:py-18">
      <SectionHeader
        eyebrow="Framework"
        title="A runtime loop designed for inspectable, retry-aware research execution."
        description="HealthFlow keeps the workflow narrow and legible: plan with memory, execute in a workspace, evaluate outcomes, then reflect reusable knowledge back into memory."
      />
      <div class="grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
        <AppCard v-for="stage in frameworkStages" :key="stage.title">
          <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ stage.title }}</div>
          <p class="mt-4 text-base leading-7 text-slate-600">{{ stage.description }}</p>
        </AppCard>
      </div>
    </section>

    <section class="space-y-10 py-14 sm:py-18">
      <SectionHeader
        eyebrow="Highlights"
        title="Built for benchmark-facing, artifact-rich evaluation."
        description="The platform focuses on showing the work: task prompts, expected outputs, generated answers, and the files a reviewer needs to judge."
      />
      <div class="grid gap-4 lg:grid-cols-3">
        <AppCard v-for="item in resultHighlights" :key="item.title">
          <div class="text-2xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.title }}</div>
          <p class="mt-4 text-base leading-7 text-slate-600">{{ item.body }}</p>
        </AppCard>
      </div>
    </section>
  </AppShell>
</template>
