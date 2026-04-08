<script setup lang="ts">
import { benchmarkCards } from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppCard from '../components/ui/AppCard.vue'
import SectionHeader from '../components/ui/SectionHeader.vue'
</script>

<template>
  <AppShell>
    <section class="py-12 sm:py-16">
      <SectionHeader
        eyebrow="Benchmarks"
        title="Two benchmark families, one evaluation surface."
        description="MedAgentBoard centers on artifact-heavy workflows. EHRFlowBench centers on paper-derived report generation. The platform keeps both under one reviewer workflow and one export format."
      />
    </section>

    <section class="grid gap-5 lg:grid-cols-2">
      <AppCard v-for="item in benchmarkCards" :key="item.id" class="h-full">
        <div class="space-y-5">
          <div>
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.eyebrow }}</div>
            <h2 class="mt-3 text-3xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.title }}</h2>
          </div>
          <p class="text-base leading-8 text-slate-600">{{ item.body }}</p>
          <ul class="space-y-3 text-sm leading-7 text-slate-700">
            <li v-for="bullet in item.bullets" :key="bullet" class="flex gap-3">
              <span class="mt-2 h-2 w-2 rounded-full bg-slate-950" />
              <span>{{ bullet }}</span>
            </li>
          </ul>
        </div>
      </AppCard>
    </section>

    <section class="grid gap-5 py-16 lg:grid-cols-[1.1fr_0.9fr]">
      <AppCard>
        <div class="space-y-5">
          <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Evaluation protocol</div>
          <h2 class="text-3xl font-semibold tracking-[-0.04em] text-slate-950">Blind reviewer flow</h2>
          <ol class="space-y-4 text-base leading-8 text-slate-600">
            <li>1. Reviewer enters a local `reviewer_id` and receives a stable per-question slot permutation.</li>
            <li>2. Each task shows the prompt, expected outputs, any reference material, and anonymous candidate answers.</li>
            <li>3. The reviewer selects the best candidate or marks `None / 都不好`, adds an optional note, and exports JSON locally.</li>
          </ol>
        </div>
      </AppCard>
      <AppCard>
        <div class="space-y-5">
          <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Static snapshot</div>
          <h2 class="text-3xl font-semibold tracking-[-0.04em] text-slate-950">Data ingress for v1</h2>
          <p class="text-base leading-8 text-slate-600">
            `npm run snapshot` resolves selected qids from local processed benchmark files, copies previewable artifacts
            from `benchmark_results`, and emits a frontend-ready `evaluation.snapshot.json`.
          </p>
        </div>
      </AppCard>
    </section>
  </AppShell>
</template>
