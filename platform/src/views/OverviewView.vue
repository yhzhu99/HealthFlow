<script setup lang="ts">
import { ref } from 'vue'

import {
  benchmarkOverview,
  citation,
  frameworkStages,
  projectMeta,
} from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'

const isExternalLink = (href: string) => href.startsWith('http')

const citationButtonLabel = ref('Copy BibTeX')
let citationResetTimer: ReturnType<typeof setTimeout> | null = null

const copyCitation = async () => {
  try {
    await navigator.clipboard.writeText(citation)
    citationButtonLabel.value = 'Copied'
  } catch {
    citationButtonLabel.value = 'Copy failed'
  }

  if (citationResetTimer) {
    clearTimeout(citationResetTimer)
  }
  citationResetTimer = setTimeout(() => {
    citationButtonLabel.value = 'Copy BibTeX'
  }, 1800)
}
</script>

<template>
  <AppShell content-width="wide">
    <section class="py-8 sm:py-10 xl:py-12">
      <div class="space-y-6">
        <div
          class="inline-flex rounded-full border border-slate-950/10 bg-white/84 px-4 py-1.5 text-xs font-semibold tracking-[0.24em] text-slate-500 uppercase backdrop-blur"
        >
          {{ projectMeta.eyebrow }}
        </div>

        <h1 class="max-w-none text-[clamp(2.75rem,4.95vw,5.5rem)] font-semibold leading-[0.94] tracking-[-0.065em] text-slate-950">
          {{ projectMeta.title }}
        </h1>

        <div class="max-w-none text-base leading-8 text-slate-900 sm:text-lg">
          <template v-for="(author, index) in projectMeta.authors" :key="author.name">
            <span>
              {{ author.name }}<sup class="ml-0.5 text-[0.7em] align-super">{{ author.marks.join(',') }}</sup>
            </span>
            <span v-if="index < projectMeta.authors.length - 1">, </span>
          </template>
        </div>

        <AppCard class="!p-0 border-slate-200/80 bg-white/84">
          <div class="divide-y divide-slate-200">
            <div class="px-5 py-4 sm:px-6">
              <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Affiliations</div>
              <div class="mt-4 space-y-3 text-sm leading-7 text-slate-700 sm:text-[0.97rem]">
                <div v-for="item in projectMeta.affiliations" :key="item.mark" class="flex gap-3">
                  <span class="w-6 shrink-0 font-semibold text-slate-950">{{ item.mark }}</span>
                  <span>{{ item.text }}</span>
                </div>
              </div>
            </div>

            <div class="px-5 py-4 sm:px-6">
              <div class="space-y-2 text-sm leading-7 text-slate-600">
                <div v-for="item in projectMeta.notes" :key="item.mark" class="flex gap-3">
                  <span class="w-6 shrink-0 font-semibold text-slate-950">{{ item.mark }}</span>
                  <span>{{ item.text }}</span>
                </div>
              </div>
            </div>
          </div>
        </AppCard>

        <div class="max-w-none space-y-3">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Abstract</div>
          <p class="max-w-none text-base leading-8 text-slate-600 sm:text-lg sm:leading-9">
            {{ projectMeta.abstract }}
          </p>
        </div>

        <div class="flex flex-wrap gap-3">
          <template v-for="link in projectMeta.links" :key="link.href">
            <RouterLink v-if="!isExternalLink(link.href)" :to="link.href">
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </RouterLink>
            <a
              v-else
              :href="link.href"
              :target="isExternalLink(link.href) ? '_blank' : undefined"
              :rel="isExternalLink(link.href) ? 'noreferrer' : undefined"
            >
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </a>
          </template>
        </div>
      </div>
    </section>

    <section class="pb-6 sm:pb-8">
      <AppCard class="!p-0 border-slate-200/80 bg-white/84">
        <div class="divide-y divide-slate-200">
          <div
            v-for="item in projectMeta.details"
            :key="item.label"
            class="flex flex-col gap-2 px-5 py-4 sm:flex-row sm:gap-6 sm:px-6"
          >
            <div class="w-full shrink-0 text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase sm:w-44">
              {{ item.label }}
            </div>
            <div class="text-base leading-8 text-slate-900">{{ item.value }}</div>
          </div>
        </div>
      </AppCard>
    </section>

    <section id="framework" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Framework</div>
        <h2 class="text-3xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-4xl">
          Strategically self-evolving multi-agent workflow.
        </h2>
      </div>

      <AppCard class="!p-0 border-slate-200/80 bg-white/84">
        <div class="divide-y divide-slate-200">
          <div
            v-for="(stage, index) in frameworkStages"
            :key="stage.title"
            class="flex flex-col gap-3 px-5 py-5 sm:flex-row sm:gap-6 sm:px-6"
          >
            <div class="w-full shrink-0 sm:w-48">
              <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Stage {{ index + 1 }}</div>
              <div class="mt-2 text-xl font-semibold tracking-[-0.03em] text-slate-950">{{ stage.title }}</div>
            </div>
            <p class="text-base leading-8 text-slate-600">{{ stage.description }}</p>
          </div>
        </div>
      </AppCard>
    </section>

    <section id="benchmarks" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Benchmarks</div>
        <h2 class="text-3xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-4xl">
          Five benchmark suites with EHRFlowBench as the primary open-ended testbed.
        </h2>
        <p class="text-base leading-8 text-slate-600 sm:text-lg">
          {{ benchmarkOverview.intro }}
        </p>
      </div>

      <div class="flex flex-wrap gap-3">
        <div
          v-for="item in benchmarkOverview.benchmarkNames"
          :key="item"
          class="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700"
        >
          {{ item }}
        </div>
      </div>

      <AppCard class="border-slate-200/80 bg-white/84">
        <div class="space-y-5">
          <div>
            <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ benchmarkOverview.ehrflowbench.subtitle }}</div>
            <h3 class="mt-3 text-3xl font-semibold tracking-[-0.04em] text-slate-950">{{ benchmarkOverview.ehrflowbench.title }}</h3>
          </div>
          <p class="text-base leading-8 text-slate-600">{{ benchmarkOverview.ehrflowbench.body }}</p>
          <ul class="space-y-3 text-sm leading-7 text-slate-700 sm:text-base">
            <li v-for="item in benchmarkOverview.ehrflowbench.pipeline" :key="item" class="flex gap-3">
              <span class="mt-2 h-2 w-2 rounded-full bg-slate-900" />
              <span>{{ item }}</span>
            </li>
          </ul>
        </div>
      </AppCard>

      <AppCard class="border-slate-200/80 bg-white/84">
        <div class="space-y-5">
          <div>
            <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ benchmarkOverview.medagentboard.subtitle }}</div>
            <h3 class="mt-3 text-3xl font-semibold tracking-[-0.04em] text-slate-950">{{ benchmarkOverview.medagentboard.title }}</h3>
          </div>
          <p class="text-base leading-8 text-slate-600">{{ benchmarkOverview.medagentboard.body }}</p>
          <ul class="space-y-3 text-sm leading-7 text-slate-700 sm:text-base">
            <li v-for="item in benchmarkOverview.medagentboard.highlights" :key="item" class="flex gap-3">
              <span class="mt-2 h-2 w-2 rounded-full bg-slate-900" />
              <span>{{ item }}</span>
            </li>
          </ul>
        </div>
      </AppCard>

      <AppCard class="border-slate-200/80 bg-white/84">
        <div class="space-y-5">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Additional Benchmarks</div>
          <div class="space-y-4">
            <div
              v-for="item in benchmarkOverview.additionalBenchmarks"
              :key="item.title"
              class="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-5 py-4"
            >
              <div class="text-lg font-semibold tracking-[-0.03em] text-slate-950">{{ item.title }}</div>
              <p class="mt-2 text-sm leading-7 text-slate-600 sm:text-base">{{ item.body }}</p>
            </div>
          </div>
          <div class="border-t border-slate-200 pt-5">
            <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Main Baselines</div>
            <p class="mt-3 text-base leading-8 text-slate-600">
              {{ benchmarkOverview.baselineNames.join(', ') }}
            </p>
          </div>
        </div>
      </AppCard>
    </section>

    <section id="citation" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Citation</div>
        <h2 class="text-3xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-4xl">
          Current manuscript metadata.
        </h2>
        <p class="text-base leading-8 text-slate-600 sm:text-lg">
          The public paper record is intentionally hidden for now. The citation block below tracks the current manuscript title and author list.
        </p>
      </div>

      <AppCard class="border-slate-200/80 bg-white/84">
        <div class="space-y-4">
          <AppButton variant="secondary" @click="copyCitation">
            {{ citationButtonLabel }}
          </AppButton>
          <pre class="overflow-x-auto rounded-[1.75rem] bg-slate-950 px-5 py-5 text-sm leading-7 text-slate-100"><code>{{ citation }}</code></pre>
        </div>
      </AppCard>
    </section>
  </AppShell>
</template>
