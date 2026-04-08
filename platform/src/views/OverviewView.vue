<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

import {
  benchmarkDeck,
  citation,
  frameworkStages,
  projectFacts,
  projectMeta,
  resultDeck,
  sectionNav,
} from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'

const isExternalLink = (href: string) => href.startsWith('http')
const isHashLink = (href: string) => href.startsWith('#')

const activeFrameworkId = ref(frameworkStages[0]?.id ?? '')
const activeBenchmarkId = ref(benchmarkDeck[0]?.id ?? '')
const activeResultId = ref(resultDeck[0]?.id ?? '')

const activeFramework = computed(
  () => frameworkStages.find((stage) => stage.id === activeFrameworkId.value) ?? frameworkStages[0],
)
const activeFrameworkIndex = computed(() =>
  frameworkStages.findIndex((stage) => stage.id === activeFramework.value.id) + 1,
)
const activeBenchmark = computed(
  () => benchmarkDeck.find((benchmark) => benchmark.id === activeBenchmarkId.value) ?? benchmarkDeck[0],
)
const activeResult = computed(() => resultDeck.find((item) => item.id === activeResultId.value) ?? resultDeck[0])

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

onBeforeUnmount(() => {
  if (citationResetTimer) {
    clearTimeout(citationResetTimer)
  }
})
</script>

<template>
  <AppShell content-width="wide">
    <section id="overview" class="py-7 sm:py-8 xl:py-10">
      <div class="space-y-7">
        <div class="flex flex-wrap items-center gap-3">
          <div
            class="inline-flex rounded-full border border-slate-950/10 bg-white/86 px-4 py-1.5 text-[11px] font-semibold tracking-[0.26em] text-slate-500 uppercase backdrop-blur"
          >
            {{ projectMeta.eyebrow }}
          </div>
          <div class="text-sm text-slate-500">Manuscript preview for benchmarked autonomous EHR analysis.</div>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2">
            <a
              v-for="item in sectionNav"
              :key="item.id"
              :href="`#${item.id}`"
              class="rounded-full border border-slate-200 bg-white/84 px-4 py-2 text-sm font-semibold text-slate-600 transition hover:border-slate-300 hover:text-slate-950"
            >
              {{ item.label }}
            </a>
          </div>
        </div>

        <h1
          class="max-w-none text-balance text-[clamp(1.95rem,2.85vw,3.45rem)] font-semibold leading-[0.94] tracking-[-0.07em] text-slate-950"
        >
          {{ projectMeta.title }}
        </h1>

        <div class="max-w-[92rem] text-sm leading-8 text-slate-900 sm:text-base lg:text-lg">
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

        <div class="space-y-3">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Abstract</div>
          <p class="max-w-[96rem] text-base leading-8 text-pretty text-slate-600 sm:text-lg sm:leading-9">
            {{ projectMeta.abstract }}
          </p>
        </div>

        <div class="flex flex-wrap gap-3">
          <template v-for="link in projectMeta.links" :key="link.href">
            <a
              v-if="isExternalLink(link.href) || isHashLink(link.href)"
              :href="link.href"
              :target="isExternalLink(link.href) ? '_blank' : undefined"
              :rel="isExternalLink(link.href) ? 'noreferrer' : undefined"
            >
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </a>
            <RouterLink v-else :to="link.href">
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </RouterLink>
          </template>
        </div>

        <div class="grid gap-3 lg:grid-cols-3">
          <AppCard
            v-for="fact in projectFacts"
            :key="fact.label"
            class="border-slate-200/80 bg-white/86 !p-5"
          >
            <div class="space-y-3">
              <div class="text-[11px] font-semibold tracking-[0.18em] text-slate-500 uppercase">{{ fact.label }}</div>
              <div class="text-2xl font-semibold tracking-[-0.05em] text-slate-950">{{ fact.value }}</div>
              <p class="text-sm leading-7 text-slate-600">{{ fact.note }}</p>
            </div>
          </AppCard>
        </div>
      </div>
    </section>

    <section id="framework" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Framework</div>
        <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
          A governed four-agent loop instead of a single opaque prompt.
        </h2>
        <p class="max-w-[90rem] text-base leading-8 text-slate-600 sm:text-lg">
          HealthFlow keeps planning, execution, critique, and memory writeback explicit. The homepage now mirrors that structure directly through a stage switcher instead of a static list.
        </p>
      </div>

      <div class="overflow-x-auto pb-1">
        <div class="flex w-max gap-2">
          <button
            v-for="stage in frameworkStages"
            :key="stage.id"
            type="button"
            class="rounded-full border px-4 py-2 text-sm font-semibold transition"
            :class="
              activeFrameworkId === stage.id
                ? 'border-slate-950 bg-slate-950 text-white shadow-sm'
                : 'border-slate-200 bg-white/86 text-slate-600 hover:border-slate-300 hover:text-slate-950'
            "
            @click="activeFrameworkId = stage.id"
          >
            {{ stage.shortLabel }}
          </button>
        </div>
      </div>

      <AppCard class="border-slate-200/80 bg-white/86">
        <div class="space-y-6">
          <div class="flex flex-wrap items-center gap-3 text-sm text-slate-500">
            <span class="rounded-full bg-slate-950 px-3 py-1 text-xs font-semibold tracking-[0.16em] text-white uppercase">
              Stage {{ activeFrameworkIndex }}
            </span>
            <span class="rounded-full border border-slate-200 bg-white px-3 py-1 font-semibold text-slate-700">
              {{ activeFramework.kicker }}
            </span>
          </div>

          <div class="space-y-3">
            <h3 class="font-display text-[clamp(2rem,2.7vw,3.15rem)] leading-[0.96] tracking-[-0.04em] text-slate-950">
              {{ activeFramework.title }}
            </h3>
            <p class="max-w-[86rem] text-base leading-8 text-slate-600 sm:text-lg">
              {{ activeFramework.description }}
            </p>
          </div>

          <div class="grid gap-3 md:grid-cols-3">
            <div
              v-for="item in activeFramework.outputs"
              :key="item"
              class="rounded-[1.4rem] border border-slate-200 bg-slate-50/90 px-4 py-4"
            >
              <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Key output</div>
              <div class="mt-3 text-base font-semibold tracking-[-0.03em] text-slate-900">{{ item }}</div>
            </div>
          </div>
        </div>
      </AppCard>
    </section>

    <section id="benchmarks" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Benchmarks</div>
        <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
          Five benchmarks, with EHRFlowBench as the primary open-ended testbed.
        </h2>
        <p class="max-w-[90rem] text-base leading-8 text-slate-600 sm:text-lg">
          Instead of flattening everything into two cards, the homepage now lets each benchmark speak in its own panel, including the full EHRFlowBench construction pipeline.
        </p>
      </div>

      <div class="overflow-x-auto pb-1">
        <div class="flex w-max gap-2">
          <button
            v-for="benchmark in benchmarkDeck"
            :key="benchmark.id"
            type="button"
            class="rounded-full border px-4 py-2 text-sm font-semibold transition"
            :class="
              activeBenchmarkId === benchmark.id
                ? 'border-slate-950 bg-slate-950 text-white shadow-sm'
                : 'border-slate-200 bg-white/86 text-slate-600 hover:border-slate-300 hover:text-slate-950'
            "
            @click="activeBenchmarkId = benchmark.id"
          >
            {{ benchmark.label }}
          </button>
        </div>
      </div>

      <AppCard class="border-slate-200/80 bg-white/86">
        <div class="space-y-6">
          <div class="flex flex-wrap items-center gap-3">
            <span class="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-[11px] font-semibold tracking-[0.16em] text-sky-700 uppercase">
              {{ activeBenchmark.kicker }}
            </span>
          </div>

          <div class="space-y-3">
            <h3 class="font-display text-[clamp(2rem,2.7vw,3.15rem)] leading-[0.96] tracking-[-0.04em] text-slate-950">
              {{ activeBenchmark.title }}
            </h3>
            <p class="max-w-[86rem] text-base leading-8 text-slate-600 sm:text-lg">
              {{ activeBenchmark.body }}
            </p>
          </div>

          <div class="grid gap-3 sm:grid-cols-3">
            <div
              v-for="metric in activeBenchmark.metrics"
              :key="`${activeBenchmark.id}-${metric.label}`"
              class="rounded-[1.4rem] border border-slate-200 bg-slate-50/90 px-4 py-4"
            >
              <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ metric.label }}</div>
              <div class="mt-3 text-2xl font-semibold tracking-[-0.05em] text-slate-950">{{ metric.value }}</div>
            </div>
          </div>

          <div class="grid gap-3 lg:grid-cols-2">
            <div
              v-for="(item, index) in activeBenchmark.bullets"
              :key="item"
              class="rounded-[1.4rem] border border-slate-200 bg-white px-4 py-4"
            >
              <div class="flex items-start gap-3">
                <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-950 text-sm font-semibold text-white">
                  {{ index + 1 }}
                </div>
                <p class="text-sm leading-7 text-slate-600 sm:text-base">{{ item }}</p>
              </div>
            </div>
          </div>
        </div>
      </AppCard>
    </section>

    <section id="results" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Results</div>
        <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
          Interactive summary of the manuscript’s main takeaways.
        </h2>
        <p class="max-w-[90rem] text-base leading-8 text-slate-600 sm:text-lg">
          The homepage now exposes open-ended workflow performance, executable artifact quality, five-benchmark coverage, and blinded human review as separate switchable views.
        </p>
      </div>

      <div class="overflow-x-auto pb-1">
        <div class="flex w-max gap-2">
          <button
            v-for="item in resultDeck"
            :key="item.id"
            type="button"
            class="rounded-full border px-4 py-2 text-sm font-semibold transition"
            :class="
              activeResultId === item.id
                ? 'border-slate-950 bg-slate-950 text-white shadow-sm'
                : 'border-slate-200 bg-white/86 text-slate-600 hover:border-slate-300 hover:text-slate-950'
            "
            @click="activeResultId = item.id"
          >
            {{ item.label }}
          </button>
        </div>
      </div>

      <AppCard class="border-slate-200/80 bg-white/86">
        <div class="space-y-6">
          <div class="space-y-3">
            <h3 class="font-display text-[clamp(2rem,2.7vw,3.15rem)] leading-[0.96] tracking-[-0.04em] text-slate-950">
              {{ activeResult.title }}
            </h3>
            <p class="max-w-[86rem] text-base leading-8 text-slate-600 sm:text-lg">
              {{ activeResult.summary }}
            </p>
          </div>

          <div class="grid gap-3 sm:grid-cols-3">
            <div
              v-for="stat in activeResult.stats"
              :key="`${activeResult.id}-${stat.label}`"
              class="rounded-[1.4rem] border border-slate-200 bg-slate-50/90 px-4 py-4"
            >
              <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ stat.label }}</div>
              <div class="mt-3 text-2xl font-semibold tracking-[-0.05em] text-slate-950">{{ stat.value }}</div>
            </div>
          </div>
        </div>
      </AppCard>
    </section>

    <section id="citation" class="space-y-5 py-8 sm:py-10">
      <div class="space-y-3">
        <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Citation</div>
        <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
          One clean citation block, no duplicate paper-record language.
        </h2>
      </div>

      <AppCard class="border-slate-200/80 bg-white/86">
        <div class="space-y-5">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div class="text-sm text-slate-600">
              Current manuscript metadata and a direct code link.
            </div>
            <div class="flex flex-wrap gap-2">
              <AppButton variant="secondary" @click="copyCitation">
                {{ citationButtonLabel }}
              </AppButton>
              <a href="https://github.com/yhzhu99/HealthFlow" target="_blank" rel="noreferrer">
                <AppButton variant="ghost">Code Repository</AppButton>
              </a>
            </div>
          </div>

          <pre class="overflow-x-auto rounded-[1.6rem] bg-slate-950 px-5 py-5 text-sm leading-7 text-slate-100"><code>{{ citation }}</code></pre>
        </div>
      </AppCard>
    </section>
  </AppShell>
</template>
