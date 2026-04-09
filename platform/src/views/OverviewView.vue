<script setup lang="ts">
import { computed, ref } from 'vue'

import {
  benchmarkDeck,
  frameworkStages,
  projectFacts,
  projectMeta,
  resultDeck,
  sectionNav,
} from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppCard from '../components/ui/AppCard.vue'

const activeFrameworkId = ref(frameworkStages[0]?.id ?? '')
const activeBenchmarkId = ref(benchmarkDeck[0]?.id ?? '')
const activeResultId = ref(resultDeck[0]?.id ?? '')

const switcherButtonClass = (isActive: boolean) =>
  isActive
    ? 'border-sky-300 bg-sky-50 text-sky-900 shadow-[0_14px_32px_rgba(56,189,248,0.14)] ring-4 ring-sky-100'
    : 'border-slate-200 bg-white/86 text-slate-600 hover:-translate-y-0.5 hover:border-slate-300 hover:bg-white hover:text-slate-950'

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

const isNumericAuthorMark = (mark: string) => /^\d+$/.test(mark)
const formatAuthorMarks = (marks: string[]) => {
  const numericMarks = marks.filter(isNumericAuthorMark)
  const symbolicMarks = marks.filter((mark) => !isNumericAuthorMark(mark))
  return [...numericMarks, ...symbolicMarks]
}
const formattedAuthors = computed(() =>
  projectMeta.authors.map((author) => ({
    ...author,
    formattedMarks: formatAuthorMarks(author.marks),
  })),
)
</script>

<template>
  <AppShell content-width="wide">
    <section id="overview" class="py-7 sm:py-8 xl:py-10">
      <div class="mx-auto max-w-[1240px] space-y-7">
        <div class="flex items-center">
          <div
            class="inline-flex rounded-full border border-slate-950/10 bg-white/86 px-4 py-1.5 text-[11px] font-semibold tracking-[0.26em] text-slate-500 uppercase backdrop-blur"
          >
            {{ projectMeta.eyebrow }}
          </div>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto xl:mx-0">
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

        <div class="space-y-4 sm:space-y-5">
          <h1
            class="max-w-[34ch] text-[clamp(1.72rem,2.18vw,2.92rem)] font-semibold leading-[0.92] tracking-[-0.07em] text-slate-950"
          >
            {{ projectMeta.title }}
          </h1>

          <div class="max-w-[72rem] space-y-4">
            <div class="text-[0.98rem] leading-[1.82] text-slate-900 sm:text-[1.02rem] lg:text-[1.06rem]">
              <template v-for="(author, index) in formattedAuthors" :key="author.name">
                <span class="inline-flex items-start">
                  <span>{{ author.name }}</span>
                  <span
                    v-if="author.formattedMarks.length"
                    class="ml-1 inline-flex -translate-y-[0.38em] items-start gap-[0.14rem] text-[0.66em] font-semibold tracking-[0.02em] text-slate-500"
                  >
                    <template v-for="(mark, markIndex) in author.formattedMarks" :key="`${author.name}-${mark}-${markIndex}`">
                      <span :class="isNumericAuthorMark(mark) ? 'text-slate-500' : 'text-slate-900'">{{ mark }}</span>
                      <span v-if="markIndex < author.formattedMarks.length - 1" class="text-slate-400">,</span>
                    </template>
                  </span>
                </span>
                <span>
                  <span v-if="index < formattedAuthors.length - 1">, </span>
                </span>
              </template>
            </div>

            <div class="max-w-[72rem] space-y-2.5 border-t border-slate-200/80 pt-3 text-sm leading-6 text-slate-600">
              <div
                v-for="item in projectMeta.affiliations"
                :key="item.mark"
                class="flex gap-2.5"
              >
                <span class="w-4 shrink-0 font-semibold text-slate-950">{{ item.mark }}</span>
                <span>{{ item.text }}</span>
              </div>

              <div class="space-y-1.5 pt-1">
                <div
                  v-for="item in projectMeta.notes"
                  :key="item.mark"
                  class="flex gap-2.5"
                >
                  <span class="w-4 shrink-0 font-semibold text-slate-950">{{ item.mark }}</span>
                  <span>{{ item.text }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="mx-auto max-w-[70rem] space-y-4">
          <div class="text-center font-display text-[clamp(1.45rem,1.8vw,1.92rem)] font-semibold tracking-[-0.04em] text-slate-950">
            Abstract
          </div>
          <AppCard class="border-slate-200/80 bg-white/84 !p-5 text-left sm:!p-6">
            <p class="text-base leading-8 text-pretty text-slate-600 sm:text-lg sm:leading-9">
              {{ projectMeta.abstract }}
            </p>
          </AppCard>
        </div>
      </div>

      <div class="mx-auto mt-6 grid max-w-[1240px] gap-3 xl:mt-8 lg:grid-cols-3">
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
    </section>

    <section id="framework" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Framework</div>
          <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
            A governed four-agent loop instead of a single opaque prompt.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            HealthFlow keeps planning, execution, critique, and memory writeback explicit. The homepage now mirrors that structure directly through a stage switcher instead of a static list.
          </p>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto" role="tablist" aria-label="Framework sections">
            <button
              v-for="stage in frameworkStages"
              :key="stage.id"
              :id="`framework-tab-${stage.id}`"
              :aria-selected="activeFrameworkId === stage.id"
              type="button"
              role="tab"
              class="rounded-full border px-4 py-2 text-sm font-semibold transition"
              :class="switcherButtonClass(activeFrameworkId === stage.id)"
              @click="activeFrameworkId = stage.id"
            >
              {{ stage.shortLabel }}
            </button>
          </div>
        </div>

        <AppCard
          class="border-slate-200/80 bg-white/86"
          role="tabpanel"
          :aria-labelledby="`framework-tab-${activeFramework.id}`"
        >
          <div class="space-y-6">
            <div class="flex flex-wrap items-center justify-center gap-3 text-sm text-slate-500">
              <span class="rounded-full bg-sky-600 px-3 py-1 text-xs font-semibold tracking-[0.16em] text-white uppercase">
                Stage {{ activeFrameworkIndex }}
              </span>
              <span class="rounded-full border border-slate-200 bg-white px-3 py-1 font-semibold text-slate-700">
                {{ activeFramework.kicker }}
              </span>
            </div>

            <div class="mx-auto max-w-[60rem] space-y-3 text-center">
              <h3 class="font-display text-[clamp(2rem,2.7vw,3.15rem)] leading-[0.96] tracking-[-0.04em] text-slate-950">
                {{ activeFramework.title }}
              </h3>
              <p class="text-base leading-8 text-slate-600 sm:text-lg">
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
      </div>
    </section>

    <section id="benchmarks" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Benchmarks</div>
          <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
            Five benchmarks, with EHRFlowBench as the primary open-ended testbed.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            Instead of flattening everything into two cards, the homepage now lets each benchmark speak in its own panel, including the full EHRFlowBench construction pipeline.
          </p>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto" role="tablist" aria-label="Benchmark sections">
            <button
              v-for="benchmark in benchmarkDeck"
              :key="benchmark.id"
              :id="`benchmark-tab-${benchmark.id}`"
              :aria-selected="activeBenchmarkId === benchmark.id"
              type="button"
              role="tab"
              class="rounded-full border px-4 py-2 text-sm font-semibold transition"
              :class="switcherButtonClass(activeBenchmarkId === benchmark.id)"
              @click="activeBenchmarkId = benchmark.id"
            >
              {{ benchmark.label }}
            </button>
          </div>
        </div>

        <AppCard
          class="border-slate-200/80 bg-white/86"
          role="tabpanel"
          :aria-labelledby="`benchmark-tab-${activeBenchmark.id}`"
        >
          <div class="space-y-6">
            <div class="flex flex-wrap items-center justify-center gap-3">
              <span class="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-[11px] font-semibold tracking-[0.16em] text-sky-700 uppercase">
                {{ activeBenchmark.kicker }}
              </span>
            </div>

            <div class="mx-auto max-w-[60rem] space-y-3 text-center">
              <h3 class="font-display text-[clamp(2rem,2.7vw,3.15rem)] leading-[0.96] tracking-[-0.04em] text-slate-950">
                {{ activeBenchmark.title }}
              </h3>
              <p class="text-base leading-8 text-slate-600 sm:text-lg">
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
      </div>
    </section>

    <section id="results" class="py-8 sm:py-10">
      <div class="mx-auto max-w-[1280px] space-y-5">
        <div class="mx-auto max-w-[58rem] space-y-3 text-center">
          <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">Results</div>
          <h2 class="font-display text-[clamp(2.2rem,3.2vw,3.55rem)] leading-[0.95] tracking-[-0.04em] text-slate-950">
            Interactive summary of the manuscript’s main takeaways.
          </h2>
          <p class="text-base leading-8 text-slate-600 sm:text-lg">
            The homepage now exposes open-ended workflow performance, executable artifact quality, five-benchmark coverage, and blinded human review as separate switchable views.
          </p>
        </div>

        <div class="overflow-x-auto pb-1">
          <div class="flex w-max gap-2 md:mx-auto" role="tablist" aria-label="Result sections">
            <button
              v-for="item in resultDeck"
              :key="item.id"
              :id="`result-tab-${item.id}`"
              :aria-selected="activeResultId === item.id"
              type="button"
              role="tab"
              class="rounded-full border px-4 py-2 text-sm font-semibold transition"
              :class="switcherButtonClass(activeResultId === item.id)"
              @click="activeResultId = item.id"
            >
              {{ item.label }}
            </button>
          </div>
        </div>

        <AppCard
          class="border-slate-200/80 bg-white/86"
          role="tabpanel"
          :aria-labelledby="`result-tab-${activeResult.id}`"
        >
          <div class="space-y-6">
            <div class="mx-auto max-w-[60rem] space-y-3 text-center">
              <h3 class="font-display text-[clamp(2rem,2.7vw,3.15rem)] leading-[0.96] tracking-[-0.04em] text-slate-950">
                {{ activeResult.title }}
              </h3>
              <p class="text-base leading-8 text-slate-600 sm:text-lg">
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
      </div>
    </section>
  </AppShell>
</template>
