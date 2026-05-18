<script setup lang="ts">
import { computed, ref } from 'vue'

import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'
import {
  appDemoArtifacts,
  appDemoChatMessages,
  appDemoDescription,
  appDemoInputFile,
  appDemoMetrics,
  appDemoStages,
  appDemoTitle,
  coreAppDemoArtifactIds,
  liveRuntimeUrl,
  resolveAppDemoArtifact,
  type AppDemoArtifact,
  type AppDemoChartBar,
  type AppDemoChartPoint,
  type AppDemoStage,
} from '../content/appDemo'
import { toBasePath } from '../lib/assets'
import { renderMarkdown } from '../lib/markdown'

const firstArtifact = appDemoArtifacts[0] as AppDemoArtifact
const firstStage = appDemoStages[0] as AppDemoStage

const selectedArtifactId = ref(firstArtifact.id)
const selectedStageId = ref(firstStage.id)

const selectedArtifact = computed(() => resolveAppDemoArtifact(selectedArtifactId.value) ?? firstArtifact)
const selectedStage = computed(() => appDemoStages.find((stage) => stage.id === selectedStageId.value) ?? firstStage)
const selectedStageIndex = computed(() => appDemoStages.findIndex((stage) => stage.id === selectedStage.value.id) + 1)

const inputFileHref = computed(() => toBasePath(appDemoInputFile.href))
const selectedMarkdownHtml = computed(() =>
  selectedArtifact.value.kind === 'markdown' ? renderMarkdown(selectedArtifact.value.content) : '',
)
const selectedJsonText = computed(() =>
  selectedArtifact.value.kind === 'json' ? JSON.stringify(selectedArtifact.value.value, null, 2) : '',
)
const selectedLogLines = computed(() => (selectedArtifact.value.kind === 'log' ? selectedArtifact.value.lines : []))
const selectedTableColumns = computed(() => (selectedArtifact.value.kind === 'table' ? selectedArtifact.value.columns : []))
const selectedTableRows = computed(() => (selectedArtifact.value.kind === 'table' ? selectedArtifact.value.rows : []))
const selectedChart = computed(() => (selectedArtifact.value.kind === 'chart' ? selectedArtifact.value.chart : null))

const isCoreArtifact = (artifactId: string) => coreAppDemoArtifactIds.includes(artifactId)

const artifactButtonClass = (artifactId: string) =>
  selectedArtifactId.value === artifactId
    ? 'border-sky-300 bg-sky-50 text-sky-950 shadow-[0_14px_34px_rgba(56,189,248,0.14)]'
    : 'border-slate-200 bg-white/82 text-slate-600 hover:border-slate-300 hover:text-slate-950'

const stageButtonClass = (stageId: string) =>
  selectedStageId.value === stageId
    ? 'border-sky-300 bg-sky-50 text-sky-950 shadow-[0_14px_34px_rgba(56,189,248,0.14)]'
    : 'border-slate-200 bg-white/82 text-slate-600 hover:border-slate-300 hover:text-slate-950'

const formatCell = (value: string | number) => (typeof value === 'number' ? value.toLocaleString('en-US') : value)

const chartBox = {
  width: 280,
  height: 188,
  pad: 24,
}

const chartCoordinate = (point: AppDemoChartPoint) => {
  const plotWidth = chartBox.width - chartBox.pad * 2
  const plotHeight = chartBox.height - chartBox.pad * 2
  const x = chartBox.pad + point.x * plotWidth
  const y = chartBox.height - chartBox.pad - point.y * plotHeight
  return { x, y }
}

const chartPolyline = (points: AppDemoChartPoint[]) =>
  points
    .map((point) => chartCoordinate(point))
    .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ')

const chartMarkerX = (point: AppDemoChartPoint) => chartCoordinate(point).x
const chartMarkerY = (point: AppDemoChartPoint) => chartCoordinate(point).y
const maxBarValue = (bars: AppDemoChartBar[]) => Math.max(...bars.map((bar) => bar.value), 1)
const barHeight = (bar: AppDemoChartBar, bars: AppDemoChartBar[]) => `${Math.max(10, (bar.value / maxBarValue(bars)) * 100)}%`
</script>

<template>
  <AppShell content-width="wide">
    <section class="py-5 sm:py-7">
      <div class="grid gap-4 xl:grid-cols-[minmax(0,1fr)_27rem]">
        <div class="space-y-4">
          <AppCard class="border-slate-200/80 bg-white/86 !p-5 sm:!p-6">
            <div class="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
              <div class="max-w-4xl space-y-3">
                <div class="flex flex-wrap items-center gap-2">
                  <span class="rounded-full bg-slate-950 px-3 py-1 text-xs font-semibold tracking-[0.16em] text-white uppercase">
                    Static App Demo
                  </span>
                  <span class="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold tracking-[0.12em] text-emerald-800 uppercase">
                    Cloudflare Pages Ready
                  </span>
                </div>
                <h1 class="font-display text-[clamp(2.1rem,4.2vw,4.6rem)] leading-[0.95] text-slate-950">
                  {{ appDemoTitle }}
                </h1>
                <p class="max-w-3xl text-base leading-8 text-slate-600 sm:text-lg">
                  {{ appDemoDescription }}
                </p>
              </div>

              <div class="flex flex-wrap gap-2 lg:justify-end">
                <a :href="liveRuntimeUrl" target="_blank" rel="noreferrer">
                  <AppButton>Open live runtime</AppButton>
                </a>
                <a :href="inputFileHref" download>
                  <AppButton variant="secondary">Download cohort</AppButton>
                </a>
              </div>
            </div>
          </AppCard>

          <div class="grid gap-4 2xl:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
            <AppCard class="border-slate-200/80 bg-white/86 !p-0">
              <div class="border-b border-slate-200/80 px-5 py-4 sm:px-6">
                <div class="text-xs font-semibold tracking-[0.18em] text-slate-500 uppercase">Conversation</div>
              </div>
              <div class="space-y-4 p-4 sm:p-5">
                <article
                  v-for="message in appDemoChatMessages"
                  :key="message.label"
                  class="rounded-[1.25rem] border p-4"
                  :class="
                    message.role === 'user'
                      ? 'border-slate-200 bg-slate-50/80'
                      : 'border-sky-200 bg-[linear-gradient(135deg,rgba(240,249,255,0.96),rgba(236,253,245,0.9))]'
                  "
                >
                  <div class="mb-2 text-xs font-semibold tracking-[0.16em] text-slate-500 uppercase">{{ message.label }}</div>
                  <p class="text-sm leading-7 text-slate-700 sm:text-base sm:leading-8">{{ message.body }}</p>
                  <div v-if="message.attachments?.length" class="mt-4 flex flex-wrap gap-2">
                    <span
                      v-for="attachment in message.attachments"
                      :key="attachment"
                      class="rounded-full border border-slate-200 bg-white/86 px-3 py-1 text-xs font-semibold text-slate-600"
                    >
                      {{ attachment }}
                    </span>
                  </div>
                </article>
              </div>
            </AppCard>

            <AppCard class="border-slate-200/80 bg-white/86 !p-0">
              <div class="flex flex-col gap-3 border-b border-slate-200/80 px-5 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <div class="text-xs font-semibold tracking-[0.18em] text-slate-500 uppercase">Artifact Preview</div>
                  <div class="mt-1 text-lg font-semibold tracking-[-0.03em] text-slate-950">{{ selectedArtifact.label }}</div>
                </div>
                <span class="w-fit rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-500">
                  {{ selectedArtifact.path }}
                </span>
              </div>

              <div class="p-4 sm:p-5">
                <p class="mb-4 text-sm leading-7 text-slate-600">{{ selectedArtifact.description }}</p>

                <div
                  v-if="selectedArtifact.kind === 'markdown'"
                  class="prose prose-slate max-w-none rounded-[1.25rem] border border-slate-200 bg-white/86 p-4 text-sm"
                  v-html="selectedMarkdownHtml"
                />

                <pre
                  v-else-if="selectedArtifact.kind === 'json'"
                  class="max-h-[31rem] overflow-auto rounded-[1.25rem] border border-slate-200 bg-slate-950 p-4 text-xs leading-6 text-slate-100"
                >{{ selectedJsonText }}</pre>

                <div
                  v-else-if="selectedArtifact.kind === 'table'"
                  class="overflow-hidden rounded-[1.25rem] border border-slate-200 bg-white/88"
                >
                  <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-slate-200 text-left text-sm">
                      <thead class="bg-slate-50">
                        <tr>
                          <th
                            v-for="column in selectedTableColumns"
                            :key="column"
                            class="whitespace-nowrap px-4 py-3 text-xs font-semibold tracking-[0.12em] text-slate-500 uppercase"
                          >
                            {{ column }}
                          </th>
                        </tr>
                      </thead>
                      <tbody class="divide-y divide-slate-100">
                        <tr v-for="(row, rowIndex) in selectedTableRows" :key="rowIndex" class="hover:bg-slate-50/80">
                          <td
                            v-for="column in selectedTableColumns"
                            :key="column"
                            class="whitespace-nowrap px-4 py-3 text-slate-700"
                          >
                            {{ formatCell(row[column] ?? '') }}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                <div
                  v-else-if="selectedChart"
                  class="rounded-[1.25rem] border border-slate-200 bg-slate-50/80 p-4"
                >
                  <div class="mb-4 flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div class="text-base font-semibold tracking-[-0.03em] text-slate-950">{{ selectedChart.title }}</div>
                      <div class="text-xs font-semibold tracking-[0.14em] text-slate-500 uppercase">
                        {{ selectedChart.xLabel }} / {{ selectedChart.yLabel }}
                      </div>
                    </div>
                    <span class="rounded-full bg-slate-950 px-3 py-1 text-xs font-semibold text-white">{{ selectedChart.metric }}</span>
                  </div>

                  <svg
                    v-if="selectedChart.type === 'line'"
                    viewBox="0 0 280 188"
                    role="img"
                    :aria-label="selectedChart.title"
                    class="h-auto w-full rounded-[1rem] bg-white"
                  >
                    <path d="M24 164 H256 M24 164 V24" fill="none" stroke="#cbd5e1" stroke-width="1.5" />
                    <path d="M24 164 L256 24" fill="none" stroke="#cbd5e1" stroke-dasharray="5 5" stroke-width="1.25" />
                    <polyline
                      :points="chartPolyline(selectedChart.points)"
                      fill="none"
                      stroke="#0284c7"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="5"
                    />
                    <circle
                      v-for="point in selectedChart.points"
                      :key="`${point.x}-${point.y}`"
                      :cx="chartMarkerX(point)"
                      :cy="chartMarkerY(point)"
                      r="4"
                      fill="#0f172a"
                    />
                  </svg>

                  <div
                    v-else
                    class="grid h-72 grid-cols-3 items-end gap-4 rounded-[1rem] bg-white px-4 py-5"
                  >
                    <div
                      v-for="bar in selectedChart.bars"
                      :key="bar.label"
                      class="flex h-full min-w-0 flex-col justify-end gap-2"
                    >
                      <div class="flex min-h-0 flex-1 items-end">
                        <div
                          class="w-full rounded-t-xl bg-[linear-gradient(180deg,#0ea5e9,#14b8a6)] shadow-[0_14px_30px_rgba(14,165,233,0.18)]"
                          :style="{ height: barHeight(bar, selectedChart.bars) }"
                        />
                      </div>
                      <div class="text-center">
                        <div class="text-sm font-semibold text-slate-950">{{ bar.count }}</div>
                        <div class="truncate text-xs font-semibold tracking-[0.12em] text-slate-500 uppercase">{{ bar.label }}</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div
                  v-else
                  class="rounded-[1.25rem] border border-slate-200 bg-slate-950 p-4 font-mono text-xs leading-6 text-slate-100"
                >
                  <div v-for="line in selectedLogLines" :key="line">{{ line }}</div>
                </div>
              </div>
            </AppCard>
          </div>
        </div>

        <aside class="space-y-4 xl:sticky xl:top-24 xl:self-start">
          <AppCard class="border-slate-200/80 bg-white/86 !p-5">
            <div class="mb-4 flex items-center justify-between gap-3">
              <div>
                <div class="text-xs font-semibold tracking-[0.18em] text-slate-500 uppercase">Run Stages</div>
                <div class="mt-1 text-lg font-semibold tracking-[-0.03em] text-slate-950">
                  {{ selectedStageIndex }} / {{ appDemoStages.length }}
                </div>
              </div>
              <span class="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">Completed</span>
            </div>

            <div class="space-y-2">
              <button
                v-for="stage in appDemoStages"
                :key="stage.id"
                type="button"
                class="w-full rounded-[1rem] border px-4 py-3 text-left transition"
                :class="stageButtonClass(stage.id)"
                @click="selectedStageId = stage.id"
              >
                <div class="flex items-center justify-between gap-3">
                  <span class="text-sm font-semibold">{{ stage.label }}</span>
                  <span class="text-[11px] font-semibold tracking-[0.14em] uppercase">{{ stage.status }}</span>
                </div>
                <div class="mt-1 text-xs leading-5 text-slate-500">{{ stage.title }}</div>
              </button>
            </div>

            <div class="mt-5 rounded-[1rem] border border-slate-200 bg-slate-50/80 p-4">
              <div class="text-sm font-semibold text-slate-950">{{ selectedStage.summary }}</div>
              <ul class="mt-3 space-y-2 text-sm leading-6 text-slate-600">
                <li v-for="detail in selectedStage.details" :key="detail" class="flex gap-2">
                  <span class="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-500" />
                  <span>{{ detail }}</span>
                </li>
              </ul>
            </div>
          </AppCard>

          <AppCard class="border-slate-200/80 bg-white/86 !p-5">
            <div class="mb-4">
              <div class="text-xs font-semibold tracking-[0.18em] text-slate-500 uppercase">Workspace</div>
              <div class="mt-1 text-lg font-semibold tracking-[-0.03em] text-slate-950">{{ appDemoInputFile.name }}</div>
              <div class="mt-2 flex flex-wrap gap-2 text-xs font-semibold text-slate-500">
                <span class="rounded-full border border-slate-200 bg-white px-3 py-1">{{ appDemoInputFile.rows }} rows</span>
                <span class="rounded-full border border-slate-200 bg-white px-3 py-1">{{ appDemoInputFile.columns.length }} columns</span>
              </div>
            </div>

            <div class="mb-4 grid grid-cols-2 gap-2">
              <div
                v-for="metric in appDemoMetrics"
                :key="metric.label"
                class="rounded-[1rem] border border-slate-200 bg-slate-50/70 p-3"
              >
                <div class="text-[11px] font-semibold tracking-[0.14em] text-slate-500 uppercase">{{ metric.label }}</div>
                <div class="mt-1 text-xl font-semibold tracking-[-0.04em] text-slate-950">{{ metric.value }}</div>
              </div>
            </div>

            <div class="space-y-2">
              <button
                v-for="artifact in appDemoArtifacts"
                :key="artifact.id"
                type="button"
                class="w-full rounded-[1rem] border px-4 py-3 text-left transition"
                :class="artifactButtonClass(artifact.id)"
                @click="selectedArtifactId = artifact.id"
              >
                <div class="flex items-center justify-between gap-3">
                  <span class="truncate text-sm font-semibold">{{ artifact.label }}</span>
                  <span
                    class="shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold tracking-[0.12em] uppercase"
                    :class="isCoreArtifact(artifact.id) ? 'bg-sky-100 text-sky-800' : 'bg-slate-100 text-slate-500'"
                  >
                    {{ artifact.kind }}
                  </span>
                </div>
                <div class="mt-1 truncate text-xs text-slate-500">{{ artifact.path }}</div>
              </button>
            </div>
          </AppCard>
        </aside>
      </div>
    </section>
  </AppShell>
</template>
