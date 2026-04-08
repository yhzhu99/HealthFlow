<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import ArtifactViewer from '../components/evaluation/ArtifactViewer.vue'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'
import {
  blindOrderCandidates,
  createReviewerState,
  DEFAULT_REVIEWER_ID,
  evaluationStorageKey,
  resolveReviewerId,
  type BenchmarkId,
  type EvaluationSnapshot,
  type ReviewerResponse,
  type ReviewerState,
  type SnapshotCandidate,
} from '../domain/evaluation'
import { downloadJson } from '../lib/download'
import { renderMarkdown } from '../lib/markdown'
import { loadEvaluationSnapshot } from '../lib/snapshot'
import { readJson, writeJson } from '../lib/storage'

const LAST_REVIEWER_KEY = 'healthflow:evaluation:last-reviewer'

type CompareTab =
  | {
      key: 'reference'
      label: 'Reference'
      kind: 'reference'
    }
  | {
      key: string
      label: string
      kind: 'framework'
      candidate: SnapshotCandidate
    }

const snapshot = ref<EvaluationSnapshot | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)

const reviewerDraft = ref(resolveReviewerId(readJson<string>(LAST_REVIEWER_KEY, DEFAULT_REVIEWER_ID)))
const reviewerState = ref<ReviewerState | null>(null)
const draftChoice = ref<string | null>(null)
const draftNote = ref('')
const activeCompareKeyByDataset = ref<Partial<Record<BenchmarkId, string>>>({})

const benchmarks = computed(() => snapshot.value?.benchmarks ?? [])

const activeBenchmarkId = computed<BenchmarkId | null>(() => {
  if (reviewerState.value?.activeBenchmarkId) return reviewerState.value.activeBenchmarkId
  return (benchmarks.value[0]?.id as BenchmarkId | undefined) ?? null
})

const activeBenchmark = computed(() => benchmarks.value.find((benchmark) => benchmark.id === activeBenchmarkId.value) ?? null)

const benchmarkQuestions = computed(() => {
  if (!snapshot.value || !activeBenchmarkId.value) return []
  return snapshot.value.questions.filter((question) => question.datasetId === activeBenchmarkId.value)
})

const benchmarkRuns = computed(() => {
  if (!snapshot.value || !activeBenchmarkId.value) return []
  return snapshot.value.runs.filter((run) => run.datasetId === activeBenchmarkId.value)
})

const answeredIds = computed(() => {
  const questionIds = new Set(benchmarkQuestions.value.map((question) => question.id))
  return new Set(
    Object.keys(reviewerState.value?.responses ?? {}).filter((questionId) => questionIds.has(questionId)),
  )
})

const currentQuestionIndex = computed(() => {
  if (!reviewerState.value || !activeBenchmarkId.value) return 0
  return reviewerState.value.currentIndexByDataset[activeBenchmarkId.value] ?? 0
})

const currentQuestion = computed(() => benchmarkQuestions.value[currentQuestionIndex.value] ?? null)

const progressPercent = computed(() =>
  benchmarkQuestions.value.length ? (answeredIds.value.size / benchmarkQuestions.value.length) * 100 : 0,
)

const candidateSlots = computed(() => {
  if (!currentQuestion.value || !reviewerState.value) return []
  return blindOrderCandidates(
    currentQuestion.value.candidates,
    reviewerState.value.reviewerId,
    currentQuestion.value.datasetId,
    currentQuestion.value.qid,
  )
})

const frameworkCandidates = computed(() => {
  if (!currentQuestion.value) return []

  const runOrder = new Map(benchmarkRuns.value.map((run, index) => [run.id, index]))
  const slotByRunId = new Map(candidateSlots.value.map((item) => [item.candidate.runId, item.slot]))

  return [...currentQuestion.value.candidates]
    .sort((left, right) => {
      const leftOrder = runOrder.get(left.runId) ?? Number.MAX_SAFE_INTEGER
      const rightOrder = runOrder.get(right.runId) ?? Number.MAX_SAFE_INTEGER
      return leftOrder - rightOrder
    })
    .map((candidate) => ({
      candidate,
      slot: slotByRunId.get(candidate.runId) ?? null,
    }))
})

const activeRunId = computed(() => {
  if (!reviewerState.value || !activeBenchmarkId.value) return null
  return reviewerState.value.activeRunIdByDataset[activeBenchmarkId.value] ?? frameworkCandidates.value[0]?.candidate.runId ?? null
})

const compareTabs = computed<CompareTab[]>(() => {
  if (!currentQuestion.value) return []

  return [
    {
      key: 'reference',
      label: 'Reference',
      kind: 'reference',
    },
    ...frameworkCandidates.value.map((item) => ({
      key: item.candidate.runId,
      label: item.candidate.runLabel,
      kind: 'framework' as const,
      candidate: item.candidate,
    })),
  ]
})

const activeCompareKey = computed(() => {
  if (!activeBenchmarkId.value) return null
  const storedKey = activeCompareKeyByDataset.value[activeBenchmarkId.value]
  if (storedKey && compareTabs.value.some((tab) => tab.key === storedKey)) {
    return storedKey
  }

  if (activeRunId.value && compareTabs.value.some((tab) => tab.key === activeRunId.value)) {
    return activeRunId.value
  }

  return compareTabs.value[0]?.key ?? null
})

const activeCompareTab = computed(
  () => compareTabs.value.find((tab) => tab.key === activeCompareKey.value) ?? compareTabs.value[0] ?? null,
)

const activeFrameworkCandidate = computed(() =>
  activeCompareTab.value?.kind === 'framework' ? activeCompareTab.value.candidate : null,
)

const currentSelectionLabel = computed(() => {
  if (draftChoice.value === 'none') return 'None / 都不好'
  return frameworkCandidates.value.find((item) => item.candidate.runId === draftChoice.value)?.candidate.runLabel ?? 'Unselected'
})

const renderedTask = computed(() => renderMarkdown(currentQuestion.value?.task ?? ''))
const renderedReferenceText = computed(() => renderMarkdown(currentQuestion.value?.reference.text ?? ''))
const renderedActiveAnswer = computed(() =>
  renderMarkdown(activeFrameworkCandidate.value?.answerText || 'No final answer was recorded.'),
)

const ensureCurrentIndex = () => {
  if (!reviewerState.value || !activeBenchmarkId.value || benchmarkQuestions.value.length === 0) return
  const storedIndex = reviewerState.value.currentIndexByDataset[activeBenchmarkId.value]
  if (storedIndex != null && storedIndex < benchmarkQuestions.value.length) return

  const firstUnansweredIndex = benchmarkQuestions.value.findIndex(
    (question) => !reviewerState.value?.responses[question.id],
  )
  reviewerState.value.currentIndexByDataset[activeBenchmarkId.value] =
    firstUnansweredIndex >= 0 ? firstUnansweredIndex : 0
}

const ensureActiveRun = () => {
  if (!reviewerState.value || !activeBenchmarkId.value) return
  const storedRunId = reviewerState.value.activeRunIdByDataset[activeBenchmarkId.value]
  if (storedRunId && frameworkCandidates.value.some((item) => item.candidate.runId === storedRunId)) return

  const firstRunId = frameworkCandidates.value[0]?.candidate.runId
  if (firstRunId) {
    reviewerState.value.activeRunIdByDataset[activeBenchmarkId.value] = firstRunId
  }
}

const ensureActiveCompareTab = () => {
  if (!activeBenchmarkId.value || compareTabs.value.length === 0) return
  const currentKey = activeCompareKeyByDataset.value[activeBenchmarkId.value]
  if (currentKey && compareTabs.value.some((tab) => tab.key === currentKey)) return

  if (activeRunId.value && compareTabs.value.some((tab) => tab.key === activeRunId.value)) {
    activeCompareKeyByDataset.value[activeBenchmarkId.value] = activeRunId.value
    return
  }

  activeCompareKeyByDataset.value[activeBenchmarkId.value] = compareTabs.value[0]?.key ?? 'reference'
}

const persistReviewerState = () => {
  if (!reviewerState.value) return
  writeJson(evaluationStorageKey(reviewerState.value.reviewerId), reviewerState.value)
  writeJson(LAST_REVIEWER_KEY, reviewerState.value.reviewerId)
}

const resolveSavedRunId = (saved: ReviewerResponse | undefined) => {
  if (!saved || !currentQuestion.value) return null
  if (saved.choice === 'none') return 'none'

  if (saved.selectedRunId && currentQuestion.value.candidates.some((candidate) => candidate.runId === saved.selectedRunId)) {
    return saved.selectedRunId
  }

  if (saved.selectedModelId) {
    const matchedByModel = currentQuestion.value.candidates.find((candidate) => candidate.modelId === saved.selectedModelId)
    if (matchedByModel) return matchedByModel.runId
  }

  if (saved.selectedSlot) {
    const matchedBySlot = candidateSlots.value.find((item) => item.slot === saved.selectedSlot)
    if (matchedBySlot) return matchedBySlot.candidate.runId
  }

  return null
}

const hydrateDraft = () => {
  if (!currentQuestion.value || !reviewerState.value) {
    draftChoice.value = null
    draftNote.value = ''
    return
  }

  const saved = reviewerState.value.responses[currentQuestion.value.id]
  draftChoice.value = resolveSavedRunId(saved)
  draftNote.value = saved?.note ?? ''

  if (draftChoice.value && draftChoice.value !== 'none' && activeBenchmarkId.value) {
    reviewerState.value.activeRunIdByDataset[activeBenchmarkId.value] = draftChoice.value
  }
}

const activateReviewer = (requestedId?: string) => {
  const reviewerId = resolveReviewerId(requestedId ?? reviewerDraft.value)
  reviewerDraft.value = reviewerId
  reviewerState.value =
    readJson<ReviewerState | null>(evaluationStorageKey(reviewerId), null) ?? createReviewerState(reviewerId)

  if (!reviewerState.value.activeBenchmarkId && benchmarks.value[0]) {
    reviewerState.value.activeBenchmarkId = benchmarks.value[0].id
  }

  ensureCurrentIndex()
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
  persistReviewerState()
}

const saveCurrentResponse = () => {
  if (!currentQuestion.value || !reviewerState.value || !draftChoice.value) return false
  const slotMapping = Object.fromEntries(candidateSlots.value.map((item) => [item.slot, item.candidate.modelId]))
  const selectedRunId = draftChoice.value === 'none' ? null : draftChoice.value
  const selectedSlot = selectedRunId
    ? candidateSlots.value.find((item) => item.candidate.runId === selectedRunId)?.slot ?? null
    : null
  const selectedCandidate = selectedRunId
    ? currentQuestion.value.candidates.find((candidate) => candidate.runId === selectedRunId) ?? null
    : null

  reviewerState.value.responses[currentQuestion.value.id] = {
    questionId: currentQuestion.value.id,
    datasetId: currentQuestion.value.datasetId,
    qid: currentQuestion.value.qid,
    choice: draftChoice.value,
    selectedSlot,
    selectedRunId,
    selectedRunLabel: selectedCandidate?.runLabel ?? null,
    selectedModelId: selectedCandidate?.modelId ?? null,
    slotMapping,
    note: draftNote.value.trim(),
    updatedAt: new Date().toISOString(),
  }
  persistReviewerState()
  return true
}

const setActiveBenchmark = (benchmarkId: BenchmarkId) => {
  if (!reviewerState.value) return
  saveCurrentResponse()
  reviewerState.value.activeBenchmarkId = benchmarkId
  ensureCurrentIndex()
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
  persistReviewerState()
}

const setActiveRun = (runId: string) => {
  if (!reviewerState.value || !activeBenchmarkId.value) return
  reviewerState.value.activeRunIdByDataset[activeBenchmarkId.value] = runId
  persistReviewerState()
}

const setActiveCompareTab = (key: string) => {
  if (!activeBenchmarkId.value) return
  activeCompareKeyByDataset.value[activeBenchmarkId.value] = key
  if (key !== 'reference') {
    setActiveRun(key)
  }
}

const selectQuestion = (index: number) => {
  if (!reviewerState.value || !activeBenchmarkId.value) return
  saveCurrentResponse()
  reviewerState.value.currentIndexByDataset[activeBenchmarkId.value] = index
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
  persistReviewerState()
}

const goToRelativeQuestion = (delta: number) => {
  if (!reviewerState.value || !activeBenchmarkId.value) return
  const nextIndex = Math.max(0, Math.min(benchmarkQuestions.value.length - 1, currentQuestionIndex.value + delta))
  if (nextIndex === currentQuestionIndex.value) return
  selectQuestion(nextIndex)
}

const exportResponses = () => {
  if (!reviewerState.value || !snapshot.value) return
  const orderedResponses = snapshot.value.questions
    .map((question) => reviewerState.value?.responses[question.id])
    .filter((item): item is NonNullable<typeof item> => Boolean(item))

  downloadJson(`${reviewerState.value.reviewerId}_evaluation.json`, {
    reviewer_id: reviewerState.value.reviewerId,
    snapshot_version: snapshot.value.snapshotVersion,
    exported_at: new Date().toISOString(),
    responses: orderedResponses,
  })
}

watch([activeBenchmarkId, benchmarkQuestions], () => {
  ensureCurrentIndex()
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
})

watch(currentQuestion, () => {
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
})

watch(frameworkCandidates, () => {
  ensureActiveRun()
  ensureActiveCompareTab()
})

onMounted(async () => {
  try {
    snapshot.value = await loadEvaluationSnapshot()
    if (snapshot.value) {
      activateReviewer(reviewerDraft.value)
    }
  } catch (caughtError) {
    loadError.value = caughtError instanceof Error ? caughtError.message : String(caughtError)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <AppShell content-width="wide">
    <div v-if="loading" class="pt-4 sm:pt-5">
      <AppCard class="!p-4">
        <div class="py-8 text-center text-slate-500">Loading evaluation snapshot...</div>
      </AppCard>
    </div>

    <div v-else-if="loadError" class="pt-4 sm:pt-5">
      <AppCard class="!p-4">
        <div class="space-y-3">
          <div class="text-lg font-semibold text-rose-700">Snapshot load failed</div>
          <p class="text-sm leading-7 text-rose-700">{{ loadError }}</p>
        </div>
      </AppCard>
    </div>

    <div v-else-if="!snapshot" class="space-y-4 pt-4 sm:pt-5">
      <AppCard class="!p-4">
        <div class="space-y-4">
          <div class="text-lg font-semibold text-slate-950">No snapshot found yet</div>
          <p class="text-base leading-8 text-slate-600">
            Build one from local benchmark outputs after configuring
            <code class="rounded bg-slate-100 px-2 py-1 text-sm">platform/content/evaluation.config.json</code>.
          </p>
          <pre class="overflow-x-auto rounded-[1.5rem] bg-slate-950 px-4 py-5 text-sm leading-7 text-slate-100"><code>cd platform
npm run snapshot
npm run dev</code></pre>
        </div>
      </AppCard>
    </div>

    <div v-else class="space-y-4 pt-4 sm:pt-5">
      <AppCard class="!p-4 border-slate-200/80 bg-white/86">
        <div class="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div class="flex flex-wrap items-center gap-2">
            <div class="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-[11px] font-semibold tracking-[0.18em] text-sky-700 uppercase">
              Human Evaluation
            </div>

            <button
              v-for="benchmark in benchmarks"
              :key="benchmark.id"
              type="button"
              class="rounded-full px-4 py-2 text-sm font-semibold transition"
              :class="
                activeBenchmarkId === benchmark.id
                  ? 'border border-slate-900 bg-slate-950 text-white shadow-sm'
                  : 'border border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-950'
              "
              @click="setActiveBenchmark(benchmark.id)"
            >
              {{ benchmark.label }}
              <span class="ml-1.5 text-[11px] opacity-70">{{ benchmark.taskCount }}</span>
            </button>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <input
              v-model="reviewerDraft"
              type="text"
              placeholder="reviewer_id"
              class="min-w-[176px] rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-900 outline-none transition focus:border-slate-950"
            />
            <AppButton variant="secondary" @click="activateReviewer()">Switch Reviewer</AppButton>
            <AppButton @click="exportResponses">Export JSON</AppButton>
          </div>
        </div>

        <p v-if="activeBenchmark" class="mt-3 text-xs leading-6 text-slate-500">
          {{ activeBenchmark.description }}
        </p>
      </AppCard>

      <template v-if="reviewerState && currentQuestion">
        <div class="grid gap-4 xl:grid-cols-[176px_minmax(0,1fr)_300px] 2xl:grid-cols-[188px_minmax(0,1fr)_320px]">
          <AppCard class="!p-4 border-slate-200/80 bg-white/86 xl:sticky xl:top-20 xl:self-start">
            <div class="space-y-4">
              <div class="space-y-2">
                <div class="flex items-center justify-between text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">
                  <span>Progress</span>
                  <span>{{ answeredIds.size }}/{{ benchmarkQuestions.length }}</span>
                </div>
                <div class="h-2 rounded-full bg-slate-200">
                  <div class="h-full rounded-full bg-slate-950 transition-all" :style="{ width: `${progressPercent}%` }" />
                </div>
              </div>

              <div class="space-y-2">
                <div class="flex items-center justify-between text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">
                  <span>Questions</span>
                  <span>{{ currentQuestionIndex + 1 }}/{{ benchmarkQuestions.length }}</span>
                </div>

                <div class="grid grid-cols-4 gap-2">
                  <button
                    v-for="(question, index) in benchmarkQuestions"
                    :key="question.id"
                    type="button"
                    class="rounded-xl px-0 py-2.5 text-sm font-semibold transition"
                    :class="
                      question.id === currentQuestion.id
                        ? 'bg-slate-950 text-white'
                        : answeredIds.has(question.id)
                          ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 hover:bg-emerald-100'
                          : 'bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                    "
                    @click="selectQuestion(index)"
                  >
                    {{ index + 1 }}
                  </button>
                </div>
              </div>
            </div>
          </AppCard>

          <AppCard class="!p-4 border-slate-200/80 bg-white/86">
            <div class="space-y-4">
              <div class="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                <span class="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-900">
                  {{ currentQuestion.datasetLabel }} · Q{{ currentQuestion.qid }}
                </span>
                <span v-if="currentQuestion.taskType">{{ currentQuestion.taskType }}</span>
                <span v-if="currentQuestion.taskBrief">{{ currentQuestion.taskBrief }}</span>
                <span v-if="currentQuestion.paperTitle">{{ currentQuestion.paperTitle }}</span>
              </div>

              <div class="grid gap-4 xl:grid-cols-[minmax(0,1fr)_272px]">
                <div class="prose prose-slate max-w-none" v-html="renderedTask" />

                <div class="space-y-3">
                  <div class="rounded-[1.25rem] border border-slate-200 bg-slate-50 p-4">
                    <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Expected Outputs</div>
                    <div class="mt-3 flex flex-wrap gap-2">
                      <template v-if="currentQuestion.expectedOutputs.length">
                        <div
                          v-for="item in currentQuestion.expectedOutputs"
                          :key="`${currentQuestion.id}-${item.fileName}`"
                          class="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[11px] text-slate-600"
                        >
                          <span class="font-semibold text-slate-900">{{ item.fileName }}</span>
                          <span class="ml-1.5">{{ item.mediaType }}</span>
                        </div>
                      </template>
                      <div
                        v-else
                        class="rounded-full border border-dashed border-slate-200 px-3 py-1.5 text-[11px] text-slate-500"
                      >
                        No structured expected outputs
                      </div>
                    </div>
                  </div>

                  <div
                    v-if="currentQuestion.reportRequirements.length"
                    class="rounded-[1.25rem] border border-slate-200 bg-slate-50 p-4"
                  >
                    <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Report Requirements</div>
                    <ul class="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                      <li v-for="item in currentQuestion.reportRequirements" :key="item" class="flex gap-3">
                        <span class="mt-2 h-1.5 w-1.5 rounded-full bg-slate-900" />
                        <span>{{ item }}</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              <div class="flex flex-wrap items-center justify-between gap-3">
                <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Compare</div>
                <div class="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-600">
                  {{ frameworkCandidates.length }} frameworks + reference
                </div>
              </div>

              <div class="flex flex-wrap gap-2">
                <button
                  v-for="tab in compareTabs"
                  :key="tab.key"
                  type="button"
                  class="rounded-full border px-3 py-2 text-sm font-semibold transition"
                  :class="
                    activeCompareKey === tab.key
                      ? 'border-slate-950 bg-slate-950 text-white shadow-sm'
                      : 'border border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300 hover:bg-white'
                  "
                  @click="setActiveCompareTab(tab.key)"
                >
                  {{ tab.label }}
                </button>
              </div>

              <template v-if="activeCompareTab">
                <div
                  v-if="activeCompareTab.kind === 'framework' && activeFrameworkCandidate"
                  class="rounded-[1.4rem] border border-slate-200 bg-slate-50/80 p-4"
                >
                  <div class="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span class="rounded-full bg-white px-3 py-1 font-semibold text-slate-900">
                      {{ activeFrameworkCandidate.runLabel }}
                    </span>
                    <span class="rounded-full border border-slate-200 bg-white px-3 py-1">
                      {{ activeFrameworkCandidate.backend ?? 'snapshot' }}
                    </span>
                    <span
                      v-if="draftChoice === activeFrameworkCandidate.runId"
                      class="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-emerald-700"
                    >
                      Selected
                    </span>
                  </div>
                  <p class="mt-3 text-sm leading-7 text-slate-600">
                    {{ activeFrameworkCandidate.summary ?? 'No summary recorded for this framework output.' }}
                  </p>
                </div>

                <div
                  v-else
                  class="rounded-[1.4rem] border border-slate-200 bg-slate-50/80 p-4"
                >
                  <div class="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span class="rounded-full bg-white px-3 py-1 font-semibold text-slate-900">Reference</span>
                    <span class="rounded-full border border-slate-200 bg-white px-3 py-1">
                      {{ currentQuestion.reference.mode }}
                    </span>
                  </div>
                  <p v-if="currentQuestion.reference.note" class="mt-3 text-sm leading-7 text-slate-600">
                    {{ currentQuestion.reference.note }}
                  </p>
                </div>

                <div v-if="activeCompareTab.kind === 'framework' && activeFrameworkCandidate" class="space-y-4">
                  <div class="prose prose-slate max-w-none rounded-[1.4rem] border border-slate-200 bg-white px-5 py-4" v-html="renderedActiveAnswer" />

                  <ArtifactViewer
                    :artifacts="activeFrameworkCandidate.artifacts"
                    :title="`${activeFrameworkCandidate.runLabel} Artifacts`"
                  />
                </div>

                <div v-else class="space-y-4">
                  <div
                    v-if="currentQuestion.reference.text"
                    class="prose prose-slate max-w-none rounded-[1.4rem] border border-slate-200 bg-white px-5 py-4"
                    v-html="renderedReferenceText"
                  />

                  <div
                    v-if="currentQuestion.reference.requiredOutputs.length"
                    class="rounded-[1.4rem] border border-slate-200 bg-slate-50 p-4"
                  >
                    <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Reference Deliverables</div>
                    <ul class="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                      <li
                        v-for="item in currentQuestion.reference.requiredOutputs"
                        :key="`${currentQuestion.id}-reference-${item.fileName}`"
                        class="flex gap-3"
                      >
                        <span class="mt-2 h-1.5 w-1.5 rounded-full bg-slate-900" />
                        <span>{{ item.fileName }} <span class="text-slate-500">({{ item.mediaType }})</span></span>
                      </li>
                    </ul>
                  </div>

                  <ArtifactViewer
                    v-if="currentQuestion.reference.artifacts.length"
                    :artifacts="currentQuestion.reference.artifacts"
                    title="Reference Artifacts"
                  />
                </div>
              </template>
            </div>
          </AppCard>

          <AppCard class="!p-4 border-slate-200/80 bg-white/86 xl:sticky xl:top-20 xl:self-start">
            <div class="space-y-4">
              <div class="flex items-center justify-between gap-3">
                <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Decision</div>
                <div class="text-xs font-semibold text-slate-900">{{ currentSelectionLabel }}</div>
              </div>

              <div class="grid gap-2">
                <div class="rounded-[1rem] border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-6 text-slate-600">
                  Reviewer <span class="font-semibold text-slate-900">{{ reviewerState.reviewerId }}</span>
                </div>
                <div class="rounded-[1rem] border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-6 text-slate-600">
                  Progress <span class="font-semibold text-slate-900">{{ answeredIds.size }}/{{ benchmarkQuestions.length }}</span>
                </div>
              </div>

              <div class="space-y-2">
                <button
                  v-for="item in frameworkCandidates"
                  :key="item.candidate.runId"
                  type="button"
                  class="w-full rounded-[1rem] border px-3 py-2.5 text-left transition"
                  :class="
                    draftChoice === item.candidate.runId
                      ? 'border-slate-950 bg-slate-950 text-white shadow-sm'
                      : 'border border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50'
                  "
                  @click="draftChoice = item.candidate.runId"
                >
                  <div class="flex items-center justify-between gap-2">
                    <span class="text-sm font-semibold">{{ item.candidate.runLabel }}</span>
                    <span class="text-[11px] opacity-75">{{ item.candidate.artifacts.length }} files</span>
                  </div>
                </button>

                <button
                  type="button"
                  class="w-full rounded-[1rem] border px-3 py-2.5 text-left text-sm font-semibold transition"
                  :class="
                    draftChoice === 'none'
                      ? 'border-rose-600 bg-rose-600 text-white shadow-sm'
                      : 'border border-rose-200 bg-rose-50 text-rose-700 hover:border-rose-300'
                  "
                  @click="draftChoice = 'none'"
                >
                  None / 都不好
                </button>
              </div>

              <textarea
                v-model="draftNote"
                rows="6"
                placeholder="Optional note"
                class="w-full rounded-[1rem] border border-slate-200 bg-white px-3 py-3 text-sm leading-7 text-slate-900 outline-none transition focus:border-slate-950"
              />

              <div class="grid gap-2">
                <AppButton :disabled="!draftChoice" @click="saveCurrentResponse">Save</AppButton>
                <AppButton
                  variant="secondary"
                  :disabled="!draftChoice"
                  @click="saveCurrentResponse(); goToRelativeQuestion(1)"
                >
                  Save + Next
                </AppButton>
              </div>

              <div class="grid grid-cols-2 gap-2">
                <AppButton variant="ghost" @click="goToRelativeQuestion(-1)">Previous</AppButton>
                <AppButton variant="ghost" @click="goToRelativeQuestion(1)">Next</AppButton>
              </div>
            </div>
          </AppCard>
        </div>
      </template>
    </div>
  </AppShell>
</template>
