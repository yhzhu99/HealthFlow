<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import ArtifactViewer from '../components/evaluation/ArtifactViewer.vue'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'
import {
  blindOrderCandidates,
  buildBlindMapping,
  buildEvaluationExportPayload,
  EVALUATION_SESSION_STORAGE_KEY,
  LEGACY_LAST_REVIEWER_KEY,
  legacyReviewerStorageKey,
  type BenchmarkId,
  type DevEvaluationManifest,
  type DiagnosticEvaluationDiagnostics,
  type EvaluationQuestionSummary,
  type EvaluationResponse,
  type EvaluationSessionState,
  type EvaluationSnapshot,
  type SnapshotCandidate,
  type SnapshotQuestion,
  restoreEvaluationSessionStateWithLegacySupport,
} from '../domain/evaluation'
import { downloadJson } from '../lib/download'
import { renderMarkdown } from '../lib/markdown'
import {
  loadEvaluationSnapshot,
  loadLocalEvaluationCase,
  loadLocalEvaluationManifest,
  localEvaluationManifestUrl,
} from '../lib/snapshot'
import { readJson, writeJson } from '../lib/storage'

const TASK_PANEL_EXPANDED_KEY = 'healthflow:evaluation:task-panel-expanded'

type EvaluationSourceMode = 'booting' | 'live' | 'static' | 'diagnostic' | 'error'
type CaseLoadState = 'idle' | 'loading' | 'ready' | 'error'

type CompareTab =
  | {
      key: 'reference'
      label: 'Reference Foundset'
      kind: 'reference'
    }
  | {
      key: string
      label: string
      kind: 'candidate'
      slot: string
      candidate: SnapshotCandidate
    }

const isDevMode = import.meta.env.DEV

const snapshot = ref<EvaluationSnapshot | null>(null)
const localManifest = ref<DevEvaluationManifest | null>(null)
const currentQuestionDetail = ref<SnapshotQuestion | null>(null)
const caseLoadState = ref<CaseLoadState>('idle')
const caseLoadError = ref<string | null>(null)
const sourceMode = ref<EvaluationSourceMode>('booting')
const loadError = ref<string | null>(null)
const diagnostics = ref<DiagnosticEvaluationDiagnostics | null>(null)
const sourceWarnings = ref<string[]>([])
const localManifestUrlValue = localEvaluationManifestUrl()
let loadRequestId = 0
let caseRequestId = 0

const resolveStoredBoolean = (value: unknown, fallback: boolean) => (typeof value === 'boolean' ? value : fallback)
const compactMarkdown = (value: string) =>
  value
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/[#>*_\-\|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const sessionState = ref<EvaluationSessionState | null>(null)
const draftChoice = ref<string | null>(null)
const draftNote = ref('')
const taskPanelExpanded = ref(resolveStoredBoolean(readJson<unknown>(TASK_PANEL_EXPANDED_KEY, true), true))
const exportModalOpen = ref(false)
const participantNameDraft = ref('')
const exportError = ref<string | null>(null)

const allBenchmarks = computed(() => (sourceMode.value === 'live' && isDevMode ? localManifest.value?.benchmarks : snapshot.value?.benchmarks) ?? [])

const allRuns = computed(() => (sourceMode.value === 'live' && isDevMode ? localManifest.value?.runs : snapshot.value?.runs) ?? [])

const allQuestionSummaries = computed<EvaluationQuestionSummary[]>(() => {
  if (sourceMode.value === 'live' && isDevMode) {
    return localManifest.value?.questions ?? []
  }

  return (snapshot.value?.questions ?? []).map((question) => ({
    caseId: question.id,
    id: question.id,
    datasetId: question.datasetId,
    datasetLabel: question.datasetLabel,
    qid: question.qid,
    taskBrief: question.taskBrief ?? null,
    taskType: question.taskType ?? null,
    paperTitle: question.paperTitle ?? null,
    candidateCount: question.candidates.length,
  }))
})

const benchmarks = computed(() => allBenchmarks.value)
const activeSnapshotVersion = computed(() => (sourceMode.value === 'live' && isDevMode ? localManifest.value?.snapshotVersion : snapshot.value?.snapshotVersion) ?? null)
const isWorkspaceReady = computed(() => {
  if (sourceMode.value === 'live' && isDevMode) return Boolean(localManifest.value)
  if (sourceMode.value === 'static') return Boolean(snapshot.value)
  return false
})
const sourceBadgeLabel = computed(() => (sourceMode.value === 'static' ? 'Build Bundle' : 'Local Cases'))
const sourceDescription = computed(() =>
  sourceMode.value === 'static'
    ? 'Serving the build-generated evaluation bundle created from platform/evaluation-data.'
    : 'Reading a lightweight evaluation manifest first, then loading each case on demand from platform/evaluation-data.',
)

const activeBenchmarkId = computed<BenchmarkId | null>(() => {
  if (sessionState.value?.activeBenchmarkId) return sessionState.value.activeBenchmarkId
  return (benchmarks.value[0]?.id as BenchmarkId | undefined) ?? null
})

const benchmarkQuestions = computed(() => {
  if (!activeBenchmarkId.value) return []
  return allQuestionSummaries.value.filter((question) => question.datasetId === activeBenchmarkId.value)
})

const benchmarkRuns = computed(() => {
  if (!activeBenchmarkId.value) return []
  return allRuns.value.filter((run) => run.datasetId === activeBenchmarkId.value)
})

const answeredIds = computed(() => {
  const questionIds = new Set(benchmarkQuestions.value.map((question) => question.id))
  return new Set(Object.keys(sessionState.value?.responses ?? {}).filter((questionId) => questionIds.has(questionId)))
})

const currentQuestionIndex = computed(() => {
  if (!sessionState.value || !activeBenchmarkId.value) return 0
  return sessionState.value.currentIndexByDataset[activeBenchmarkId.value] ?? 0
})

const currentQuestionSummary = computed(() => benchmarkQuestions.value[currentQuestionIndex.value] ?? null)
const currentQuestion = computed(() =>
  sourceMode.value === 'live' && isDevMode
    ? currentQuestionDetail.value
    : (snapshot.value?.questions.find((question) => question.id === currentQuestionSummary.value?.id) ?? null),
)

const currentQuestionLabel = computed(() => currentQuestion.value?.datasetLabel ?? currentQuestionSummary.value?.datasetLabel ?? '')
const currentQuestionQid = computed(() => currentQuestion.value?.qid ?? currentQuestionSummary.value?.qid ?? '')
const currentTaskType = computed(() => currentQuestion.value?.taskType ?? currentQuestionSummary.value?.taskType ?? null)
const currentTaskBrief = computed(() => currentQuestion.value?.taskBrief ?? currentQuestionSummary.value?.taskBrief ?? null)
const currentPaperTitle = computed(() => currentQuestion.value?.paperTitle ?? currentQuestionSummary.value?.paperTitle ?? null)

const progressPercent = computed(() =>
  benchmarkQuestions.value.length ? (answeredIds.value.size / benchmarkQuestions.value.length) * 100 : 0,
)

const candidateSlots = computed(() => {
  if (!currentQuestion.value || !sessionState.value) return []

  return blindOrderCandidates(
    currentQuestion.value.candidates,
    sessionState.value.sessionId,
    currentQuestion.value.datasetId,
    currentQuestion.value.qid,
  )
})

const submissionCandidates = computed(() => candidateSlots.value)

const activeRunId = computed(() => {
  if (!sessionState.value || !activeBenchmarkId.value) return null
  return sessionState.value.activeRunIdByDataset[activeBenchmarkId.value] ?? benchmarkRuns.value[0]?.id ?? null
})

const compareTabs = computed<CompareTab[]>(() => {
  if (!currentQuestion.value) return []

  return [
    {
      key: 'reference',
      label: 'Reference Foundset',
      kind: 'reference',
    },
    ...submissionCandidates.value.map((item) => ({
      key: item.candidate.runId,
      label: `Submission ${item.slot}`,
      kind: 'candidate' as const,
      slot: item.slot,
      candidate: item.candidate,
    })),
  ]
})

const activeCompareKey = computed(() => {
  if (!activeBenchmarkId.value || !sessionState.value) return null
  const storedKey = sessionState.value.activeCompareKeyByDataset[activeBenchmarkId.value]
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

const activeCandidate = computed(() =>
  activeCompareTab.value?.kind === 'candidate' ? activeCompareTab.value.candidate : null,
)

const currentSelectionLabel = computed(() => {
  if (draftChoice.value === 'none') return 'No Acceptable Submission'
  const selectedSlot = submissionCandidates.value.find((item) => item.candidate.runId === draftChoice.value)?.slot
  return selectedSlot ? `Submission ${selectedSlot}` : 'Unselected'
})

const renderedTask = computed(() => renderMarkdown(currentQuestion.value?.task ?? ''))
const renderedReferenceText = computed(() => renderMarkdown(currentQuestion.value?.reference.text ?? ''))
const renderedActiveAnswer = computed(() =>
  renderMarkdown(activeCandidate.value?.answerText || 'No final answer was recorded.'),
)
const taskPreviewText = computed(() => {
  const compactTask = compactMarkdown(currentQuestion.value?.task ?? '')
  if (!compactTask) return 'Open the task to review the full prompt.'
  return compactTask.length > 240 ? `${compactTask.slice(0, 237).trimEnd()}...` : compactTask
})
const taskSupportSummary = computed(() => {
  if (!currentQuestion.value) return []

  const summary = [
    `${currentQuestion.value.expectedOutputs.length} expected output${currentQuestion.value.expectedOutputs.length === 1 ? '' : 's'}`,
  ]

  if (currentQuestion.value.reportRequirements.length) {
    summary.push(
      `${currentQuestion.value.reportRequirements.length} requirement${currentQuestion.value.reportRequirements.length === 1 ? '' : 's'}`,
    )
  }

  if (currentQuestion.value.reference.requiredOutputs.length) {
    summary.push(
      `${currentQuestion.value.reference.requiredOutputs.length} reference deliverable${currentQuestion.value.reference.requiredOutputs.length === 1 ? '' : 's'}`,
    )
  }

  return summary
})

const hasSavedResponses = computed(() => answeredIds.value.size > 0)

const toggleTaskPanel = () => {
  taskPanelExpanded.value = !taskPanelExpanded.value
}

const ensureActiveBenchmark = () => {
  if (!sessionState.value) return
  if (sessionState.value.activeBenchmarkId && benchmarks.value.some((benchmark) => benchmark.id === sessionState.value?.activeBenchmarkId)) {
    return
  }

  sessionState.value.activeBenchmarkId = benchmarks.value[0]?.id ?? null
}

const ensureCurrentIndex = () => {
  if (!sessionState.value || !activeBenchmarkId.value || benchmarkQuestions.value.length === 0) return
  const storedIndex = sessionState.value.currentIndexByDataset[activeBenchmarkId.value]
  if (storedIndex != null && storedIndex < benchmarkQuestions.value.length) return

  const firstUnansweredIndex = benchmarkQuestions.value.findIndex((question) => !sessionState.value?.responses[question.id])
  sessionState.value.currentIndexByDataset[activeBenchmarkId.value] = firstUnansweredIndex >= 0 ? firstUnansweredIndex : 0
}

const ensureActiveRun = () => {
  if (!sessionState.value || !activeBenchmarkId.value) return
  const storedRunId = sessionState.value.activeRunIdByDataset[activeBenchmarkId.value]
  if (storedRunId && benchmarkRuns.value.some((run) => run.id === storedRunId)) return

  const firstRunId = benchmarkRuns.value[0]?.id
  if (firstRunId) {
    sessionState.value.activeRunIdByDataset[activeBenchmarkId.value] = firstRunId
  }
}

const ensureActiveCompareTab = () => {
  if (!activeBenchmarkId.value || compareTabs.value.length === 0 || !sessionState.value) return
  const currentKey = sessionState.value.activeCompareKeyByDataset[activeBenchmarkId.value]
  if (currentKey && compareTabs.value.some((tab) => tab.key === currentKey)) return

  if (activeRunId.value && compareTabs.value.some((tab) => tab.key === activeRunId.value)) {
    sessionState.value.activeCompareKeyByDataset[activeBenchmarkId.value] = activeRunId.value
    return
  }

  sessionState.value.activeCompareKeyByDataset[activeBenchmarkId.value] = compareTabs.value[0]?.key ?? 'reference'
}

const persistSessionState = () => {
  if (!sessionState.value) return
  writeJson(EVALUATION_SESSION_STORAGE_KEY, sessionState.value)
}

const resolveRunIdFromChoice = (choice: string | null | undefined) => {
  if (!choice || !currentQuestion.value) return null
  if (choice === 'none') return 'none'

  if (currentQuestion.value.candidates.some((candidate) => candidate.runId === choice)) {
    return choice
  }

  const matchedByModel = currentQuestion.value.candidates.find((candidate) => candidate.modelId === choice)
  if (matchedByModel) {
    return matchedByModel.runId
  }

  const matchedBySlot = candidateSlots.value.find((item) => item.slot === choice)
  if (matchedBySlot) {
    return matchedBySlot.candidate.runId
  }

  return null
}

const resolveSavedRunId = (saved: EvaluationResponse | undefined) => {
  if (!saved || !currentQuestion.value) return null
  if (saved.choice === 'none') return 'none'
  return resolveRunIdFromChoice(saved.selectedRunId ?? saved.selectedModelId ?? saved.selectedBlindSlot ?? saved.choice)
}

const persistDraft = () => {
  if (!currentQuestion.value || !sessionState.value) return

  if (!draftChoice.value && !draftNote.value.trim()) {
    delete sessionState.value.drafts[currentQuestion.value.id]
    persistSessionState()
    return
  }

  sessionState.value.drafts[currentQuestion.value.id] = {
    choice: draftChoice.value,
    note: draftNote.value,
    updatedAt: new Date().toISOString(),
  }
  persistSessionState()
}

const hydrateDraft = () => {
  if (!currentQuestion.value || !sessionState.value) {
    draftChoice.value = null
    draftNote.value = ''
    return
  }

  const savedDraft = sessionState.value.drafts[currentQuestion.value.id]
  const savedResponse = sessionState.value.responses[currentQuestion.value.id]
  const restoredDraftChoice = resolveRunIdFromChoice(savedDraft?.choice)
  const restoredSavedChoice = resolveSavedRunId(savedResponse)

  draftChoice.value = restoredDraftChoice ?? restoredSavedChoice
  draftNote.value = savedDraft?.note ?? savedResponse?.note ?? ''

  if (draftChoice.value && draftChoice.value !== 'none' && activeBenchmarkId.value) {
    sessionState.value.activeRunIdByDataset[activeBenchmarkId.value] = draftChoice.value
  }
}

const activateSession = () => {
  const legacyReviewerId = readJson<unknown>(LEGACY_LAST_REVIEWER_KEY, null)
  const normalizedLegacyReviewerId =
    typeof legacyReviewerId === 'string' && legacyReviewerId.trim() ? legacyReviewerId.trim() : null
  const legacyState = normalizedLegacyReviewerId
    ? readJson<unknown>(legacyReviewerStorageKey(normalizedLegacyReviewerId), null)
    : null

  sessionState.value = restoreEvaluationSessionStateWithLegacySupport({
    sessionValue: readJson<unknown>(EVALUATION_SESSION_STORAGE_KEY, null),
    legacyReviewerValue: legacyState,
    legacyReviewerId: normalizedLegacyReviewerId,
  })

  ensureActiveBenchmark()
  ensureCurrentIndex()
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
  persistSessionState()
}

const saveCurrentResponse = () => {
  if (!currentQuestion.value || !sessionState.value || !draftChoice.value) return false

  const selectedRunId = draftChoice.value === 'none' ? null : draftChoice.value
  const selectedSlot = selectedRunId
    ? candidateSlots.value.find((item) => item.candidate.runId === selectedRunId)?.slot ?? null
    : null
  const selectedCandidate = selectedRunId
    ? currentQuestion.value.candidates.find((candidate) => candidate.runId === selectedRunId) ?? null
    : null
  const now = new Date().toISOString()

  sessionState.value.responses[currentQuestion.value.id] = {
    questionId: currentQuestion.value.id,
    datasetId: currentQuestion.value.datasetId,
    qid: currentQuestion.value.qid,
    choice: draftChoice.value,
    selectedBlindSlot: selectedSlot,
    selectedRunId,
    selectedRunLabel: selectedCandidate?.runLabel ?? null,
    selectedModelId: selectedCandidate?.modelId ?? null,
    selectedBackend: selectedCandidate?.backend ?? null,
    blindMapping: buildBlindMapping(candidateSlots.value),
    note: draftNote.value.trim(),
    updatedAt: now,
  }
  sessionState.value.drafts[currentQuestion.value.id] = {
    choice: draftChoice.value,
    note: draftNote.value,
    updatedAt: now,
  }
  persistSessionState()
  return true
}

const setActiveBenchmark = (benchmarkId: BenchmarkId) => {
  if (!sessionState.value) return
  saveCurrentResponse()
  sessionState.value.activeBenchmarkId = benchmarkId
  ensureCurrentIndex()
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
  persistSessionState()
}

const setActiveRun = (runId: string) => {
  if (!sessionState.value || !activeBenchmarkId.value) return
  sessionState.value.activeRunIdByDataset[activeBenchmarkId.value] = runId
  persistSessionState()
}

const setActiveCompareTab = (key: string) => {
  if (!activeBenchmarkId.value || !sessionState.value) return
  sessionState.value.activeCompareKeyByDataset[activeBenchmarkId.value] = key
  if (key !== 'reference') {
    setActiveRun(key)
  } else {
    persistSessionState()
  }
}

const selectQuestion = (index: number) => {
  if (!sessionState.value || !activeBenchmarkId.value) return
  saveCurrentResponse()
  sessionState.value.currentIndexByDataset[activeBenchmarkId.value] = index
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
  persistSessionState()
}

const goToRelativeQuestion = (delta: number) => {
  if (!sessionState.value || !activeBenchmarkId.value || benchmarkQuestions.value.length === 0) return
  const nextIndex = Math.max(0, Math.min(benchmarkQuestions.value.length - 1, currentQuestionIndex.value + delta))
  if (nextIndex === currentQuestionIndex.value) return
  selectQuestion(nextIndex)
}

const openExportModal = () => {
  if (!sessionState.value) return
  participantNameDraft.value = sessionState.value.lastParticipantName
  exportError.value = null
  exportModalOpen.value = true
}

const closeExportModal = () => {
  exportModalOpen.value = false
  exportError.value = null
}

const exportResponses = () => {
  if (!sessionState.value || !activeSnapshotVersion.value) return

  const participantName = participantNameDraft.value.trim()
  if (!participantName) {
    exportError.value = 'Name is required before export.'
    return
  }

  sessionState.value.lastParticipantName = participantName
  persistSessionState()

  const payload = buildEvaluationExportPayload({
    participantName,
    snapshotVersion: activeSnapshotVersion.value,
    sessionState: sessionState.value,
    questions: allQuestionSummaries.value,
  })

  const filenameStem = participantName.replace(/[^a-z0-9]+/gi, '-').replace(/(^-|-$)/g, '') || 'evaluation'
  downloadJson(`${filenameStem}_evaluation.json`, payload)
  closeExportModal()
}

const loadCurrentQuestion = async () => {
  if (!(sourceMode.value === 'live' && isDevMode) || !currentQuestionSummary.value) {
    currentQuestionDetail.value = null
    caseLoadState.value = 'idle'
    caseLoadError.value = null
    return
  }

  const requestId = ++caseRequestId
  caseLoadState.value = 'loading'
  caseLoadError.value = null
  currentQuestionDetail.value = null
  draftChoice.value = null
  draftNote.value = ''

  try {
    const question = await loadLocalEvaluationCase(currentQuestionSummary.value.datasetId, currentQuestionSummary.value.caseId)
    if (requestId !== caseRequestId) return
    currentQuestionDetail.value = question
    caseLoadState.value = 'ready'
    ensureActiveRun()
    hydrateDraft()
    ensureActiveCompareTab()
  } catch (caughtError) {
    if (requestId !== caseRequestId) return
    currentQuestionDetail.value = null
    caseLoadState.value = 'error'
    caseLoadError.value = caughtError instanceof Error ? caughtError.message : String(caughtError)
  }
}

const reloadCurrentQuestion = async () => {
  await loadCurrentQuestion()
}

watch([activeBenchmarkId, benchmarkQuestions], () => {
  ensureActiveBenchmark()
  ensureCurrentIndex()
  ensureActiveRun()
  ensureActiveCompareTab()
})

watch(currentQuestion, () => {
  ensureActiveRun()
  hydrateDraft()
  ensureActiveCompareTab()
})

watch(submissionCandidates, () => {
  ensureActiveRun()
  ensureActiveCompareTab()
})

watch(taskPanelExpanded, (expanded) => {
  writeJson(TASK_PANEL_EXPANDED_KEY, expanded)
})

watch(
  () => currentQuestionSummary.value?.id,
  async () => {
    if (sourceMode.value === 'live' && isDevMode) {
      await loadCurrentQuestion()
    } else {
      hydrateDraft()
    }
  },
)

watch([draftChoice, draftNote, () => currentQuestion.value?.id], () => {
  persistDraft()
})

const applySnapshot = (loadedSnapshot: EvaluationSnapshot) => {
  snapshot.value = loadedSnapshot
  localManifest.value = null
  currentQuestionDetail.value = null
  caseLoadState.value = 'idle'
  caseLoadError.value = null
  sourceMode.value = 'static'
  activateSession()
}

const applyLocalManifest = (manifest: DevEvaluationManifest) => {
  snapshot.value = null
  localManifest.value = manifest
  currentQuestionDetail.value = null
  caseLoadState.value = 'idle'
  caseLoadError.value = null
  sourceMode.value = 'live'
  activateSession()
}

const loadSnapshot = async () => {
  const requestId = ++loadRequestId
  sourceMode.value = 'booting'
  loadError.value = null
  diagnostics.value = null
  sourceWarnings.value = []
  caseLoadState.value = 'idle'
  caseLoadError.value = null

  try {
    if (isDevMode) {
      const payload = await loadLocalEvaluationManifest()
      if (requestId !== loadRequestId) return

      if (payload.mode === 'live') {
        diagnostics.value = null
        sourceWarnings.value = payload.diagnostics.warnings
        applyLocalManifest(payload.manifest)
        return
      }

      snapshot.value = null
      localManifest.value = null
      currentQuestionDetail.value = null
      diagnostics.value = payload.diagnostics
      sourceWarnings.value = payload.diagnostics.warnings
      sourceMode.value = 'diagnostic'
      return
    }

    const loadedSnapshot = await loadEvaluationSnapshot()
    if (requestId !== loadRequestId) return

    if (!loadedSnapshot) {
      throw new Error('No build-generated evaluation snapshot was found. Rebuild the app from the local evaluation-data tree.')
    }

    diagnostics.value = null
    sourceWarnings.value = []
    applySnapshot(loadedSnapshot)
  } catch (caughtError) {
    if (requestId !== loadRequestId) return
    snapshot.value = null
    localManifest.value = null
    currentQuestionDetail.value = null
    loadError.value = caughtError instanceof Error ? caughtError.message : String(caughtError)
    sourceMode.value = 'error'
  }
}

onMounted(async () => {
  await loadSnapshot()
})
</script>

<template>
  <AppShell content-width="wide">
    <div v-if="sourceMode === 'booting'" class="pt-4 sm:pt-5">
      <AppCard class="!p-4">
        <div class="space-y-4 py-4 text-center">
          <div class="text-lg font-semibold text-slate-950">Loading evaluation workspace...</div>
          <p class="text-sm leading-7 text-slate-500">
            <span v-if="isDevMode">
              Resolving benchmark cases from
              <code class="rounded bg-slate-100 px-2 py-1 text-[12px]">{{ localManifestUrlValue }}</code>
            </span>
            <span v-else>Loading the build-generated evaluation bundle.</span>
          </p>
        </div>
      </AppCard>
    </div>

    <div v-else-if="sourceMode === 'error'" class="pt-4 sm:pt-5">
      <AppCard class="!p-4">
        <div class="space-y-4">
          <div class="text-lg font-semibold text-rose-700">Evaluation runtime failed</div>
          <p class="text-sm leading-7 text-rose-700">{{ loadError }}</p>
          <div class="flex flex-wrap gap-2">
            <AppButton @click="loadSnapshot">Reload Evaluation Data</AppButton>
          </div>
        </div>
      </AppCard>
    </div>

    <div v-else-if="sourceMode === 'diagnostic'" class="space-y-4 pt-4 sm:pt-5">
      <AppCard class="!p-4">
        <div class="space-y-4">
          <div class="text-lg font-semibold text-slate-950">Local evaluation data not ready</div>
          <p class="text-base leading-8 text-slate-600">
            The dev server is looking for case directories under
            <code class="rounded bg-slate-100 px-2 py-1 text-sm">{{ diagnostics?.root }}</code>.
            Fix the listed issues and reload the local evaluation workspace.
          </p>
          <div class="grid gap-3 lg:grid-cols-3">
            <div class="rounded-[1.4rem] border border-slate-200 bg-slate-50/90 p-4">
              <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Missing</div>
              <ul class="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                <li v-if="!diagnostics?.missing.length" class="text-slate-500">No missing paths reported.</li>
                <li v-for="item in diagnostics?.missing ?? []" :key="`missing-${item}`">{{ item }}</li>
              </ul>
            </div>
            <div class="rounded-[1.4rem] border border-slate-200 bg-slate-50/90 p-4">
              <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Invalid</div>
              <ul class="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                <li v-if="!diagnostics?.invalid.length" class="text-slate-500">No invalid manifests reported.</li>
                <li v-for="item in diagnostics?.invalid ?? []" :key="`invalid-${item}`">{{ item }}</li>
              </ul>
            </div>
            <div class="rounded-[1.4rem] border border-slate-200 bg-slate-50/90 p-4">
              <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Warnings</div>
              <ul class="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                <li v-if="!diagnostics?.warnings.length" class="text-slate-500">No warnings reported.</li>
                <li v-for="item in diagnostics?.warnings ?? []" :key="`warning-${item}`">{{ item }}</li>
              </ul>
            </div>
          </div>
          <div class="flex flex-wrap gap-2">
            <AppButton @click="loadSnapshot">Reload Local Data</AppButton>
          </div>
        </div>
      </AppCard>
    </div>

    <div v-else-if="isWorkspaceReady" class="space-y-3 pt-3 sm:pt-4">
      <AppCard class="!p-3 border-slate-200/80 bg-white/88">
        <div class="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
          <div class="flex flex-wrap items-center gap-2">
            <div
              class="rounded-full border px-3 py-1 text-[10px] font-semibold tracking-[0.18em] uppercase"
              :class="
                sourceMode === 'static'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                  : 'border-sky-200 bg-sky-50 text-sky-700'
              "
            >
              {{ sourceBadgeLabel }}
            </div>
            <div class="text-sm text-slate-500">{{ sourceDescription }}</div>

            <button
              v-for="benchmark in benchmarks"
              :key="benchmark.id"
              type="button"
              class="rounded-full px-3 py-1.5 text-sm font-semibold transition"
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
            <div class="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs leading-6 text-slate-600">
              Saved locally in this browser
            </div>
            <AppButton v-if="isDevMode" variant="ghost" @click="loadSnapshot">Reload Local Data</AppButton>
            <AppButton :disabled="!hasSavedResponses" @click="openExportModal">Export JSON</AppButton>
          </div>
        </div>
      </AppCard>

      <AppCard v-if="sourceWarnings.length" class="!p-3 border-amber-200/80 bg-amber-50/80">
        <div class="space-y-2">
          <div class="text-[11px] font-semibold tracking-[0.16em] text-amber-700 uppercase">Local data warnings</div>
          <ul class="space-y-1.5 text-sm leading-6 text-amber-900">
            <li v-for="item in sourceWarnings" :key="item">{{ item }}</li>
          </ul>
        </div>
      </AppCard>

      <template v-if="sessionState && currentQuestionSummary">
        <div class="grid gap-3 xl:grid-cols-[136px_minmax(0,1fr)_280px] 2xl:grid-cols-[144px_minmax(0,1fr)_292px]">
          <AppCard class="!p-3 border-slate-200/80 bg-white/88 xl:sticky xl:top-20 xl:self-start">
            <div class="space-y-3">
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

                <div class="grid grid-cols-3 gap-1.5">
                  <button
                    v-for="(question, index) in benchmarkQuestions"
                    :key="question.id"
                    type="button"
                    class="rounded-xl px-0 py-2 text-sm font-semibold transition"
                    :class="
                      question.id === currentQuestionSummary.id
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

          <AppCard class="!p-3 border-slate-200/80 bg-white/88">
            <div class="space-y-3">
              <div class="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                <span class="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-900">
                  {{ currentQuestionLabel }} · Q{{ currentQuestionQid }}
                </span>
                <span v-if="currentTaskType">{{ currentTaskType }}</span>
                <span v-if="currentTaskBrief">{{ currentTaskBrief }}</span>
                <span v-if="currentPaperTitle">{{ currentPaperTitle }}</span>
              </div>

              <div v-if="currentQuestion" class="space-y-3">
                <div class="rounded-[1.25rem] border border-slate-200 bg-slate-50/80 p-3">
                  <div class="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                    <div class="space-y-2">
                      <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Task</div>
                      <p class="text-sm leading-7 text-slate-600">
                        {{
                          taskPanelExpanded
                            ? 'The full prompt is visible below.'
                            : taskPreviewText
                        }}
                      </p>
                      <div class="flex flex-wrap gap-2">
                        <span
                          v-for="item in taskSupportSummary"
                          :key="item"
                          class="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] text-slate-600"
                        >
                          {{ item }}
                        </span>
                      </div>
                    </div>

                    <AppButton
                      variant="secondary"
                      :aria-expanded="taskPanelExpanded"
                      :aria-label="taskPanelExpanded ? 'Collapse task details' : 'Expand task details'"
                      @click="toggleTaskPanel"
                    >
                      {{ taskPanelExpanded ? 'Collapse Task' : 'Expand Task' }}
                    </AppButton>
                  </div>

                  <div
                    v-if="taskPanelExpanded"
                    class="mt-4 grid gap-3 xl:grid-cols-[minmax(0,1fr)_240px] 2xl:grid-cols-[minmax(0,1fr)_256px]"
                  >
                    <div class="prose prose-slate max-w-none" v-html="renderedTask" />

                    <div class="space-y-2.5">
                      <div class="rounded-[1.2rem] border border-slate-200 bg-white p-3">
                        <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Expected Outputs</div>
                        <div class="mt-3 flex flex-wrap gap-2">
                          <template v-if="currentQuestion.expectedOutputs.length">
                            <div
                              v-for="item in currentQuestion.expectedOutputs"
                              :key="`${currentQuestion.id}-${item.fileName}`"
                              class="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[11px] text-slate-600"
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
                        class="rounded-[1.2rem] border border-slate-200 bg-white p-3"
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
                </div>
              </div>

              <div
                v-else-if="caseLoadState === 'loading'"
                class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 px-5 py-8 text-sm leading-7 text-slate-600"
              >
                Loading Q{{ currentQuestionQid }} materials and submission outputs...
              </div>

              <div
                v-else-if="caseLoadState === 'error'"
                class="rounded-[1.35rem] border border-rose-200 bg-rose-50 px-5 py-6 text-sm leading-7 text-rose-700"
              >
                <div>{{ caseLoadError }}</div>
                <div class="mt-4">
                  <AppButton variant="secondary" @click="reloadCurrentQuestion">Reload This Case</AppButton>
                </div>
              </div>

              <div class="flex flex-wrap items-center justify-between gap-2">
                <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Compare</div>
                <div class="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                  {{ submissionCandidates.length }} submissions + reference foundset
                </div>
              </div>

              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="tab in compareTabs"
                  :key="tab.key"
                  type="button"
                  class="rounded-full border px-3 py-1.5 text-sm font-semibold transition"
                  :class="
                    activeCompareKey === tab.key
                      ? tab.kind === 'reference'
                        ? 'border-amber-700 bg-amber-600 text-white shadow-sm'
                        : 'border-slate-950 bg-slate-950 text-white shadow-sm'
                      : tab.kind === 'reference'
                        ? 'border-amber-200 bg-amber-50 text-amber-800 hover:border-amber-300 hover:bg-amber-100'
                        : 'border border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300 hover:bg-white'
                  "
                  @click="setActiveCompareTab(tab.key)"
                >
                  {{ tab.label }}
                </button>
              </div>

              <template v-if="currentQuestion && activeCompareTab">
                <div
                  v-if="activeCompareTab.kind === 'candidate' && activeCandidate"
                  class="rounded-[1.25rem] border border-slate-200 bg-slate-50/80 p-3"
                >
                  <div class="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span class="rounded-full bg-white px-3 py-1 font-semibold text-slate-900">
                      {{ activeCompareTab.label }}
                    </span>
                    <span
                      v-if="draftChoice === activeCandidate.runId"
                      class="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-emerald-700"
                    >
                      Selected
                    </span>
                  </div>
                  <p class="mt-3 text-sm leading-7 text-slate-600">
                    {{ activeCandidate.summary ?? 'No summary recorded for this submission.' }}
                  </p>
                </div>

                <div
                  v-else
                  class="rounded-[1.25rem] border border-amber-200 bg-amber-50/90 p-3"
                >
                  <div class="flex flex-wrap items-center gap-2 text-xs text-amber-800">
                    <span class="rounded-full bg-white px-3 py-1 font-semibold text-amber-900">Reference Foundset</span>
                    <span class="rounded-full border border-amber-200 bg-white px-3 py-1">
                      {{ currentQuestion.reference.mode }}
                    </span>
                  </div>
                  <p v-if="currentQuestion.reference.note" class="mt-3 text-sm leading-7 text-amber-900/80">
                    {{ currentQuestion.reference.note }}
                  </p>
                </div>

                <div v-if="activeCompareTab.kind === 'candidate' && activeCandidate" class="space-y-4">
                  <div
                    class="prose prose-slate max-w-none rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3"
                    v-html="renderedActiveAnswer"
                  />

                  <ArtifactViewer
                    :artifacts="activeCandidate.artifacts"
                    :title="`${activeCompareTab.label} Files`"
                  />
                </div>

                <div v-else class="space-y-4">
                  <div
                    v-if="currentQuestion.reference.text"
                    class="prose prose-slate max-w-none rounded-[1.25rem] border border-amber-200 bg-white px-4 py-3"
                    v-html="renderedReferenceText"
                  />

                  <div
                    v-if="currentQuestion.reference.requiredOutputs.length"
                    class="rounded-[1.25rem] border border-amber-200 bg-amber-50 p-3"
                  >
                    <div class="text-[11px] font-semibold tracking-[0.16em] text-amber-700 uppercase">Reference Deliverables</div>
                    <ul class="mt-3 space-y-2 text-sm leading-6 text-amber-900">
                      <li
                        v-for="item in currentQuestion.reference.requiredOutputs"
                        :key="`${currentQuestion.id}-reference-${item.fileName}`"
                        class="flex gap-3"
                      >
                        <span class="mt-2 h-1.5 w-1.5 rounded-full bg-amber-700" />
                        <span>{{ item.fileName }} <span class="text-amber-700/70">({{ item.mediaType }})</span></span>
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

          <AppCard class="!p-3 border-slate-200/80 bg-white/88 xl:sticky xl:top-20 xl:self-start">
            <div class="space-y-3">
              <div class="flex items-center justify-between gap-3">
                <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Decision</div>
                <div class="text-xs font-semibold text-slate-900">{{ currentSelectionLabel }}</div>
              </div>

              <div class="flex flex-wrap gap-2">
                <div class="rounded-[1rem] border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs leading-6 text-slate-600">
                  Progress <span class="font-semibold text-slate-900">{{ answeredIds.size }}/{{ benchmarkQuestions.length }}</span>
                </div>
                <div class="rounded-[1rem] border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs leading-6 text-slate-600">
                  All edits persist locally
                </div>
              </div>

              <div v-if="currentQuestion" class="space-y-2">
                <button
                  v-for="item in submissionCandidates"
                  :key="item.candidate.runId"
                  type="button"
                  class="w-full rounded-[1rem] border px-3 py-2 text-left transition"
                  :class="
                    draftChoice === item.candidate.runId
                      ? 'border-slate-950 bg-slate-950 text-white shadow-sm'
                      : 'border border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50'
                  "
                  @click="draftChoice = item.candidate.runId"
                >
                  <div class="flex items-center justify-between gap-2">
                    <span class="text-sm font-semibold">Submission {{ item.slot }}</span>
                    <span class="text-[11px] opacity-75">{{ item.candidate.artifacts.length }} files</span>
                  </div>
                </button>

                <button
                  type="button"
                  class="w-full rounded-[1rem] border px-3 py-2 text-left text-sm font-semibold transition"
                  :class="
                    draftChoice === 'none'
                      ? 'border-rose-600 bg-rose-600 text-white shadow-sm'
                      : 'border border-rose-200 bg-rose-50 text-rose-700 hover:border-rose-300'
                  "
                  @click="draftChoice = 'none'"
                >
                  No Acceptable Submission
                </button>
              </div>

              <div
                v-else-if="caseLoadState === 'loading'"
                class="rounded-[1rem] border border-slate-200 bg-slate-50 px-4 py-5 text-sm leading-7 text-slate-500"
              >
                Waiting for this case to load before voting.
              </div>

              <div
                v-else-if="caseLoadState === 'error'"
                class="rounded-[1rem] border border-rose-200 bg-rose-50 px-4 py-5 text-sm leading-7 text-rose-700"
              >
                Unable to load the active case.
              </div>

              <textarea
                v-model="draftNote"
                rows="5"
                placeholder="Quick note"
                :disabled="!currentQuestion"
                class="w-full rounded-[1rem] border border-slate-200 bg-white px-3 py-3 text-sm leading-7 text-slate-900 outline-none transition focus:border-slate-950"
              />

              <div class="grid gap-2">
                <AppButton :disabled="!currentQuestion || !draftChoice" @click="saveCurrentResponse">Save</AppButton>
                <AppButton
                  variant="secondary"
                  :disabled="!currentQuestion || !draftChoice"
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

    <div
      v-if="exportModalOpen"
      class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-4 py-8 backdrop-blur-sm"
      @click.self="closeExportModal"
    >
      <AppCard class="w-full max-w-md border-slate-200/90 bg-white !p-5 shadow-2xl">
        <div class="space-y-4">
          <div>
            <div class="text-[11px] font-semibold tracking-[0.16em] text-slate-500 uppercase">Export JSON</div>
            <div class="mt-2 text-lg font-semibold text-slate-950">Attach your name to this evaluation export</div>
            <p class="mt-2 text-sm leading-7 text-slate-600">
              The downloaded JSON will keep both the blinded submission labels and the de-anonymized framework mapping.
            </p>
          </div>

          <div class="space-y-2">
            <label class="text-sm font-semibold text-slate-900" for="participant-name">Name</label>
            <input
              id="participant-name"
              v-model="participantNameDraft"
              type="text"
              placeholder="Your name"
              class="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-950"
            />
          </div>

          <div v-if="exportError" class="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {{ exportError }}
          </div>

          <div class="flex justify-end gap-2">
            <AppButton variant="ghost" @click="closeExportModal">Cancel</AppButton>
            <AppButton @click="exportResponses">Download JSON</AppButton>
          </div>
        </div>
      </AppCard>
    </div>
  </AppShell>
</template>
