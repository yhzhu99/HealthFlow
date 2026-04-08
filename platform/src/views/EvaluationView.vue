<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import CandidateCard from '../components/evaluation/CandidateCard.vue'
import ArtifactViewer from '../components/evaluation/ArtifactViewer.vue'
import ProgressSummary from '../components/evaluation/ProgressSummary.vue'
import QuestionNavigator from '../components/evaluation/QuestionNavigator.vue'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'
import SectionHeader from '../components/ui/SectionHeader.vue'
import {
  blindOrderCandidates,
  createReviewerState,
  evaluationStorageKey,
  type BenchmarkId,
  type EvaluationSnapshot,
  type ReviewerState,
} from '../domain/evaluation'
import { downloadJson } from '../lib/download'
import { renderMarkdown } from '../lib/markdown'
import { loadEvaluationSnapshot } from '../lib/snapshot'
import { readJson, writeJson } from '../lib/storage'

const LAST_REVIEWER_KEY = 'healthflow:evaluation:last-reviewer'

const snapshot = ref<EvaluationSnapshot | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)

const reviewerDraft = ref(readJson<string>(LAST_REVIEWER_KEY, ''))
const reviewerState = ref<ReviewerState | null>(null)
const draftChoice = ref<string | null>(null)
const draftNote = ref('')

const benchmarks = computed(() => snapshot.value?.benchmarks ?? [])

const activeBenchmarkId = computed<BenchmarkId | null>(() => {
  if (reviewerState.value?.activeBenchmarkId) return reviewerState.value.activeBenchmarkId
  return (benchmarks.value[0]?.id as BenchmarkId | undefined) ?? null
})

const benchmarkQuestions = computed(() => {
  if (!snapshot.value || !activeBenchmarkId.value) return []
  return snapshot.value.questions.filter((question) => question.datasetId === activeBenchmarkId.value)
})

const answeredIds = computed(() => {
  const questionIds = new Set(benchmarkQuestions.value.map((question) => question.id))
  return new Set(
    Object.keys(reviewerState.value?.responses ?? {}).filter((questionId) => questionIds.has(questionId)),
  )
})

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

const currentQuestionIndex = computed(() => {
  if (!reviewerState.value || !activeBenchmarkId.value) return 0
  return reviewerState.value.currentIndexByDataset[activeBenchmarkId.value] ?? 0
})

const currentQuestion = computed(() => benchmarkQuestions.value[currentQuestionIndex.value] ?? null)

const blindCandidates = computed(() => {
  if (!currentQuestion.value || !reviewerState.value) return []
  return blindOrderCandidates(
    currentQuestion.value.candidates,
    reviewerState.value.reviewerId,
    currentQuestion.value.datasetId,
    currentQuestion.value.qid,
  )
})

const renderedTask = computed(() => renderMarkdown(currentQuestion.value?.task ?? ''))
const renderedReferenceText = computed(() => renderMarkdown(currentQuestion.value?.reference.text ?? ''))

const persistReviewerState = () => {
  if (!reviewerState.value) return
  writeJson(evaluationStorageKey(reviewerState.value.reviewerId), reviewerState.value)
  writeJson(LAST_REVIEWER_KEY, reviewerState.value.reviewerId)
}

const hydrateDraft = () => {
  if (!currentQuestion.value || !reviewerState.value) {
    draftChoice.value = null
    draftNote.value = ''
    return
  }

  const saved = reviewerState.value.responses[currentQuestion.value.id]
  draftChoice.value = saved?.choice ?? null
  draftNote.value = saved?.note ?? ''
}

const activateReviewer = () => {
  const reviewerId = reviewerDraft.value.trim()
  if (!reviewerId) return

  reviewerState.value =
    readJson<ReviewerState | null>(evaluationStorageKey(reviewerId), null) ?? createReviewerState(reviewerId)

  if (!reviewerState.value.activeBenchmarkId && benchmarks.value[0]) {
    reviewerState.value.activeBenchmarkId = benchmarks.value[0].id
  }

  ensureCurrentIndex()
  hydrateDraft()
  persistReviewerState()
}

const saveCurrentResponse = () => {
  if (!currentQuestion.value || !reviewerState.value || !draftChoice.value) return false
  const slotMapping = Object.fromEntries(blindCandidates.value.map((item) => [item.slot, item.candidate.modelId]))
  const selectedSlot = draftChoice.value === 'none' ? null : draftChoice.value

  reviewerState.value.responses[currentQuestion.value.id] = {
    questionId: currentQuestion.value.id,
    datasetId: currentQuestion.value.datasetId,
    qid: currentQuestion.value.qid,
    choice: draftChoice.value,
    selectedSlot,
    selectedModelId: selectedSlot ? slotMapping[selectedSlot] ?? null : null,
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
  hydrateDraft()
  persistReviewerState()
}

const selectQuestion = (index: number) => {
  if (!reviewerState.value || !activeBenchmarkId.value) return
  saveCurrentResponse()
  reviewerState.value.currentIndexByDataset[activeBenchmarkId.value] = index
  hydrateDraft()
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
  hydrateDraft()
})

watch(currentQuestion, () => {
  hydrateDraft()
})

onMounted(async () => {
  try {
    snapshot.value = await loadEvaluationSnapshot()
    if (reviewerDraft.value.trim() && snapshot.value) {
      activateReviewer()
    }
  } catch (caughtError) {
    loadError.value = caughtError instanceof Error ? caughtError.message : String(caughtError)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <AppShell>
    <section class="py-12 sm:py-16">
      <SectionHeader
        eyebrow="Evaluation"
        title="Blind reviewer workspace for 20 benchmark tasks."
        description="Review anonymous candidate outputs, inspect generated artifacts, and export your choices locally without exposing model identities inside the UI."
      />
    </section>

    <AppCard v-if="loading">
      <div class="py-10 text-center text-slate-500">Loading evaluation snapshot…</div>
    </AppCard>

    <AppCard v-else-if="loadError">
      <div class="space-y-3">
        <div class="text-lg font-semibold text-rose-700">Snapshot load failed</div>
        <p class="text-sm leading-7 text-rose-700">{{ loadError }}</p>
      </div>
    </AppCard>

    <div v-else-if="!snapshot" class="space-y-6">
      <AppCard>
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

    <div v-else class="space-y-8">
      <AppCard>
        <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div class="space-y-2">
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Reviewer</div>
            <p class="text-base leading-7 text-slate-600">
              Enter a stable reviewer id. Candidate slot order is deterministic per reviewer, dataset, and qid.
            </p>
          </div>
          <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
            <input
              v-model="reviewerDraft"
              type="text"
              placeholder="reviewer_id"
              class="rounded-full border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none ring-0 transition focus:border-slate-950"
            />
            <AppButton @click="activateReviewer">Load Reviewer</AppButton>
          </div>
        </div>
      </AppCard>

      <template v-if="reviewerState">
        <div class="flex flex-wrap gap-3">
          <button
            v-for="benchmark in benchmarks"
            :key="benchmark.id"
            type="button"
            class="rounded-full px-4 py-2 text-sm font-semibold transition"
            :class="
              activeBenchmarkId === benchmark.id
                ? 'bg-slate-950 text-white'
                : 'border border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-950'
            "
            @click="setActiveBenchmark(benchmark.id)"
          >
            {{ benchmark.label }}
            <span class="ml-2 text-xs opacity-70">
              {{
                benchmarkQuestions.filter((question) => question.datasetId === benchmark.id).length ||
                snapshot.questions.filter((question) => question.datasetId === benchmark.id).length
              }}
            </span>
          </button>
        </div>

        <div class="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
          <div class="space-y-6">
            <AppCard>
              <ProgressSummary :answered="answeredIds.size" :total="benchmarkQuestions.length" />
            </AppCard>
            <AppCard>
              <QuestionNavigator
                :questions="benchmarkQuestions"
                :current-question-id="currentQuestion?.id ?? null"
                :answered-ids="answeredIds"
                @select="selectQuestion"
              />
            </AppCard>
            <AppCard>
              <div class="space-y-4">
                <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Export</div>
                <p class="text-sm leading-7 text-slate-600">
                  Responses remain in browser local storage until you export them.
                </p>
                <AppButton variant="secondary" @click="exportResponses">Export JSON</AppButton>
              </div>
            </AppCard>
          </div>

          <div v-if="currentQuestion" class="space-y-6">
            <AppCard>
              <div class="flex flex-col gap-8 lg:flex-row lg:items-start lg:justify-between">
                <div class="max-w-3xl space-y-5">
                  <div class="flex flex-wrap items-center gap-3 text-sm text-slate-500">
                    <span class="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-900">
                      {{ currentQuestion.datasetLabel }} · Q{{ currentQuestion.qid }}
                    </span>
                    <span v-if="currentQuestion.taskType">{{ currentQuestion.taskType }}</span>
                    <span v-if="currentQuestion.taskBrief">{{ currentQuestion.taskBrief }}</span>
                  </div>
                  <div class="prose prose-slate max-w-none" v-html="renderedTask" />

                  <div v-if="currentQuestion.options" class="space-y-3 rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                    <div class="text-sm font-semibold text-slate-900">Options</div>
                    <ol class="space-y-2 text-sm leading-7 text-slate-700">
                      <li v-for="(value, key) in currentQuestion.options" :key="key"><strong>{{ key }}.</strong> {{ value }}</li>
                    </ol>
                  </div>

                  <div class="space-y-3 rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                    <div class="text-sm font-semibold text-slate-900">Expected Outputs</div>
                    <ul class="space-y-2 text-sm leading-7 text-slate-700">
                      <li v-for="item in currentQuestion.expectedOutputs" :key="`${currentQuestion.id}-${item.fileName}`">
                        <code class="rounded bg-white px-2 py-1 text-xs">{{ item.fileName }}</code>
                        <span class="ml-2 text-slate-500">{{ item.mediaType }}</span>
                      </li>
                      <li v-if="currentQuestion.expectedOutputs.length === 0">No structured expected outputs were recorded.</li>
                    </ul>
                  </div>
                </div>

                <div class="w-full max-w-sm space-y-4">
                  <div class="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                    <div class="text-sm font-semibold text-slate-900">Your choice</div>
                    <div class="mt-4 flex flex-wrap gap-2">
                      <button
                        v-for="slot in blindCandidates"
                        :key="slot.slot"
                        type="button"
                        class="rounded-full px-3 py-2 text-sm font-semibold transition"
                        :class="
                          draftChoice === slot.slot
                            ? 'bg-slate-950 text-white'
                            : 'border border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-950'
                        "
                        @click="draftChoice = slot.slot"
                      >
                        Candidate {{ slot.slot }}
                      </button>
                      <button
                        type="button"
                        class="rounded-full px-3 py-2 text-sm font-semibold transition"
                        :class="
                          draftChoice === 'none'
                            ? 'bg-rose-600 text-white'
                            : 'border border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-950'
                        "
                        @click="draftChoice = 'none'"
                      >
                        None / 都不好
                      </button>
                    </div>
                    <textarea
                      v-model="draftNote"
                      rows="4"
                      placeholder="Optional note"
                      class="mt-4 w-full rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3 text-sm leading-7 text-slate-900 outline-none transition focus:border-slate-950"
                    />
                    <div class="mt-4 flex flex-wrap gap-2">
                      <AppButton :disabled="!draftChoice" @click="saveCurrentResponse">Save</AppButton>
                      <AppButton variant="secondary" :disabled="!draftChoice" @click="saveCurrentResponse(); goToRelativeQuestion(1)">
                        Save & Next
                      </AppButton>
                    </div>
                    <div class="mt-4 flex gap-2">
                      <AppButton variant="ghost" @click="goToRelativeQuestion(-1)">Previous</AppButton>
                      <AppButton variant="ghost" @click="goToRelativeQuestion(1)">Next</AppButton>
                    </div>
                  </div>

                  <div class="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                    <div class="text-sm font-semibold text-slate-900">Reference</div>
                    <p v-if="currentQuestion.reference.note" class="mt-3 text-sm leading-7 text-slate-600">
                      {{ currentQuestion.reference.note }}
                    </p>
                    <div
                      v-if="currentQuestion.reference.text"
                      class="prose prose-slate mt-4 max-w-none rounded-[1.25rem] bg-white px-4 py-4"
                      v-html="renderedReferenceText"
                    />
                    <div v-if="currentQuestion.reference.artifacts.length" class="mt-4">
                      <ArtifactViewer :artifacts="currentQuestion.reference.artifacts" title="Reference Artifacts" />
                    </div>
                  </div>
                </div>
              </div>
            </AppCard>

            <div class="grid gap-5 xl:grid-cols-2">
              <CandidateCard
                v-for="slot in blindCandidates"
                :key="slot.slot"
                :slot="slot"
                :selected-choice="draftChoice"
                @select="draftChoice = $event"
              />
            </div>
          </div>
        </div>
      </template>
    </div>
  </AppShell>
</template>
