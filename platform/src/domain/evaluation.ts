export type BenchmarkId = string

export type SnapshotArtifactKind = 'markdown' | 'pdf' | 'image' | 'csv' | 'json' | 'text' | 'download'

export interface SnapshotExpectedOutput {
  fileName: string
  mediaType: string
  referencePath?: string | null
}

export interface SnapshotArtifact {
  id: string
  label: string
  kind: SnapshotArtifactKind
  mediaType: string
  relativePath: string
  originalPath: string
  previewPriority: number
}

export interface SnapshotReference {
  mode: 'none' | 'text' | 'artifacts' | 'manifest'
  text?: string | null
  note?: string | null
  artifacts: SnapshotArtifact[]
  requiredOutputs: SnapshotExpectedOutput[]
}

export interface SnapshotCandidate {
  id: string
  runId: string
  runLabel: string
  modelId: string
  backend?: string | null
  answerText: string
  summary?: string | null
  score?: number | null
  success: boolean
  reportPath?: string | null
  artifacts: SnapshotArtifact[]
}

export interface SnapshotQuestion {
  id: string
  datasetId: BenchmarkId
  datasetLabel: string
  qid: string
  task: string
  taskBrief?: string | null
  taskType?: string | null
  paperTitle?: string | null
  options?: Record<string, string> | null
  reportRequirements: string[]
  expectedOutputs: SnapshotExpectedOutput[]
  reference: SnapshotReference
  candidates: SnapshotCandidate[]
}

export interface SnapshotBenchmark {
  id: BenchmarkId
  label: string
  description: string
  taskCount: number
}

export interface EvaluationQuestionSummary {
  caseId: string
  id: string
  datasetId: BenchmarkId
  datasetLabel: string
  qid: string
  taskBrief?: string | null
  taskType?: string | null
  paperTitle?: string | null
  candidateCount: number
}

export interface SnapshotRun {
  id: string
  label: string
  modelId: string
  datasetId: BenchmarkId
}

export interface EvaluationSnapshot {
  snapshotVersion: string
  generatedAt: string
  site: {
    title: string
    reviewerExportKeyPrefix: string
  }
  benchmarks: SnapshotBenchmark[]
  runs: SnapshotRun[]
  questions: SnapshotQuestion[]
}

export interface DevEvaluationManifest {
  snapshotVersion: string
  generatedAt: string
  site: {
    title: string
    reviewerExportKeyPrefix: string
  }
  benchmarks: SnapshotBenchmark[]
  runs: SnapshotRun[]
  questions: EvaluationQuestionSummary[]
}

export interface EvaluationDiagnostics {
  root: string
  warnings: string[]
}

export interface LiveEvaluationDiagnostics extends EvaluationDiagnostics {}

export interface DiagnosticEvaluationDiagnostics extends EvaluationDiagnostics {
  missing: string[]
  invalid: string[]
}

export type DevEvaluationManifestPayload =
  | {
      mode: 'live'
      manifest: DevEvaluationManifest
      diagnostics: LiveEvaluationDiagnostics
    }
  | {
      mode: 'diagnostic'
      manifest: null
      diagnostics: DiagnosticEvaluationDiagnostics
    }

export interface DevEvaluationCasePayload {
  question: SnapshotQuestion
}

export interface BlindCandidateSlot {
  slot: string
  candidate: SnapshotCandidate
}

export interface ReviewerResponse {
  questionId: string
  datasetId: BenchmarkId
  qid: string
  choice: string
  selectedSlot: string | null
  selectedRunId?: string | null
  selectedRunLabel?: string | null
  selectedModelId: string | null
  slotMapping: Record<string, string>
  note: string
  updatedAt: string
}

export interface ReviewerState {
  reviewerId: string
  activeBenchmarkId: BenchmarkId | null
  activeRunIdByDataset: Partial<Record<BenchmarkId, string>>
  currentIndexByDataset: Partial<Record<BenchmarkId, number>>
  responses: Record<string, ReviewerResponse>
}

export const BLIND_SLOTS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
export const DEFAULT_REVIEWER_ID = 'demo-reviewer'

export const evaluationStorageKey = (reviewerId: string) => `healthflow:evaluation:${reviewerId.trim()}`

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const normalizeNullableString = (value: unknown) => {
  if (typeof value !== 'string') return null
  const normalizedValue = value.trim()
  return normalizedValue || null
}

const normalizeStringMap = (value: unknown): Record<string, string> => {
  if (!isRecord(value)) return {}

  const normalizedEntries = Object.entries(value).flatMap(([key, itemValue]) => {
    const normalizedValue = normalizeNullableString(itemValue)
    return normalizedValue ? [[key, normalizedValue] as const] : []
  })

  return Object.fromEntries(normalizedEntries)
}

const normalizeIndexMap = (value: unknown): Partial<Record<BenchmarkId, number>> => {
  if (!isRecord(value)) return {}

  const normalizedEntries = Object.entries(value).flatMap(([key, itemValue]) => {
    if (typeof itemValue !== 'number' || !Number.isInteger(itemValue) || itemValue < 0) return []
    return [[key, itemValue] as const]
  })

  return Object.fromEntries(normalizedEntries) as Partial<Record<BenchmarkId, number>>
}

const normalizeReviewerResponse = (questionId: string, value: unknown): ReviewerResponse | null => {
  if (!isRecord(value)) return null

  const datasetId = normalizeNullableString(value.datasetId)
  const qid = normalizeNullableString(value.qid)
  const choice = normalizeNullableString(value.choice)

  if (!datasetId || !qid || !choice) {
    return null
  }

  return {
    questionId: normalizeNullableString(value.questionId) ?? questionId,
    datasetId,
    qid,
    choice,
    selectedSlot: normalizeNullableString(value.selectedSlot),
    selectedRunId: normalizeNullableString(value.selectedRunId),
    selectedRunLabel: normalizeNullableString(value.selectedRunLabel),
    selectedModelId: normalizeNullableString(value.selectedModelId),
    slotMapping: normalizeStringMap(value.slotMapping),
    note: typeof value.note === 'string' ? value.note : '',
    updatedAt: normalizeNullableString(value.updatedAt) ?? new Date(0).toISOString(),
  }
}

const normalizeResponseMap = (value: unknown): Record<string, ReviewerResponse> => {
  if (!isRecord(value)) return {}

  const normalizedEntries = Object.entries(value).flatMap(([questionId, responseValue]) => {
    const response = normalizeReviewerResponse(questionId, responseValue)
    return response ? [[questionId, response] as const] : []
  })

  return Object.fromEntries(normalizedEntries)
}

export const resolveReviewerId = (reviewerId: unknown) =>
  typeof reviewerId === 'string' && reviewerId.trim() ? reviewerId.trim() : DEFAULT_REVIEWER_ID

export const createReviewerState = (reviewerId: string): ReviewerState => ({
  reviewerId: resolveReviewerId(reviewerId),
  activeBenchmarkId: null,
  activeRunIdByDataset: {},
  currentIndexByDataset: {},
  responses: {},
})

export const restoreReviewerState = (value: unknown, preferredReviewerId?: unknown): ReviewerState => {
  const normalizedPreferredReviewerId = normalizeNullableString(preferredReviewerId)
  const normalizedStoredReviewerId = isRecord(value) ? normalizeNullableString(value.reviewerId) : null
  const reviewerId = resolveReviewerId(normalizedPreferredReviewerId ?? normalizedStoredReviewerId)
  const restoredState = createReviewerState(reviewerId)

  if (!isRecord(value)) {
    return restoredState
  }

  restoredState.activeBenchmarkId = normalizeNullableString(value.activeBenchmarkId)
  restoredState.activeRunIdByDataset = normalizeStringMap(value.activeRunIdByDataset)
  restoredState.currentIndexByDataset = normalizeIndexMap(value.currentIndexByDataset)
  restoredState.responses = normalizeResponseMap(value.responses)

  return restoredState
}

export const stableHash = (value: string) => {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

export const blindOrderCandidates = (
  candidates: SnapshotCandidate[],
  reviewerId: string,
  datasetId: BenchmarkId,
  qid: string,
): BlindCandidateSlot[] => {
  const reviewerToken = reviewerId.trim().toLowerCase() || 'anonymous'

  return [...candidates]
    .sort((left, right) => {
      const leftKey = stableHash(`${reviewerToken}:${datasetId}:${qid}:${left.modelId}`)
      const rightKey = stableHash(`${reviewerToken}:${datasetId}:${qid}:${right.modelId}`)
      if (leftKey === rightKey) {
        return left.modelId.localeCompare(right.modelId)
      }
      return leftKey - rightKey
    })
    .map((candidate, index) => ({
      slot: BLIND_SLOTS[index] ?? `Slot-${index + 1}`,
      candidate,
    }))
}
