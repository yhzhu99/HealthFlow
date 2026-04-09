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
  benchmarks: SnapshotBenchmark[]
  runs: SnapshotRun[]
  questions: SnapshotQuestion[]
}

export interface DevEvaluationManifest {
  snapshotVersion: string
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

export interface BlindSlotMappingEntry {
  slot: string
  runId: string
  runLabel: string
  modelId: string
  backend: string | null
}

export interface EvaluationDraft {
  choice: string | null
  note: string
  updatedAt: string
}

export interface EvaluationResponse {
  questionId: string
  datasetId: BenchmarkId
  qid: string
  choice: string
  selectedBlindSlot: string | null
  selectedRunId: string | null
  selectedRunLabel: string | null
  selectedModelId: string | null
  selectedBackend: string | null
  blindMapping: BlindSlotMappingEntry[]
  note: string
  updatedAt: string
}

export interface EvaluationSessionState {
  sessionId: string
  activeBenchmarkId: BenchmarkId | null
  activeRunIdByDataset: Partial<Record<BenchmarkId, string>>
  activeCompareKeyByDataset: Partial<Record<BenchmarkId, string>>
  currentIndexByDataset: Partial<Record<BenchmarkId, number>>
  responses: Record<string, EvaluationResponse>
  drafts: Record<string, EvaluationDraft>
  lastParticipantName: string
}

export interface ExportedEvaluationResponse {
  questionId: string
  datasetId: BenchmarkId
  qid: string
  choice: string
  selectedBlindSlot: string | null
  selectedRunId: string | null
  selectedRunLabel: string | null
  selectedModelId: string | null
  selectedBackend: string | null
  blindMapping: BlindSlotMappingEntry[]
  note: string
  updatedAt: string
}

export interface ExportedEvaluationPayload {
  participant_name: string
  session_id: string
  snapshot_version: string
  exported_at: string
  responses: ExportedEvaluationResponse[]
}

export const BLIND_SLOTS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
export const EVALUATION_SESSION_STORAGE_KEY = 'healthflow:evaluation:session'
export const LEGACY_LAST_REVIEWER_KEY = 'healthflow:evaluation:last-reviewer'

export const legacyReviewerStorageKey = (reviewerId: string) => `healthflow:evaluation:${reviewerId.trim()}`

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

const normalizeBlindMappingEntry = (value: unknown): BlindSlotMappingEntry | null => {
  if (!isRecord(value)) return null

  const slot = normalizeNullableString(value.slot)
  const modelId = normalizeNullableString(value.modelId)
  if (!slot || !modelId) {
    return null
  }

  return {
    slot,
    runId: normalizeNullableString(value.runId) ?? modelId,
    runLabel: normalizeNullableString(value.runLabel) ?? normalizeNullableString(value.runId) ?? modelId,
    modelId,
    backend: normalizeNullableString(value.backend),
  }
}

const normalizeBlindMapping = (value: unknown): BlindSlotMappingEntry[] => {
  if (Array.isArray(value)) {
    return value.flatMap((item) => {
      const entry = normalizeBlindMappingEntry(item)
      return entry ? [entry] : []
    })
  }

  if (!isRecord(value)) {
    return []
  }

  return Object.entries(normalizeStringMap(value)).map(([slot, modelId]) => ({
    slot,
    runId: modelId,
    runLabel: modelId,
    modelId,
    backend: null,
  }))
}

const normalizeDraft = (value: unknown): EvaluationDraft | null => {
  if (!isRecord(value)) return null

  return {
    choice: normalizeNullableString(value.choice),
    note: typeof value.note === 'string' ? value.note : '',
    updatedAt: normalizeNullableString(value.updatedAt) ?? new Date(0).toISOString(),
  }
}

const normalizeDraftMap = (value: unknown): Record<string, EvaluationDraft> => {
  if (!isRecord(value)) return {}

  const normalizedEntries = Object.entries(value).flatMap(([questionId, draftValue]) => {
    const draft = normalizeDraft(draftValue)
    return draft ? [[questionId, draft] as const] : []
  })

  return Object.fromEntries(normalizedEntries)
}

const normalizeEvaluationResponse = (questionId: string, value: unknown): EvaluationResponse | null => {
  if (!isRecord(value)) return null

  const datasetId = normalizeNullableString(value.datasetId)
  const qid = normalizeNullableString(value.qid)
  const choice = normalizeNullableString(value.choice)
  const selectedModelId = normalizeNullableString(value.selectedModelId)

  if (!datasetId || !qid || !choice) {
    return null
  }

  return {
    questionId: normalizeNullableString(value.questionId) ?? questionId,
    datasetId,
    qid,
    choice,
    selectedBlindSlot: normalizeNullableString(value.selectedBlindSlot) ?? normalizeNullableString(value.selectedSlot),
    selectedRunId: normalizeNullableString(value.selectedRunId) ?? (choice === 'none' ? null : selectedModelId),
    selectedRunLabel:
      normalizeNullableString(value.selectedRunLabel) ??
      normalizeNullableString(value.selectedRunId) ??
      (choice === 'none' ? null : selectedModelId),
    selectedModelId,
    selectedBackend: normalizeNullableString(value.selectedBackend),
    blindMapping: normalizeBlindMapping(value.blindMapping ?? value.slotMapping),
    note: typeof value.note === 'string' ? value.note : '',
    updatedAt: normalizeNullableString(value.updatedAt) ?? new Date(0).toISOString(),
  }
}

const normalizeResponseMap = (value: unknown): Record<string, EvaluationResponse> => {
  if (!isRecord(value)) return {}

  const normalizedEntries = Object.entries(value).flatMap(([questionId, responseValue]) => {
    const response = normalizeEvaluationResponse(questionId, responseValue)
    return response ? [[questionId, response] as const] : []
  })

  return Object.fromEntries(normalizedEntries)
}

const createSessionId = () => {
  const cryptoObject = globalThis.crypto
  if (cryptoObject?.randomUUID) {
    return cryptoObject.randomUUID()
  }

  return `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

export const resolveSessionId = (sessionId: unknown) =>
  typeof sessionId === 'string' && sessionId.trim() ? sessionId.trim() : createSessionId()

export const createEvaluationSessionState = (sessionId?: string): EvaluationSessionState => ({
  sessionId: resolveSessionId(sessionId),
  activeBenchmarkId: null,
  activeRunIdByDataset: {},
  activeCompareKeyByDataset: {},
  currentIndexByDataset: {},
  responses: {},
  drafts: {},
  lastParticipantName: '',
})

export const restoreEvaluationSessionState = (value: unknown, preferredSessionId?: unknown): EvaluationSessionState => {
  const normalizedPreferredSessionId = normalizeNullableString(preferredSessionId)
  const normalizedStoredSessionId = isRecord(value) ? normalizeNullableString(value.sessionId) : null
  const sessionId = resolveSessionId(normalizedPreferredSessionId ?? normalizedStoredSessionId)
  const restoredState = createEvaluationSessionState(sessionId)

  if (!isRecord(value)) {
    return restoredState
  }

  restoredState.activeBenchmarkId = normalizeNullableString(value.activeBenchmarkId)
  restoredState.activeRunIdByDataset = normalizeStringMap(value.activeRunIdByDataset)
  restoredState.activeCompareKeyByDataset = normalizeStringMap(value.activeCompareKeyByDataset)
  restoredState.currentIndexByDataset = normalizeIndexMap(value.currentIndexByDataset)
  restoredState.responses = normalizeResponseMap(value.responses)
  restoredState.drafts = normalizeDraftMap(value.drafts)
  restoredState.lastParticipantName = normalizeNullableString(value.lastParticipantName) ?? ''

  return restoredState
}

export const restoreEvaluationSessionStateWithLegacySupport = ({
  sessionValue,
  legacyReviewerValue,
  legacyReviewerId,
}: {
  sessionValue: unknown
  legacyReviewerValue: unknown
  legacyReviewerId?: unknown
}): EvaluationSessionState => {
  if (isRecord(sessionValue)) {
    return restoreEvaluationSessionState(sessionValue)
  }

  const normalizedLegacyReviewerId =
    normalizeNullableString(legacyReviewerId) ??
    (isRecord(legacyReviewerValue) ? normalizeNullableString(legacyReviewerValue.reviewerId) : null)

  if (!isRecord(legacyReviewerValue)) {
    return restoreEvaluationSessionState(sessionValue, normalizedLegacyReviewerId)
  }

  const restoredLegacyState = restoreEvaluationSessionState({}, normalizedLegacyReviewerId)
  restoredLegacyState.activeBenchmarkId = normalizeNullableString(legacyReviewerValue.activeBenchmarkId)
  restoredLegacyState.activeRunIdByDataset = normalizeStringMap(legacyReviewerValue.activeRunIdByDataset)
  restoredLegacyState.currentIndexByDataset = normalizeIndexMap(legacyReviewerValue.currentIndexByDataset)
  restoredLegacyState.responses = normalizeResponseMap(legacyReviewerValue.responses)

  return restoredLegacyState
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
  sessionId: string,
  datasetId: BenchmarkId,
  qid: string,
): BlindCandidateSlot[] => {
  const sessionToken = resolveSessionId(sessionId).toLowerCase()

  return [...candidates]
    .sort((left, right) => {
      const leftKey = stableHash(`${sessionToken}:${datasetId}:${qid}:${left.modelId}`)
      const rightKey = stableHash(`${sessionToken}:${datasetId}:${qid}:${right.modelId}`)
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

export const buildBlindMapping = (slots: BlindCandidateSlot[]): BlindSlotMappingEntry[] =>
  slots.map((item) => ({
    slot: item.slot,
    runId: item.candidate.runId,
    runLabel: item.candidate.runLabel,
    modelId: item.candidate.modelId,
    backend: item.candidate.backend ?? null,
  }))

export const buildEvaluationExportPayload = ({
  participantName,
  snapshotVersion,
  exportedAt = new Date().toISOString(),
  sessionState,
  questions,
}: {
  participantName: string
  snapshotVersion: string
  exportedAt?: string
  sessionState: EvaluationSessionState
  questions: EvaluationQuestionSummary[]
}): ExportedEvaluationPayload => {
  const normalizedParticipantName = participantName.trim()
  if (!normalizedParticipantName) {
    throw new Error('Participant name is required for export.')
  }

  const orderedResponses = questions
    .map((question) => sessionState.responses[question.id])
    .filter((item): item is EvaluationResponse => Boolean(item))

  const missingResponseCount = questions.length - orderedResponses.length
  if (missingResponseCount > 0) {
    throw new Error(`All questions must be answered before export. ${missingResponseCount} pending.`)
  }

  return {
    participant_name: normalizedParticipantName,
    session_id: sessionState.sessionId,
    snapshot_version: snapshotVersion,
    exported_at: exportedAt,
    responses: orderedResponses.map((response) => ({
      questionId: response.questionId,
      datasetId: response.datasetId,
      qid: response.qid,
      choice: response.choice === 'none' ? 'none' : response.selectedBlindSlot ?? response.choice,
      selectedBlindSlot: response.choice === 'none' ? null : response.selectedBlindSlot,
      selectedRunId: response.selectedRunId,
      selectedRunLabel: response.selectedRunLabel,
      selectedModelId: response.selectedModelId,
      selectedBackend: response.selectedBackend,
      blindMapping: response.blindMapping,
      note: response.note,
      updatedAt: response.updatedAt,
    })),
  }
}
