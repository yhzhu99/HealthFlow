export type BenchmarkId = 'medagentboard' | 'ehrflowbench'

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

export interface BlindCandidateSlot {
  slot: string
  candidate: SnapshotCandidate
}

export const BLIND_SLOTS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')

export const evaluationStorageKey = (reviewerId: string) => `healthflow:evaluation:${reviewerId.trim()}`

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
