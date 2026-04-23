import type {
  DevEvaluationCasePayload,
  DevEvaluationManifestPayload,
  EvaluationSnapshot,
  SnapshotQuestion,
  StaticEvaluationPayload,
} from '../domain/evaluation'
import { toBasePath } from './assets'

const DEFAULT_TIMEOUT_MS = 8000
const DEFAULT_RETRIES = 1
const DEFAULT_RETRY_DELAY_MS = 350
const DEFAULT_LOCAL_TIMEOUT_MS = 12000
const LOCAL_CASE_ROUTE_PREFIX = '/__eval/cases'

export interface LoadEvaluationSnapshotOptions {
  timeoutMs?: number
  retries?: number
  retryDelayMs?: number
}

export interface LoadLocalEvaluationPayloadOptions {
  timeoutMs?: number
}

export const evaluationSnapshotUrl = () => toBasePath('data/evaluation.snapshot.json')
export const evaluationStaticPayloadUrl = () => toBasePath('data/evaluation.payload.json')
export const localEvaluationManifestUrl = () => toBasePath('/__eval/manifest')
export const localEvaluationCaseUrl = (benchmarkId: string, caseId: string) =>
  toBasePath(`${LOCAL_CASE_ROUTE_PREFIX}/${encodeURIComponent(benchmarkId)}/${encodeURIComponent(caseId)}`)

export const toAssetUrl = (relativePath: string) => {
  if (!relativePath) return relativePath
  if (/^(?:[a-z]+:)?\/\//i.test(relativePath)) return relativePath
  return toBasePath(relativePath)
}

const delay = async (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const fetchSnapshotResponse = async (url: string, timeoutMs: number) => {
  const controller = new AbortController()
  const timeoutId = globalThis.setTimeout(() => controller.abort(), timeoutMs)

  try {
    return await fetch(url, {
      cache: 'no-store',
      signal: controller.signal,
    })
  } catch (caughtError) {
    if (caughtError instanceof DOMException && caughtError.name === 'AbortError') {
      throw new Error(`Timed out after ${Math.ceil(timeoutMs / 1000)}s while loading evaluation snapshot`)
    }

    throw caughtError instanceof Error ? caughtError : new Error(String(caughtError))
  } finally {
    globalThis.clearTimeout(timeoutId)
  }
}

export const loadEvaluationSnapshot = async (
  options: LoadEvaluationSnapshotOptions = {},
): Promise<EvaluationSnapshot | null> => {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const retries = options.retries ?? DEFAULT_RETRIES
  const retryDelayMs = options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS
  const url = evaluationSnapshotUrl()

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetchSnapshotResponse(url, timeoutMs)

      if (response.status === 404) {
        return null
      }

      if (!response.ok) {
        throw new Error(`Failed to load evaluation snapshot: ${response.status} ${response.statusText}`)
      }

      return (await response.json()) as EvaluationSnapshot
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : String(caughtError)
      lastError = new Error(`${message} (${url})`)

      if (attempt >= retries) {
        throw lastError
      }

      await delay(retryDelayMs)
    }
  }

  throw lastError ?? new Error(`Failed to load evaluation snapshot (${url})`)
}

export const loadStaticEvaluationPayload = async (
  options: LoadEvaluationSnapshotOptions = {},
): Promise<StaticEvaluationPayload | null> => {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const retries = options.retries ?? DEFAULT_RETRIES
  const retryDelayMs = options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS
  const url = evaluationStaticPayloadUrl()

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetchSnapshotResponse(url, timeoutMs)

      if (response.status === 404) {
        return null
      }

      if (!response.ok) {
        throw new Error(`Failed to load static evaluation payload: ${response.status} ${response.statusText}`)
      }

      return (await response.json()) as StaticEvaluationPayload
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : String(caughtError)
      lastError = new Error(`${message} (${url})`)

      if (attempt >= retries) {
        throw lastError
      }

      await delay(retryDelayMs)
    }
  }

  throw lastError ?? new Error(`Failed to load static evaluation payload (${url})`)
}

export const loadLocalEvaluationManifest = async (
  options: LoadLocalEvaluationPayloadOptions = {},
): Promise<DevEvaluationManifestPayload> => {
  const response = await fetchSnapshotResponse(localEvaluationManifestUrl(), options.timeoutMs ?? DEFAULT_LOCAL_TIMEOUT_MS)
  if (!response.ok) {
    throw new Error(`Failed to load local evaluation manifest: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as DevEvaluationManifestPayload
}

export const loadLocalEvaluationCase = async (
  benchmarkId: string,
  caseId: string,
  options: LoadLocalEvaluationPayloadOptions = {},
): Promise<SnapshotQuestion> => {
  const response = await fetchSnapshotResponse(
    localEvaluationCaseUrl(benchmarkId, caseId),
    options.timeoutMs ?? DEFAULT_LOCAL_TIMEOUT_MS,
  )

  if (!response.ok) {
    throw new Error(`Failed to load local evaluation case: ${response.status} ${response.statusText}`)
  }

  return ((await response.json()) as DevEvaluationCasePayload).question
}
