import type { EvaluationSnapshot } from '../domain/evaluation'
import { toBasePath } from './assets'

const DEFAULT_TIMEOUT_MS = 8000
const DEFAULT_RETRIES = 1
const DEFAULT_RETRY_DELAY_MS = 350

export interface LoadEvaluationSnapshotOptions {
  timeoutMs?: number
  retries?: number
  retryDelayMs?: number
}

export const evaluationSnapshotUrl = () => toBasePath('data/evaluation.snapshot.json')

export const toPublicAssetUrl = (relativePath: string) => {
  if (!relativePath) return relativePath
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
