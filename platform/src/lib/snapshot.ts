import type { EvaluationSnapshot } from '../domain/evaluation'
import { toBasePath } from './assets'

export const toPublicAssetUrl = (relativePath: string) => {
  if (!relativePath) return relativePath
  return toBasePath(relativePath)
}

export const loadEvaluationSnapshot = async (): Promise<EvaluationSnapshot | null> => {
  const response = await fetch(toBasePath('data/evaluation.snapshot.json'), {
    cache: 'no-store',
  })

  if (response.status === 404) {
    return null
  }

  if (!response.ok) {
    throw new Error(`Failed to load evaluation snapshot: ${response.status} ${response.statusText}`)
  }

  return (await response.json()) as EvaluationSnapshot
}
