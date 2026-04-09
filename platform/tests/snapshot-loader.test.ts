import { afterEach, describe, expect, it, vi } from 'vitest'

import { evaluationSnapshotUrl, loadEvaluationSnapshot } from '../src/lib/snapshot'

const originalFetch = globalThis.fetch

const mockSnapshot = {
  snapshotVersion: 'test-snapshot',
  benchmarks: [],
  runs: [],
  questions: [],
}

afterEach(() => {
  globalThis.fetch = originalFetch
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('snapshot loader', () => {
  it('loads the committed snapshot url and parses json', async () => {
    globalThis.fetch = vi.fn(async () =>
      new Response(JSON.stringify(mockSnapshot), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ) as typeof fetch

    const snapshot = await loadEvaluationSnapshot({ retries: 0 })

    expect(globalThis.fetch).toHaveBeenCalledWith(
      evaluationSnapshotUrl(),
      expect.objectContaining({ cache: 'no-store', signal: expect.any(AbortSignal) }),
    )
    expect(snapshot?.snapshotVersion).toBe('test-snapshot')
  })

  it('retries once after a transient fetch failure', async () => {
    let attempts = 0
    globalThis.fetch = vi.fn(async () => {
      attempts += 1
      if (attempts === 1) {
        throw new Error('Temporary network failure')
      }

      return new Response(JSON.stringify(mockSnapshot), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }) as typeof fetch

    const snapshot = await loadEvaluationSnapshot({ retries: 1, retryDelayMs: 0 })

    expect(snapshot?.snapshotVersion).toBe('test-snapshot')
    expect(globalThis.fetch).toHaveBeenCalledTimes(2)
  })

  it('returns null immediately when the snapshot is missing', async () => {
    globalThis.fetch = vi.fn(async () => new Response('', { status: 404 })) as typeof fetch

    const snapshot = await loadEvaluationSnapshot({ retries: 1 })

    expect(snapshot).toBeNull()
    expect(globalThis.fetch).toHaveBeenCalledTimes(1)
  })

  it('surfaces a timeout instead of hanging forever', async () => {
    vi.useFakeTimers()
    globalThis.fetch = vi.fn(
      async (_input: string | URL | Request, init?: RequestInit) =>
        await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true },
          )
        }),
    ) as typeof fetch

    const loadingPromise = expect(
      loadEvaluationSnapshot({ timeoutMs: 25, retries: 0 }),
    ).rejects.toThrow(/Timed out after 1s while loading evaluation snapshot/)
    await vi.advanceTimersByTimeAsync(25)
    await loadingPromise
  })
})
