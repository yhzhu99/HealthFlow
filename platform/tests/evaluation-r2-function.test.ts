import { describe, expect, it, vi } from 'vitest'

const loadPagesFunction = async (): Promise<{
  onRequestGet: (context: Record<string, any>) => Promise<Response>
  onRequestHead: (context: Record<string, any>) => Promise<Response>
}> => {
  // @ts-ignore Cloudflare Pages Functions are plain JavaScript modules.
  return await import('../functions/evaluation-data/[[path]].js')
}

const createR2Object = (body = 'payload', contentType?: string) => ({
  body,
  httpEtag: '"test-etag"',
  writeHttpMetadata(headers: Headers) {
    if (contentType) {
      headers.set('content-type', contentType)
    }
  },
})

describe('evaluation-data R2 Pages Function', () => {
  it('serves the evaluation payload from the R2 binding', async () => {
    const { onRequestGet } = await loadPagesFunction()
    const bucket = {
      get: vi.fn(async () => createR2Object('{"mode":"live"}')),
    }

    const response = await onRequestGet({
      env: { EVALUATION_DATA_BUCKET: bucket },
      params: { path: ['data', 'evaluation.payload.json'] },
    })

    expect(bucket.get).toHaveBeenCalledWith('evaluation-data/data/evaluation.payload.json')
    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toBe('application/json; charset=utf-8')
    expect(response.headers.get('cache-control')).toBe('no-store')
    expect(await response.text()).toBe('{"mode":"live"}')
  })

  it('uses an optional bucket prefix override for custom R2 layouts', async () => {
    const { onRequestHead } = await loadPagesFunction()
    const bucket = {
      head: vi.fn(async () => createR2Object('', 'text/markdown; charset=utf-8')),
    }

    const response = await onRequestHead({
      env: {
        EVALUATION_DATA_BUCKET: bucket,
        EVALUATION_DATA_BUCKET_PREFIX: 'healthflow',
      },
      params: { path: ['benchmarks', 'demo', 'cases', '0001', 'reference', 'files', 'report.md'] },
    })

    expect(bucket.head).toHaveBeenCalledWith('healthflow/benchmarks/demo/cases/0001/reference/files/report.md')
    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toBe('text/markdown; charset=utf-8')
  })

  it('accepts the generic BUCKET binding name', async () => {
    const { onRequestGet } = await loadPagesFunction()
    const bucket = {
      get: vi.fn(async () => createR2Object('{"mode":"live"}')),
    }

    const response = await onRequestGet({
      env: { BUCKET: bucket },
      params: { path: ['data', 'evaluation.payload.json'] },
    })

    expect(response.status).toBe(200)
    expect(bucket.get).toHaveBeenCalledWith('evaluation-data/data/evaluation.payload.json')
  })

  it('returns a clear 500 when no R2 binding is configured', async () => {
    const { onRequestGet } = await loadPagesFunction()

    const response = await onRequestGet({
      env: {},
      params: { path: ['data', 'evaluation.payload.json'] },
    })

    expect(response.status).toBe(500)
    expect(await response.text()).toContain('Missing R2 binding')
  })

  it('rejects unknown roots and traversal segments', async () => {
    const { onRequestGet } = await loadPagesFunction()
    const bucket = {
      get: vi.fn(),
    }

    const unknownRoot = await onRequestGet({
      env: { EVALUATION_DATA_BUCKET: bucket },
      params: { path: ['private', 'secret.json'] },
    })
    const traversal = await onRequestGet({
      env: { EVALUATION_DATA_BUCKET: bucket },
      params: { path: ['benchmarks', '..', 'secret.json'] },
    })

    expect(unknownRoot.status).toBe(404)
    expect(traversal.status).toBe(404)
    expect(bucket.get).not.toHaveBeenCalled()
  })
})
