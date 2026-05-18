const ALLOWED_ROOTS = new Set(['benchmarks', 'data'])
const R2_BINDING_NAMES = ['EVALUATION_DATA_BUCKET', 'DATA_BUCKET', 'BUCKET']

const toPathSegments = (value) => {
  if (Array.isArray(value)) return value
  if (typeof value === 'string') return value.split('/').filter(Boolean)
  return []
}

const contentTypeFor = (key) => {
  const lowerKey = key.toLowerCase()
  if (lowerKey.endsWith('.json')) return 'application/json; charset=utf-8'
  if (lowerKey.endsWith('.jsonl')) return 'application/x-ndjson; charset=utf-8'
  if (lowerKey.endsWith('.md')) return 'text/markdown; charset=utf-8'
  if (lowerKey.endsWith('.txt') || lowerKey.endsWith('.log') || lowerKey.endsWith('.py')) return 'text/plain; charset=utf-8'
  if (lowerKey.endsWith('.csv')) return 'text/csv; charset=utf-8'
  if (lowerKey.endsWith('.tsv')) return 'text/tab-separated-values; charset=utf-8'
  if (lowerKey.endsWith('.pdf')) return 'application/pdf'
  if (lowerKey.endsWith('.png')) return 'image/png'
  if (lowerKey.endsWith('.jpg') || lowerKey.endsWith('.jpeg')) return 'image/jpeg'
  if (lowerKey.endsWith('.gif')) return 'image/gif'
  if (lowerKey.endsWith('.svg')) return 'image/svg+xml'
  if (lowerKey.endsWith('.webp')) return 'image/webp'
  return 'application/octet-stream'
}

const joinKey = (prefix, relativePath) => {
  const normalizedPrefix = (prefix ?? 'evaluation-data').replace(/^\/+|\/+$/g, '')
  return normalizedPrefix ? `${normalizedPrefix}/${relativePath}` : relativePath
}

const isNoStoreKey = (objectKey) => {
  const keyWithoutPrefix = objectKey.replace(/^\/+/, '')
  return keyWithoutPrefix.endsWith('data/evaluation.payload.json') || keyWithoutPrefix.endsWith('data/evaluation.snapshot.json')
}

const getObjectKey = (context) => {
  const bucket = R2_BINDING_NAMES.map((name) => context.env[name]).find(Boolean)
  if (!bucket) {
    return { response: new Response(`Missing R2 binding. Configure one of: ${R2_BINDING_NAMES.join(', ')}`, { status: 500 }) }
  }

  const segments = toPathSegments(context.params.path)
  const root = segments[0]
  if (!root || !ALLOWED_ROOTS.has(root) || segments.some((segment) => segment === '.' || segment === '..')) {
    return { response: new Response('Not found', { status: 404 }) }
  }

  const relativePath = segments.join('/')
  const prefix = context.env.EVALUATION_DATA_BUCKET_PREFIX ?? context.env.DATA_BUCKET_PREFIX
  return { bucket, objectKey: joinKey(prefix, relativePath) }
}

const responseHeaders = (object, objectKey) => {
  const headers = new Headers()
  object.writeHttpMetadata(headers)
  headers.set('etag', object.httpEtag)
  headers.set('content-type', headers.get('content-type') ?? contentTypeFor(objectKey))
  headers.set('cache-control', isNoStoreKey(objectKey) ? 'no-store' : 'public, max-age=300')
  return headers
}

export async function onRequestGet(context) {
  try {
    const result = getObjectKey(context)
    if (result.response) return result.response

    const object = await result.bucket.get(result.objectKey)
    if (!object) {
      return new Response(`R2 object not found: ${result.objectKey}`, { status: 404 })
    }

    return new Response(object.body, { headers: responseHeaders(object, result.objectKey) })
  } catch (caughtError) {
    return new Response(`Evaluation data function failed: ${caughtError instanceof Error ? caughtError.message : String(caughtError)}`, {
      status: 500,
    })
  }
}

export async function onRequestHead(context) {
  try {
    const result = getObjectKey(context)
    if (result.response) return result.response

    const object = await result.bucket.head(result.objectKey)
    if (!object) {
      return new Response(null, { status: 404 })
    }

    return new Response(null, { headers: responseHeaders(object, result.objectKey) })
  } catch {
    return new Response(null, { status: 500 })
  }
}
