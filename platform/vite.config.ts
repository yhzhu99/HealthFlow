import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import type { Plugin } from 'vite'
import { defineConfig } from 'vitest/config'

import {
  buildEvaluationCasePayload,
  buildEvaluationManifestPayload,
  evaluationDataRootForProject,
  evaluationCaseRouteFromSegments,
  evaluationManifestRoute,
} from './dev/evaluation-data'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const devArtifactContentType = (filePath: string) => {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.md') return 'text/markdown; charset=utf-8'
  if (extension === '.pdf') return 'application/pdf'
  if (extension === '.csv') return 'text/csv; charset=utf-8'
  if (extension === '.tsv') return 'text/tab-separated-values; charset=utf-8'
  if (extension === '.json') return 'application/json; charset=utf-8'
  if (extension === '.jsonl') return 'application/x-ndjson; charset=utf-8'
  if (extension === '.png') return 'image/png'
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg'
  if (extension === '.gif') return 'image/gif'
  if (extension === '.svg') return 'image/svg+xml'
  if (extension === '.webp') return 'image/webp'
  return 'text/plain; charset=utf-8'
}

const resolveEvaluationArtifactPath = (projectRoot: string, routePath: string) => {
  const decodedSegments = routePath
    .split('/')
    .filter(Boolean)
    .map((segment) => decodeURIComponent(segment))

  if (decodedSegments.length < 4) {
    return null
  }

  if (decodedSegments.some((segment) => segment === '.' || segment === '..' || segment.includes(path.sep))) {
    return null
  }

  const [benchmarkId, caseId, scope, ...rest] = decodedSegments
  if (rest.length === 0) {
    return null
  }

  const evaluationRoot = evaluationDataRootForProject(projectRoot)
  if (scope === 'reference') {
    return path.join(evaluationRoot, 'benchmarks', benchmarkId, 'cases', caseId, 'reference', 'files', ...rest)
  }

  return path.join(evaluationRoot, 'benchmarks', benchmarkId, 'cases', caseId, 'frameworks', scope, 'files', ...rest)
}

const evaluationDevPlugin = (): Plugin => ({
  name: 'healthflow-evaluation-dev',
  configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const requestUrl = req.url ? new URL(req.url, 'http://localhost') : null
      if (!requestUrl) {
        next()
        return
      }

      const projectRoot = __dirname
      const evaluationCasePrefix = evaluationCaseRouteFromSegments([])
      if (requestUrl.pathname === evaluationManifestRoute()) {
        try {
          const payload = await buildEvaluationManifestPayload({
            projectRoot,
          })
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.end(JSON.stringify(payload))
          return
        } catch (caughtError) {
          res.statusCode = 500
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.end(
            JSON.stringify({
              error: caughtError instanceof Error ? caughtError.message : String(caughtError),
            }),
          )
          return
        }
      }

      if (requestUrl.pathname.startsWith(evaluationCasePrefix)) {
        const decodedSegments = requestUrl.pathname
          .replace(evaluationCasePrefix, '')
          .split('/')
          .filter(Boolean)
          .map((segment) => decodeURIComponent(segment))

        if (decodedSegments.length !== 2) {
          res.statusCode = 400
          res.end('Invalid evaluation case path')
          return
        }

        try {
          const payload = await buildEvaluationCasePayload({
            projectRoot,
            benchmarkId: decodedSegments[0] ?? '',
            caseId: decodedSegments[1] ?? '',
            mode: 'dev',
          })
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.end(JSON.stringify(payload))
          return
        } catch (caughtError) {
          res.statusCode = 404
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.end(
            JSON.stringify({
              error: caughtError instanceof Error ? caughtError.message : String(caughtError),
            }),
          )
          return
        }
      }

      if (requestUrl.pathname.startsWith('/__eval/artifacts/')) {
        const artifactPath = resolveEvaluationArtifactPath(projectRoot, requestUrl.pathname.replace('/__eval/artifacts/', ''))
        if (!artifactPath) {
          res.statusCode = 400
          res.end('Invalid evaluation artifact path')
          return
        }

        try {
          const payload = await readFile(artifactPath)
          res.statusCode = 200
          res.setHeader('Content-Type', devArtifactContentType(artifactPath))
          res.end(payload)
          return
        } catch {
          res.statusCode = 404
          res.end('Evaluation artifact not found')
          return
        }
      }

      next()
    })
  },
})

export default defineConfig(({ command }) => ({
  plugins: [vue(), tailwindcss(), command === 'serve' ? evaluationDevPlugin() : null].filter(Boolean),
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
}))
