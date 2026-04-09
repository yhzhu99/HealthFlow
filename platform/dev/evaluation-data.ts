import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'

import type {
  BenchmarkId,
  DevEvaluationCasePayload,
  DevEvaluationManifest,
  DevEvaluationManifestPayload,
  DiagnosticEvaluationDiagnostics,
  EvaluationQuestionSummary,
  EvaluationSnapshot,
  LiveEvaluationDiagnostics,
  SnapshotArtifact,
  SnapshotArtifactKind,
  SnapshotCandidate,
  SnapshotExpectedOutput,
  SnapshotQuestion,
  SnapshotReference,
  SnapshotRun,
} from '../src/domain/evaluation'

type AssetMode = 'dev' | 'static'

interface BenchmarkFile {
  id?: BenchmarkId
  label?: string
  description?: string
  frameworkOrder?: string[]
}

interface CaseFile {
  id?: string
  qid?: string | number
  task?: string
  taskBrief?: string | null
  taskType?: string | null
  paperTitle?: string | null
  options?: Record<string, string> | null
  reportRequirements?: string[]
  expectedOutputs?: SnapshotExpectedOutput[]
}

interface FrameworkManifestFile {
  runId?: string
  runLabel?: string
  modelId?: string
  backend?: string | null
  summary?: string | null
  score?: number | null
  success?: boolean
}

interface ArtifactCopy {
  sourcePath: string
  relativePath: string
}

interface InternalArtifact {
  artifact: SnapshotArtifact
  sourcePath: string
}

interface ScannedFramework {
  frameworkId: string
  root: string
  runId: string
  runLabel: string
  modelId: string
  manifest: FrameworkManifestFile
}

interface ScannedCase {
  caseId: string
  root: string
  benchmarkId: string
  benchmarkLabel: string
  config: CaseFile
  frameworks: ScannedFramework[]
}

interface ScannedBenchmark {
  id: string
  label: string
  description: string
  cases: ScannedCase[]
}

interface ScannedEvaluationIndex {
  root: string
  benchmarks: ScannedBenchmark[]
  warnings: string[]
  missing: string[]
  invalid: string[]
}

interface ScannedBenchmarkMeta {
  benchmarkId: string
  benchmarkLabel: string
  benchmarkDescription: string
  benchmarkRoot: string
  casesRoot: string
  frameworkOrder: Map<string, number>
}

type EvaluationSnapshotPayload =
  | {
      mode: 'live'
      snapshot: EvaluationSnapshot
      diagnostics: LiveEvaluationDiagnostics
    }
  | {
      mode: 'diagnostic'
      snapshot: null
      diagnostics: DiagnosticEvaluationDiagnostics
    }

export interface EvaluationSnapshotBundle {
  payload: EvaluationSnapshotPayload
  artifactCopies: ArtifactCopy[]
}

export interface ExportStaticEvaluationBundleResult {
  outputPath: string
  snapshot: EvaluationSnapshot
  artifactCount: number
}

const DEFAULT_SNAPSHOT_VERSION = 'healthflow-local-dev'
const EVALUATION_ROUTE_PREFIX = '/__eval'
const EVALUATION_ARTIFACT_ROUTE_PREFIX = `${EVALUATION_ROUTE_PREFIX}/artifacts`
const EVALUATION_MANIFEST_ROUTE = `${EVALUATION_ROUTE_PREFIX}/manifest`
const EVALUATION_CASE_ROUTE_PREFIX = `${EVALUATION_ROUTE_PREFIX}/cases`

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'])
const TEXT_EXTENSIONS = new Set(['.md', '.txt', '.log', '.rst'])
const STRUCTURED_EXTENSIONS = new Set(['.json', '.jsonl'])
const TABULAR_EXTENSIONS = new Set(['.csv', '.tsv'])

export const evaluationDataRootForProject = (projectRoot: string) => path.join(projectRoot, 'evaluation-data')
export const evaluationManifestRoute = () => EVALUATION_MANIFEST_ROUTE
export const evaluationCaseRouteFromSegments = (segments: string[]) =>
  `${EVALUATION_CASE_ROUTE_PREFIX}/${segments.map(encodeURIComponent).join('/')}`
export const evaluationArtifactRouteFromSegments = (segments: string[]) =>
  `${EVALUATION_ARTIFACT_ROUTE_PREFIX}/${segments.map(encodeURIComponent).join('/')}`

const sortDirectoryNames = (values: string[]) =>
  [...values].sort((left, right) =>
    left.localeCompare(right, undefined, {
      numeric: true,
      sensitivity: 'base',
    }),
  )

const relativeToPosix = (basePath: string, targetPath: string) => path.relative(basePath, targetPath).split(path.sep).join('/')

const mediaTypeForPath = (filePath: string) => {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.md') return 'text/markdown'
  if (extension === '.pdf') return 'application/pdf'
  if (extension === '.csv') return 'text/csv'
  if (extension === '.tsv') return 'text/tab-separated-values'
  if (extension === '.json') return 'application/json'
  if (extension === '.jsonl') return 'application/x-ndjson'
  if (extension === '.png') return 'image/png'
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg'
  if (extension === '.gif') return 'image/gif'
  if (extension === '.svg') return 'image/svg+xml'
  if (extension === '.webp') return 'image/webp'
  return 'text/plain'
}

const artifactKindForPath = (filePath: string): SnapshotArtifactKind => {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.md') return 'markdown'
  if (extension === '.pdf') return 'pdf'
  if (IMAGE_EXTENSIONS.has(extension)) return 'image'
  if (TABULAR_EXTENSIONS.has(extension)) return 'csv'
  if (STRUCTURED_EXTENSIONS.has(extension)) return 'json'
  if (TEXT_EXTENSIONS.has(extension)) return 'text'
  return 'download'
}

const previewPriority = (relativePath: string) => {
  const lowerPath = relativePath.toLowerCase()
  let score = 0

  if (lowerPath.endsWith('report.md')) score += 320
  if (lowerPath.endsWith('.pdf')) score += 280
  if (lowerPath.includes('/figures/') || lowerPath.includes('/images/')) score += 180
  if (lowerPath.endsWith('.png') || lowerPath.endsWith('.jpg') || lowerPath.endsWith('.svg')) score += 160
  if (lowerPath.endsWith('.csv')) score += 140
  if (lowerPath.endsWith('.json')) score += 100

  return score
}

const safeReadJson = async <T>(filePath: string): Promise<T> => JSON.parse(await readFile(filePath, 'utf-8')) as T

const readTextOrNull = async (filePath: string) => {
  try {
    const fileInfo = await stat(filePath)
    if (!fileInfo.isFile()) return null
    return await readFile(filePath, 'utf-8')
  } catch {
    return null
  }
}

const isDirectory = async (targetPath: string) => {
  try {
    return (await stat(targetPath)).isDirectory()
  } catch {
    return false
  }
}

const isFile = async (targetPath: string) => {
  try {
    return (await stat(targetPath)).isFile()
  } catch {
    return false
  }
}

const listDirectories = async (rootPath: string) => {
  const entries = await readdir(rootPath, { withFileTypes: true })
  return sortDirectoryNames(entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name))
}

const listFilesRecursive = async (rootPath: string): Promise<string[]> => {
  const entries = await readdir(rootPath, { withFileTypes: true })
  const files: string[] = []

  for (const entry of entries) {
    const resolvedPath = path.join(rootPath, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await listFilesRecursive(resolvedPath)))
      continue
    }
    if (entry.isFile()) {
      files.push(resolvedPath)
    }
  }

  return files
}

const artifactRelativePath = (mode: AssetMode, routeSegments: string[]) =>
  mode === 'dev'
    ? evaluationArtifactRouteFromSegments(routeSegments)
    : path.posix.join('evaluation-assets', ...routeSegments)

const createArtifact = ({
  mode,
  sourcePath,
  label,
  originalPath,
  routeSegments,
}: {
  mode: AssetMode
  sourcePath: string
  label: string
  originalPath: string
  routeSegments: string[]
}): InternalArtifact => ({
  sourcePath,
  artifact: {
    id: routeSegments.join(':'),
    label,
    kind: artifactKindForPath(sourcePath),
    mediaType: mediaTypeForPath(sourcePath),
    relativePath: artifactRelativePath(mode, routeSegments),
    originalPath,
    previewPriority: previewPriority(originalPath),
  },
})

const loadArtifacts = async ({
  mode,
  sourceRoot,
  routeBaseSegments,
  originalPathPrefix,
}: {
  mode: AssetMode
  sourceRoot: string
  routeBaseSegments: string[]
  originalPathPrefix: string
}): Promise<InternalArtifact[]> => {
  if (!(await isDirectory(sourceRoot))) {
    return []
  }

  const files = await listFilesRecursive(sourceRoot)
  const artifacts = files.map((sourcePath) => {
    const relativePath = relativeToPosix(sourceRoot, sourcePath)
    return createArtifact({
      mode,
      sourcePath,
      label: path.basename(sourcePath),
      originalPath: path.posix.join(originalPathPrefix, relativePath),
      routeSegments: [...routeBaseSegments, ...relativePath.split('/')],
    })
  })

  artifacts.sort((left, right) => {
    if (left.artifact.previewPriority === right.artifact.previewPriority) {
      return left.artifact.originalPath.localeCompare(right.artifact.originalPath)
    }
    return right.artifact.previewPriority - left.artifact.previewPriority
  })

  return artifacts
}

const findReportPath = (artifacts: SnapshotArtifact[]) =>
  artifacts.find((artifact) => artifact.label === 'report.md')?.relativePath ??
  artifacts.find((artifact) => artifact.originalPath.endsWith('/report.md'))?.relativePath ??
  null

const summarizeDiagnostics = (
  root: string,
  warnings: string[],
  missing: string[] = [],
  invalid: string[] = [],
): DiagnosticEvaluationDiagnostics => ({
  root,
  warnings,
  missing,
  invalid,
})

const buildLiveDiagnostics = (root: string, warnings: string[]): LiveEvaluationDiagnostics => ({
  root,
  warnings,
})

const defaultQuestionId = (benchmarkId: string, qid: string) => `${benchmarkId}:${qid}`

const buildQuestionSummary = ({
  caseId,
  benchmarkId,
  benchmarkLabel,
  config,
  candidateCount,
}: {
  caseId: string
  benchmarkId: string
  benchmarkLabel: string
  config: CaseFile
  candidateCount: number
}): EvaluationQuestionSummary => {
  const qid = String(config.qid ?? caseId)
  return {
    caseId,
    id: config.id?.trim() || defaultQuestionId(benchmarkId, qid),
    datasetId: benchmarkId,
    datasetLabel: benchmarkLabel,
    qid,
    taskBrief: config.taskBrief ?? null,
    taskType: config.taskType ?? null,
    paperTitle: config.paperTitle ?? null,
    candidateCount,
  }
}

const readBenchmarkMeta = async (
  projectRoot: string,
  benchmarkDirectory: string,
): Promise<ScannedBenchmarkMeta> => {
  const root = evaluationDataRootForProject(projectRoot)
  const benchmarkRoot = path.join(root, 'benchmarks', benchmarkDirectory)
  const benchmarkConfigPath = path.join(benchmarkRoot, 'benchmark.json')

  if (!(await isFile(benchmarkConfigPath))) {
    throw new Error(`Missing benchmark.json for ${benchmarkDirectory}`)
  }

  const benchmarkConfig = await safeReadJson<BenchmarkFile>(benchmarkConfigPath)
  const benchmarkId = String(benchmarkConfig.id ?? benchmarkDirectory)
  const benchmarkLabel = benchmarkConfig.label?.trim() || benchmarkId
  const benchmarkDescription = benchmarkConfig.description?.trim() || `${benchmarkLabel} evaluation benchmark`
  const frameworkOrder = new Map((benchmarkConfig.frameworkOrder ?? []).map((item, index) => [item, index]))
  const casesRoot = path.join(benchmarkRoot, 'cases')

  if (!(await isDirectory(casesRoot))) {
    throw new Error(`Missing cases directory for ${benchmarkId}`)
  }

  return {
    benchmarkId,
    benchmarkLabel,
    benchmarkDescription,
    benchmarkRoot,
    casesRoot,
    frameworkOrder,
  }
}

const loadFrameworkSummaries = async ({
  frameworksRoot,
  benchmarkId,
  caseId,
  frameworkOrder,
  warnings,
}: {
  frameworksRoot: string
  benchmarkId: string
  caseId: string
  frameworkOrder: Map<string, number>
  warnings: string[]
}): Promise<ScannedFramework[]> => {
  if (!(await isDirectory(frameworksRoot))) {
    throw new Error(`Missing frameworks directory for ${benchmarkId}/${caseId}`)
  }

  const frameworkDirectories = await listDirectories(frameworksRoot)
  if (frameworkDirectories.length === 0) {
    throw new Error(`No frameworks found for ${benchmarkId}/${caseId}`)
  }

  const orderedFrameworkDirectories = [...frameworkDirectories].sort((left, right) => {
    const leftOrder = frameworkOrder.get(left) ?? Number.MAX_SAFE_INTEGER
    const rightOrder = frameworkOrder.get(right) ?? Number.MAX_SAFE_INTEGER
    if (leftOrder !== rightOrder) return leftOrder - rightOrder
    return left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' })
  })

  const frameworks: ScannedFramework[] = []
  for (const frameworkDirectory of orderedFrameworkDirectories) {
    const frameworkRoot = path.join(frameworksRoot, frameworkDirectory)
    const frameworkManifestPath = path.join(frameworkRoot, 'manifest.json')
    if (!(await isFile(frameworkManifestPath))) {
      warnings.push(`Skipping ${benchmarkId}/${caseId}/${frameworkDirectory}: missing manifest.json`)
      continue
    }

    let manifest: FrameworkManifestFile
    try {
      manifest = await safeReadJson<FrameworkManifestFile>(frameworkManifestPath)
    } catch (caughtError) {
      warnings.push(`Skipping ${benchmarkId}/${caseId}/${frameworkDirectory}: invalid manifest.json (${String(caughtError)})`)
      continue
    }

    frameworks.push({
      frameworkId: frameworkDirectory,
      root: frameworkRoot,
      runId: manifest.runId?.trim() || `${benchmarkId}-${frameworkDirectory}`,
      runLabel: manifest.runLabel?.trim() || frameworkDirectory,
      modelId: manifest.modelId?.trim() || frameworkDirectory,
      manifest,
    })
  }

  if (frameworks.length === 0) {
    throw new Error(`No valid framework candidates found for ${benchmarkId}/${caseId}`)
  }

  return frameworks
}

const loadCaseSummary = async ({
  projectRoot,
  benchmarkDirectory,
  caseId,
  warnings,
}: {
  projectRoot: string
  benchmarkDirectory: string
  caseId: string
  warnings: string[]
}): Promise<ScannedCase> => {
  const meta = await readBenchmarkMeta(projectRoot, benchmarkDirectory)
  const caseRoot = path.join(meta.casesRoot, caseId)
  const caseConfigPath = path.join(caseRoot, 'case.json')

  if (!(await isFile(caseConfigPath))) {
    throw new Error(`Missing case.json for ${meta.benchmarkId}/${caseId}`)
  }

  let caseConfig: CaseFile
  try {
    caseConfig = await safeReadJson<CaseFile>(caseConfigPath)
  } catch (caughtError) {
    throw new Error(`Invalid case.json for ${meta.benchmarkId}/${caseId}: ${String(caughtError)}`)
  }

  const frameworks = await loadFrameworkSummaries({
    frameworksRoot: path.join(caseRoot, 'frameworks'),
    benchmarkId: meta.benchmarkId,
    caseId,
    frameworkOrder: meta.frameworkOrder,
    warnings,
  })

  return {
    caseId,
    root: caseRoot,
    benchmarkId: meta.benchmarkId,
    benchmarkLabel: meta.benchmarkLabel,
    config: caseConfig,
    frameworks,
  }
}

const scanEvaluationIndex = async (projectRoot: string): Promise<ScannedEvaluationIndex> => {
  const root = evaluationDataRootForProject(projectRoot)
  const benchmarksRoot = path.join(root, 'benchmarks')
  const missing: string[] = []
  const invalid: string[] = []
  const warnings: string[] = []

  if (!(await isDirectory(root))) {
    missing.push(`Missing evaluation root: ${root}`)
  }

  if (!(await isDirectory(benchmarksRoot))) {
    missing.push(`Missing benchmarks directory: ${benchmarksRoot}`)
  }

  if (missing.length > 0) {
    return {
      root,
      benchmarks: [],
      warnings,
      missing,
      invalid,
    }
  }

  const benchmarkDirectories = await listDirectories(benchmarksRoot)
  if (benchmarkDirectories.length === 0) {
    missing.push(`No benchmark directories found under ${benchmarksRoot}`)
    return {
      root,
      benchmarks: [],
      warnings,
      missing,
      invalid,
    }
  }

  const benchmarks: ScannedBenchmark[] = []

  for (const benchmarkDirectory of benchmarkDirectories) {
    let meta: ScannedBenchmarkMeta
    try {
      meta = await readBenchmarkMeta(projectRoot, benchmarkDirectory)
    } catch (caughtError) {
      invalid.push(caughtError instanceof Error ? caughtError.message : String(caughtError))
      continue
    }

    const caseDirectories = await listDirectories(meta.casesRoot)
    if (caseDirectories.length === 0) {
      invalid.push(`No cases found for ${meta.benchmarkId}`)
      continue
    }

    const cases: ScannedCase[] = []
    for (const caseId of caseDirectories) {
      try {
        cases.push(
          await loadCaseSummary({
            projectRoot,
            benchmarkDirectory,
            caseId,
            warnings,
          }),
        )
      } catch (caughtError) {
        invalid.push(caughtError instanceof Error ? caughtError.message : String(caughtError))
      }
    }

    if (cases.length === 0) {
      invalid.push(`No valid cases found for ${meta.benchmarkId}`)
      continue
    }

    cases.sort((left, right) =>
      buildQuestionSummary({
        caseId: left.caseId,
        benchmarkId: left.benchmarkId,
        benchmarkLabel: left.benchmarkLabel,
        config: left.config,
        candidateCount: left.frameworks.length,
      }).qid.localeCompare(
        buildQuestionSummary({
          caseId: right.caseId,
          benchmarkId: right.benchmarkId,
          benchmarkLabel: right.benchmarkLabel,
          config: right.config,
          candidateCount: right.frameworks.length,
        }).qid,
        undefined,
        {
          numeric: true,
          sensitivity: 'base',
        },
      ),
    )

    benchmarks.push({
      id: meta.benchmarkId,
      label: meta.benchmarkLabel,
      description: meta.benchmarkDescription,
      cases,
    })
  }

  return {
    root,
    benchmarks,
    warnings,
    missing,
    invalid,
  }
}

const buildReference = async ({
  mode,
  caseRecord,
  artifactCopies,
}: {
  mode: AssetMode
  caseRecord: ScannedCase
  artifactCopies: ArtifactCopy[]
}): Promise<SnapshotReference> => {
  const referenceAnswerPath = path.join(caseRecord.root, 'reference', 'answer.md')
  const referenceAnswer = await readTextOrNull(referenceAnswerPath)
  const referenceArtifacts = await loadArtifacts({
    mode,
    sourceRoot: path.join(caseRecord.root, 'reference', 'files'),
    routeBaseSegments: [caseRecord.benchmarkId, caseRecord.caseId, 'reference'],
    originalPathPrefix: 'reference/files',
  })

  artifactCopies.push(...referenceArtifacts.map(({ sourcePath, artifact }) => ({ sourcePath, relativePath: artifact.relativePath })))

  const artifacts = referenceArtifacts.map((item) => item.artifact)
  return {
    mode: artifacts.length > 0 ? 'artifacts' : referenceAnswer ? 'text' : 'none',
    text: referenceAnswer,
    note: null,
    artifacts,
    requiredOutputs: caseRecord.config.expectedOutputs ?? [],
  }
}

const buildCandidate = async ({
  mode,
  caseRecord,
  framework,
  artifactCopies,
}: {
  mode: AssetMode
  caseRecord: ScannedCase
  framework: ScannedFramework
  artifactCopies: ArtifactCopy[]
}): Promise<SnapshotCandidate> => {
  const answerText =
    (await readTextOrNull(path.join(framework.root, 'answer.md'))) ??
    framework.manifest.summary?.trim() ??
    'No final answer was recorded.'

  const frameworkArtifacts = await loadArtifacts({
    mode,
    sourceRoot: path.join(framework.root, 'files'),
    routeBaseSegments: [caseRecord.benchmarkId, caseRecord.caseId, framework.frameworkId],
    originalPathPrefix: `frameworks/${framework.frameworkId}/files`,
  })

  artifactCopies.push(...frameworkArtifacts.map(({ sourcePath, artifact }) => ({ sourcePath, relativePath: artifact.relativePath })))

  const candidateArtifacts = frameworkArtifacts.map((item) => item.artifact)
  return {
    id: `${framework.runId}:${String(caseRecord.config.qid ?? caseRecord.caseId)}`,
    runId: framework.runId,
    runLabel: framework.runLabel,
    modelId: framework.modelId,
    backend: framework.manifest.backend ?? null,
    answerText,
    summary: framework.manifest.summary ?? null,
    score: framework.manifest.score ?? null,
    success: framework.manifest.success ?? true,
    reportPath: findReportPath(candidateArtifacts),
    artifacts: candidateArtifacts,
  }
}

const buildQuestionFromCase = async ({
  mode,
  caseRecord,
  artifactCopies,
}: {
  mode: AssetMode
  caseRecord: ScannedCase
  artifactCopies: ArtifactCopy[]
}): Promise<SnapshotQuestion> => {
  const reference = await buildReference({
    mode,
    caseRecord,
    artifactCopies,
  })

  const candidates: SnapshotCandidate[] = []
  for (const framework of caseRecord.frameworks) {
    candidates.push(
      await buildCandidate({
        mode,
        caseRecord,
        framework,
        artifactCopies,
      }),
    )
  }

  const qid = String(caseRecord.config.qid ?? caseRecord.caseId)
  return {
    id: caseRecord.config.id?.trim() || defaultQuestionId(caseRecord.benchmarkId, qid),
    datasetId: caseRecord.benchmarkId,
    datasetLabel: caseRecord.benchmarkLabel,
    qid,
    task: caseRecord.config.task?.trim() || 'Task description missing.',
    taskBrief: caseRecord.config.taskBrief ?? null,
    taskType: caseRecord.config.taskType ?? null,
    paperTitle: caseRecord.config.paperTitle ?? null,
    options: caseRecord.config.options ?? null,
    reportRequirements: caseRecord.config.reportRequirements ?? [],
    expectedOutputs: caseRecord.config.expectedOutputs ?? [],
    reference,
    candidates,
  }
}

const buildRunsAndQuestionsFromIndex = (index: ScannedEvaluationIndex) => {
  const runsByKey = new Map<string, SnapshotRun>()
  const questions: EvaluationQuestionSummary[] = []

  for (const benchmark of index.benchmarks) {
    for (const caseRecord of benchmark.cases) {
      for (const framework of caseRecord.frameworks) {
        const runKey = `${benchmark.id}:${framework.runId}`
        if (!runsByKey.has(runKey)) {
          runsByKey.set(runKey, {
            id: framework.runId,
            label: framework.runLabel,
            modelId: framework.modelId,
            datasetId: benchmark.id,
          })
        }
      }

      questions.push(
        buildQuestionSummary({
          caseId: caseRecord.caseId,
          benchmarkId: caseRecord.benchmarkId,
          benchmarkLabel: caseRecord.benchmarkLabel,
          config: caseRecord.config,
          candidateCount: caseRecord.frameworks.length,
        }),
      )
    }
  }

  return {
    runs: [...runsByKey.values()],
    questions,
  }
}

export const buildEvaluationManifestPayload = async ({
  projectRoot,
}: {
  projectRoot: string
}): Promise<DevEvaluationManifestPayload> => {
  const index = await scanEvaluationIndex(projectRoot)

  if (index.benchmarks.length === 0) {
    return {
      mode: 'diagnostic',
      manifest: null,
      diagnostics: summarizeDiagnostics(index.root, index.warnings, index.missing, index.invalid),
    }
  }

  const { runs, questions } = buildRunsAndQuestionsFromIndex(index)
  const manifest: DevEvaluationManifest = {
    snapshotVersion: DEFAULT_SNAPSHOT_VERSION,
    benchmarks: index.benchmarks.map((benchmark) => ({
      id: benchmark.id,
      label: benchmark.label,
      description: benchmark.description,
      taskCount: benchmark.cases.length,
    })),
    runs,
    questions,
  }

  if (index.missing.length > 0 || index.invalid.length > 0) {
    index.warnings.push(...index.missing, ...index.invalid)
  }

  return {
    mode: 'live',
    manifest,
    diagnostics: buildLiveDiagnostics(index.root, index.warnings),
  }
}

export const buildEvaluationCasePayload = async ({
  projectRoot,
  benchmarkId,
  caseId,
  mode = 'dev',
}: {
  projectRoot: string
  benchmarkId: string
  caseId: string
  mode?: AssetMode
}): Promise<DevEvaluationCasePayload> => {
  const warnings: string[] = []
  const caseRecord = await loadCaseSummary({
    projectRoot,
    benchmarkDirectory: benchmarkId,
    caseId,
    warnings,
  })

  const artifactCopies: ArtifactCopy[] = []
  const question = await buildQuestionFromCase({
    mode,
    caseRecord,
    artifactCopies,
  })

  return { question }
}

export const buildEvaluationSnapshotBundle = async ({
  projectRoot,
  mode,
}: {
  projectRoot: string
  mode: AssetMode
}): Promise<EvaluationSnapshotBundle> => {
  const manifestPayload = await buildEvaluationManifestPayload({ projectRoot })

  if (manifestPayload.mode !== 'live') {
    return {
      payload: {
        mode: 'diagnostic',
        snapshot: null,
        diagnostics: manifestPayload.diagnostics,
      },
      artifactCopies: [],
    }
  }

  const artifactCopies: ArtifactCopy[] = []
  const questions: SnapshotQuestion[] = []

  for (const summary of manifestPayload.manifest.questions) {
    const caseRecord = await loadCaseSummary({
      projectRoot,
      benchmarkDirectory: summary.datasetId,
      caseId: summary.caseId,
      warnings: [],
    })
    questions.push(
      await buildQuestionFromCase({
        mode,
        caseRecord,
        artifactCopies,
      }),
    )
  }

  const snapshot: EvaluationSnapshot = {
    snapshotVersion: manifestPayload.manifest.snapshotVersion,
    benchmarks: manifestPayload.manifest.benchmarks,
    runs: manifestPayload.manifest.runs,
    questions,
  }

  return {
    payload: {
      mode: 'live',
      snapshot,
      diagnostics: manifestPayload.diagnostics,
    },
    artifactCopies,
  }
}

export const exportStaticEvaluationBundle = async ({
  projectRoot,
  outputRoot,
}: {
  projectRoot: string
  outputRoot: string
}): Promise<ExportStaticEvaluationBundleResult> => {
  const bundle = await buildEvaluationSnapshotBundle({
    projectRoot,
    mode: 'static',
  })

  if (bundle.payload.mode !== 'live') {
    const diagnostics = bundle.payload.diagnostics
    const issues = [...diagnostics.missing, ...diagnostics.invalid, ...diagnostics.warnings]
    throw new Error(
      ['Unable to export evaluation snapshot from platform/evaluation-data.', ...issues].join('\n'),
    )
  }

  const snapshot = bundle.payload.snapshot
  const snapshotOutputPath = path.join(outputRoot, 'data', 'evaluation.snapshot.json')
  const artifactRoot = path.join(outputRoot, 'evaluation-assets')

  await rm(artifactRoot, { recursive: true, force: true })
  await mkdir(path.dirname(snapshotOutputPath), { recursive: true })
  await mkdir(artifactRoot, { recursive: true })

  for (const artifact of bundle.artifactCopies) {
    const destinationPath = path.join(outputRoot, artifact.relativePath)
    await mkdir(path.dirname(destinationPath), { recursive: true })
    await cp(artifact.sourcePath, destinationPath)
  }

  await writeFile(snapshotOutputPath, JSON.stringify(snapshot, null, 2))

  return {
    outputPath: snapshotOutputPath,
    snapshot,
    artifactCount: bundle.artifactCopies.length,
  }
}
