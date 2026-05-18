<script setup lang="ts">
import { computed, ref } from 'vue'

import {
  appDemoArtifacts,
  appDemoInputFile,
  appDemoPrompt,
  appDemoStages,
  appDemoTitle,
  liveRuntimeUrl,
  resolveAppDemoArtifact,
  type AppDemoArtifact,
  type AppDemoChartBar,
  type AppDemoChartPoint,
} from '../content/appDemo'
import { toBasePath } from '../lib/assets'
import { renderMarkdown } from '../lib/markdown'

type DetailPanel = 'workspace' | 'memory' | 'logs'

interface WorkspaceRow {
  kind: 'folder' | 'file'
  depth: number
  label: string
  artifact?: AppDemoArtifact
}

const selectedArtifactId = ref('report')
const selectedPanel = ref<DetailPanel>('workspace')

const logoUrl = computed(() => toBasePath('branding/healthflow-logo.svg'))
const inputFileHref = computed(() => toBasePath(appDemoInputFile.href))
const selectedArtifact = computed(() => resolveAppDemoArtifact(selectedArtifactId.value) ?? appDemoArtifacts[0])
const selectedMarkdownHtml = computed(() =>
  selectedArtifact.value.kind === 'markdown' ? renderMarkdown(selectedArtifact.value.content) : '',
)
const selectedJsonText = computed(() =>
  selectedArtifact.value.kind === 'json' ? JSON.stringify(selectedArtifact.value.value, null, 2) : '',
)
const selectedTableColumns = computed(() => (selectedArtifact.value.kind === 'table' ? selectedArtifact.value.columns : []))
const selectedTableRows = computed(() => (selectedArtifact.value.kind === 'table' ? selectedArtifact.value.rows : []))
const selectedLogLines = computed(() => (selectedArtifact.value.kind === 'log' ? selectedArtifact.value.lines : []))
const selectedChart = computed(() => (selectedArtifact.value.kind === 'chart' ? selectedArtifact.value.chart : null))

const historyItems = [
  { title: appDemoTitle, active: true },
  { title: 'New Task', active: false },
]

const memoryCards = [
  {
    title: 'Predictive modeling workflow',
    chips: ['workflow', 'ehr', 'tabular'],
    body: 'Confirm class balance and outcome prevalence before fitting the baseline ICU model.',
  },
  {
    title: 'Calibration safeguard',
    chips: ['safeguard', 'risk model'],
    body: 'Always pair AUROC with calibration evidence before surfacing a risk model to clinicians.',
  },
  {
    title: 'Clinical packet pattern',
    chips: ['reflection', 'report'],
    body: 'Ship ROC, calibration, risk distribution, cohort profile, and vignettes together for presentation-ready EHR modeling runs.',
  },
]

const traceLines = [
  'Execution logs for this task will appear here.',
  ...(
    appDemoArtifacts.find((artifact) => artifact.kind === 'log')?.kind === 'log'
      ? appDemoArtifacts.find((artifact) => artifact.kind === 'log')?.lines ?? []
      : []
  ),
]

const workspaceRows = computed<WorkspaceRow[]>(() => {
  const rows: WorkspaceRow[] = []
  const seenFolders = new Set<string>()
  const sortedArtifacts = [...appDemoArtifacts].sort((left, right) => {
    if (left.id === 'report') return -1
    if (right.id === 'report') return 1
    return left.path.localeCompare(right.path)
  })

  for (const artifact of sortedArtifacts) {
    const parts = artifact.path.split('/').filter(Boolean)
    for (let index = 0; index < parts.length - 1; index += 1) {
      const folderKey = parts.slice(0, index + 1).join('/')
      if (seenFolders.has(folderKey)) continue
      seenFolders.add(folderKey)
      rows.push({ kind: 'folder', depth: index, label: parts[index] ?? '' })
    }
    rows.push({ kind: 'file', depth: Math.max(parts.length - 1, 0), label: parts.at(-1) ?? artifact.label, artifact })
  }

  return rows
})

const artifactBadge = (artifact: AppDemoArtifact) => {
  if (artifact.kind === 'chart') return 'png'
  if (artifact.kind === 'table') return 'csv'
  if (artifact.kind === 'markdown') return 'md'
  if (artifact.kind === 'json') return 'json'
  return 'log'
}

const artifactBadgeClass = (artifact: AppDemoArtifact) => {
  if (artifact.kind === 'chart') return 'is-image'
  if (artifact.kind === 'table') return 'is-table'
  if (artifact.id === 'report') return 'is-report'
  return 'is-text'
}

const formatCell = (value: string | number) => (typeof value === 'number' ? value.toLocaleString('en-US') : value)

const chartBox = {
  width: 280,
  height: 188,
  pad: 24,
}

const chartCoordinate = (point: AppDemoChartPoint) => {
  const plotWidth = chartBox.width - chartBox.pad * 2
  const plotHeight = chartBox.height - chartBox.pad * 2
  const x = chartBox.pad + point.x * plotWidth
  const y = chartBox.height - chartBox.pad - point.y * plotHeight
  return { x, y }
}

const chartPolyline = (points: AppDemoChartPoint[]) =>
  points
    .map((point) => chartCoordinate(point))
    .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ')

const chartMarkerX = (point: AppDemoChartPoint) => chartCoordinate(point).x
const chartMarkerY = (point: AppDemoChartPoint) => chartCoordinate(point).y
const maxBarValue = (bars: AppDemoChartBar[]) => Math.max(...bars.map((bar) => bar.value), 1)
const barHeight = (bar: AppDemoChartBar, bars: AppDemoChartBar[]) => `${Math.max(10, (bar.value / maxBarValue(bars)) * 100)}%`
</script>

<template>
  <div class="hf-app-demo">
    <aside class="hf-sidebar" aria-label="History">
      <section class="hf-sidebar-brand" aria-label="HealthFlow brand">
        <img :src="logoUrl" alt="HealthFlow" />
      </section>

      <a class="hf-live-button" :href="liveRuntimeUrl" target="_blank" rel="noreferrer">Open live runtime</a>
      <button class="hf-new-task-button" type="button">New Task</button>

      <div class="hf-history-list" id="hf-history-list">
        <div class="hf-history-cards">
          <div
            v-for="item in historyItems"
            :key="item.title"
            class="hf-history-card"
            :class="{ 'is-active': item.active }"
          >
            <div class="hf-history-card-main">
              <span class="hf-history-card-title">{{ item.title }}</span>
              <span class="hf-history-card-actions">
                <button class="hf-history-card-btn" type="button">Rename</button>
                <button class="hf-history-card-btn is-danger" type="button">Delete</button>
              </span>
            </div>
          </div>
        </div>
      </div>
    </aside>

    <main class="hf-content-shell">
      <section class="hf-main">
        <section class="hf-chat-shell" aria-label="HealthFlow chat">
          <header class="hf-task-header">
            <h2>{{ appDemoTitle }}</h2>
          </header>

          <div class="hf-run-overview-shell">
            <details class="hf-run-overview is-completed" open>
              <summary class="hf-run-overview-summary">
                <div class="hf-run-overview-summary-main">
                  <div class="hf-run-overview-eyebrow">Run Overview</div>
                  <div class="hf-run-overview-summary-line">
                    <h3>Attempt 1</h3>
                    <span class="hf-run-overview-status">workflow_run</span>
                    <span class="hf-run-overview-stage">Reflect</span>
                  </div>
                  <div class="hf-run-overview-objective">Build an in-hospital mortality risk packet from the ICU cohort.</div>
                  <div class="hf-run-overview-note">
                    Clinical example completed successfully. ROC and calibration diagnostics were generated, the evaluator accepted the packet, and the final AUROC was 0.892.
                  </div>
                </div>
                <div class="hf-run-overview-toggle">Details</div>
              </summary>

              <div class="hf-run-overview-detail">
                <div class="hf-stage-chip-row">
                  <div
                    v-for="stage in appDemoStages"
                    :key="stage.id"
                    class="hf-stage-chip is-done"
                  >
                    <span class="hf-stage-chip-label">{{ stage.label }}</span>
                    <span class="hf-stage-chip-state">done</span>
                  </div>
                </div>

                <div class="hf-run-overview-grid">
                  <div class="hf-run-overview-panel">
                    <div class="hf-run-overview-label">Planned steps</div>
                    <ol class="hf-run-overview-list">
                      <li
                        v-for="(step, index) in appDemoStages[1]?.details ?? []"
                        :key="step"
                      >
                        <span>{{ String(index + 1).padStart(2, '0') }}</span>{{ step }}
                      </li>
                    </ol>
                  </div>

                  <div class="hf-run-overview-panel">
                    <div class="hf-run-overview-label">Success checks</div>
                    <ul class="hf-run-overview-watchouts">
                      <li>Inline figures render directly in the main panel.</li>
                      <li>Workspace contains tables, notes, figures, and a final report.</li>
                    </ul>
                  </div>

                  <div class="hf-run-overview-panel">
                    <div class="hf-run-overview-label">Guardrails</div>
                    <ul class="hf-run-overview-watchouts">
                      <li>Do not leak the mortality label into engineered features.</li>
                      <li>Keep generated artifacts concise and presentation-ready.</li>
                    </ul>
                  </div>
                </div>
              </div>
            </details>
          </div>

          <div class="hf-chatbot">
            <div class="hf-chat-scroll">
              <article class="hf-message-row is-user">
                <div class="hf-message hf-user-message">
                  <p>{{ appDemoPrompt }}</p>
                  <div class="hf-chat-attachments">
                    <a class="hf-chat-attachment" :href="inputFileHref" download>
                      <code>{{ appDemoInputFile.name }}</code>
                    </a>
                  </div>
                </div>
              </article>

              <article class="hf-message-row is-bot">
                <div class="hf-message hf-bot-message hf-process-snapshot">
                  <div class="hf-process-card">
                    <div class="hf-process-card-head">
                      <div class="hf-process-card-main">
                        <div class="hf-process-card-eyebrow">HealthFlow process</div>
                        <h3 class="hf-process-card-title">Clinical example completed</h3>
                      </div>
                      <div class="hf-process-card-meta">
                        <span class="hf-process-pill is-completed">success</span>
                        <span class="hf-process-pill">score 0.94</span>
                      </div>
                    </div>

                    <div class="hf-process-stage-rail">
                      <span
                        v-for="stage in appDemoStages"
                        :key="stage.id"
                        class="hf-process-stage is-done"
                      >
                        {{ stage.label }}
                      </span>
                    </div>

                    <p class="hf-process-card-subcopy">
                      Built a compact ICU mortality packet for 18 admissions. The baseline model reached AUROC 0.892 with good rank ordering, and the highest-risk patients consistently combined older age, low systolic blood pressure, high lactate, and tachycardia.
                    </p>

                    <div class="hf-process-card-grid">
                      <div class="hf-process-card-block">
                        <div class="hf-process-card-label">Key outputs</div>
                        <ul class="hf-process-card-list">
                          <li>ROC, calibration, and risk distribution figures.</li>
                          <li>Predictions, feature importance, and cohort profile tables.</li>
                          <li>Patient vignettes and final clinical report.</li>
                        </ul>
                      </div>
                      <div class="hf-process-card-block">
                        <div class="hf-process-card-label">Clinical takeaway</div>
                        <p class="hf-process-card-copy">
                          Older patients with hypotension, tachycardia, and elevated lactate clustered in the highest-risk bucket.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </article>

              <article class="hf-message-row is-bot">
                <div class="hf-inline-gallery">
                  <button
                    v-for="artifact in appDemoArtifacts.filter((item) => item.kind === 'chart')"
                    :key="artifact.id"
                    type="button"
                    class="hf-gallery-item"
                    :class="{ 'is-active': selectedArtifactId === artifact.id }"
                    @click="selectedPanel = 'workspace'; selectedArtifactId = artifact.id"
                  >
                    <span class="hf-gallery-title">{{ artifact.label }}</span>
                    <span v-if="artifact.kind === 'chart'" class="hf-mini-chart">
                      <svg
                        v-if="artifact.chart.type === 'line'"
                        viewBox="0 0 280 188"
                        role="img"
                        :aria-label="artifact.chart.title"
                      >
                        <path d="M24 164 H256 M24 164 V24" fill="none" stroke="#cbd5e1" stroke-width="1.5" />
                        <path d="M24 164 L256 24" fill="none" stroke="#cbd5e1" stroke-dasharray="5 5" stroke-width="1.25" />
                        <polyline
                          :points="chartPolyline(artifact.chart.points)"
                          fill="none"
                          stroke="#0284c7"
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          stroke-width="5"
                        />
                      </svg>
                      <span v-else class="hf-mini-bars">
                        <span
                          v-for="bar in artifact.chart.bars"
                          :key="bar.label"
                          class="hf-mini-bar"
                          :style="{ height: barHeight(bar, artifact.chart.bars) }"
                        />
                      </span>
                    </span>
                  </button>
                </div>
              </article>
            </div>
          </div>

          <footer class="hf-composer-shell">
            <div class="hf-composer-attachments">
              <span class="hf-composer-attachment"><code>{{ appDemoInputFile.name }}</code></span>
            </div>
            <div class="hf-composer">
              <button type="button" class="hf-upload-button" aria-label="Upload file">+</button>
              <textarea rows="1" readonly placeholder="Message HealthFlow" aria-label="Message HealthFlow" />
              <button type="button" class="hf-submit-button" aria-label="Send message">↑</button>
            </div>
          </footer>
        </section>

        <aside class="hf-workspace-shell" aria-label="Workspace details">
          <section class="hf-detail-shell">
            <nav class="hf-detail-nav" aria-label="Detail panels">
              <button
                type="button"
                class="hf-detail-switch"
                :class="selectedPanel === 'workspace' ? 'primary' : 'secondary'"
                @click="selectedPanel = 'workspace'"
              >
                Workspace
              </button>
              <button
                type="button"
                class="hf-detail-switch"
                :class="selectedPanel === 'memory' ? 'primary' : 'secondary'"
                @click="selectedPanel = 'memory'"
              >
                Memory
              </button>
              <button
                type="button"
                class="hf-detail-switch"
                :class="selectedPanel === 'logs' ? 'primary' : 'secondary'"
                @click="selectedPanel = 'logs'"
              >
                Logs
              </button>
            </nav>

            <section v-if="selectedPanel === 'workspace'" class="hf-detail-panel hf-workspace-panel">
              <div class="hf-workspace-row">
                <div class="hf-browser-pane">
                  <h3>Workspace</h3>
                  <div id="hf-workspace-tree" class="hf-tree-root">
                    <template v-for="row in workspaceRows" :key="`${row.kind}-${row.depth}-${row.label}-${row.artifact?.id ?? ''}`">
                      <div
                        v-if="row.kind === 'folder'"
                        class="hf-tree-folder"
                        :class="`hf-tree-depth-${Math.min(row.depth, 5)}`"
                      >
                        <span class="hf-tree-folder-prefix">/</span>
                        <span class="hf-tree-folder-name">{{ row.label }}</span>
                      </div>
                      <button
                        v-else-if="row.artifact"
                        type="button"
                        class="hf-tree-file"
                        :class="[`hf-tree-depth-${Math.min(row.depth, 5)}`, { 'is-active': selectedArtifactId === row.artifact.id }]"
                        @click="selectedArtifactId = row.artifact.id"
                      >
                        <span class="hf-tree-file-main">
                          <span class="hf-tree-node-badge" :class="artifactBadgeClass(row.artifact)">{{ artifactBadge(row.artifact) }}</span>
                          <span class="hf-tree-file-name">{{ row.label }}</span>
                        </span>
                      </button>
                    </template>
                  </div>
                </div>

                <div class="hf-preview-pane">
                  <div class="hf-preview-header">
                    <h3>Preview</h3>
                    <span>{{ selectedArtifact.path }}</span>
                  </div>

                  <div
                    v-if="selectedArtifact.kind === 'markdown'"
                    class="prose"
                    v-html="selectedMarkdownHtml"
                  />

                  <pre v-else-if="selectedArtifact.kind === 'json'" class="hf-code-preview">{{ selectedJsonText }}</pre>

                  <div v-else-if="selectedArtifact.kind === 'table'" class="hf-table-preview">
                    <table>
                      <thead>
                        <tr>
                          <th v-for="column in selectedTableColumns" :key="column">{{ column }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="(row, rowIndex) in selectedTableRows" :key="rowIndex">
                          <td v-for="column in selectedTableColumns" :key="column">
                            {{ formatCell(row[column] ?? '') }}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div v-else-if="selectedChart" class="hf-chart-preview">
                    <div class="hf-chart-head">
                      <span>{{ selectedChart.title }}</span>
                      <strong>{{ selectedChart.metric }}</strong>
                    </div>
                    <svg
                      v-if="selectedChart.type === 'line'"
                      viewBox="0 0 280 188"
                      role="img"
                      :aria-label="selectedChart.title"
                    >
                      <path d="M24 164 H256 M24 164 V24" fill="none" stroke="#cbd5e1" stroke-width="1.5" />
                      <path d="M24 164 L256 24" fill="none" stroke="#cbd5e1" stroke-dasharray="5 5" stroke-width="1.25" />
                      <polyline
                        :points="chartPolyline(selectedChart.points)"
                        fill="none"
                        stroke="#0284c7"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="5"
                      />
                      <circle
                        v-for="point in selectedChart.points"
                        :key="`${point.x}-${point.y}`"
                        :cx="chartMarkerX(point)"
                        :cy="chartMarkerY(point)"
                        r="4"
                        fill="#0f172a"
                      />
                    </svg>
                    <div v-else class="hf-chart-bars">
                      <div
                        v-for="bar in selectedChart.bars"
                        :key="bar.label"
                        class="hf-chart-bar-cell"
                      >
                        <div class="hf-chart-bar-track">
                          <div class="hf-chart-bar-fill" :style="{ height: barHeight(bar, selectedChart.bars) }" />
                        </div>
                        <strong>{{ bar.count }}</strong>
                        <span>{{ bar.label }}</span>
                      </div>
                    </div>
                  </div>

                  <div v-else class="hf-log-preview">
                    <p v-for="line in selectedLogLines" :key="line">{{ line }}</p>
                  </div>
                </div>
              </div>
            </section>

            <section v-else-if="selectedPanel === 'memory'" class="hf-detail-panel">
              <div class="hf-memory-pane">
                <h3>Memory</h3>
                <div class="hf-memory-stack">
                  <article v-for="card in memoryCards" :key="card.title" class="hf-memory-card">
                    <h4>{{ card.title }}</h4>
                    <p>{{ card.body }}</p>
                    <div class="hf-memory-chip-row">
                      <span v-for="chip in card.chips" :key="chip" class="hf-memory-chip">{{ chip }}</span>
                    </div>
                  </article>
                </div>
              </div>
            </section>

            <section v-else class="hf-detail-panel">
              <div class="hf-advanced-toolbar">
                <h3>Logs</h3>
                <label><input type="checkbox" checked /> Generate report.md for each turn</label>
              </div>
              <div class="hf-trace-panel">
                <div class="hf-trace-line" v-for="line in traceLines" :key="line">{{ line }}</div>
              </div>
            </section>
          </section>
        </aside>
      </section>
    </main>
  </div>
</template>

<style scoped>
.hf-app-demo {
  --hf-border: rgba(15, 23, 42, 0.12);
  --hf-border-strong: rgba(15, 86, 140, 0.28);
  --hf-surface: #f4f8fb;
  --hf-surface-strong: rgba(255, 255, 255, 0.99);
  --hf-surface-muted: #eef3f7;
  --hf-surface-contrast: #e6edf3;
  --hf-text: #102033;
  --hf-text-muted: #536579;
  --hf-accent: #155a8a;
  --hf-accent-strong: #0f4a74;
  --hf-radius-shell: 6px;
  --hf-radius-panel: 4px;
  --hf-radius-control: 3px;
  --hf-radius-chip: 2px;
  display: grid;
  grid-template-columns: 336px minmax(0, 1fr);
  gap: 0.85rem;
  width: 100%;
  height: 100dvh;
  padding: 0.42rem;
  overflow: hidden;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 28%),
    linear-gradient(180deg, #f6fbff 0%, #f2f7fb 45%, #f7fafc 100%);
  color: var(--hf-text);
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}

.hf-sidebar,
.hf-chat-shell,
.hf-detail-shell {
  border: 1px solid var(--hf-border);
  border-radius: var(--hf-radius-shell);
  background: var(--hf-surface-strong);
  overflow: hidden;
}

.hf-sidebar {
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 0.95rem;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(239, 244, 248, 0.98) 100%);
}

.hf-sidebar-brand {
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 0 0.8rem;
  padding: 0.35rem 0 0.95rem;
  border-bottom: 1px solid rgba(15, 23, 42, 0.06);
}

.hf-sidebar-brand img {
  display: block;
  width: min(100%, 268px);
  height: auto;
}

.hf-live-button,
.hf-new-task-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  min-height: 2.58rem;
  margin-bottom: 0.62rem;
  border: 1px solid rgba(15, 107, 220, 0.18);
  border-radius: var(--hf-radius-control);
  background: linear-gradient(135deg, var(--hf-accent) 0%, var(--hf-accent-strong) 100%);
  color: #ffffff;
  font-size: 0.92rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.hf-new-task-button {
  background: #ffffff;
  color: var(--hf-accent);
}

.hf-history-list {
  min-height: 0;
  overflow: auto;
}

.hf-history-cards {
  display: flex;
  flex-direction: column;
  gap: 0.48rem;
}

.hf-history-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-control);
  background: rgba(255, 255, 255, 0.96);
  transition: border-color 140ms ease, background 140ms ease;
}

.hf-history-card:hover,
.hf-history-card.is-active {
  border-color: rgba(11, 132, 219, 0.34);
  background: linear-gradient(180deg, rgba(239, 248, 255, 0.96) 0%, rgba(255, 255, 255, 0.99) 100%);
}

.hf-history-card-main {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  width: 100%;
  padding: 0.72rem 0.76rem;
}

.hf-history-card-title {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  color: #0f172a;
  font-size: 0.9rem;
  font-weight: 600;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hf-history-card-actions {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 0.35rem;
}

.hf-history-card-btn {
  height: 1.8rem;
  padding: 0 0.52rem;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-control);
  background: rgba(248, 250, 252, 0.96);
  color: #475569;
  font-size: 0.7rem;
  font-weight: 600;
}

.hf-history-card-btn.is-danger {
  border-color: rgba(239, 68, 68, 0.16);
  background: rgba(254, 242, 242, 0.98);
  color: #dc2626;
}

.hf-content-shell {
  min-width: 0;
  min-height: 0;
  height: calc(100dvh - 0.84rem);
}

.hf-main {
  display: grid;
  grid-template-columns: minmax(620px, 1fr) minmax(360px, 32vw);
  gap: 0.85rem;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

.hf-chat-shell,
.hf-detail-shell {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.hf-task-header {
  flex: 0 0 auto;
  margin: 0;
  padding: 1.2rem 1.35rem 1rem;
  border-bottom: 1px solid var(--hf-border);
}

.hf-task-header h2 {
  margin: 0;
  overflow: hidden;
  color: var(--hf-text);
  font-size: 2.14rem;
  font-weight: 700;
  letter-spacing: -0.045em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hf-run-overview-shell {
  flex: 0 0 auto;
  padding: 0 1rem 0.72rem;
}

.hf-run-overview {
  border: 1px solid var(--hf-border);
  border-radius: var(--hf-radius-panel);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(243, 247, 250, 0.98) 100%);
  overflow: hidden;
}

.hf-run-overview summary::-webkit-details-marker {
  display: none;
}

.hf-run-overview-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.78rem;
  padding: 0.8rem 0.9rem;
  list-style: none;
  cursor: pointer;
  background: linear-gradient(180deg, rgba(230, 237, 243, 0.82) 0%, rgba(255, 255, 255, 0.22) 100%);
}

.hf-run-overview-summary-main {
  min-width: 0;
  flex: 1 1 auto;
}

.hf-run-overview-eyebrow,
.hf-run-overview-label,
.hf-process-card-eyebrow,
.hf-process-card-label {
  color: var(--hf-text-muted);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hf-run-overview-eyebrow {
  margin-bottom: 0.24rem;
}

.hf-run-overview-summary-line {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.hf-run-overview-summary-line h3 {
  margin: 0;
  font-size: 1.32rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.hf-run-overview-status,
.hf-run-overview-stage,
.hf-run-overview-toggle,
.hf-process-pill,
.hf-process-stage {
  flex: 0 0 auto;
  padding: 0.22rem 0.5rem;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-chip);
  background: rgba(255, 255, 255, 0.86);
  color: var(--hf-text-muted);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.hf-run-overview-status {
  border-color: var(--hf-border);
  background: var(--hf-surface-contrast);
  color: var(--hf-text);
}

.hf-run-overview-objective {
  color: var(--hf-text);
  font-size: 1.14rem;
  font-weight: 600;
  line-height: 1.5;
}

.hf-run-overview-note {
  margin-top: 0.24rem;
  color: var(--hf-text-muted);
  font-size: 1.08rem;
  line-height: 1.52;
}

.hf-run-overview-detail {
  padding: 0 0.9rem 0.88rem;
  border-top: 1px solid var(--hf-border);
}

.hf-stage-chip-row {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.34rem;
  padding-top: 0.72rem;
}

.hf-stage-chip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  min-width: 0;
  min-height: 2.55rem;
  padding: 0.48rem 0.56rem;
  border: 1px solid rgba(5, 150, 105, 0.18);
  border-radius: var(--hf-radius-chip);
  background: rgba(236, 253, 245, 0.96);
}

.hf-stage-chip-label {
  font-size: 0.98rem;
  font-weight: 700;
}

.hf-stage-chip-state {
  color: #047857;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.hf-run-overview-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.56rem;
  padding-top: 0.72rem;
}

.hf-run-overview-panel {
  min-width: 0;
  padding: 0.72rem 0.78rem;
  border: 1px solid var(--hf-border);
  border-radius: var(--hf-radius-control);
  background: rgba(244, 248, 251, 0.92);
}

.hf-run-overview-list,
.hf-run-overview-watchouts {
  margin: 0;
  padding: 0;
  list-style: none;
}

.hf-run-overview-list li,
.hf-run-overview-watchouts li {
  display: flex;
  gap: 0.65rem;
  align-items: flex-start;
  padding: 0.32rem 0;
  color: var(--hf-text);
  font-size: 0.96rem;
  line-height: 1.5;
}

.hf-run-overview-list li + li,
.hf-run-overview-watchouts li + li {
  border-top: 1px solid rgba(15, 23, 42, 0.06);
}

.hf-run-overview-list li span {
  flex: 0 0 auto;
  min-width: 1.65rem;
  color: var(--hf-accent);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.06em;
}

.hf-chatbot {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
}

.hf-chat-scroll {
  display: flex;
  flex-direction: column;
  gap: 0.56rem;
  height: 100%;
  overflow: auto;
  padding: 0.2rem 0.52rem 0.7rem;
}

.hf-message-row {
  display: flex;
  width: 100%;
  padding: 0 0.3rem;
}

.hf-message-row.is-user {
  justify-content: flex-end;
}

.hf-message {
  border-radius: var(--hf-radius-control);
  box-shadow: none;
}

.hf-user-message {
  width: min(88%, 62rem);
  max-width: min(88%, 62rem);
  padding: 0.72rem 0.82rem;
  border: 1px solid rgba(15, 23, 42, 0.06);
  background: linear-gradient(180deg, rgba(232, 238, 244, 0.96) 0%, rgba(245, 249, 252, 0.98) 100%);
  color: var(--hf-text);
}

.hf-bot-message {
  width: 100%;
  border: 1px solid rgba(15, 23, 42, 0.07);
  background: rgba(246, 249, 252, 0.98);
}

.hf-message p,
.hf-message li {
  font-size: 1.22rem;
  line-height: 1.7;
}

.hf-user-message p {
  margin: 0;
}

.hf-chat-attachments,
.hf-composer-attachments {
  display: flex;
  flex-wrap: wrap;
  gap: 0.42rem;
  margin-top: 0.68rem;
}

.hf-composer-attachments {
  margin: 0 0 0.55rem;
}

.hf-chat-attachment,
.hf-composer-attachment {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  min-width: 0;
  padding: 0.36rem 0.7rem;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-chip);
  background: rgba(248, 250, 252, 0.98);
  color: #334155;
  font-size: 0.78rem;
  line-height: 1.2;
}

.hf-chat-attachment code,
.hf-composer-attachment code {
  padding: 0;
  border: none;
  background: transparent;
  color: inherit;
  font-size: inherit;
}

.hf-process-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-panel);
  background: linear-gradient(180deg, rgba(244, 248, 251, 0.98) 0%, rgba(255, 255, 255, 0.98) 100%);
  padding: 0.8rem 0.9rem;
}

.hf-process-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.65rem;
}

.hf-process-card-main {
  min-width: 0;
  flex: 1 1 auto;
}

.hf-process-card-title {
  margin: 0;
  color: var(--hf-text);
  font-size: 1.22rem;
  font-weight: 700;
  line-height: 1.42;
}

.hf-process-card-meta {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  flex-wrap: wrap;
}

.hf-process-pill.is-completed,
.hf-process-stage.is-done {
  border-color: rgba(5, 150, 105, 0.18);
  background: rgba(236, 253, 245, 0.96);
  color: #047857;
}

.hf-process-stage-rail {
  display: flex;
  gap: 0.3rem;
  flex-wrap: wrap;
  margin-top: 0.6rem;
}

.hf-process-card-subcopy {
  margin: 0.48rem 0 0;
  color: var(--hf-text);
  font-size: 1.05rem;
  line-height: 1.54;
}

.hf-process-card-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(0, 1fr);
  gap: 0.56rem;
  margin-top: 0.66rem;
}

.hf-process-card-block {
  min-width: 0;
  padding: 0.62rem 0.68rem;
  border: 1px solid rgba(15, 23, 42, 0.06);
  border-radius: var(--hf-radius-control);
  background: rgba(255, 255, 255, 0.88);
}

.hf-process-card-list {
  margin: 0;
  padding-left: 1rem;
  color: var(--hf-text);
  font-size: 1.03rem;
  line-height: 1.54;
}

.hf-process-card-copy {
  margin: 0;
  color: var(--hf-text);
  font-size: 1.08rem;
  line-height: 1.58;
}

.hf-inline-gallery {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.54rem;
  width: 100%;
}

.hf-gallery-item {
  min-width: 0;
  padding: 0.2rem;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-panel);
  background: linear-gradient(180deg, rgba(244, 248, 251, 0.98) 0%, rgba(255, 255, 255, 0.99) 100%);
  overflow: hidden;
}

.hf-gallery-item.is-active,
.hf-gallery-item:hover {
  border-color: rgba(11, 132, 219, 0.34);
}

.hf-gallery-title {
  display: flex;
  justify-content: center;
  padding: 0.08rem 0.16rem 0.18rem;
  color: var(--hf-text);
  font-size: 0.98rem;
  font-weight: 700;
  text-align: center;
}

.hf-mini-chart {
  display: block;
  aspect-ratio: 1.48;
  width: 100%;
  border-radius: var(--hf-radius-control);
  background: #ffffff;
}

.hf-mini-chart svg {
  width: 100%;
  height: 100%;
}

.hf-mini-bars {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: end;
  gap: 0.45rem;
  height: 100%;
  padding: 1rem;
}

.hf-mini-bar {
  display: block;
  min-height: 14px;
  border-radius: 4px 4px 0 0;
  background: linear-gradient(180deg, #0ea5e9, #14b8a6);
}

.hf-composer-shell {
  flex: 0 0 auto;
  margin-top: auto;
  padding: 0.75rem 0.9rem 0.9rem;
  border-top: 1px solid var(--hf-border);
  background: linear-gradient(180deg, rgba(247, 250, 252, 0) 0%, rgba(247, 250, 252, 0.92) 30%, rgba(247, 250, 252, 0.98) 100%);
}

.hf-composer {
  display: grid;
  grid-template-columns: 2.75rem minmax(0, 1fr) 2.75rem;
  align-items: end;
  gap: 0.5rem;
  padding: 0.42rem 0.48rem 0.42rem 0.3rem;
  border: 1px solid rgba(15, 23, 42, 0.1);
  border-radius: var(--hf-radius-panel);
  background: rgba(255, 255, 255, 0.98);
}

.hf-composer textarea {
  min-height: 2.75rem;
  resize: none;
  border: none;
  outline: none;
  background: transparent;
  color: #64748b;
  font-size: 1.06rem;
  line-height: 1.6;
  padding: 0.72rem 0;
}

.hf-upload-button,
.hf-submit-button {
  width: 2.75rem;
  min-width: 2.75rem;
  height: 2.75rem;
  border-radius: var(--hf-radius-control);
  font-size: 1.08rem;
  font-weight: 700;
}

.hf-upload-button {
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: rgba(248, 250, 252, 0.96);
  color: #0f172a;
}

.hf-submit-button {
  border: none;
  background: linear-gradient(135deg, var(--hf-accent) 0%, var(--hf-accent-strong) 100%);
  color: #ffffff;
}

.hf-workspace-shell {
  min-width: 0;
  min-height: 0;
}

.hf-detail-nav {
  display: flex;
  flex: 0 0 auto;
  gap: 0.55rem;
  padding: 1rem;
  border-bottom: 1px solid var(--hf-border);
  background: rgba(249, 251, 253, 0.92);
}

.hf-detail-switch {
  flex: 1 1 0;
  min-width: 0;
  height: 2.5rem;
  border-radius: var(--hf-radius-control);
  font-weight: 600;
  letter-spacing: -0.01em;
}

.hf-detail-switch.primary {
  border: 1px solid rgba(15, 107, 220, 0.18);
  background: linear-gradient(135deg, var(--hf-accent) 0%, var(--hf-accent-strong) 100%);
  color: #ffffff;
}

.hf-detail-switch.secondary {
  border: 1px solid rgba(15, 23, 42, 0.06);
  background: rgba(255, 255, 255, 0.86);
  color: #334155;
}

.hf-detail-panel {
  flex: 1 1 auto;
  min-height: 0;
  padding: 1rem;
  overflow: hidden;
}

.hf-workspace-row {
  display: flex;
  flex: 1 1 auto;
  gap: 0.9rem;
  align-items: stretch;
  min-height: 0;
  height: 100%;
}

.hf-browser-pane,
.hf-preview-pane,
.hf-memory-pane,
.hf-advanced-toolbar,
.hf-trace-panel {
  border: 1px solid var(--hf-border);
  border-radius: var(--hf-radius-panel);
  background: var(--hf-surface-muted);
}

.hf-browser-pane,
.hf-preview-pane,
.hf-memory-pane,
.hf-advanced-toolbar {
  padding: 1rem 1rem 1.05rem;
}

.hf-browser-pane,
.hf-preview-pane,
.hf-memory-pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  gap: 0.75rem;
}

.hf-browser-pane {
  flex: 0 0 41%;
  min-width: 11.8rem;
  overflow: auto;
}

.hf-preview-pane {
  flex: 1 1 auto;
  min-width: 0;
  overflow: auto;
}

.hf-browser-pane h3,
.hf-preview-pane h3,
.hf-memory-pane h3,
.hf-advanced-toolbar h3 {
  margin: 0;
}

.hf-tree-root {
  display: flex;
  flex-direction: column;
  gap: 0.12rem;
  padding: 0.08rem 0;
}

.hf-tree-folder,
.hf-tree-file {
  position: relative;
  min-width: 0;
}

.hf-tree-folder::before,
.hf-tree-file::before {
  content: '';
  position: absolute;
  left: -0.42rem;
  top: 0.38rem;
  bottom: 0.38rem;
  width: 1px;
  background: rgba(15, 23, 42, 0.08);
}

.hf-tree-depth-0::before {
  display: none;
}

.hf-tree-folder {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  min-height: 1.88rem;
  padding: 0.3rem 0.08rem 0.14rem;
  color: #334155;
  font-size: 1.08rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.hf-tree-folder-prefix {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 0.78rem;
  color: var(--hf-accent);
  font-family: 'IBM Plex Mono', 'SFMono-Regular', monospace;
  font-size: 0.78rem;
  font-weight: 700;
}

.hf-tree-folder-name,
.hf-tree-file-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hf-tree-file {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.55rem;
  min-height: 2.18rem;
  padding: 0.34rem 0.42rem;
  border: 1px solid transparent;
  border-radius: var(--hf-radius-control);
  background: transparent;
  cursor: pointer;
  text-align: left;
}

.hf-tree-file:hover,
.hf-tree-file.is-active {
  border-color: rgba(11, 132, 219, 0.34);
  background: linear-gradient(180deg, rgba(239, 248, 255, 0.96) 0%, rgba(255, 255, 255, 0.99) 100%);
}

.hf-tree-file-main {
  display: flex;
  align-items: center;
  gap: 0.42rem;
  min-width: 0;
  flex: 1 1 auto;
}

.hf-tree-node-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  min-width: 1.56rem;
  height: 1.02rem;
  padding: 0 0.24rem;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-chip);
  background: rgba(248, 250, 252, 0.98);
  color: #526171;
  font-family: 'IBM Plex Mono', 'SFMono-Regular', monospace;
  font-size: 0.55rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.hf-tree-node-badge.is-image {
  border-color: rgba(37, 99, 235, 0.16);
  background: rgba(239, 246, 255, 0.96);
  color: #1d4ed8;
}

.hf-tree-node-badge.is-table {
  border-color: rgba(5, 150, 105, 0.16);
  background: rgba(236, 253, 245, 0.98);
  color: #047857;
}

.hf-tree-node-badge.is-report,
.hf-tree-node-badge.is-text {
  border-color: rgba(100, 116, 139, 0.16);
  background: rgba(248, 250, 252, 0.98);
  color: #475569;
}

.hf-tree-file-name {
  color: #0f172a;
  font-size: 1.04rem;
  font-weight: 600;
  line-height: 1.35;
}

.hf-tree-depth-1 {
  margin-left: 1.15rem;
}

.hf-tree-depth-2 {
  margin-left: 2.3rem;
}

.hf-tree-depth-3 {
  margin-left: 3.45rem;
}

.hf-tree-depth-4 {
  margin-left: 4.6rem;
}

.hf-tree-depth-5 {
  margin-left: 5.75rem;
}

.hf-preview-header,
.hf-chart-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.hf-preview-header span,
.hf-chart-head strong {
  color: var(--hf-text-muted);
  font-size: 0.76rem;
  font-weight: 700;
}

.hf-preview-pane :deep(.prose) {
  max-width: none;
  color: var(--hf-text);
}

.hf-preview-pane :deep(.prose p),
.hf-preview-pane :deep(.prose li),
.hf-preview-pane :deep(.prose div),
.hf-preview-pane :deep(.prose span),
.hf-preview-pane :deep(.prose strong),
.hf-preview-pane :deep(.prose em) {
  font-size: 1.18rem;
  line-height: 1.66;
}

.hf-code-preview,
.hf-log-preview,
.hf-trace-panel {
  margin: 0;
  overflow: auto;
  padding: 0.88rem;
  border-radius: var(--hf-radius-control);
  background: #102033;
  color: #f8fafc;
  font-family: 'IBM Plex Mono', 'SFMono-Regular', monospace;
  font-size: 0.8rem;
  line-height: 1.65;
}

.hf-table-preview {
  overflow: auto;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-control);
  background: #ffffff;
}

.hf-table-preview table {
  min-width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.hf-table-preview th,
.hf-table-preview td {
  padding: 0.55rem 0.62rem;
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
  text-align: left;
  white-space: nowrap;
}

.hf-table-preview th {
  color: var(--hf-text-muted);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hf-chart-preview {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-control);
  background: #ffffff;
  padding: 0.8rem;
}

.hf-chart-preview svg {
  width: 100%;
  height: auto;
}

.hf-chart-head {
  margin-bottom: 0.65rem;
}

.hf-chart-head span {
  font-size: 1rem;
  font-weight: 700;
}

.hf-chart-bars {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  height: 18rem;
  align-items: end;
}

.hf-chart-bar-cell {
  display: flex;
  min-width: 0;
  height: 100%;
  flex-direction: column;
  justify-content: flex-end;
  gap: 0.35rem;
  text-align: center;
}

.hf-chart-bar-track {
  display: flex;
  min-height: 0;
  flex: 1 1 auto;
  align-items: end;
}

.hf-chart-bar-fill {
  width: 100%;
  min-height: 16px;
  border-radius: 6px 6px 0 0;
  background: linear-gradient(180deg, #0ea5e9, #14b8a6);
}

.hf-memory-pane {
  overflow: auto;
}

.hf-memory-stack {
  display: flex;
  flex-direction: column;
  gap: 0.72rem;
}

.hf-memory-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-panel);
  background: rgba(255, 255, 255, 0.96);
  padding: 0.9rem 0.95rem;
}

.hf-memory-card h4 {
  margin: 0;
  color: var(--hf-text);
  font-size: 1.02rem;
  font-weight: 700;
}

.hf-memory-card p {
  margin: 0.36rem 0 0;
  color: var(--hf-text);
  font-size: 0.98rem;
  line-height: 1.58;
}

.hf-memory-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.58rem;
}

.hf-memory-chip {
  display: inline-flex;
  align-items: center;
  min-width: 0;
  padding: 0.24rem 0.5rem;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: var(--hf-radius-chip);
  background: rgba(244, 248, 251, 0.96);
  color: var(--hf-text-muted);
  font-size: 0.78rem;
  font-weight: 700;
}

.hf-advanced-toolbar {
  margin-bottom: 0.9rem;
}

.hf-advanced-toolbar label {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin-top: 0.72rem;
  color: var(--hf-text-muted);
  font-size: 0.92rem;
}

.hf-trace-panel {
  height: calc(100% - 5rem);
  background: var(--hf-surface-muted);
  color: var(--hf-text);
}

.hf-trace-line {
  padding: 0.42rem 0;
  border-bottom: 1px solid rgba(15, 23, 42, 0.06);
  font-size: 0.92rem;
  line-height: 1.5;
}

@media (max-width: 1280px) {
  .hf-app-demo {
    grid-template-columns: 300px minmax(0, 1fr);
  }

  .hf-main {
    grid-template-columns: minmax(0, 1fr);
    overflow: auto;
  }

  .hf-workspace-shell {
    min-height: 34rem;
  }
}

@media (max-width: 900px) {
  .hf-app-demo {
    height: auto;
    min-height: 100dvh;
    grid-template-columns: 1fr;
    overflow: auto;
  }

  .hf-content-shell {
    height: auto;
  }

  .hf-main {
    height: auto;
  }

  .hf-chat-shell,
  .hf-detail-shell {
    min-height: 34rem;
  }

  .hf-run-overview-grid,
  .hf-process-card-grid,
  .hf-workspace-row {
    grid-template-columns: 1fr;
  }

  .hf-workspace-row {
    flex-direction: column;
  }

  .hf-stage-chip-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .hf-inline-gallery {
    grid-template-columns: 1fr;
  }

  .hf-user-message {
    width: min(96%, 62rem);
    max-width: min(96%, 62rem);
  }
}
</style>
