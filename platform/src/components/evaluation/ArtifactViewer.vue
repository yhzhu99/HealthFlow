<script setup lang="ts">
import { computed, defineAsyncComponent, ref, watch } from 'vue'

import type { SnapshotArtifact } from '../../domain/evaluation'
import { parseDelimitedText } from '../../lib/csv'
import { renderMarkdown } from '../../lib/markdown'
import { toPublicAssetUrl } from '../../lib/snapshot'

const PdfPreview = defineAsyncComponent(() => import('./PdfPreview.vue'))

const props = defineProps<{
  artifacts: SnapshotArtifact[]
  title?: string
}>()

const selectedIndex = ref(0)
const content = ref('')
const error = ref<string | null>(null)
const loading = ref(false)

const selectedArtifact = computed(() => props.artifacts[selectedIndex.value] ?? null)
const selectedUrl = computed(() => (selectedArtifact.value ? toPublicAssetUrl(selectedArtifact.value.relativePath) : ''))
const parsedTable = computed(() => parseDelimitedText(content.value))
const renderedMarkdown = computed(() => renderMarkdown(content.value))

watch(
  () => props.artifacts,
  (artifacts) => {
    if (selectedIndex.value > artifacts.length - 1) {
      selectedIndex.value = 0
    }
  },
)

watch(
  selectedArtifact,
  async (artifact) => {
    content.value = ''
    error.value = null
    loading.value = false
    if (!artifact || ['image', 'pdf', 'download'].includes(artifact.kind)) return

    loading.value = true
    try {
      const response = await fetch(toPublicAssetUrl(artifact.relativePath), { cache: 'force-cache' })
      if (!response.ok) {
        throw new Error(`Failed to load ${artifact.label}`)
      }
      content.value = await response.text()
    } catch (caughtError) {
      error.value = caughtError instanceof Error ? caughtError.message : String(caughtError)
    } finally {
      loading.value = false
    }
  },
  { immediate: true },
)
</script>

<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between gap-3">
      <div class="text-sm font-semibold text-slate-900">{{ props.title ?? 'Artifacts' }}</div>
      <div class="text-xs text-slate-500">{{ props.artifacts.length }} file{{ props.artifacts.length === 1 ? '' : 's' }}</div>
    </div>

    <div class="flex gap-2 overflow-x-auto pb-1">
      <button
        v-for="(artifact, index) in props.artifacts"
        :key="artifact.id"
        type="button"
        class="shrink-0 rounded-full border px-3 py-1.5 text-xs font-semibold transition"
        :class="
          index === selectedIndex
            ? 'border-slate-950 bg-slate-950 text-white'
            : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-900'
        "
        @click="selectedIndex = index"
      >
        {{ artifact.label }}
      </button>
    </div>

    <div v-if="!selectedArtifact" class="rounded-[1.5rem] border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-500">
      No previewable artifacts were found for this candidate.
    </div>

    <div v-else class="space-y-4">
      <div class="flex items-center justify-between rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500">
        <span>{{ selectedArtifact.originalPath }}</span>
        <a :href="selectedUrl" target="_blank" rel="noreferrer" class="font-semibold text-slate-900">Open file</a>
      </div>

      <div v-if="loading" class="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
        Loading artifact preview…
      </div>

      <div v-else-if="error" class="rounded-[1.5rem] border border-rose-200 bg-rose-50 px-4 py-6 text-sm text-rose-700">
        {{ error }}
      </div>

      <div
        v-else-if="selectedArtifact.kind === 'image'"
        class="overflow-hidden rounded-[1.5rem] border border-slate-200 bg-white px-4 py-4"
      >
        <img :src="selectedUrl" :alt="selectedArtifact.label" class="mx-auto max-h-[760px] w-full object-contain" />
      </div>

      <PdfPreview v-else-if="selectedArtifact.kind === 'pdf'" :src="selectedUrl" />

      <div
        v-else-if="selectedArtifact.kind === 'markdown'"
        class="prose prose-slate max-w-none rounded-[1.5rem] border border-slate-200 bg-white px-6 py-5"
        v-html="renderedMarkdown"
      />

      <div v-else-if="selectedArtifact.kind === 'csv'" class="overflow-hidden rounded-[1.5rem] border border-slate-200 bg-white">
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-slate-200 text-sm">
            <thead class="bg-slate-50 text-left text-slate-600">
              <tr>
                <th v-for="header in parsedTable.headers" :key="header" class="px-4 py-3 font-semibold">{{ header }}</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100">
              <tr v-for="(row, rowIndex) in parsedTable.rows" :key="`${selectedArtifact.id}-${rowIndex}`">
                <td v-for="(cell, cellIndex) in row" :key="`${rowIndex}-${cellIndex}`" class="px-4 py-3 text-slate-700">
                  {{ cell }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <pre
        v-else
        class="overflow-x-auto rounded-[1.5rem] border border-slate-200 bg-slate-950 px-4 py-5 text-sm leading-7 text-slate-100"
      ><code>{{ content }}</code></pre>
    </div>
  </div>
</template>
