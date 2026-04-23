<script setup lang="ts">
import { GlobalWorkerOptions, getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs'
import { onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps<{
  src: string
}>()

GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/legacy/build/pdf.worker.min.mjs', import.meta.url).toString()

const container = ref<HTMLDivElement | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const pageCount = ref(0)
let renderToken = 0

const clearContainer = () => {
  if (container.value) {
    container.value.innerHTML = ''
  }
}

const renderDocument = async () => {
  const currentToken = ++renderToken
  clearContainer()
  loading.value = true
  error.value = null

  try {
    const documentProxy = await getDocument(props.src).promise
    pageCount.value = documentProxy.numPages

    for (let pageNumber = 1; pageNumber <= documentProxy.numPages; pageNumber += 1) {
      if (currentToken !== renderToken || !container.value) return

      const page = await documentProxy.getPage(pageNumber)
      const viewport = page.getViewport({ scale: 1.15 })
      const wrapper = document.createElement('div')
      wrapper.className = 'overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm'
      const canvas = document.createElement('canvas')
      const context = canvas.getContext('2d')
      if (!context) continue

      canvas.width = viewport.width
      canvas.height = viewport.height
      wrapper.appendChild(canvas)
      container.value.appendChild(wrapper)

      await page.render({
        canvas,
        canvasContext: context,
        viewport,
      }).promise
    }
  } catch (caughtError) {
    error.value = caughtError instanceof Error ? caughtError.message : String(caughtError)
  } finally {
    if (currentToken === renderToken) {
      loading.value = false
    }
  }
}

watch(
  () => props.src,
  () => {
    void renderDocument()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  renderToken += 1
  clearContainer()
})
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
      <span>{{ loading ? 'Rendering PDF…' : `${pageCount} page${pageCount === 1 ? '' : 's'}` }}</span>
      <a :href="props.src" target="_blank" rel="noreferrer" class="font-semibold text-slate-900">Open in new tab</a>
    </div>
    <div v-if="error" class="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
      {{ error }}
    </div>
    <div ref="container" class="space-y-4" />
  </div>
</template>
