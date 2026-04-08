<script setup lang="ts">
import type { SnapshotQuestion } from '../../domain/evaluation'

defineProps<{
  questions: SnapshotQuestion[]
  currentQuestionId: string | null
  answeredIds: Set<string>
}>()

defineEmits<{
  select: [index: number]
}>()
</script>

<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <div class="text-sm font-semibold text-slate-900">Question Navigator</div>
      <div class="text-xs text-slate-500">{{ answeredIds.size }}/{{ questions.length }} answered</div>
    </div>
    <div class="grid grid-cols-5 gap-2">
      <button
        v-for="(question, index) in questions"
        :key="question.id"
        type="button"
        class="rounded-2xl px-3 py-3 text-sm font-semibold transition"
        :class="
          question.id === currentQuestionId
            ? 'bg-slate-950 text-white'
            : answeredIds.has(question.id)
              ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 hover:bg-emerald-100'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900'
        "
        @click="$emit('select', index)"
      >
        {{ index + 1 }}
      </button>
    </div>
  </div>
</template>
