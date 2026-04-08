<script setup lang="ts">
import { computed } from 'vue'

import type { BlindCandidateSlot } from '../../domain/evaluation'
import { renderMarkdown } from '../../lib/markdown'
import AppButton from '../ui/AppButton.vue'
import AppCard from '../ui/AppCard.vue'
import ArtifactViewer from './ArtifactViewer.vue'

const props = defineProps<{
  slot: BlindCandidateSlot
  selectedChoice: string | null
}>()

const emit = defineEmits<{
  select: [choice: string]
}>()

const renderedAnswer = computed(() => renderMarkdown(props.slot.candidate.answerText || 'No final answer was recorded.'))
</script>

<template>
  <AppCard class="h-full">
    <div class="flex h-full flex-col gap-5">
      <div class="flex items-start justify-between gap-4">
        <div>
          <div class="text-xs font-semibold tracking-[0.24em] text-slate-500 uppercase">Candidate {{ props.slot.slot }}</div>
          <div class="mt-2 text-sm text-slate-500">
            {{ props.slot.candidate.artifacts.length }} artifact{{ props.slot.candidate.artifacts.length === 1 ? '' : 's' }}
          </div>
        </div>
        <AppButton
          :variant="props.selectedChoice === props.slot.slot ? 'primary' : 'secondary'"
          @click="emit('select', props.slot.slot)"
        >
          {{ props.selectedChoice === props.slot.slot ? 'Selected' : `Select ${props.slot.slot}` }}
        </AppButton>
      </div>

      <div class="prose prose-slate max-w-none text-sm leading-7 text-slate-700" v-html="renderedAnswer" />

      <ArtifactViewer :artifacts="props.slot.candidate.artifacts" :title="`Candidate ${props.slot.slot} Artifacts`" />
    </div>
  </AppCard>
</template>
