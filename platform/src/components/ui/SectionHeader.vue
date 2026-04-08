<script setup lang="ts">
defineProps<{
  eyebrow: string
  title: string
  description?: string
}>()
</script>

<template>
  <div class="max-w-3xl space-y-4">
    <p class="text-sm font-semibold tracking-[0.24em] text-slate-500 uppercase">{{ eyebrow }}</p>
    <h2 class="text-4xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-5xl">{{ title }}</h2>
    <p v-if="description" class="text-lg leading-8 text-slate-600">
      {{ description }}
    </p>
  </div>
</template>
