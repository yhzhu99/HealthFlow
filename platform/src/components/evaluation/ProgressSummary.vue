<script setup lang="ts">
defineProps<{
  answered: number
  total: number
}>()
</script>

<template>
  <div class="space-y-3 rounded-[1.5rem] border border-slate-200 bg-slate-50 p-4">
    <div class="flex items-center justify-between text-sm">
      <span class="font-semibold text-slate-900">Progress</span>
      <span class="text-slate-500">{{ answered }}/{{ total }}</span>
    </div>
    <div class="h-2 rounded-full bg-slate-200">
      <div class="h-full rounded-full bg-slate-950 transition-all" :style="{ width: `${total ? (answered / total) * 100 : 0}%` }" />
    </div>
  </div>
</template>
