<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    variant?: 'primary' | 'secondary' | 'ghost'
    type?: 'button' | 'submit'
    disabled?: boolean
  }>(),
  {
    variant: 'primary',
    type: 'button',
    disabled: false,
  },
)
</script>

<template>
  <button
    :type="props.type"
    :disabled="props.disabled"
    class="inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50"
    :class="
      props.variant === 'primary'
        ? 'bg-slate-950 text-white hover:bg-slate-800'
        : props.variant === 'secondary'
          ? 'border border-slate-950/10 bg-white text-slate-900 hover:border-slate-950/20'
          : 'bg-transparent text-slate-600 hover:bg-slate-950/5 hover:text-slate-950'
    "
  >
    <slot />
  </button>
</template>
