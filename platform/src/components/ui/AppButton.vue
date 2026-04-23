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
    class="inline-flex items-center justify-center rounded-full px-4 py-2.5 text-sm font-semibold tracking-[-0.01em] transition duration-200 disabled:cursor-not-allowed disabled:opacity-50"
    :class="
      props.variant === 'primary'
        ? 'border border-sky-300 bg-[linear-gradient(135deg,rgba(224,242,254,0.96),rgba(220,252,231,0.96))] text-sky-950 shadow-[0_12px_28px_rgba(56,189,248,0.16)] hover:-translate-y-0.5 hover:shadow-[0_18px_36px_rgba(56,189,248,0.2)] active:translate-y-0'
        : props.variant === 'secondary'
          ? 'border border-sky-200 bg-white/92 text-sky-900 shadow-[0_10px_24px_rgba(56,189,248,0.08)] hover:-translate-y-0.5 hover:border-sky-300 hover:bg-sky-50/80 active:translate-y-0'
          : 'border border-transparent bg-transparent text-sky-700 hover:bg-sky-50 hover:text-sky-900'
    "
  >
    <slot />
  </button>
</template>
