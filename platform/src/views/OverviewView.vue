<script setup lang="ts">
import { ref } from 'vue'

import {
  benchmarkSuites,
  citation,
  citationLinks,
  featuredBenchmarks,
  frameworkStages,
  projectMeta,
} from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'
import SectionHeader from '../components/ui/SectionHeader.vue'

const isExternalLink = (href: string) => href.startsWith('http')

const citationButtonLabel = ref('Copy BibTeX')
let citationResetTimer: ReturnType<typeof setTimeout> | null = null

const copyCitation = async () => {
  try {
    await navigator.clipboard.writeText(citation)
    citationButtonLabel.value = 'Copied'
  } catch {
    citationButtonLabel.value = 'Copy failed'
  }

  if (citationResetTimer) {
    clearTimeout(citationResetTimer)
  }
  citationResetTimer = setTimeout(() => {
    citationButtonLabel.value = 'Copy BibTeX'
  }, 1800)
}
</script>

<template>
  <AppShell content-width="wide">
    <section class="py-12 sm:py-16 xl:py-20">
      <div class="max-w-6xl space-y-8">
        <div
          class="inline-flex rounded-full border border-slate-950/10 bg-white/84 px-4 py-1.5 text-xs font-semibold tracking-[0.24em] text-slate-500 uppercase backdrop-blur"
        >
          {{ projectMeta.eyebrow }}
        </div>

        <div class="space-y-6">
          <h1 class="max-w-5xl text-[clamp(2.65rem,4.4vw,4.6rem)] font-semibold leading-[0.97] tracking-[-0.065em] text-slate-950">
            {{ projectMeta.title }}
          </h1>

          <div class="max-w-3xl space-y-4 text-lg leading-8 text-slate-600">
            <p v-for="paragraph in projectMeta.abstract" :key="paragraph">
              {{ paragraph }}
            </p>
          </div>
        </div>

        <AppCard class="max-w-4xl border-slate-200/80 bg-white/84">
          <div class="space-y-5">
            <div>
              <div class="text-sm font-semibold tracking-[0.18em] text-slate-500 uppercase">Authors</div>
              <p class="mt-3 text-base leading-8 text-slate-900">
                {{ projectMeta.authors.join(', ') }}
              </p>
            </div>

            <div class="border-t border-slate-200 pt-5">
              <div class="text-sm font-semibold tracking-[0.18em] text-slate-500 uppercase">Affiliations</div>
              <p class="mt-3 text-sm leading-7 text-slate-600">
                {{ projectMeta.affiliations.join(' · ') }}
              </p>
            </div>
          </div>
        </AppCard>

        <div class="flex flex-wrap gap-3">
          <template v-for="link in projectMeta.links" :key="link.href">
            <RouterLink v-if="!isExternalLink(link.href)" :to="link.href">
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </RouterLink>
            <a
              v-else
              :href="link.href"
              :target="isExternalLink(link.href) ? '_blank' : undefined"
              :rel="isExternalLink(link.href) ? 'noreferrer' : undefined"
            >
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </a>
          </template>
        </div>
      </div>
    </section>

    <section class="pb-8 sm:pb-12">
      <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <AppCard
          v-for="fact in projectMeta.facts"
          :key="fact.label"
          class="border-slate-200/80 bg-white/82"
        >
          <div class="space-y-3">
            <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ fact.label }}</div>
            <p class="text-lg font-semibold tracking-[-0.04em] text-slate-950">
              {{ fact.value }}
            </p>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="framework" class="space-y-10 py-16 sm:py-18">
      <SectionHeader
        eyebrow="Framework"
        title="Meta -> Executor -> Evaluator -> Reflector."
        description="HealthFlow keeps the runtime small and inspectable: plan the work, execute it in a visible workspace, judge the attempt, then write back reusable experience."
      />

      <div class="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
        <AppCard v-for="stage in frameworkStages" :key="stage.title" class="border-slate-200/80 bg-white/82">
          <div class="space-y-4">
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ stage.title }}</div>
            <p class="text-base leading-8 text-slate-600">{{ stage.description }}</p>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="benchmarks" class="space-y-10 py-16 sm:py-18">
      <SectionHeader
        eyebrow="Benchmarks"
        title="Five benchmark suites, with two platform-facing review surfaces."
        description="The repository spans five benchmark suites. The platform focuses on the two surfaces that need direct artifact or report review: MedAgentBoard and EHRFlowBench."
      />

      <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <AppCard v-for="item in benchmarkSuites" :key="item.title" class="border-slate-200/80 bg-white/82">
          <div class="space-y-3">
            <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.category }}</div>
            <h2 class="text-2xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.title }}</h2>
            <p class="text-sm leading-7 text-slate-600">{{ item.body }}</p>
          </div>
        </AppCard>
      </div>

      <div class="grid gap-5 xl:grid-cols-2">
        <AppCard v-for="item in featuredBenchmarks" :key="item.title" class="border-slate-200/80 bg-white/84">
          <div class="space-y-5">
            <div>
              <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.eyebrow }}</div>
              <h3 class="mt-3 text-3xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.title }}</h3>
            </div>
            <p class="max-w-2xl text-base leading-8 text-slate-600">{{ item.body }}</p>
            <ul class="space-y-3 text-sm leading-7 text-slate-700">
              <li v-for="bullet in item.bullets" :key="bullet" class="flex gap-3">
                <span class="mt-2 h-2 w-2 rounded-full bg-slate-900" />
                <span>{{ bullet }}</span>
              </li>
            </ul>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="citation" class="space-y-10 py-16 sm:py-18">
      <SectionHeader
        eyebrow="Citation"
        title="Paper metadata and a usable BibTeX block."
        description="This stays simple: direct paper link, direct code link, and a citation block that can be copied immediately."
      />

      <AppCard class="border-slate-200/80 bg-white/84">
        <div class="grid gap-8 xl:grid-cols-[320px_minmax(0,1fr)] xl:items-start">
          <div class="space-y-4">
            <div
              v-for="item in citationLinks"
              :key="item.label"
              class="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-5 py-4"
            >
              <div class="text-xs font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.label }}</div>
              <a
                :href="item.href"
                target="_blank"
                rel="noreferrer"
                class="mt-2 block text-lg font-semibold tracking-[-0.03em] text-slate-950"
              >
                {{ item.href.replace('https://', '') }}
              </a>
              <p class="mt-2 text-sm leading-7 text-slate-600">{{ item.body }}</p>
            </div>

            <AppButton variant="secondary" class="w-full" @click="copyCitation">
              {{ citationButtonLabel }}
            </AppButton>
          </div>

          <div class="space-y-4">
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">BibTeX</div>
            <pre class="overflow-x-auto rounded-[1.75rem] bg-slate-950 px-5 py-5 text-sm leading-7 text-slate-100"><code>{{ citation }}</code></pre>
          </div>
        </div>
      </AppCard>
    </section>
  </AppShell>
</template>
