<script setup lang="ts">
import {
  abstractParagraphs,
  benchmarkCards,
  citation,
  frameworkStages,
  heroSignals,
  projectMeta,
  resourceLinks,
  resultHighlights,
} from '../content/site'
import AppShell from '../components/layout/AppShell.vue'
import AppButton from '../components/ui/AppButton.vue'
import AppCard from '../components/ui/AppCard.vue'
import SectionHeader from '../components/ui/SectionHeader.vue'

const isExternalLink = (href: string) => href.startsWith('http')
</script>

<template>
  <AppShell content-width="wide">
    <section class="grid items-start gap-8 py-12 sm:py-16 xl:grid-cols-[minmax(0,1.1fr)_420px] xl:gap-10 xl:py-22">
      <div class="space-y-8">
        <div class="inline-flex rounded-full border border-slate-950/10 bg-white/85 px-4 py-1.5 text-xs font-semibold tracking-[0.24em] text-slate-500 uppercase backdrop-blur">
          Paper + Platform
        </div>

        <div class="space-y-6">
          <h1 class="max-w-5xl text-[clamp(3.35rem,6vw,6.25rem)] font-semibold leading-[0.94] tracking-[-0.07em] text-slate-950">
            {{ projectMeta.title }}
          </h1>
          <p class="max-w-3xl text-lg leading-9 text-slate-600 sm:text-[1.28rem]">
            {{ projectMeta.subtitle }}
          </p>
        </div>

        <div class="max-w-4xl space-y-4 rounded-[2rem] border border-white/70 bg-white/72 px-6 py-6 shadow-[0_22px_60px_rgba(15,23,42,0.06)] backdrop-blur-xl">
          <p class="text-base leading-8 text-slate-800">
            {{ projectMeta.authors.join(', ') }}
          </p>
          <p class="text-sm leading-7 text-slate-500">
            Affiliations: {{ projectMeta.affiliations.join(' · ') }}
          </p>
        </div>

        <div class="flex flex-wrap gap-3">
          <template v-for="link in projectMeta.links" :key="link.href">
            <RouterLink v-if="!isExternalLink(link.href)" :to="link.href">
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </RouterLink>
            <a v-else :href="link.href" target="_blank" rel="noreferrer">
              <AppButton :variant="link.kind">{{ link.label }}</AppButton>
            </a>
          </template>
        </div>
      </div>

      <AppCard class="h-fit border-slate-200/80 bg-white/80">
        <div class="space-y-8">
          <div class="space-y-3">
            <div class="text-sm font-semibold tracking-[0.22em] text-slate-500 uppercase">Project Facts</div>
            <div class="space-y-4">
              <div
                v-for="fact in projectMeta.facts"
                :key="fact.label"
                class="rounded-[1.5rem] border border-slate-200 bg-slate-50 px-5 py-4"
              >
                <div class="text-xs font-semibold tracking-[0.18em] text-slate-500 uppercase">{{ fact.label }}</div>
                <div class="mt-2 text-lg font-semibold tracking-[-0.04em] text-slate-950">{{ fact.value }}</div>
              </div>
            </div>
          </div>

          <div class="rounded-[1.5rem] border border-slate-200 bg-slate-950 px-5 py-5 text-slate-100">
            <div class="text-xs font-semibold tracking-[0.22em] text-slate-400 uppercase">Citation Link</div>
            <a
              href="https://arxiv.org/abs/2508.02621"
              target="_blank"
              rel="noreferrer"
              class="mt-3 block text-xl font-semibold tracking-[-0.04em] text-white"
            >
              arXiv:2508.02621
            </a>
            <p class="mt-3 text-sm leading-7 text-slate-300">
              Direct paper record for title, author list, abstract, and BibTeX-compatible citation metadata.
            </p>
          </div>
        </div>
      </AppCard>
    </section>

    <section class="grid gap-4 pb-8 lg:grid-cols-3">
      <AppCard
        v-for="item in heroSignals"
        :key="item.title"
        class="border-slate-200/80 bg-white/72"
      >
        <div class="space-y-3">
          <div class="text-sm font-semibold tracking-[0.18em] text-slate-500 uppercase">{{ item.title }}</div>
          <p class="text-base leading-8 text-slate-600">{{ item.body }}</p>
        </div>
      </AppCard>
    </section>

    <section id="abstract" class="space-y-12 py-16 sm:py-20">
      <SectionHeader
        eyebrow="Abstract"
        title="A runtime built to learn from trajectories, not just finish a single task."
        description="HealthFlow treats planning, execution, critique, and reflection as first-class stages while keeping the product surface small enough for paper readers and benchmark reviewers."
      />
      <div class="grid gap-5 xl:grid-cols-2">
        <AppCard v-for="paragraph in abstractParagraphs" :key="paragraph" class="border-slate-200/80 bg-white/78">
          <p class="max-w-3xl text-base leading-8 text-slate-600">{{ paragraph }}</p>
        </AppCard>
      </div>
    </section>

    <section id="framework" class="space-y-12 py-16 sm:py-20">
      <SectionHeader
        eyebrow="Framework"
        title="Four stages, one inspectable loop."
        description="The runtime underneath the interface is deliberate about how work is planned, executed, judged, and remembered."
      />
      <div class="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
        <AppCard v-for="stage in frameworkStages" :key="stage.title" class="border-slate-200/80 bg-white/78">
          <div class="space-y-4">
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ stage.title }}</div>
            <p class="text-base leading-8 text-slate-600">{{ stage.description }}</p>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="benchmarks" class="space-y-12 py-16 sm:py-20">
      <SectionHeader
        eyebrow="Benchmarks"
        title="Two benchmark families, one clean review flow."
        description="The homepage explains both benchmark surfaces, and the evaluation page turns them into a single workspace with dataset and framework tabs."
      />
      <div class="grid gap-5 xl:grid-cols-2">
        <AppCard v-for="item in benchmarkCards" :key="item.title" class="border-slate-200/80 bg-white/78">
          <div class="space-y-5">
            <div>
              <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.eyebrow }}</div>
              <h2 class="mt-3 text-3xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.title }}</h2>
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

    <section id="results" class="space-y-12 py-16 sm:py-20">
      <SectionHeader
        eyebrow="Why It Matters"
        title="Designed for reproducibility, inspection, and reviewer trust."
        description="The emphasis is not only on getting an answer, but on leaving behind the evidence needed to understand how that answer was produced."
      />
      <div class="grid gap-5 xl:grid-cols-3">
        <AppCard v-for="item in resultHighlights" :key="item.title" class="border-slate-200/80 bg-white/78">
          <div class="space-y-4">
            <div class="text-2xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.title }}</div>
            <p class="text-base leading-8 text-slate-600">{{ item.body }}</p>
          </div>
        </AppCard>
      </div>
    </section>

    <section id="resources" class="space-y-12 py-16 sm:py-20">
      <SectionHeader
        eyebrow="Resources"
        title="Paper, code, evaluation, and citation in one place."
        description="The homepage should function like a proper research project page: clear metadata, direct links, and a citation block that can actually be used."
      />

      <div class="grid gap-5 xl:grid-cols-3">
        <a
          v-for="item in resourceLinks"
          :key="item.title"
          :href="item.href"
          :target="isExternalLink(item.href) ? '_blank' : undefined"
          :rel="isExternalLink(item.href) ? 'noreferrer' : undefined"
        >
          <AppCard class="h-full border-slate-200/80 bg-white/78 transition hover:translate-y-[-2px] hover:shadow-[0_32px_80px_rgba(15,23,42,0.1)]">
            <div class="space-y-4">
              <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">{{ item.title }}</div>
              <div class="text-2xl font-semibold tracking-[-0.04em] text-slate-950">{{ item.label }}</div>
              <p class="text-base leading-8 text-slate-600">{{ item.body }}</p>
            </div>
          </AppCard>
        </a>
      </div>

      <AppCard class="border-slate-200/80 bg-white/80">
        <div class="grid gap-6 xl:grid-cols-[minmax(0,1fr)_240px] xl:items-start">
          <div class="space-y-4">
            <div class="text-sm font-semibold tracking-[0.2em] text-slate-500 uppercase">Citation</div>
            <pre class="overflow-x-auto rounded-[1.5rem] bg-slate-950 px-5 py-5 text-sm leading-7 text-slate-100"><code>{{ citation }}</code></pre>
          </div>

          <div class="space-y-3">
            <a href="https://arxiv.org/abs/2508.02621" target="_blank" rel="noreferrer">
              <AppButton class="w-full">Open Paper Record</AppButton>
            </a>
            <a href="https://github.com/yhzhu99/HealthFlow" target="_blank" rel="noreferrer">
              <AppButton variant="secondary" class="w-full">Open Code Repository</AppButton>
            </a>
          </div>
        </div>
      </AppCard>
    </section>
  </AppShell>
</template>
