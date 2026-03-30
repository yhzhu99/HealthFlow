const { createApp } = Vue;

createApp({
  data() {
    return {
      options: {
        llm_options: [],
        executor_options: [],
        default_executor: "healthflow_agent",
      },
      form: {
        task: "",
        active_llm: "",
        active_executor: "healthflow_agent",
        files: [],
      },
      loading: false,
      result: null,
      artifact: null,
      artifactTitle: "",
      error: "",
    };
  },
  computed: {
    canRun() {
      return !!this.form.task.trim() && !!this.form.active_llm && !this.loading;
    },
  },
  async mounted() {
    await this.fetchOptions();
  },
  methods: {
    async fetchOptions() {
      try {
        const response = await fetch("/api/options");
        const payload = await response.json();
        this.options = payload;
        this.form.active_llm = payload.llm_options[0] || "";
        this.form.active_executor = payload.default_executor || "healthflow_agent";
      } catch (error) {
        this.error = `Failed to load configuration options: ${error}`;
      }
    },
    handleFiles(event) {
      this.form.files = Array.from(event.target.files || []);
    },
    async runTask() {
      this.loading = true;
      this.error = "";
      this.result = null;
      this.artifact = null;

      try {
        const body = new FormData();
        body.append("task", this.form.task);
        body.append("active_llm", this.form.active_llm);
        body.append("active_executor", this.form.active_executor);
        for (const file of this.form.files) {
          body.append("files", file);
        }

        const response = await fetch("/api/run", {
          method: "POST",
          body,
        });
        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.detail || "Run failed");
        }
        this.result = await response.json();
      } catch (error) {
        this.error = String(error);
      } finally {
        this.loading = false;
      }
    },
    async loadArtifact(title, path) {
      if (!path) {
        return;
      }
      this.artifactTitle = title;
      this.artifact = { kind: "loading", content: "Loading..." };
      try {
        const response = await fetch(`/api/artifact?path=${encodeURIComponent(path)}`);
        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.detail || "Artifact load failed");
        }
        this.artifact = await response.json();
      } catch (error) {
        this.artifact = { kind: "text", content: String(error) };
      }
    },
    prettyJson(value) {
      return JSON.stringify(value, null, 2);
    },
  },
  template: `
    <main class="page">
      <section class="hero">
        <div class="hero-copy">
          <p class="eyebrow">HealthFlow</p>
          <h1>EHR-specific orchestration with auditable memory and verifier gating</h1>
          <p class="lede">
            Vue 3 frontend, Python runtime. HealthFlow stays Python-native for orchestration, verification, and memory,
            while this web app provides a cleaner user-facing interface for task execution and artifact inspection.
          </p>
        </div>
        <div class="hero-card">
          <div class="metric">
            <span class="metric-label">Default backend</span>
            <strong>{{ options.default_executor }}</strong>
          </div>
          <div class="metric">
            <span class="metric-label">Reasoning models</span>
            <strong>{{ options.llm_options.length }}</strong>
          </div>
          <div class="metric">
            <span class="metric-label">Artifacts</span>
            <strong>run manifest + verification + memory context</strong>
          </div>
        </div>
      </section>

      <section class="panel run-panel">
        <div class="panel-head">
          <h2>Run HealthFlow</h2>
          <span class="badge" :class="{ busy: loading }">{{ loading ? "Running" : "Ready" }}</span>
        </div>

        <div class="grid">
          <label class="field field-wide">
            <span>Task</span>
            <textarea
              v-model="form.task"
              rows="7"
              placeholder="Analyze the uploaded patients.csv to identify the top 3 risk factors for readmission."
            ></textarea>
          </label>

          <label class="field">
            <span>Reasoning LLM</span>
            <select v-model="form.active_llm">
              <option v-for="llm in options.llm_options" :key="llm" :value="llm">{{ llm }}</option>
            </select>
          </label>

          <label class="field">
            <span>Executor backend</span>
            <select v-model="form.active_executor">
              <option v-for="executor in options.executor_options" :key="executor" :value="executor">{{ executor }}</option>
            </select>
          </label>

          <label class="field field-wide">
            <span>Upload EHR inputs</span>
            <input type="file" multiple @change="handleFiles" />
            <small>{{ form.files.length ? form.files.map(file => file.name).join(", ") : "No files selected." }}</small>
          </label>
        </div>

        <div class="actions">
          <button class="primary" :disabled="!canRun" @click="runTask">{{ loading ? "Running..." : "Start task" }}</button>
          <p class="error" v-if="error">{{ error }}</p>
        </div>
      </section>

      <section v-if="result" class="results">
        <div class="result-head">
          <div>
            <p class="eyebrow">{{ result.success ? "Completed" : "Failed" }}</p>
            <h2>{{ result.final_summary }}</h2>
          </div>
          <div class="result-stats">
            <div>
              <span>Backend</span>
              <strong>{{ result.backend }}</strong>
            </div>
            <div>
              <span>Reasoning model</span>
              <strong>{{ result.reasoning_model }}</strong>
            </div>
            <div>
              <span>Verification</span>
              <strong>{{ result.verification_passed }}</strong>
            </div>
            <div>
              <span>Time</span>
              <strong>{{ result.execution_time }}s</strong>
            </div>
          </div>
        </div>

        <div class="columns">
          <article class="panel">
            <div class="panel-head">
              <h3>Answer</h3>
            </div>
            <pre class="artifact-text">{{ result.answer }}</pre>
          </article>

          <article class="panel">
            <div class="panel-head">
              <h3>Artifacts</h3>
            </div>
            <div class="artifact-actions">
              <button @click="loadArtifact('Execution log', result.log_path)">Execution log</button>
              <button @click="loadArtifact('Verification', result.verification_path)">Verification</button>
              <button @click="loadArtifact('Memory context', result.memory_context_path)">Memory context</button>
              <button @click="loadArtifact('Run manifest', result.run_manifest_path)">Run manifest</button>
              <button @click="loadArtifact('Run result', result.run_result_path)">Run result</button>
            </div>
            <dl class="meta-list">
              <div>
                <dt>Workspace</dt>
                <dd>{{ result.workspace_path }}</dd>
              </div>
              <div>
                <dt>Task family</dt>
                <dd>{{ result.task_family }}</dd>
              </div>
              <div>
                <dt>Dataset signature</dt>
                <dd>{{ result.dataset_signature }}</dd>
              </div>
            </dl>
          </article>
        </div>

        <article v-if="artifact" class="panel artifact-panel">
          <div class="panel-head">
            <h3>{{ artifactTitle }}</h3>
            <span class="badge">{{ artifact.kind }}</span>
          </div>
          <pre class="artifact-text">{{ artifact.kind === 'json' ? prettyJson(artifact.content) : artifact.content }}</pre>
        </article>
      </section>
    </main>
  `,
}).mount("#app");
