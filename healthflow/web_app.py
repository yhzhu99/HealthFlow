from __future__ import annotations

import asyncio
import html
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import artifact_preview_kind, artifact_preview_language, collect_task_artifacts, read_structured_preview
from .session import HealthFlowProgressEvent, TaskSessionSummary
from .session_client import TaskSessionClient

_TRACE_ASSISTANT_TEXT = "Advanced execution details for this task will appear here."
_EMPTY_WORKSPACE_TEXT = "No workspace files yet."
_EMPTY_PREVIEW_TEXT = "Select a file to preview it."
_UPLOAD_ONLY_USER_MESSAGE = "Please inspect the uploaded files and continue the current task."
_DRAFT_BROWSER_TASK_ID = "__healthflow_draft__"
_BRANDING_DIR = Path(__file__).resolve().parent.parent / "assets" / "branding"
_DEMO_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo"
_RUN_STAGE_ORDER = ["memory", "planner", "executor", "evaluator", "reflection"]
_RUN_STAGE_LABELS = {
    "memory": "Memory",
    "planner": "Planner",
    "executor": "Executor",
    "evaluator": "Evaluator",
    "reflection": "Reflection",
}
_DEMO_CASE = {
    "id": "predictive_modeling_demo",
    "title": "EHR Predictive Modeling Demo",
    "description": "Run a compact in-hospital mortality modeling case with figures rendered directly in the main panel.",
    "file_name": "ehr_predictive_demo.csv",
    "prompt": (
        "Use the attached demo EHR cohort to build a compact predictive modeling diagnostic packet for "
        "in-hospital mortality. Inspect the schema first, use a reproducible patient-level split, train a "
        "reasonable baseline model, and save `metrics.json`, `predictions.csv`, `figures/roc_curve.png`, "
        "`figures/calibration.png`, and a concise `final_report.md`. In the final answer, summarize cohort "
        "size, the strongest risk pattern you observed, and reference the generated figures and report."
    ),
}
_WEB_APP_HEAD = """
<script>
(() => {
  const composerSelector = "#hf-prompt-input";
  const previewSelector = "#hf-composer-attachments";
  const historyListSelector = "#hf-history-list";
  const workspaceTreeSelector = "#hf-workspace-tree";

  const setComponentValue = (selector, value) => {
    const root = document.querySelector(selector);
    const input = root?.querySelector("textarea, input");
    if (!input) {
      return false;
    }
    const nextValue = String(value ?? "");
    const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value");
    descriptor?.set?.call(input, nextValue);
    input.value = nextValue;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  };

  const clickComponentButton = (selector) => {
    const root = document.querySelector(selector);
    const button = root?.querySelector("button") || root;
    if (!(button instanceof HTMLElement)) {
      return false;
    }
    button.click();
    return true;
  };

  const escapeHtml = (value) =>
    String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const attachmentMarkup = (names) =>
    names
      .map(
        (name) =>
          `<span class="hf-composer-attachment"><code>${escapeHtml(name)}</code></span>`
      )
      .join("");

  const syncComposerAttachments = () => {
    const composer = document.querySelector(composerSelector);
    const preview = document.querySelector(previewSelector);
    const fileInput = composer?.querySelector("input[data-testid='file-upload']");
    if (!preview) {
      return;
    }
    const names = Array.from(fileInput?.files || [])
      .map((file) => String(file?.name || "").split(/[\\\\/]/).pop())
      .filter(Boolean);
    preview.innerHTML = attachmentMarkup(names);
    preview.classList.toggle("is-empty", names.length === 0);
  };

  const clearUploadedFilesAfterSubmit = () => {
    const composer = document.querySelector(composerSelector);
    const fileInput = composer?.querySelector("input[data-testid='file-upload']");
    if (!fileInput || !fileInput.files?.length) {
      return;
    }
    window.setTimeout(() => {
      const refreshedComposer = document.querySelector(composerSelector);
      const refreshedInput = refreshedComposer?.querySelector("input[data-testid='file-upload']");
      if (!refreshedInput || !refreshedInput.files?.length) {
        return;
      }
      refreshedInput.value = "";
      refreshedInput.dispatchEvent(new Event("change", { bubbles: true }));
      syncComposerAttachments();
    }, 240);
  };

  const bindComposerAttachments = () => {
    const composer = document.querySelector(composerSelector);
    const preview = document.querySelector(previewSelector);
    const fileInput = composer?.querySelector("input[data-testid='file-upload']");
    if (!composer || !preview || !fileInput) {
      return;
    }
    if (composer.dataset.hfAttachmentBound === "1") {
      syncComposerAttachments();
      return;
    }
    const scheduleSync = () => window.requestAnimationFrame(syncComposerAttachments);
    composer.dataset.hfAttachmentBound = "1";
    fileInput.addEventListener("change", scheduleSync);
    composer.addEventListener("click", (event) => {
      if (event.target instanceof Element && event.target.closest(".submit-button")) {
        clearUploadedFilesAfterSubmit();
      }
      window.setTimeout(scheduleSync, 0);
    });
    composer.addEventListener("drop", () => window.setTimeout(scheduleSync, 0));
    composer.addEventListener("keydown", (event) => {
      if (
        event.target instanceof HTMLTextAreaElement
        && event.key === "Enter"
        && !event.shiftKey
      ) {
        clearUploadedFilesAfterSubmit();
      }
    });
    new MutationObserver(scheduleSync).observe(composer, {
      childList: true,
      subtree: true,
      attributes: true,
    });
    scheduleSync();
  };

  const bindHistoryInteractions = () => {
    const historyList = document.querySelector(historyListSelector);
    if (!historyList || historyList.dataset.hfBound === "1") {
      return;
    }
    historyList.dataset.hfBound = "1";
    historyList.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : null;
      if (!target) {
        return;
      }
      const card = target.closest("[data-task-id]");
      if (!(card instanceof HTMLElement)) {
        return;
      }
      const taskId = card.dataset.taskId || "";
      const actionTarget = target.closest("[data-history-action]");
      const action =
        actionTarget instanceof HTMLElement && actionTarget.dataset.historyAction
          ? actionTarget.dataset.historyAction
          : "open";
      if (!taskId) {
        return;
      }
      if (!setComponentValue("#hf-history-target-task", taskId)) {
        return;
      }
      if (action === "rename") {
        clickComponentButton("#hf-history-rename-trigger");
      } else if (action === "delete") {
        clickComponentButton("#hf-history-delete-trigger");
      } else {
        clickComponentButton("#hf-history-open-trigger");
      }
    });
  };

  const bindWorkspaceInteractions = () => {
    const tree = document.querySelector(workspaceTreeSelector);
    if (!tree || tree.dataset.hfBound === "1") {
      return;
    }
    tree.dataset.hfBound = "1";
    tree.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : null;
      const fileNode = target?.closest("[data-file-path]");
      if (!(fileNode instanceof HTMLElement)) {
        return;
      }
      const filePath = fileNode.dataset.filePath || "";
      if (!filePath) {
        return;
      }
      if (!setComponentValue("#hf-workspace-target-file", filePath)) {
        return;
      }
      clickComponentButton("#hf-workspace-open-trigger");
    });
  };

  const boot = () => {
    bindComposerAttachments();
    bindHistoryInteractions();
    bindWorkspaceInteractions();
  };
  document.addEventListener("DOMContentLoaded", boot);
  window.addEventListener("load", boot);
  new MutationObserver(() => window.requestAnimationFrame(boot)).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
})();
</script>
"""
_WEB_APP_CSS = """
:root {
    --hf-border: rgba(15, 23, 42, 0.1);
    --hf-border-strong: rgba(15, 86, 140, 0.24);
    --hf-surface: #f8fbfd;
    --hf-surface-strong: rgba(255, 255, 255, 0.99);
    --hf-surface-muted: #f2f6f9;
    --hf-surface-contrast: #edf3f8;
    --hf-text: #102033;
    --hf-text-muted: #536579;
    --hf-accent: #155a8a;
    --hf-accent-strong: #0f4a74;
    --hf-accent-soft: rgba(21, 90, 138, 0.08);
    --hf-danger-soft: rgba(185, 28, 28, 0.08);
    --hf-shadow-shell: 0 10px 24px rgba(15, 23, 42, 0.05);
    --hf-shadow-soft: 0 4px 12px rgba(15, 23, 42, 0.04);
    --hf-radius-shell: 16px;
    --hf-radius-panel: 12px;
    --hf-radius-control: 10px;
    --hf-radius-chip: 8px;
}

html,
body {
    margin: 0;
    min-height: 100dvh;
    height: 100dvh;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 28%),
        linear-gradient(180deg, #f6fbff 0%, #f2f7fb 45%, #f7fafc 100%);
}

body {
    color: var(--hf-text);
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.gradio-container,
.gradio-container > main.app,
.gradio-container > main.app > .wrap.sidebar-parent {
    width: 100%;
    min-height: 100dvh;
    height: 100dvh;
    max-width: none !important;
}

.gradio-container {
    padding: 0.42rem !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.gradio-container > main.app {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.gradio-container > main.app > .wrap.sidebar-parent {
    gap: 0.85rem;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    overflow: hidden;
}

.gradio-container * {
    box-sizing: border-box;
}

main.app::before,
main.app::after,
.wrap.sidebar-parent::before,
.wrap.sidebar-parent::after {
    display: none !important;
}

aside {
    height: 100%;
    border: 1px solid var(--hf-border);
    border-right: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-shell);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 246, 249, 0.98) 100%);
    box-shadow: var(--hf-shadow-shell);
    overflow: hidden;
}

aside > div {
    min-height: 0;
}

.hf-sidebar-brand {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 0 0.8rem;
    padding: 0.35rem 0 0.95rem;
    border-bottom: 1px solid rgba(15, 23, 42, 0.06);
}

.hf-sidebar-brand svg {
    display: block;
    width: min(100%, 268px);
    height: auto;
}

.hf-sidebar-brand__fallback {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.045em;
    color: #102033;
}

.hf-main {
    min-height: 0;
    height: 100%;
    gap: 0.85rem;
    align-items: stretch;
    flex-wrap: nowrap !important;
    overflow: hidden;
}

.hf-content-shell {
    min-height: 0;
    height: calc(100dvh - 0.84rem);
}

.hf-chat-shell {
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap !important;
    min-height: 0;
    height: 100%;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-shell);
    background: var(--hf-surface-strong);
    box-shadow: var(--hf-shadow-shell);
    overflow: hidden;
}

.hf-task-header {
    flex: 0 0 auto;
    margin: 0;
    padding: 1.2rem 1.35rem 1rem;
    border: none !important;
    border-bottom: 1px solid var(--hf-border) !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.hf-task-header > :last-child {
    min-width: 0;
    overflow: hidden;
}

.hf-task-header h2 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.045em;
    color: var(--hf-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.hf-task-header p {
    display: none;
}

.hf-run-overview-shell {
    flex: 0 0 auto;
    margin: 0;
    padding: 0 1rem 0.85rem;
}

.hf-run-overview {
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-panel);
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(245, 249, 252, 0.98) 100%);
    box-shadow: none;
    overflow: hidden;
}

.hf-run-overview-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1rem 0.85rem;
    border-bottom: 1px solid var(--hf-border);
    background: linear-gradient(180deg, rgba(237, 243, 248, 0.92) 0%, rgba(255, 255, 255, 0) 100%);
}

.hf-run-overview-eyebrow {
    margin-bottom: 0.35rem;
    color: var(--hf-text-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hf-run-overview-head h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.hf-run-overview-head p {
    margin: 0.18rem 0 0;
    color: var(--hf-text-muted);
    font-size: 0.82rem;
    line-height: 1.45;
}

.hf-run-overview-status {
    flex: 0 0 auto;
    padding: 0.28rem 0.56rem;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-chip);
    background: var(--hf-surface-contrast);
    color: var(--hf-text);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hf-run-overview-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    padding: 0.9rem 1rem 1rem;
}

.hf-run-overview-panel {
    min-width: 0;
    padding: 0.8rem 0.85rem;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-control);
    background: var(--hf-surface);
}

.hf-run-overview-label {
    margin-bottom: 0.45rem;
    color: var(--hf-text-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hf-run-overview-objective {
    color: var(--hf-text);
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.5;
}

.hf-run-overview-note {
    margin-top: 0.5rem;
    color: var(--hf-text-muted);
    font-size: 0.78rem;
    line-height: 1.45;
}

.hf-stage-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.hf-stage-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    min-width: 0;
    width: calc(50% - 0.225rem);
    padding: 0.5rem 0.6rem;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-chip);
    background: #ffffff;
}

.hf-stage-chip-label {
    font-size: 0.78rem;
    font-weight: 700;
}

.hf-stage-chip-state {
    color: var(--hf-text-muted);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hf-stage-chip.is-active {
    border-color: var(--hf-border-strong);
    background: rgba(224, 238, 248, 0.95);
}

.hf-stage-chip.is-active .hf-stage-chip-state {
    color: var(--hf-accent);
}

.hf-stage-chip.is-done {
    border-color: rgba(5, 150, 105, 0.18);
    background: rgba(236, 253, 245, 0.96);
}

.hf-stage-chip.is-done .hf-stage-chip-state {
    color: #047857;
}

.hf-stage-chip.is-failed,
.hf-stage-chip.is-cancelled {
    border-color: rgba(185, 28, 28, 0.18);
    background: rgba(254, 242, 242, 0.96);
}

.hf-stage-chip.is-failed .hf-stage-chip-state,
.hf-stage-chip.is-cancelled .hf-stage-chip-state {
    color: #b91c1c;
}

.hf-stage-chip.is-skipped {
    background: var(--hf-surface-contrast);
}

.hf-run-overview-list,
.hf-run-overview-watchouts {
    margin: 0;
    padding: 0;
    list-style: none;
}

.hf-run-overview-list li,
.hf-run-overview-watchouts li {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    padding: 0.38rem 0;
    color: var(--hf-text);
    font-size: 0.82rem;
    line-height: 1.45;
}

.hf-run-overview-list li + li,
.hf-run-overview-watchouts li + li {
    border-top: 1px solid rgba(15, 23, 42, 0.06);
}

.hf-run-overview-list li span {
    flex: 0 0 auto;
    min-width: 1.65rem;
    color: var(--hf-accent);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}

.hf-starter-shell {
    flex: 0 0 auto;
    gap: 0.7rem;
    margin: 0 1rem 0.8rem;
    padding: 0.95rem 1rem 1rem;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-panel);
    background:
        linear-gradient(135deg, rgba(225, 236, 246, 0.72) 0%, rgba(248, 251, 253, 0.98) 54%, rgba(255, 255, 255, 0.98) 100%);
}

.hf-starter-card h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.025em;
}

.hf-starter-card p {
    margin: 0.35rem 0 0;
    color: var(--hf-text-muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

.hf-starter-eyebrow {
    margin-bottom: 0.45rem;
    color: var(--hf-accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hf-starter-meta,
.hf-starter-deliverables {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.hf-starter-meta {
    margin-top: 0.85rem;
}

.hf-starter-meta-item,
.hf-starter-deliverable {
    display: inline-flex;
    align-items: center;
    min-width: 0;
    padding: 0.34rem 0.55rem;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-chip);
    background: rgba(255, 255, 255, 0.92);
    color: var(--hf-text);
    font-size: 0.75rem;
    line-height: 1.2;
}

.hf-starter-deliverable code,
.hf-starter-meta-item code {
    padding: 0;
    border: none;
    background: transparent;
    color: inherit;
    font-size: inherit;
}

.hf-starter-button {
    min-height: 2.8rem;
    border-radius: var(--hf-radius-control) !important;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.hf-chatbot,
.hf-trace-panel {
    flex: 1 1 auto;
    min-height: 0;
    height: 100% !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden;
}

.hf-composer-shell {
    flex: 0 0 auto;
    margin-top: auto;
    padding: 0.75rem 0.9rem 0.9rem;
    border-top: 1px solid var(--hf-border);
    background: linear-gradient(180deg, rgba(247, 250, 252, 0) 0%, rgba(247, 250, 252, 0.92) 30%, rgba(247, 250, 252, 0.98) 100%);
}

.hf-composer {
    border: 1px solid rgba(15, 23, 42, 0.1) !important;
    border-radius: calc(var(--hf-radius-panel) + 2px) !important;
    background: rgba(255, 255, 255, 0.98) !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.hf-composer-attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0 0 0.55rem;
}

.hf-composer-attachments.is-empty {
    display: none;
}

.hf-composer-attachment,
.hf-chat-attachment {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    min-width: 0;
    padding: 0.36rem 0.7rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: var(--hf-radius-chip);
    background: rgba(248, 250, 252, 0.96);
    color: #334155;
    font-size: 0.78rem;
    line-height: 1.2;
    box-shadow: none !important;
}

.hf-composer-attachment code,
.hf-chat-attachment code {
    padding: 0;
    border: none;
    background: transparent;
    color: inherit;
    font-size: inherit;
}

.hf-composer .input-container {
    gap: 0.5rem;
    padding: 0.42rem 0.48rem 0.42rem 0.3rem;
    align-items: flex-end;
}

.hf-composer .scroll-hide,
.hf-composer .thumbnails,
.hf-composer .thumbnail-list {
    scrollbar-width: none;
}

.hf-composer .scroll-hide::-webkit-scrollbar,
.hf-composer .thumbnails::-webkit-scrollbar,
.hf-composer .thumbnail-list::-webkit-scrollbar {
    display: none;
}

.hf-composer textarea {
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding-top: 0.72rem !important;
    padding-bottom: 0.72rem !important;
}

.hf-composer button.upload-button,
.hf-composer button.submit-button {
    width: 2.75rem;
    min-width: 2.75rem;
    height: 2.75rem;
    border-radius: var(--hf-radius-control);
}

.hf-composer button.upload-button {
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: rgba(248, 250, 252, 0.96);
    color: #0f172a;
}

.hf-composer button.submit-button {
    background: linear-gradient(135deg, var(--hf-accent) 0%, var(--hf-accent-strong) 100%);
    color: #ffffff;
    box-shadow: none;
}

.hf-workspace-shell {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
}

.hf-detail-shell {
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap !important;
    flex: 1 1 auto;
    min-height: 0;
    height: 100%;
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-shell);
    background: var(--hf-surface-strong);
    box-shadow: var(--hf-shadow-shell);
    overflow: hidden;
}

.hf-detail-nav {
    flex: 0 0 auto;
    gap: 0.55rem;
    padding: 1rem;
    border-bottom: 1px solid var(--hf-border);
    background: rgba(249, 251, 253, 0.92);
}

.hf-detail-switch {
    flex: 1 1 0;
    min-width: 0;
    height: 2.5rem;
    border-radius: var(--hf-radius-control);
    font-weight: 600;
    letter-spacing: -0.01em;
    box-shadow: none !important;
}

.hf-detail-switch.primary {
    border: 1px solid rgba(15, 107, 220, 0.18);
    background: linear-gradient(135deg, var(--hf-accent) 0%, var(--hf-accent-strong) 100%);
    color: #ffffff !important;
}

.hf-detail-switch.secondary {
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: rgba(255, 255, 255, 0.86);
    color: #334155 !important;
}

.hf-detail-panel {
    flex: 1 1 auto;
    min-height: 0;
    padding: 1rem;
    gap: 0.9rem;
    overflow: hidden;
}

.hf-workspace-row {
    display: flex;
    flex: 1 1 auto;
    flex-wrap: nowrap !important;
    gap: 0.9rem;
    align-items: stretch;
    min-height: 0;
}

.hf-workspace-row > .column {
    height: 100%;
    min-height: 0;
}

.hf-browser-pane,
.hf-preview-pane,
.hf-advanced-toolbar {
    border: 1px solid var(--hf-border);
    border-radius: var(--hf-radius-panel);
    background: var(--hf-surface-muted);
    box-shadow: none !important;
}

.hf-browser-pane,
.hf-preview-pane,
.hf-advanced-toolbar {
    padding: 1rem 1rem 1.05rem;
}

.hf-browser-pane,
.hf-preview-pane {
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap !important;
    gap: 0.75rem;
    min-height: 0;
}

.hf-browser-pane {
    flex: 0 0 40%;
    min-width: 10.25rem;
    overflow: hidden;
}

.hf-preview-pane {
    flex: 1 1 auto;
    min-width: 0;
    overflow: auto;
}

.hf-browser-pane h3,
.hf-preview-pane h3 {
    margin: 0;
}

.hf-history-list {
    gap: 0.65rem;
}

#hf-history-list,
#hf-workspace-tree {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: auto !important;
    padding: 0 !important;
}

.hf-history-cards,
.hf-tree-root {
    display: flex;
    flex-direction: column;
}

.hf-history-cards {
    gap: 0.48rem;
}

.hf-history-card {
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: var(--hf-radius-control);
    background: rgba(255, 255, 255, 0.96);
    box-shadow: none !important;
    transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}

.hf-history-card:hover,
.hf-tree-file:hover {
    border-color: rgba(11, 132, 219, 0.22);
    background: rgba(250, 252, 255, 0.98);
}

.hf-history-card.is-active,
.hf-tree-file.is-active {
    border-color: rgba(11, 132, 219, 0.34);
    background: linear-gradient(180deg, rgba(239, 248, 255, 0.96) 0%, rgba(255, 255, 255, 0.99) 100%);
}

.hf-history-card-main {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
    padding: 0.72rem 0.76rem;
    cursor: pointer;
}

.hf-history-card-title {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.35;
    color: #0f172a;
    white-space: nowrap;
}

.hf-history-card-actions {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.hf-history-card-btn {
    height: 1.8rem;
    padding: 0 0.52rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 10px;
    background: rgba(248, 250, 252, 0.96);
    color: #475569;
    font-size: 0.7rem;
    font-weight: 600;
    cursor: pointer;
}

.hf-history-card-btn.is-danger {
    border-color: rgba(239, 68, 68, 0.16);
    background: rgba(254, 242, 242, 0.98);
    color: #dc2626;
}

.hf-workspace-empty {
    margin: 0;
    color: #64748b;
    font-size: 0.9rem;
}

.hf-tree-root {
    gap: 0.25rem;
}

.hf-tree-folder {
    margin-top: 0.36rem;
    padding: 0.16rem 0.45rem;
    color: #64748b;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.hf-tree-file {
    display: flex;
    flex-direction: column;
    gap: 0.16rem;
    padding: 0.54rem 0.64rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: var(--hf-radius-chip);
    background: rgba(255, 255, 255, 0.96);
    cursor: pointer;
}

.hf-tree-file-name {
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.35;
    color: #0f172a;
    word-break: break-word;
}

.hf-tree-file-meta {
    font-size: 0.72rem;
    line-height: 1.35;
    color: #64748b;
    word-break: break-word;
}

.hf-tree-depth-0 {
    margin-left: 0;
}

.hf-tree-depth-1 {
    margin-left: 0.85rem;
}

.hf-tree-depth-2 {
    margin-left: 1.7rem;
}

.hf-tree-depth-3 {
    margin-left: 2.55rem;
}

.hf-tree-depth-4 {
    margin-left: 3.4rem;
}

.hf-tree-depth-5 {
    margin-left: 4.25rem;
}

.hf-workspace-folder {
    margin: 0;
    padding: 0.16rem 0 0.08rem;
    color: #526171;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    text-transform: uppercase;
}

.hf-workspace-entry {
    gap: 0.2rem;
}

.hf-workspace-entry .gradio-button {
    justify-content: flex-start;
    width: 100%;
    min-height: 2.55rem;
    border-radius: 14px;
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: rgba(255, 255, 255, 0.92);
    box-shadow: none !important;
    text-align: left;
}

.hf-workspace-entry .gradio-button.primary {
    border-color: rgba(11, 132, 219, 0.26);
    background: linear-gradient(180deg, rgba(239, 248, 255, 0.96) 0%, rgba(255, 255, 255, 0.98) 100%);
    color: #0f172a !important;
}

.hf-workspace-entry-meta {
    margin: 0;
    color: #64748b;
    font-size: 0.75rem;
    line-height: 1.35;
}

.hf-workspace-depth-0 {
    margin-left: 0;
}

.hf-workspace-depth-1 {
    margin-left: 0.7rem;
}

.hf-workspace-depth-2 {
    margin-left: 1.4rem;
}

.hf-workspace-depth-3 {
    margin-left: 2.1rem;
}

.hf-workspace-depth-4 {
    margin-left: 2.8rem;
}

.hf-workspace-depth-5 {
    margin-left: 3.5rem;
}

.hf-history-item {
    position: relative;
    gap: 0.2rem;
    min-height: 4.1rem;
    padding: 0.78rem 0.82rem;
    border: 1px solid rgba(16, 32, 51, 0.08);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.88);
    cursor: pointer;
    overflow: hidden;
}

.hf-history-item.is-active {
    border-color: rgba(11, 132, 219, 0.38);
    background: linear-gradient(180deg, rgba(239, 248, 255, 0.96) 0%, rgba(255, 255, 255, 0.98) 100%);
    box-shadow: 0 10px 30px rgba(11, 132, 219, 0.08);
}

.hf-history-row {
    align-items: center;
    gap: 0.45rem;
    flex-wrap: nowrap !important;
}

.hf-history-hitbox,
.hf-history-action,
.hf-history-inline button {
    border-radius: 12px;
}

.hf-history-hitbox {
    position: absolute;
    inset: 0;
    z-index: 2;
    min-height: 100%;
    padding: 0;
    border: none;
    background: transparent;
    color: transparent !important;
    box-shadow: none !important;
}

.hf-history-hitbox .wrap {
    opacity: 0;
}

.hf-history-hitbox:hover,
.hf-history-item.is-active .hf-history-hitbox {
    background: linear-gradient(180deg, rgba(11, 132, 219, 0.03) 0%, rgba(11, 132, 219, 0.05) 100%);
}

.hf-history-content {
    position: relative;
    z-index: 1;
    pointer-events: none;
}

.hf-history-title {
    flex: 1 1 auto;
    min-width: 0;
    margin: 0;
    pointer-events: none;
}

.hf-history-title-text {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.35;
    color: #0f172a;
}

.hf-history-action {
    flex: 0 0 auto;
    position: relative;
    z-index: 3;
    min-width: 0 !important;
    pointer-events: auto !important;
    padding: 0 0.55rem;
    height: 2rem;
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: rgba(248, 250, 252, 0.9);
    color: #334155 !important;
    font-size: 0.72rem;
    box-shadow: none;
}

.hf-history-action-danger {
    border-color: rgba(239, 68, 68, 0.16);
    background: rgba(254, 242, 242, 0.98);
    color: #dc2626 !important;
}

.hf-history-meta {
    display: none;
}

.hf-history-inline {
    gap: 0.45rem;
    margin-top: 0.15rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(16, 32, 51, 0.08);
}

.hf-history-inline .gradio-button {
    min-width: 0;
}

.hf-history-empty {
    margin: 0;
    font-size: 0.92rem;
    color: #526171;
}

.hf-history-panel {
    gap: 0.75rem;
    margin-top: 0.25rem;
    padding: 0.9rem;
    border: 1px solid rgba(16, 32, 51, 0.08);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.92);
    box-shadow: none;
}

.hf-history-panel-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0f172a;
}

.hf-history-panel-subtitle {
    margin-top: 0.18rem;
    color: #526171;
    font-size: 0.8rem;
    line-height: 1.4;
}

.hf-chatbot > .wrap,
.hf-trace-panel > .wrap {
    height: 100%;
    padding: 0.25rem 0.2rem 0.1rem !important;
}

.hf-chatbot .message-row,
.hf-chatbot .message-wrap,
.hf-chatbot .bubble-wrap,
.hf-trace-panel .message-row,
.hf-trace-panel .message-wrap,
.hf-trace-panel .bubble-wrap {
    max-width: 100% !important;
}

.hf-chatbot :is(.bot.message, .message.bot, .bot .message, .bot .panel-full-width),
.hf-trace-panel :is(.bot.message, .message.bot, .bot .message, .bot .panel-full-width) {
    width: 100%;
    max-width: none !important;
}

.hf-chatbot :is(.user.message, .message.user, .user .message) {
    margin-left: auto;
    max-width: min(88%, 42rem) !important;
}

.hf-chatbot :is(.message, .panel-full-width),
.hf-trace-panel :is(.message, .panel-full-width) {
    border-radius: var(--hf-radius-control) !important;
    box-shadow: none !important;
}

.hf-chatbot .message-row,
.hf-trace-panel .message-row {
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}

.hf-chatbot :is(.bot.message, .message.bot, .bot .message, .bot .panel-full-width),
.hf-trace-panel :is(.bot.message, .message.bot, .bot .message, .bot .panel-full-width) {
    border: 1px solid rgba(15, 23, 42, 0.07);
    background: rgba(246, 249, 252, 0.98) !important;
}

.hf-chatbot :is(.user.message, .message.user, .user .message) {
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: rgba(226, 236, 245, 0.7) !important;
    color: var(--hf-text) !important;
}

.hf-chatbot :is(.user.message, .message.user, .user .message) :is(.prose, p, span, div, code, li, strong, em) {
    color: var(--hf-text) !important;
}

.hf-chat-shell button[aria-label="Clear"],
.hf-chat-shell button[title="Clear"] {
    display: none !important;
}

.hf-chat-attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
    margin-top: 0.68rem;
}

.hf-chat-attachment {
    background: rgba(248, 250, 252, 0.98);
}

.hf-chatbot .prose,
.hf-trace-panel .prose,
.hf-preview-pane .prose,
.hf-browser-pane .prose,
.hf-task-header .prose {
    max-width: none;
}

.hf-history-meta p,
.hf-history-empty p {
    margin: 0;
}

.hf-preview-pane .empty,
.hf-browser-pane .empty {
    color: var(--hf-text-muted);
}

footer {
    display: none !important;
}

@media (max-width: 1280px) {
    html,
    body {
        overflow: auto;
    }

    .gradio-container,
    .gradio-container > main.app,
    .gradio-container > main.app > .wrap.sidebar-parent {
        height: auto;
        min-height: 100%;
    }

    .hf-main {
        height: auto;
        flex-wrap: wrap !important;
        overflow: visible;
    }

    .hf-content-shell {
        height: auto;
    }

    .hf-chat-shell,
    .hf-detail-shell {
        min-height: 34rem;
    }

    .hf-run-overview-grid {
        grid-template-columns: 1fr;
    }

    .hf-workspace-row {
        flex-direction: column;
    }

    .hf-browser-pane {
        min-width: 0;
        min-height: 12rem;
    }
}

@media (max-width: 900px) {
    html,
    body {
        overflow: auto;
    }

    .gradio-container {
        padding: 0.45rem !important;
    }

    .hf-main {
        min-height: auto;
        gap: 0.7rem;
    }

    .hf-chat-shell,
    .hf-detail-shell,
    aside {
        border-radius: 14px;
    }

    .hf-composer-shell {
        padding: 0.85rem;
    }

    .hf-task-header {
        padding: 1rem 1rem 0.85rem;
    }

    .hf-detail-panel {
        padding: 0.85rem;
    }

    .hf-browser-pane {
        flex-basis: 12rem;
        min-height: 12rem;
    }

    .hf-stage-chip {
        width: 100%;
    }

    .hf-run-overview-shell,
    .hf-starter-shell {
        margin-left: 0.85rem;
        margin-right: 0.85rem;
    }
}
"""


class WebTaskSessionStore:
    def __init__(self, system_factory: Callable[[], Any]):
        self._system_factory = system_factory
        self._clients: dict[str, TaskSessionClient] = {}
        self._listing_system: Any | None = None

    def get_client(self, task_id: str | None = None) -> TaskSessionClient:
        normalized_task_id = str(task_id).strip() if task_id else None
        if normalized_task_id and normalized_task_id in self._clients:
            return self._clients[normalized_task_id]

        client = TaskSessionClient(self._system_factory(), task_id=normalized_task_id)
        self._clients[client.task_id] = client
        return client

    def new_client(self) -> TaskSessionClient:
        client = TaskSessionClient(self._system_factory())
        self._clients[client.task_id] = client
        return client

    def list_recent_tasks(self, limit: int = 20) -> list[TaskSessionSummary]:
        if self._listing_system is None:
            self._listing_system = self._system_factory()
        if hasattr(self._listing_system, "list_task_sessions"):
            return list(self._listing_system.list_task_sessions(limit=limit))
        return []

    def has_task(self, task_id: str | None) -> bool:
        normalized_task_id = str(task_id).strip() if task_id else ""
        if not normalized_task_id:
            return False
        if self._listing_system is None:
            self._listing_system = self._system_factory()
        if hasattr(self._listing_system, "load_task_session"):
            try:
                self._listing_system.load_task_session(normalized_task_id)
            except FileNotFoundError:
                return False
            return True
        return any(item.task_id == normalized_task_id for item in self.list_recent_tasks(limit=0))

    def rename_task(self, task_id: str, title: str) -> TaskSessionClient:
        client = self.get_client(task_id)
        client.rename(title)
        return client

    def delete_task(self, task_id: str) -> None:
        if task_id in self._clients:
            self._clients[task_id].delete()
            self._clients.pop(task_id, None)
            return
        if not self.has_task(task_id):
            raise FileNotFoundError(f"Task session '{task_id}' does not exist.")
        client = self.get_client(task_id)
        client.delete()
        self._clients.pop(task_id, None)


def _status_label(status: str | None) -> str:
    normalized = (status or "ready").strip().lower()
    if normalized in {"success", "completed"}:
        return "success"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    if normalized in {"failed", "failure", "error"}:
        return "failed"
    if normalized in {"needs_retry", "retry"}:
        return "needs retry"
    return normalized or "ready"


def _truncate_text(text: str, max_chars: int = 72) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return "Untitled task"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _relative_time_text(value: str | None) -> str:
    timestamp = _parse_utc_timestamp(value)
    if timestamp is None:
        return "recently"
    delta = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    total_seconds = max(int(delta.total_seconds()), 0)
    if total_seconds < 60:
        return "just now"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes}m ago"
    if total_seconds < 86400:
        hours = total_seconds // 3600
        return f"{hours}h ago"
    days = total_seconds // 86400
    return f"{days}d ago"


def _task_title_text(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    for item in recent_tasks:
        if item.task_id == client.task_id:
            return str(item.title or "").strip() or "Untitled task"
    return client.display_title.strip() or client.original_goal.strip() or "Untitled task"


def _task_title(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    return _truncate_text(_task_title_text(client, recent_tasks))


def _load_branding_svg(filename: str) -> str:
    try:
        return (_BRANDING_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _branding_header_html() -> str:
    logo_svg = _load_branding_svg("healthflow-logo.svg")
    if logo_svg:
        return f'<section class="hf-hero" aria-label="HealthFlow brand">{logo_svg}</section>'
    else:
        return '<p class="hf-hero__fallback">HealthFlow</p>'


def _sidebar_brand_html() -> str:
    logo_svg = _load_branding_svg("healthflow-logo.svg")
    if logo_svg:
        return f'<section class="hf-sidebar-brand" aria-label="HealthFlow brand">{logo_svg}</section>'
    return '<p class="hf-sidebar-brand__fallback">HealthFlow</p>'


def _starter_card_html() -> str:
    deliverables = [
        "metrics.json",
        "predictions.csv",
        "figures/roc_curve.png",
        "figures/calibration.png",
        "final_report.md",
    ]
    deliverables_html = "".join(
        f'<span class="hf-starter-deliverable"><code>{html.escape(item)}</code></span>'
        for item in deliverables
    )
    return (
        '<section class="hf-starter-card">'
        '<div class="hf-starter-eyebrow">Showcase Case</div>'
        f'<h3>{html.escape(str(_DEMO_CASE["title"]))}</h3>'
        f'<p>{html.escape(str(_DEMO_CASE["description"]))}</p>'
        '<div class="hf-starter-meta">'
        '<span class="hf-starter-meta-item">Built-in demo file</span>'
        f'<span class="hf-starter-meta-item"><code>{html.escape(str(_DEMO_CASE["file_name"]))}</code></span>'
        "</div>"
        f'<div class="hf-starter-deliverables">{deliverables_html}</div>'
        "</section>"
    )


def _empty_run_overview() -> dict[str, Any]:
    return {
        "attempt": None,
        "mode": "idle",
        "current_stage": "",
        "latest_message": "",
        "objective": "",
        "assumptions_to_check": [],
        "recommended_steps": [],
        "recommended_workflows": [],
        "avoidances": [],
        "success_signals": [],
        "stage_status": {stage: "pending" for stage in _RUN_STAGE_ORDER},
    }


def _copy_run_overview(state: dict[str, Any] | None) -> dict[str, Any]:
    overview = dict(state or _empty_run_overview())
    overview["stage_status"] = dict(overview.get("stage_status") or {})
    for key in (
        "assumptions_to_check",
        "recommended_steps",
        "recommended_workflows",
        "avoidances",
        "success_signals",
    ):
        overview[key] = list(overview.get(key) or [])
    return overview


def _run_stage_badge(status: str) -> str:
    normalized = str(status or "pending").strip().lower()
    if normalized in {"done", "success", "completed"}:
        return "done"
    if normalized in {"failed", "error"}:
        return "failed"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    if normalized in {"active", "running", "in_progress"}:
        return "active"
    if normalized in {"skipped"}:
        return "skipped"
    return "pending"


def _stage_sequence_after(stage: str) -> list[str]:
    if stage not in _RUN_STAGE_ORDER:
        return []
    index = _RUN_STAGE_ORDER.index(stage)
    return _RUN_STAGE_ORDER[index + 1 :]


def _run_stage_sequence_before(stage: str) -> list[str]:
    if stage not in _RUN_STAGE_ORDER:
        return []
    index = _RUN_STAGE_ORDER.index(stage)
    return _RUN_STAGE_ORDER[:index]


def _apply_progress_to_overview(state: dict[str, Any] | None, event: HealthFlowProgressEvent) -> dict[str, Any]:
    overview = _copy_run_overview(state)
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    if event.attempt is not None:
        overview["attempt"] = event.attempt

    stage = str(event.stage or "").strip().lower()
    stage_status = overview["stage_status"]

    if stage in _RUN_STAGE_ORDER and event.kind == "stage_started":
        for previous_stage in _run_stage_sequence_before(stage):
            if _run_stage_badge(stage_status.get(previous_stage, "pending")) == "active":
                stage_status[previous_stage] = "done"
        stage_status[stage] = "active"
        for next_stage in _stage_sequence_after(stage):
            if _run_stage_badge(stage_status.get(next_stage, "pending")) == "active":
                stage_status[next_stage] = "pending"
        overview["current_stage"] = stage
        overview["mode"] = "running"
        overview["latest_message"] = str(event.message or "").strip()

    if stage in _RUN_STAGE_ORDER and event.kind == "stage_finished":
        resolved_status = _run_stage_badge(event.status)
        if stage == "evaluator" and resolved_status in {"failed", "cancelled"}:
            stage_status[stage] = resolved_status
        elif resolved_status == "pending":
            stage_status[stage] = "done"
        elif resolved_status == "active":
            stage_status[stage] = "done"
        else:
            stage_status[stage] = resolved_status
        overview["current_stage"] = stage
        overview["latest_message"] = str(event.message or "").strip()

    if stage == "planner" and metadata:
        overview["objective"] = str(metadata.get("objective") or overview.get("objective") or "").strip()
        for key in (
            "assumptions_to_check",
            "recommended_steps",
            "recommended_workflows",
            "avoidances",
            "success_signals",
        ):
            overview[key] = [str(item).strip() for item in list(metadata.get(key) or []) if str(item).strip()]

    if event.kind == "turn_finished":
        resolved_mode = "completed"
        resolved_status = _status_label(event.status)
        if resolved_status == "failed":
            resolved_mode = "failed"
        elif resolved_status == "cancelled":
            resolved_mode = "cancelled"
        overview["mode"] = resolved_mode
        overview["latest_message"] = str(event.message or "").strip()
        if resolved_mode == "completed" and _run_stage_badge(stage_status.get("reflection")) == "pending":
            stage_status["reflection"] = "skipped"

    if event.kind == "turn_cancelled":
        overview["mode"] = "cancelled"
        overview["latest_message"] = str(event.message or "").strip()
        current_stage = overview.get("current_stage")
        if current_stage in _RUN_STAGE_ORDER:
            stage_status[current_stage] = "cancelled"

    return overview


def _run_overview_from_task_root(task_root: Path | None) -> dict[str, Any]:
    if task_root is None:
        return _empty_run_overview()

    trajectory_path = task_root / "runtime" / "run" / "trajectory.json"
    summary_path = task_root / "runtime" / "run" / "summary.json"
    if not trajectory_path.exists():
        return _empty_run_overview()

    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return _empty_run_overview()

    overview = _empty_run_overview()
    attempts = list(trajectory.get("attempts") or [])
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            summary = {}

    if not attempts:
        if trajectory.get("response_mode") == "direct_response":
            overview["mode"] = "completed"
            overview["objective"] = str(summary.get("answer") or trajectory.get("user_request") or "").strip()
            overview["latest_message"] = str(summary.get("final_summary") or "").strip()
        return overview

    latest_attempt = dict(attempts[-1] or {})
    plan = dict(latest_attempt.get("plan") or {})
    evaluation = dict(latest_attempt.get("evaluation") or {})
    execution = dict(latest_attempt.get("execution") or {})
    gate = dict(latest_attempt.get("gate") or {})

    overview["attempt"] = int(latest_attempt.get("attempt") or len(attempts))
    overview["objective"] = str(plan.get("objective") or trajectory.get("user_request") or "").strip()
    for key in (
        "assumptions_to_check",
        "recommended_steps",
        "recommended_workflows",
        "avoidances",
        "success_signals",
    ):
        overview[key] = [str(item).strip() for item in list(plan.get(key) or []) if str(item).strip()]

    stage_status = overview["stage_status"]
    stage_status["memory"] = "done" if latest_attempt.get("memory") else "pending"
    stage_status["planner"] = "done" if plan else "pending"
    if execution:
        if execution.get("cancelled"):
            stage_status["executor"] = "cancelled"
        else:
            stage_status["executor"] = "done" if execution.get("success") else "failed"
    if evaluation:
        stage_status["evaluator"] = _status_label(evaluation.get("status"))
    if "reflection" in trajectory or trajectory.get("memory_updates") or trajectory.get("new_experiences"):
        stage_status["reflection"] = "done"
    else:
        stage_status["reflection"] = "skipped" if summary else "pending"

    if summary.get("cancelled"):
        overview["mode"] = "cancelled"
    elif summary.get("success"):
        overview["mode"] = "completed"
    elif gate.get("retry_recommended") or _status_label(summary.get("evaluation_status")) == "failed":
        overview["mode"] = "failed"
    else:
        overview["mode"] = "completed"

    overview["current_stage"] = next(
        (
            stage
            for stage in reversed(_RUN_STAGE_ORDER)
            if _run_stage_badge(stage_status.get(stage, "pending")) not in {"pending", "skipped"}
        ),
        "",
    )
    overview["latest_message"] = str(summary.get("final_summary") or evaluation.get("feedback") or "").strip()
    return overview


def _run_overview_html(state: dict[str, Any] | None) -> str:
    overview = _copy_run_overview(state)
    status_class = f"is-{html.escape(str(overview.get('mode') or 'idle'))}"
    attempt = overview.get("attempt")
    attempt_text = f"Attempt {attempt}" if attempt else "Run idle"
    current_stage = str(overview.get("current_stage") or "").strip().lower()
    current_stage_label = _RUN_STAGE_LABELS.get(current_stage, "Ready")
    objective = html.escape(str(overview.get("objective") or "").strip() or "No active plan yet.")
    latest_message = html.escape(str(overview.get("latest_message") or "").strip() or "Planner details and runtime progress will appear here.")
    stage_html = []
    for stage in _RUN_STAGE_ORDER:
        badge = _run_stage_badge(str((overview.get("stage_status") or {}).get(stage) or "pending"))
        label = html.escape(_RUN_STAGE_LABELS[stage])
        badge_label = {
            "done": "done",
            "active": "running",
            "failed": "failed",
            "cancelled": "cancelled",
            "skipped": "skipped",
            "pending": "pending",
        }[badge]
        stage_html.append(
            f'<div class="hf-stage-chip is-{badge}"><span class="hf-stage-chip-label">{label}</span><span class="hf-stage-chip-state">{badge_label}</span></div>'
        )

    step_items = []
    for index, item in enumerate(list(overview.get("recommended_steps") or [])[:6], start=1):
        step_items.append(f"<li><span>{index:02d}</span>{html.escape(str(item))}</li>")
    if not step_items:
        step_items.append("<li><span>--</span>Planner steps will appear after plan generation.</li>")

    avoidance_items = []
    for item in list(overview.get("avoidances") or [])[:3]:
        avoidance_items.append(f"<li>{html.escape(str(item))}</li>")
    if not avoidance_items:
        avoidance_items.append("<li>Artifact rendering and progress checkpoints stay visible in the main panel.</li>")

    return (
        f'<section class="hf-run-overview {status_class}">'
        '<div class="hf-run-overview-head">'
        '<div>'
        '<div class="hf-run-overview-eyebrow">Run Overview</div>'
        f'<h3>{attempt_text}</h3>'
        f'<p>{html.escape(current_stage_label)} is the current foreground stage.</p>'
        "</div>"
        f'<div class="hf-run-overview-status">{html.escape(str(overview.get("mode") or "idle"))}</div>'
        "</div>"
        '<div class="hf-run-overview-grid">'
        '<div class="hf-run-overview-panel">'
        '<div class="hf-run-overview-label">Current objective</div>'
        f'<div class="hf-run-overview-objective">{objective}</div>'
        f'<div class="hf-run-overview-note">{latest_message}</div>'
        "</div>"
        '<div class="hf-run-overview-panel">'
        '<div class="hf-run-overview-label">Runtime stages</div>'
        f'<div class="hf-stage-chip-row">{"".join(stage_html)}</div>'
        "</div>"
        '<div class="hf-run-overview-panel">'
        '<div class="hf-run-overview-label">Execution plan</div>'
        f'<ol class="hf-run-overview-list">{"".join(step_items)}</ol>'
        "</div>"
        '<div class="hf-run-overview-panel">'
        '<div class="hf-run-overview-label">Watchouts</div>'
        f'<ul class="hf-run-overview-watchouts">{"".join(avoidance_items)}</ul>'
        "</div>"
        "</div>"
        "</section>"
    )


def _build_task_choices(recent_tasks: Sequence[TaskSessionSummary]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for item in recent_tasks:
        title = _truncate_text(item.title, max_chars=44)
        status = _status_label(item.latest_turn_status)
        turns = f"{item.turn_count} turn" if item.turn_count == 1 else f"{item.turn_count} turns"
        label = f"{title} · {status} · {turns} · {_relative_time_text(item.updated_at_utc)}"
        choices.append((label, item.task_id))
    return choices


def _history_turn_text(turn_count: int) -> str:
    return "1 turn" if turn_count == 1 else f"{turn_count} turns"


def _history_entry_payload(item: TaskSessionSummary) -> dict[str, Any]:
    return {
        "task_id": item.task_id,
        "title": _truncate_text(item.title, max_chars=52),
        "status": _status_label(item.latest_turn_status),
        "turn_count": item.turn_count,
        "turns_text": _history_turn_text(item.turn_count),
        "updated_at_utc": item.updated_at_utc,
        "updated_text": _relative_time_text(item.updated_at_utc),
    }


def _build_history_entries(recent_tasks: Sequence[TaskSessionSummary]) -> list[dict[str, Any]]:
    return [_history_entry_payload(item) for item in recent_tasks]


def _history_selector_choices(recent_tasks: Sequence[TaskSessionSummary]) -> list[tuple[str, str]]:
    return [(_truncate_text(item.title, max_chars=40), item.task_id) for item in recent_tasks]


def _choice_value_or_none(
    choices: Sequence[tuple[str, str]],
    value: str | None,
) -> str | None:
    normalized_value = str(value or "").strip()
    valid_values = {str(choice_value or "").strip() for _, choice_value in choices}
    if normalized_value and normalized_value in valid_values:
        return normalized_value
    return None


def _history_selected_title_html(
    current_task_id: str | None,
    recent_tasks: Sequence[TaskSessionSummary],
) -> str:
    normalized_task_id = str(current_task_id or "").strip()
    if not normalized_task_id or _is_draft_browser_task_id(normalized_task_id):
        title = "New Task"
    else:
        title = next(
            (
                str(item.title or "").strip() or "Untitled task"
                for item in recent_tasks
                if item.task_id == normalized_task_id
            ),
            "Untitled task",
        )
    return f'<div class="hf-history-toolbar-title">{html.escape(_truncate_text(title, max_chars=44))}</div>'


def _history_list_html(
    recent_tasks: Sequence[TaskSessionSummary],
    current_task_id: str | None,
) -> str:
    normalized_current_task_id = str(current_task_id or "").strip()
    if not recent_tasks:
        return '<p class="hf-history-empty">No tasks yet.</p>'

    cards: list[str] = []
    for item in recent_tasks:
        task_id = str(item.task_id or "").strip()
        if not task_id:
            continue
        title = html.escape(_truncate_text(item.title, max_chars=48))
        is_active = " is-active" if task_id == normalized_current_task_id else ""
        escaped_task_id = html.escape(task_id, quote=True)
        cards.append(
            f"""
            <div class="hf-history-card{is_active}" data-task-id="{escaped_task_id}">
              <div class="hf-history-card-main" data-history-action="open">
                <span class="hf-history-card-title">{title}</span>
                <span class="hf-history-card-actions">
                  <button type="button" class="hf-history-card-btn" data-history-action="rename" data-task-id="{escaped_task_id}">Rename</button>
                  <button type="button" class="hf-history-card-btn is-danger" data-history-action="delete" data-task-id="{escaped_task_id}">Delete</button>
                </span>
              </div>
            </div>
            """
        )
    return '<div class="hf-history-cards">' + "".join(cards) + "</div>"


def _history_render_token(
    history_entries: Sequence[dict[str, Any]] | None,
    current_task_id: str | None,
    action_state: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "current_task_id": str(current_task_id or ""),
            "history_entries": list(history_entries or []),
            "action_state": dict(action_state or _history_action_state()),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _history_json_value(payload: Any, *, default: Any) -> Any:
    if payload is None:
        return default
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return default
    return payload


def _history_action_state(
    *,
    mode: str = "",
    task_id: str | None = None,
    draft_title: str = "",
    placeholder: str = "",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "task_id": task_id,
        "draft_title": draft_title,
        "placeholder": placeholder,
    }


def _draft_browser_task_id() -> str:
    return _DRAFT_BROWSER_TASK_ID


def _is_draft_browser_task_id(task_id: str | None) -> bool:
    return str(task_id or "").strip() == _DRAFT_BROWSER_TASK_ID


def _turn_task_id_for_submission(
    active_task_id: str | None,
    browser_task_id: str | None,
) -> str | None:
    normalized_active_task_id = str(active_task_id or "").strip()
    normalized_browser_task_id = str(browser_task_id or "").strip()
    if _is_draft_browser_task_id(normalized_browser_task_id):
        return None
    if normalized_active_task_id and not _is_draft_browser_task_id(normalized_active_task_id):
        return normalized_active_task_id
    if normalized_browser_task_id and not _is_draft_browser_task_id(normalized_browser_task_id):
        return normalized_browser_task_id
    return None


def _draft_task_title(text: str | None, file_names: Sequence[str] | None = None) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if cleaned:
        return cleaned
    normalized_file_names = [Path(str(name)).name for name in file_names or [] if str(name).strip()]
    if len(normalized_file_names) == 1:
        return f"Review {normalized_file_names[0]}"
    if normalized_file_names:
        return "Review uploaded files"
    return ""


def _resolved_recent_task_id(
    requested_task_id: str | None,
    recent_tasks: Sequence[TaskSessionSummary],
) -> str | None:
    normalized_task_id = str(requested_task_id or "").strip()
    if normalized_task_id and any(item.task_id == normalized_task_id for item in recent_tasks):
        return normalized_task_id
    if recent_tasks:
        return recent_tasks[0].task_id
    return None


def _task_id_after_deletion(
    active_task_id: str | None,
    deleted_task_id: str | None,
    remaining_tasks: Sequence[TaskSessionSummary],
) -> str | None:
    normalized_active_task_id = str(active_task_id or "").strip()
    normalized_deleted_task_id = str(deleted_task_id or "").strip()
    if normalized_deleted_task_id != normalized_active_task_id:
        return normalized_active_task_id or None
    if remaining_tasks:
        return remaining_tasks[0].task_id
    return None


def _visible_recent_tasks(
    recent_tasks: Sequence[TaskSessionSummary],
    *,
    current_task_id: str,
    current_title: str = "",
    current_updated_at: str | None = None,
    current_turn_count: int = 0,
    current_status: str | None = None,
) -> list[TaskSessionSummary]:
    normalized_current_task_id = str(current_task_id or "").strip()
    task_map = {}
    for item in recent_tasks:
        normalized_title = str(item.title or "").strip().lower()
        is_hidden_draft = item.turn_count <= 0 and normalized_title in {"", "untitled task"}
        if is_hidden_draft and item.task_id != normalized_current_task_id:
            continue
        task_map[item.task_id] = item
    if normalized_current_task_id and normalized_current_task_id not in task_map:
        task_map[normalized_current_task_id] = TaskSessionSummary(
            task_id=normalized_current_task_id,
            title=current_title.strip() or "Untitled task",
            updated_at_utc=current_updated_at or "",
            turn_count=current_turn_count,
            latest_turn_status=current_status,
        )
    tasks = list(task_map.values())
    tasks.sort(key=lambda item: (item.updated_at_utc or "", item.created_at_utc or ""), reverse=True)
    return tasks


def _build_task_header(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    title = _task_title(client, recent_tasks)
    return f"## {title}"


def _empty_task_header() -> str:
    return "## New Task"


def _demo_case_uploads() -> dict[str, bytes]:
    demo_path = _DEMO_DIR / str(_DEMO_CASE["file_name"])
    try:
        return {demo_path.name: demo_path.read_bytes()}
    except OSError:
        return {}


def _resolved_upload_name(raw_file: Any) -> str:
    for attr in ("orig_name", "name"):
        value = getattr(raw_file, attr, None)
        if value:
            return Path(str(value)).name
    if isinstance(raw_file, dict):
        for key in ("orig_name", "name", "path"):
            value = raw_file.get(key)
            if value:
                return Path(str(value)).name
    return Path(str(getattr(raw_file, "path", raw_file))).name


def _uploaded_file_names(uploaded_files: Sequence[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for item in uploaded_files or []:
        if not isinstance(item, dict):
            continue
        resolved_name = str(item.get("original_name") or item.get("sandbox_name") or "").strip()
        if resolved_name:
            names.append(Path(resolved_name).name)
    return names


def _attachment_badges_html(file_names: Sequence[str], *, variant: str) -> str:
    chips = []
    for name in file_names:
        resolved_name = Path(str(name or "")).name.strip()
        if not resolved_name:
            continue
        chips.append(
            f'<span class="hf-{variant}-attachment"><code>{html.escape(resolved_name)}</code></span>'
        )
    if not chips:
        return ""
    return f'<div class="hf-{variant}-attachments">{"".join(chips)}</div>'


def _user_message_content(text: str | None, file_names: Sequence[str] | None = None) -> str:
    normalized_file_names = [Path(str(name)).name for name in file_names or [] if str(name).strip()]
    cleaned = str(text or "").strip()
    if normalized_file_names and cleaned == _UPLOAD_ONLY_USER_MESSAGE:
        cleaned = ""
    attachment_html = _attachment_badges_html(normalized_file_names, variant="chat")
    if cleaned and attachment_html:
        return f"{cleaned}\n\n{attachment_html}"
    if cleaned:
        return cleaned
    return attachment_html


def _history_notice_update(message: str | None, *, gr: Any) -> Any:
    text = str(message or "").strip()
    return gr.update(value=text, visible=bool(text))


def _composer_attachment_preview_update(prompt_input: dict[str, Any] | None, *, gr: Any) -> Any:
    prompt_input = prompt_input or {}
    file_names = [_resolved_upload_name(item) for item in list(prompt_input.get("files") or [])]
    attachment_html = _attachment_badges_html(file_names, variant="composer")
    return gr.update(value=attachment_html, visible=bool(attachment_html))


def _empty_composer_value() -> dict[str, Any]:
    return {"text": "", "files": []}


def _result_answer_text(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "").strip() or "No answer available."
    if result.get("success"):
        return answer

    status = "cancelled" if result.get("cancelled") else "failed"
    summary = str(result.get("final_summary") or "").strip()
    lines = [answer, f"_Run status: {status}_"]
    if summary:
        lines.append(summary)
    return "\n\n".join(lines)


def _history_answer_text(record: Any) -> str:
    answer = str(getattr(record, "answer", "") or "").strip() or "No answer available."
    status = _status_label(getattr(record, "status", None))
    if status == "success":
        return answer
    feedback = str(getattr(record, "evaluation_feedback", "") or "").strip()
    lines = [answer, f"_Run status: {status}_"]
    if feedback:
        lines.append(feedback)
    return "\n\n".join(lines)


def _restore_main_history(client: TaskSessionClient) -> list[dict[str, str]]:
    history = client.load_history()
    main_history: list[dict[str, str]] = []
    for record in history:
        main_history.append(
            {
                "role": "user",
                "content": _user_message_content(
                    getattr(record, "user_message", ""),
                    _uploaded_file_names(getattr(record, "uploaded_files", None)),
                ),
            }
        )
        main_history.append({"role": "assistant", "content": _history_answer_text(record)})
    return main_history


def _restore_trace_history(client: TaskSessionClient, *, restored: bool) -> list[dict[str, str]]:
    history = client.load_history()
    if not history:
        message = (
            "Previous execution details for this task are available here."
            if restored
            else _TRACE_ASSISTANT_TEXT
        )
        return [{"role": "assistant", "content": message}]

    trace_history = [
        {
            "role": "assistant",
            "content": "Previous execution details for this task are listed below. New progress updates will appear here.",
        }
    ]
    for record in history[-5:]:
        status = _status_label(record.status)
        lines = [f"**Turn {record.turn_number}** · `{status}`"]
        if record.evaluation_feedback:
            lines.append(record.evaluation_feedback.strip())
        trace_history.append({"role": "assistant", "content": "\n\n".join(lines)})
    return trace_history


def _progress_title(event: HealthFlowProgressEvent) -> str:
    stage_label = _RUN_STAGE_LABELS.get(str(event.stage or "").strip().lower(), str(event.stage or "Run").title())
    if event.attempt:
        return f"{stage_label} · Attempt {event.attempt}"
    return stage_label


def _main_progress_messages(
    event: HealthFlowProgressEvent,
    *,
    task_root: Path | None,
    seen_image_paths: set[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    metadata_status = "pending" if event.kind == "stage_started" else "done"

    if event.kind in {"stage_started", "stage_finished"}:
        content = str(event.message or "").strip() or f"{_progress_title(event)} update."
        planner_steps = []
        if event.stage == "planner" and event.kind == "stage_finished":
            objective = str((event.metadata or {}).get("objective") or content).strip()
            planner_steps = [str(item).strip() for item in list((event.metadata or {}).get("recommended_steps") or []) if str(item).strip()]
            content = f"**Objective**\n\n{objective}"
            if planner_steps:
                content += "\n\n**Recommended steps**\n\n" + "\n".join(
                    f"{index}. {item}" for index, item in enumerate(planner_steps, start=1)
                )
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "metadata": {
                    "title": _progress_title(event),
                    "status": metadata_status,
                    "log": str(event.message or "").strip() or None,
                },
            }
        )
        return messages

    if event.kind == "artifact_delta":
        artifacts = list((event.metadata or {}).get("artifacts") or [])
        image_messages = _inline_image_messages(artifacts, task_root=task_root, seen_image_paths=seen_image_paths)
        if image_messages:
            messages.append(
                {
                    "role": "assistant",
                    "content": "Rendered artifact previews are available below.",
                    "metadata": {
                        "title": _progress_title(event),
                        "status": "done",
                        "log": f"{len(image_messages)} image artifact(s) surfaced",
                    },
                }
            )
            messages.extend(image_messages)
        return messages

    if event.kind == "turn_cancelled":
        messages.append(
            {
                "role": "assistant",
                "content": str(event.message or "The run was cancelled.").strip(),
                "metadata": {"title": "Run Cancelled", "status": "done"},
            }
        )
    return messages


def _inline_image_messages(
    artifact_paths: Sequence[str],
    *,
    task_root: Path | None,
    seen_image_paths: set[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    if task_root is None:
        return []

    messages: list[dict[str, Any]] = []
    sandbox_root = task_root / "sandbox"
    for artifact_path in artifact_paths:
        resolved_relative_path = str(artifact_path or "").strip()
        if not resolved_relative_path:
            continue
        path = sandbox_root / resolved_relative_path
        if not path.exists() or not path.is_file():
            continue
        if artifact_preview_kind(path) != "image":
            continue
        path_str = str(path)
        if path_str in seen_image_paths:
            continue
        seen_image_paths.add(path_str)
        messages.append({"role": "assistant", "content": (path_str, path.name)})
        if len(messages) >= limit:
            break
    return messages


def _collect_artifact_catalog(client: TaskSessionClient) -> list[dict[str, Any]]:
    task_root = Path(client.task_root) if client.task_root else None
    if task_root is None or not task_root.exists():
        return []
    return collect_task_artifacts(task_root, client.load_history())


def _read_text_preview(path: Path, *, max_chars: int = 50000) -> tuple[str | None, bool]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, False
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars].rstrip(), True


def _default_selected_file(
    catalog: Sequence[dict[str, Any]],
    preferred_file: str | None = None,
) -> str | None:
    if preferred_file and any(str(item.get("source_path")) == preferred_file for item in catalog):
        return preferred_file
    if not catalog:
        return None
    report_item = next((item for item in catalog if item.get("origin") == "report"), None)
    if report_item is not None:
        return str(report_item.get("source_path"))
    return str(catalog[0].get("source_path"))


def _task_root_path(task_id: str | None, session_store: WebTaskSessionStore) -> Path | None:
    if not task_id or not session_store.has_task(task_id):
        return None
    client = session_store.get_client(task_id)
    if not client.task_root:
        return None
    return Path(client.task_root)


def _task_relative_label(path: Path, task_root: Path | None) -> str:
    if task_root is None:
        return path.name
    try:
        return path.relative_to(task_root).as_posix()
    except ValueError:
        return path.name


def _preview_header_text(path: Path, *, task_root: Path | None) -> str:
    relative_label = _task_relative_label(path, task_root)
    if relative_label and relative_label != path.name:
        return f"### {path.name}\n\n`{relative_label}`"
    return f"### {path.name}"


def _workspace_tree_value(selected_file: str | None, *, task_root: Path | None) -> str | None:
    if not selected_file or task_root is None:
        return None
    sandbox_root = task_root / "sandbox"
    path = Path(selected_file)
    try:
        path.relative_to(sandbox_root)
    except ValueError:
        return None
    return str(path)


def _workspace_relative_path(item: dict[str, Any]) -> str:
    task_relative_path = str(item.get("task_relative_path") or "").strip()
    if task_relative_path.startswith("sandbox/"):
        return task_relative_path[len("sandbox/") :]
    return task_relative_path


def _workspace_tree_rows(catalog: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    non_report_items = [item for item in catalog if item.get("origin") != "report"]
    display_counts: dict[str, int] = {}
    for item in non_report_items:
        display_name = Path(str(item.get("display_name") or item.get("source_path") or "")).name.strip()
        if not display_name:
            continue
        display_counts[display_name] = display_counts.get(display_name, 0) + 1

    rows: list[dict[str, Any]] = []
    seen_folders: set[tuple[str, ...]] = set()
    sorted_items = sorted(
        non_report_items,
        key=lambda item: (_workspace_relative_path(item).lower(), str(item.get("display_name") or "").lower()),
    )

    for item in sorted_items:
        workspace_relative_path = _workspace_relative_path(item)
        if not workspace_relative_path:
            continue
        parts = list(Path(workspace_relative_path).parts)
        if not parts:
            continue
        for depth, folder_name in enumerate(parts[:-1]):
            folder_key = tuple(parts[: depth + 1])
            if folder_key in seen_folders:
                continue
            seen_folders.add(folder_key)
            rows.append(
                {
                    "kind": "folder",
                    "depth": depth,
                    "label": folder_name,
                }
            )

        display_name = Path(str(item.get("display_name") or parts[-1])).name.strip() or parts[-1]
        meta = workspace_relative_path if display_counts.get(display_name, 0) > 1 else ""
        rows.append(
            {
                "kind": "file",
                "depth": max(len(parts) - 1, 0),
                "label": display_name,
                "meta": meta,
                "source_path": str(item.get("source_path") or ""),
            }
        )
    return rows


def _workspace_browser_choices(catalog: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    display_counts: dict[str, int] = {}
    for item in catalog:
        display_name = Path(str(item.get("display_name") or item.get("source_path") or "")).name.strip()
        if not display_name:
            continue
        display_counts[display_name] = display_counts.get(display_name, 0) + 1

    sorted_items = sorted(
        catalog,
        key=lambda item: (
            0 if item.get("origin") == "report" else 1,
            _workspace_relative_path(item).lower(),
            str(item.get("display_name") or "").lower(),
        ),
    )
    choices: list[tuple[str, str]] = []
    for item in sorted_items:
        source_path = str(item.get("source_path") or "").strip()
        if not source_path:
            continue
        display_name = Path(
            str(item.get("display_name") or item.get("task_relative_path") or source_path)
        ).name.strip() or Path(source_path).name
        if item.get("origin") == "report":
            label = "report.md"
        else:
            workspace_relative_path = _workspace_relative_path(item)
            label = workspace_relative_path or display_name
            if display_counts.get(display_name, 0) > 1 and workspace_relative_path and workspace_relative_path != display_name:
                label = f"{display_name} · {workspace_relative_path}"
        choices.append((label, source_path))
    return choices


def _workspace_tree_html(
    catalog: Sequence[dict[str, Any]],
    selected_file: str | None,
) -> str:
    selected_file_value = str(selected_file or "").strip()
    report_item = next((item for item in catalog if item.get("origin") == "report"), None)
    rows: list[dict[str, Any]] = []
    if report_item is not None:
        rows.append(
            {
                "kind": "file",
                "depth": 0,
                "label": "report.md",
                "meta": "runtime/report.md",
                "source_path": str(report_item.get("source_path") or ""),
            }
        )
    rows.extend(_workspace_tree_rows(catalog))
    if not rows:
        return f'<p class="hf-workspace-empty">{html.escape(_EMPTY_WORKSPACE_TEXT)}</p>'

    html_rows: list[str] = []
    for row in rows:
        row_depth = min(int(row.get("depth") or 0), 5)
        depth_class = f"hf-tree-depth-{row_depth}"
        if row.get("kind") == "folder":
            label = html.escape(str(row.get("label") or ""))
            html_rows.append(f'<div class="hf-tree-folder {depth_class}">{label}</div>')
            continue

        source_path = str(row.get("source_path") or "").strip()
        if not source_path:
            continue
        label = html.escape(str(row.get("label") or Path(source_path).name))
        meta = html.escape(str(row.get("meta") or "").strip())
        active_class = " is-active" if source_path == selected_file_value else ""
        meta_html = f'<span class="hf-tree-file-meta">{meta}</span>' if meta else ""
        html_rows.append(
            f"""
            <div class="hf-tree-file {depth_class}{active_class}" data-file-path="{html.escape(source_path, quote=True)}">
              <span class="hf-tree-file-name">{label}</span>
              {meta_html}
            </div>
            """
        )
    return '<div class="hf-tree-root">' + "".join(html_rows) + "</div>"


def _artifact_preview_outputs(
    selected_file: str | None,
    *,
    task_root: Path | None,
    gr: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not selected_file:
        return (
            gr.update(value="### Preview", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value=_EMPTY_PREVIEW_TEXT, visible=True),
            gr.update(value=None, visible=False),
        )

    path = Path(selected_file)
    if not path.exists():
        return (
            gr.update(value="### Preview", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value=_EMPTY_PREVIEW_TEXT, visible=True),
            gr.update(value=None, visible=False),
        )

    if path.is_dir():
        return (
            gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="Choose a file in this folder to preview it.", visible=True),
            gr.update(value=None, visible=False),
        )

    download_update = gr.update(value=str(path), visible=True)
    preview_kind = artifact_preview_kind(path)

    if preview_kind == "markdown":
        content, _ = _read_text_preview(path)
        if content is None:
            return (
                gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value="", language=None, visible=False),
                gr.update(value="Preview not available for this file type.", visible=True),
                download_update,
            )
        return (
            gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
            gr.update(value=content, visible=True),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="", visible=False),
            download_update,
        )

    if preview_kind == "image":
        return (
            gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
            gr.update(value="", visible=False),
            gr.update(value=str(path), visible=True),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="", visible=False),
            download_update,
        )

    if preview_kind == "table":
        structured_preview = read_structured_preview(path)
        if structured_preview is not None:
            return (
                gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(
                    value=structured_preview.get("rows") or [],
                    headers=structured_preview.get("headers") or [],
                    visible=True,
                ),
                gr.update(value="", language=None, visible=False),
                gr.update(value="", visible=False),
                download_update,
            )

    if preview_kind == "code":
        content, _ = _read_text_preview(path)
        if content is not None:
            return (
                gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value=content, language=artifact_preview_language(path), visible=True),
                gr.update(value="", visible=False),
                download_update,
            )

    return (
        gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(value="", language=None, visible=False),
        gr.update(value="Preview not available for this file type.", visible=True),
        download_update,
    )


def _stream_task_turn(
    client: TaskSessionClient,
    user_message: str,
    *,
    report_requested: bool,
    uploaded_files: dict[str, bytes] | None = None,
):
    stream_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    closed = threading.Event()

    def _progress_callback(event: HealthFlowProgressEvent) -> None:
        if not closed.is_set():
            stream_queue.put(("progress", event))

    def _runner() -> None:
        try:
            result = asyncio.run(
                client.run_turn(
                    user_message,
                    report_requested=report_requested,
                    progress_callback=_progress_callback,
                    uploaded_files=uploaded_files,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for unexpected runtime failures
            if not closed.is_set():
                stream_queue.put(("error", exc))
        else:
            if not closed.is_set():
                stream_queue.put(("result", result))
        finally:
            if not closed.is_set():
                stream_queue.put(("done", None))

    worker = threading.Thread(target=_runner, name=f"healthflow-web-turn-{client.task_id[:8]}", daemon=True)
    worker.start()

    finished = False
    try:
        while not finished or not stream_queue.empty():
            try:
                kind, payload = stream_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "done":
                finished = True
                continue
            yield kind, payload
    finally:
        closed.set()


def launch_web_app(
    system_factory: Callable[[], Any],
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is not installed. Install the web extra first, for example: uv sync --extra web"
        ) from None

    session_store = WebTaskSessionStore(system_factory)
    listing_system = system_factory()
    session_store._listing_system = listing_system
    workspace_root = getattr(listing_system, "workspace_dir", None)
    allowed_paths = [str(Path(workspace_root).resolve())] if workspace_root else None

    def _resolve_requested_task(task_id: str | None) -> tuple[TaskSessionClient, bool]:
        normalized_task_id = str(task_id or "").strip()
        if _is_draft_browser_task_id(normalized_task_id):
            return session_store.new_client(), False
        if normalized_task_id and session_store.has_task(normalized_task_id):
            return session_store.get_client(normalized_task_id), True
        recent_tasks = _visible_recent_tasks(session_store.list_recent_tasks(limit=50), current_task_id="")
        fallback_task_id = _resolved_recent_task_id(normalized_task_id, recent_tasks)
        if fallback_task_id:
            return session_store.get_client(fallback_task_id), True
        return session_store.new_client(), False

    def _visible_recent_task_summaries(client: TaskSessionClient) -> list[TaskSessionSummary]:
        recent_tasks = session_store.list_recent_tasks(limit=50)
        return _visible_recent_tasks(
            recent_tasks,
            current_task_id=client.task_id,
            current_title=client.display_title or client.original_goal,
            current_updated_at=client.updated_at_utc,
            current_turn_count=client.turn_count,
            current_status=client.latest_turn_status,
        )

    def _history_panel_updates(
        recent_tasks: Sequence[TaskSessionSummary],
        current_task_id: str | None,
        *,
        action_mode: str = "",
        action_title: str = "",
        action_placeholder: str = "",
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        selected_task_id = str(current_task_id or "").strip()
        if _is_draft_browser_task_id(selected_task_id):
            selected_task_id = ""
        can_manage = bool(selected_task_id)
        target_summary = next((item for item in recent_tasks if item.task_id == selected_task_id), None)
        target_title = str(target_summary.title or "").strip() if target_summary is not None else ""
        target_title = target_title or "Untitled task"
        panel_visible = can_manage and action_mode in {"rename", "delete"}

        heading_html = ""
        if action_mode == "rename" and panel_visible:
            heading_html = (
                '<div class="hf-history-panel-title">Rename task</div>'
                f'<div class="hf-history-panel-subtitle">{html.escape(target_title)}</div>'
            )
        elif action_mode == "delete" and panel_visible:
            heading_html = (
                '<div class="hf-history-panel-title">Delete task?</div>'
                f'<div class="hf-history-panel-subtitle">{html.escape(target_title)}</div>'
            )

        rename_value = action_title if panel_visible and action_mode == "rename" else ""
        rename_placeholder = action_placeholder or target_title
        return (
            gr.update(value=_history_list_html(recent_tasks, selected_task_id)),
            gr.update(visible=panel_visible),
            gr.update(value=heading_html, visible=panel_visible),
            gr.update(value=rename_value, placeholder=rename_placeholder, visible=panel_visible and action_mode == "rename"),
            gr.update(visible=panel_visible and action_mode == "rename"),
            gr.update(visible=panel_visible and action_mode == "delete"),
            gr.update(visible=panel_visible),
        )

    def _workspace_browser_updates(catalog: Sequence[dict[str, Any]], selected_file: str | None) -> Any:
        return gr.update(value=_workspace_tree_html(catalog, selected_file))

    def _compose_outputs(
        client: TaskSessionClient,
        *,
        main_history: list[dict[str, Any]],
        trace_history: list[dict[str, Any]],
        preferred_file: str | None,
        run_overview: dict[str, Any] | None = None,
        history_notice: str | None = None,
        history_action_mode: str = "",
        history_action_title: str = "",
        history_action_placeholder: str = "",
    ) -> tuple[Any, ...]:
        visible_recent_tasks = _visible_recent_task_summaries(client)
        catalog = _collect_artifact_catalog(client)
        selected_file = _default_selected_file(catalog, preferred_file=preferred_file)
        task_root = Path(client.task_root) if client.task_root else None
        resolved_run_overview = _copy_run_overview(run_overview or _run_overview_from_task_root(task_root))
        preview_outputs = _artifact_preview_outputs(selected_file, task_root=task_root, gr=gr)
        history_updates = _history_panel_updates(
            visible_recent_tasks,
            client.task_id,
            action_mode=history_action_mode,
            action_title=history_action_title,
            action_placeholder=history_action_placeholder,
        )
        workspace_updates = _workspace_browser_updates(catalog, selected_file)
        starter_visible = client.turn_count <= 0 and not main_history
        return (
            main_history,
            trace_history,
            client.task_id,
            client.task_id,
            _build_task_header(client, visible_recent_tasks),
            gr.update(value=_run_overview_html(resolved_run_overview), visible=True),
            resolved_run_overview,
            gr.update(visible=starter_visible),
            *history_updates,
            _history_notice_update(history_notice, gr=gr),
            workspace_updates,
            selected_file,
            *preview_outputs,
            gr.update(value=""),
            gr.MultimodalTextbox(value=_empty_composer_value()),
        )

    def _compose_draft_outputs(
        *,
        main_history: list[dict[str, Any]] | None = None,
        trace_history: list[dict[str, Any]] | None = None,
        run_overview: dict[str, Any] | None = None,
        history_notice: str | None = None,
        history_action_mode: str = "",
        history_action_title: str = "",
        history_action_placeholder: str = "",
    ) -> tuple[Any, ...]:
        visible_recent_tasks = _visible_recent_tasks(session_store.list_recent_tasks(limit=50), current_task_id="")
        preview_outputs = _artifact_preview_outputs(None, task_root=None, gr=gr)
        history_updates = _history_panel_updates(
            visible_recent_tasks,
            None,
            action_mode=history_action_mode,
            action_title=history_action_title,
            action_placeholder=history_action_placeholder,
        )
        workspace_updates = _workspace_browser_updates([], None)
        return (
            list(main_history or []),
            list(trace_history or [{"role": "assistant", "content": _TRACE_ASSISTANT_TEXT}]),
            _draft_browser_task_id(),
            None,
            _empty_task_header(),
            gr.update(value=_run_overview_html(run_overview or _empty_run_overview()), visible=True),
            _copy_run_overview(run_overview),
            gr.update(visible=not main_history),
            *history_updates,
            _history_notice_update(history_notice, gr=gr),
            workspace_updates,
            None,
            *preview_outputs,
            gr.update(value=""),
            gr.MultimodalTextbox(value=_empty_composer_value()),
        )

    def _compose_for_task_id(
        task_id: str | None,
        *,
        main_history: list[dict[str, Any]] | None = None,
        trace_history: list[dict[str, Any]] | None = None,
        preferred_file: str | None = None,
        run_overview: dict[str, Any] | None = None,
        history_notice: str | None = None,
        history_action_mode: str = "",
        history_action_title: str = "",
        history_action_placeholder: str = "",
        restored: bool | None = None,
    ):
        client, resolved_from_history = _resolve_requested_task(task_id)
        if main_history is None:
            main_history = _restore_main_history(client)
        else:
            main_history = list(main_history)
        if trace_history is None:
            show_restored_trace = restored if restored is not None else resolved_from_history and client.turn_count > 0
            trace_history = _restore_trace_history(client, restored=show_restored_trace)
        else:
            trace_history = list(trace_history)
        return _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            preferred_file=preferred_file,
            run_overview=run_overview,
            history_notice=history_notice,
            history_action_mode=history_action_mode,
            history_action_title=history_action_title,
            history_action_placeholder=history_action_placeholder,
        )

    def _format_progress_event(event: HealthFlowProgressEvent) -> str:
        if event.kind == "log_chunk":
            body = (event.message or "").strip()
            body = body[-3000:] if len(body) > 3000 else body
            if body:
                return f"### Executor Log\n\n```text\n{body}\n```"
        if event.kind == "artifact_delta":
            artifacts = event.metadata.get("artifacts") if isinstance(event.metadata, dict) else None
            if artifacts:
                artifact_lines = "\n".join(f"- `{item}`" for item in artifacts[:10])
                return f"### Files Updated\n\n{artifact_lines}"
        message = (event.message or f"{event.stage}: {event.status}").strip()
        return f"**{event.stage.title()}** · {message}"

    def _compose_for_existing_task_or_draft(
        task_id: str | None,
        *,
        preferred_file: str | None = None,
        run_overview: dict[str, Any] | None = None,
        history_notice: str | None = None,
        history_action_mode: str = "",
        history_action_title: str = "",
        history_action_placeholder: str = "",
        fallback_to_recent: bool = False,
    ):
        normalized_task_id = str(task_id or "").strip()
        if _is_draft_browser_task_id(normalized_task_id):
            return _compose_draft_outputs(
                run_overview=run_overview,
                history_notice=history_notice,
                history_action_mode=history_action_mode,
                history_action_title=history_action_title,
                history_action_placeholder=history_action_placeholder,
            )
        if normalized_task_id and session_store.has_task(normalized_task_id):
            return _compose_for_task_id(
                normalized_task_id,
                preferred_file=preferred_file,
                run_overview=run_overview,
                history_notice=history_notice,
                history_action_mode=history_action_mode,
                history_action_title=history_action_title,
                history_action_placeholder=history_action_placeholder,
            )
        if fallback_to_recent:
            recent_tasks = _visible_recent_tasks(session_store.list_recent_tasks(limit=50), current_task_id="")
            fallback_task_id = _resolved_recent_task_id(None, recent_tasks)
            if fallback_task_id:
                return _compose_for_task_id(
                    fallback_task_id,
                    preferred_file=None,
                    run_overview=run_overview,
                    history_notice=history_notice,
                    history_action_mode=history_action_mode,
                    history_action_title=history_action_title,
                    history_action_placeholder=history_action_placeholder,
                )
        return _compose_draft_outputs(
            run_overview=run_overview,
            history_notice=history_notice,
            history_action_mode=history_action_mode,
            history_action_title=history_action_title,
            history_action_placeholder=history_action_placeholder,
        )

    def _load_session(task_id: str | None):
        return _compose_for_existing_task_or_draft(
            task_id,
            preferred_file=None,
            fallback_to_recent=True,
        )

    def _switch_task(task_id: str | None):
        return _compose_for_existing_task_or_draft(
            task_id,
            preferred_file=None,
            fallback_to_recent=True,
        )

    def _new_task():
        return _compose_draft_outputs(run_overview=_empty_run_overview())

    def _preview_outputs_for_task(selected_file: str | None, task_id: str | None):
        task_root = _task_root_path(task_id, session_store)
        return _artifact_preview_outputs(selected_file, task_root=task_root, gr=gr)

    def _on_workspace_file_selected(selected_file: str | None, task_id: str | None):
        return selected_file, *_preview_outputs_for_task(selected_file, task_id)

    def _detail_panel_updates(mode: str) -> tuple[Any, Any, Any, Any]:
        resolved_mode = "advanced" if mode == "advanced" else "workspace"
        return (
            gr.update(variant="primary" if resolved_mode == "workspace" else "secondary"),
            gr.update(variant="primary" if resolved_mode == "advanced" else "secondary"),
            gr.update(visible=resolved_mode == "workspace"),
            gr.update(visible=resolved_mode == "advanced"),
        )

    def _submit_turn(
        *,
        text: str,
        upload_payloads: dict[str, bytes],
        main_history: list[dict[str, Any]] | None,
        trace_history: list[dict[str, Any]] | None,
        browser_task_id: str | None,
        task_id: str | None,
        report_requested: bool,
        selected_file: str | None,
        run_overview: dict[str, Any] | None,
    ):
        resolved_turn_task_id = _turn_task_id_for_submission(task_id, browser_task_id)
        draft_submission = resolved_turn_task_id is None
        if draft_submission:
            client = session_store.new_client()
            resolved_draft_title = _draft_task_title(text, list(upload_payloads.keys()))
            if resolved_draft_title:
                client.rename(resolved_draft_title)
        else:
            client, _ = _resolve_requested_task(resolved_turn_task_id)

        main_history = list(main_history or _restore_main_history(client))
        trace_history = list(trace_history or _restore_trace_history(client, restored=client.turn_count > 0))
        overview_state = _copy_run_overview(run_overview or _run_overview_from_task_root(Path(client.task_root) if client.task_root else None))

        if not text.strip() and not upload_payloads:
            yield _compose_outputs(
                client,
                main_history=main_history,
                trace_history=trace_history,
                preferred_file=selected_file,
                run_overview=overview_state,
            )
            return

        attachment_names = list(upload_payloads.keys())
        user_display = _user_message_content(text, attachment_names)
        user_message = text.strip() or _UPLOAD_ONLY_USER_MESSAGE
        main_history.append({"role": "user", "content": user_display})
        trace_history.append({"role": "assistant", "content": "Working on your latest request."})
        yield _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            preferred_file=selected_file,
            run_overview=overview_state,
        )

        result: dict[str, Any] | None = None
        error: Exception | None = None
        seen_image_paths: set[str] = set()
        task_root = Path(client.task_root) if client.task_root else None
        for kind, payload in _stream_task_turn(
            client,
            user_message,
            report_requested=report_requested,
            uploaded_files=upload_payloads or None,
        ):
            if kind == "progress":
                event = payload
                trace_history.append({"role": "assistant", "content": _format_progress_event(event)})
                overview_state = _apply_progress_to_overview(overview_state, event)
                main_history.extend(
                    _main_progress_messages(event, task_root=task_root, seen_image_paths=seen_image_paths)
                )
                yield _compose_outputs(
                    client,
                    main_history=main_history,
                    trace_history=trace_history,
                    preferred_file=selected_file,
                    run_overview=overview_state,
                )
                continue
            if kind == "error":
                error = payload
                break
            if kind == "result":
                result = payload

        if error is not None:
            result = {
                "success": False,
                "answer": "The web session failed before the run completed.",
                "final_summary": str(error),
                "cancelled": False,
            }

        result = result or {
            "success": False,
            "answer": "The run stopped before a result was returned.",
            "final_summary": "",
            "cancelled": False,
        }
        summary = str(result.get("final_summary") or "").strip()
        if task_root is not None:
            image_messages = _inline_image_messages(
                [str(item.get("task_relative_path") or "").removeprefix("sandbox/") for item in _collect_artifact_catalog(client) if item.get("origin") == "generated"],
                task_root=task_root,
                seen_image_paths=seen_image_paths,
            )
            main_history.extend(image_messages)
        main_history.append({"role": "assistant", "content": _result_answer_text(result)})
        if summary:
            trace_history.append({"role": "assistant", "content": f"**Run summary**\n\n{summary}"})
        final_event = HealthFlowProgressEvent(
            kind="turn_finished" if not result.get("cancelled") else "turn_cancelled",
            stage="run",
            status="cancelled" if result.get("cancelled") else ("completed" if result.get("success") else "failed"),
            message=summary,
        )
        overview_state = _apply_progress_to_overview(overview_state, final_event)

        yield _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            preferred_file=selected_file,
            run_overview=overview_state,
        )

    def _start_history_rename(
        target_task_id: str | None,
        active_task_id: str | None,
        selected_file: str | None,
    ):
        resolved_active_task_id = active_task_id or target_task_id
        if not target_task_id or not session_store.has_task(target_task_id):
            return _compose_for_existing_task_or_draft(
                resolved_active_task_id,
                preferred_file=selected_file,
                history_notice="Task no longer exists.",
            )
        client = session_store.get_client(target_task_id)
        visible_recent_tasks = _visible_recent_task_summaries(client)
        return _compose_for_existing_task_or_draft(
            resolved_active_task_id,
            preferred_file=selected_file,
            history_action_mode="rename",
            history_action_title=client.display_title,
            history_action_placeholder=_task_title_text(client, visible_recent_tasks),
        )

    def _start_history_delete(
        target_task_id: str | None,
        active_task_id: str | None,
        selected_file: str | None,
    ):
        resolved_active_task_id = active_task_id or target_task_id
        if not target_task_id or not session_store.has_task(target_task_id):
            return _compose_for_existing_task_or_draft(
                resolved_active_task_id,
                preferred_file=selected_file,
                history_notice="Task no longer exists.",
            )
        return _compose_for_existing_task_or_draft(
            resolved_active_task_id,
            preferred_file=selected_file,
            history_action_mode="delete",
        )

    def _cancel_history_action(
        active_task_id: str | None,
        selected_file: str | None,
    ):
        return _compose_for_existing_task_or_draft(
            active_task_id,
            preferred_file=selected_file,
        )

    def _rename_task_from_history(
        task_title: str | None,
        target_task_id: str | None,
        active_task_id: str | None,
        selected_file: str | None,
    ):
        resolved_active_task_id = active_task_id or target_task_id
        if not target_task_id or not session_store.has_task(target_task_id):
            return _compose_for_existing_task_or_draft(
                resolved_active_task_id,
                preferred_file=selected_file,
                history_notice="Task no longer exists.",
            )

        session_store.rename_task(target_task_id, str(task_title or ""))
        notice = "Custom title cleared." if not str(task_title or "").strip() else "Title updated."
        return _compose_for_existing_task_or_draft(
            resolved_active_task_id,
            preferred_file=selected_file,
            history_notice=notice,
        )

    def _delete_task_from_history(
        target_task_id: str | None,
        active_task_id: str | None,
        selected_file: str | None,
    ):
        resolved_active_task_id = active_task_id or target_task_id
        if not target_task_id or not session_store.has_task(target_task_id):
            return _compose_for_existing_task_or_draft(
                resolved_active_task_id,
                preferred_file=selected_file,
                history_notice="Task no longer exists.",
            )

        deleting_active_task = str(target_task_id) == str(resolved_active_task_id or "")
        session_store.delete_task(target_task_id)
        if deleting_active_task:
            remaining_tasks = _visible_recent_tasks(session_store.list_recent_tasks(limit=50), current_task_id="")
            next_task_id = _task_id_after_deletion(resolved_active_task_id, target_task_id, remaining_tasks)
            if next_task_id:
                return _compose_for_task_id(next_task_id, preferred_file=None, history_notice="Task deleted.")
            return _compose_draft_outputs(
                history_notice="Task deleted.",
            )
        remaining_tasks = _visible_recent_tasks(session_store.list_recent_tasks(limit=50), current_task_id="")
        next_task_id = _task_id_after_deletion(resolved_active_task_id, target_task_id, remaining_tasks)
        if next_task_id:
            return _compose_for_task_id(next_task_id, preferred_file=None, history_notice="Task deleted.")
        return _compose_for_existing_task_or_draft(
            resolved_active_task_id,
            preferred_file=selected_file,
            history_notice="Task deleted.",
        )

    def _run_turn(
        prompt_input: dict[str, Any] | None,
        main_history: list[dict[str, str]] | None,
        trace_history: list[dict[str, str]] | None,
        browser_task_id: str | None,
        task_id: str | None,
        report_requested: bool,
        selected_file: str | None,
        run_overview: dict[str, Any] | None,
    ):
        prompt_input = prompt_input or {}
        text = str(prompt_input.get("text") or "")
        files = list(prompt_input.get("files") or [])

        upload_payloads: dict[str, bytes] = {}
        for raw_file in files:
            file_path = Path(getattr(raw_file, "path", raw_file))
            if not file_path.exists():
                continue
            upload_payloads[_resolved_upload_name(raw_file)] = file_path.read_bytes()

        yield from _submit_turn(
            text=text,
            upload_payloads=upload_payloads,
            main_history=main_history,
            trace_history=trace_history,
            browser_task_id=browser_task_id,
            task_id=task_id,
            report_requested=report_requested,
            selected_file=selected_file,
            run_overview=run_overview,
        )

    def _run_demo_case(
        main_history: list[dict[str, Any]] | None,
        trace_history: list[dict[str, Any]] | None,
        browser_task_id: str | None,
        task_id: str | None,
        selected_file: str | None,
        run_overview: dict[str, Any] | None,
    ):
        yield from _submit_turn(
            text=str(_DEMO_CASE["prompt"]),
            upload_payloads=_demo_case_uploads(),
            main_history=main_history,
            trace_history=trace_history,
            browser_task_id=browser_task_id,
            task_id=task_id,
            report_requested=True,
            selected_file=selected_file,
            run_overview=run_overview,
        )

    with gr.Blocks(title="HealthFlow Web", fill_width=True, fill_height=True, css=_WEB_APP_CSS, head=_WEB_APP_HEAD) as demo:
        browser_task_id = gr.BrowserState(None, storage_key="healthflow-active-task")
        active_task_state = gr.State(None)
        selected_file_state = gr.State(None)
        run_overview_state = gr.State(_empty_run_overview())

        with gr.Sidebar(label="History", open=True, width=336):
            gr.HTML(_sidebar_brand_html())
            new_task_button = gr.Button("New Task", variant="primary")
            history_notice = gr.Markdown(visible=False)
            history_list = gr.HTML(value=_history_list_html([], None), elem_id="hf-history-list")
            history_target_task = gr.Textbox(value="", visible="hidden", container=False, elem_id="hf-history-target-task")
            history_open_trigger = gr.Button("Open task", visible="hidden", elem_id="hf-history-open-trigger")
            history_rename_trigger = gr.Button("Rename task", visible="hidden", elem_id="hf-history-rename-trigger")
            history_delete_trigger = gr.Button("Delete task", visible="hidden", elem_id="hf-history-delete-trigger")
            with gr.Column(visible=False, elem_classes=["hf-history-panel"]) as history_action_panel:
                history_action_heading = gr.HTML(value="", visible=False)
                rename_input = gr.Textbox(
                    label="Task title",
                    lines=1,
                    visible=False,
                )
                with gr.Row(elem_classes=["hf-history-inline"]):
                    save_title_button = gr.Button("Save", variant="primary", size="sm", visible=False)
                    confirm_delete_button = gr.Button("Delete task", variant="stop", size="sm", visible=False)
                    cancel_action_button = gr.Button("Cancel", size="sm", visible=False)

        with gr.Column(elem_classes=["hf-content-shell"]):
            with gr.Row(elem_classes=["hf-main"]):
                with gr.Column(scale=8, min_width=640, elem_classes=["hf-chat-shell"]):
                    task_header = gr.Markdown(elem_classes=["hf-task-header"], container=False)
                    run_overview = gr.HTML(
                        value=_run_overview_html(_empty_run_overview()),
                        elem_classes=["hf-run-overview-shell"],
                    )
                    with gr.Column(visible=True, elem_classes=["hf-starter-shell"]) as starter_panel:
                        starter_html = gr.HTML(value=_starter_card_html(), container=False)
                        demo_case_button = gr.Button(
                            "Launch Predictive Modeling Demo",
                            variant="primary",
                            elem_classes=["hf-starter-button"],
                        )
                    main_chatbot = gr.Chatbot(
                        show_label=False,
                        container=False,
                        type="messages",
                        value=[],
                        scale=1,
                        layout="panel",
                        show_copy_button=True,
                        elem_classes=["hf-chatbot"],
                    )
                    with gr.Column(scale=0, elem_classes=["hf-composer-shell"]):
                        composer_attachments = gr.HTML(
                            value="",
                            visible=True,
                            container=False,
                            elem_id="hf-composer-attachments",
                            elem_classes=["hf-composer-attachments", "is-empty"],
                        )
                        prompt_input = gr.MultimodalTextbox(
                            interactive=True,
                            file_count="multiple",
                            placeholder="Message HealthFlow",
                            show_label=False,
                            elem_id="hf-prompt-input",
                            elem_classes=["hf-composer"],
                        )
                with gr.Column(scale=4, min_width=400, elem_classes=["hf-workspace-shell"]):
                    with gr.Column(elem_classes=["hf-detail-shell"]):
                        with gr.Row(elem_classes=["hf-detail-nav"]):
                            workspace_panel_button = gr.Button(
                                "Workspace",
                                variant="primary",
                                elem_classes=["hf-detail-switch"],
                            )
                            advanced_panel_button = gr.Button(
                                "Advanced",
                                variant="secondary",
                                elem_classes=["hf-detail-switch"],
                            )

                        with gr.Column(visible=True, elem_classes=["hf-detail-panel", "hf-workspace-panel"]) as workspace_panel:
                            with gr.Row(elem_classes=["hf-workspace-row"]):
                                with gr.Column(scale=4, min_width=0, elem_classes=["hf-browser-pane"]):
                                    gr.Markdown("### Workspace", container=False)
                                    workspace_browser = gr.HTML(
                                        value=_workspace_tree_html([], None),
                                        elem_id="hf-workspace-tree",
                                    )
                                    workspace_target_file = gr.Textbox(
                                        value="",
                                        visible="hidden",
                                        container=False,
                                        elem_id="hf-workspace-target-file",
                                    )
                                    workspace_open_trigger = gr.Button(
                                        "Open workspace file",
                                        visible="hidden",
                                        elem_id="hf-workspace-open-trigger",
                                    )

                                with gr.Column(scale=6, min_width=0, elem_classes=["hf-preview-pane"]):
                                    preview_header = gr.Markdown(value="### Preview", container=False)
                                    preview_markdown = gr.Markdown(visible=False, container=False)
                                    preview_image = gr.Image(
                                        label="Image preview",
                                        show_label=False,
                                        container=False,
                                        visible=False,
                                        type="filepath",
                                        show_download_button=False,
                                    )
                                    preview_table = gr.Dataframe(
                                        label="Data preview",
                                        show_label=False,
                                        visible=False,
                                        interactive=False,
                                        show_copy_button=True,
                                        max_height=430,
                                    )
                                    preview_code = gr.Code(
                                        label="Code preview",
                                        container=False,
                                        visible=False,
                                        interactive=False,
                                        lines=18,
                                        max_lines=30,
                                    )
                                    preview_empty = gr.Markdown(value=_EMPTY_PREVIEW_TEXT, container=False)
                                    download_button = gr.DownloadButton("Download file", visible=False)

                        with gr.Column(visible=False, elem_classes=["hf-detail-panel", "hf-advanced-panel"]) as advanced_panel:
                            with gr.Column(scale=0, elem_classes=["hf-advanced-toolbar"]):
                                gr.Markdown("### Advanced", container=False)
                                report_requested = gr.Checkbox(
                                    label="Generate report.md for each turn",
                                    value=False,
                                    container=False,
                                )
                            trace_chatbot = gr.Chatbot(
                                show_label=False,
                                container=False,
                                type="messages",
                                value=[{"role": "assistant", "content": _TRACE_ASSISTANT_TEXT}],
                                scale=1,
                                layout="panel",
                                show_copy_button=True,
                                elem_classes=["hf-trace-panel"],
                            )

                    workspace_panel_button.click(
                        lambda: _detail_panel_updates("workspace"),
                        None,
                        [workspace_panel_button, advanced_panel_button, workspace_panel, advanced_panel],
                        queue=False,
                        show_progress="hidden",
                    )
                    advanced_panel_button.click(
                        lambda: _detail_panel_updates("advanced"),
                        None,
                        [workspace_panel_button, advanced_panel_button, workspace_panel, advanced_panel],
                        queue=False,
                        show_progress="hidden",
                    )

        app_outputs = [
            main_chatbot,
            trace_chatbot,
            browser_task_id,
            active_task_state,
            task_header,
            run_overview,
            run_overview_state,
            starter_panel,
            history_list,
            history_action_panel,
            history_action_heading,
            rename_input,
            save_title_button,
            confirm_delete_button,
            cancel_action_button,
            history_notice,
            workspace_browser,
            selected_file_state,
            preview_header,
            preview_markdown,
            preview_image,
            preview_table,
            preview_code,
            preview_empty,
            download_button,
            composer_attachments,
            prompt_input,
        ]

        demo.load(
            _load_session,
            [browser_task_id],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )

        new_task_button.click(
            _new_task,
            None,
            app_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            lambda: _detail_panel_updates("workspace"),
            None,
            [workspace_panel_button, advanced_panel_button, workspace_panel, advanced_panel],
            queue=False,
            show_progress="hidden",
        )

        prompt_input.submit(
            _run_turn,
            [
                prompt_input,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                active_task_state,
                report_requested,
                selected_file_state,
                run_overview_state,
            ],
            app_outputs,
        )
        demo_case_button.click(
            _run_demo_case,
            [
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                active_task_state,
                selected_file_state,
                run_overview_state,
            ],
            app_outputs,
        )

        history_open_trigger.click(
            _switch_task,
            [history_target_task],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )
        history_rename_trigger.click(
            _start_history_rename,
            [
                history_target_task,
                active_task_state,
                selected_file_state,
            ],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )
        history_delete_trigger.click(
            _start_history_delete,
            [
                history_target_task,
                active_task_state,
                selected_file_state,
            ],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )
        save_title_button.click(
            _rename_task_from_history,
            [
                rename_input,
                history_target_task,
                active_task_state,
                selected_file_state,
            ],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )
        confirm_delete_button.click(
            _delete_task_from_history,
            [
                history_target_task,
                active_task_state,
                selected_file_state,
            ],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )
        cancel_action_button.click(
            _cancel_history_action,
            [
                active_task_state,
                selected_file_state,
            ],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )
        workspace_open_trigger.click(
            _on_workspace_file_selected,
            [workspace_target_file, active_task_state],
            [
                selected_file_state,
                preview_header,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
            queue=False,
            show_progress="hidden",
        )
    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=share, allowed_paths=allowed_paths)
