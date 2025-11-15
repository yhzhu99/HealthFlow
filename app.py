# app.py
# Web UI for the HealthFlow Agentic System.

import asyncio
import json
import toml
from pathlib import Path
from typing import Dict
import streamlit as st

# Ensure the project root is in the path to import healthflow modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.system import HealthFlowSystem
from healthflow.core.config import get_config, setup_logging, HealthFlowConfig, LLMProviderConfig, SystemConfig, EvaluationConfig, LoggingConfig

# --- Page Configuration ---
st.set_page_config(
    page_title="HealthFlow Agent",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions & State Management ---

@st.cache_resource
def load_config_data():
    """Loads the config.toml file. Caches the result for performance."""
    config_path = Path("config.toml")
    if config_path.exists():
        return toml.load(config_path)
    st.warning("`config.toml` not found. Relying on Streamlit secrets for configuration.")
    return {}

# Initialize session state variables
if "running" not in st.session_state:
    st.session_state.running = False
if "task_result" not in st.session_state:
    st.session_state.task_result = None

def get_system_initializer():
    """Returns the correct system initialization function based on environment."""
    # IS_DEPLOYED checks if the 'llm' secret exists, a good proxy for cloud deployment
    is_deployed = hasattr(st.secrets, 'llm') and 'llm' in st.secrets

    if is_deployed:
        st.session_state.is_deployed = True
        return initialize_system_from_secrets
    else:
        st.session_state.is_deployed = False
        return initialize_system_from_file

def initialize_system_from_secrets(active_llm_name: str) -> HealthFlowSystem:
    """Initializes HealthFlowSystem using Streamlit's secrets for deployed apps."""
    try:
        llm_config_data = st.secrets.llm[active_llm_name]
        llm_config = LLMProviderConfig(**llm_config_data)
        system_config = SystemConfig(**st.secrets.get("system", {}))
        evaluation_config = EvaluationConfig(**st.secrets.get("evaluation", {}))
        logging_config = LoggingConfig(**st.secrets.get("logging", {"log_level": "INFO", "log_file": "healthflow_streamlit.log"}))
        config = HealthFlowConfig(active_llm_name=active_llm_name, llm=llm_config, system=system_config, evaluation=evaluation_config, logging=logging_config)
        setup_logging(config)
        return HealthFlowSystem(config=config, experience_path=Path(config.system.workspace_dir) / "experience.jsonl")
    except Exception as e:
        st.error(f"Error initializing from secrets for LLM '{active_llm_name}': {e}")
        st.stop()

def initialize_system_from_file(active_llm_name: str) -> HealthFlowSystem:
    """Initializes HealthFlowSystem from local config.toml for development."""
    try:
        config = get_config(Path("config.toml"), active_llm_name)
        setup_logging(config)
        return HealthFlowSystem(config=config, experience_path=Path("workspace/experience.jsonl"))
    except Exception as e:
        st.error(f"Error initializing from config.toml: {e}")
        st.stop()

def run_healthflow_task_async(system: HealthFlowSystem, task: str, uploaded_files: Dict[str, bytes]):
    """Wrapper to run the async HealthFlow task from Streamlit's sync context."""
    return asyncio.run(system.run_task(task, uploaded_files=uploaded_files))

def read_file_from_workspace(workspace_path_str: str, file_glob: str) -> (str, str):
    """Safely reads a file from the workspace directory, handling glob patterns."""
    if not workspace_path_str:
        return None, "Workspace path not available."
    workspace = Path(workspace_path_str)
    if not workspace.exists():
        return None, f"Workspace directory not found at {workspace}"

    # Handle glob patterns like 'task_list_v*.md'
    files = sorted(list(workspace.glob(file_glob)), reverse=True)
    if not files:
        return None, f"No file matching '{file_glob}' found in workspace."

    file_path = files[0] # Get the latest version if multiple
    try:
        return file_path.name, file_path.read_text(encoding="utf-8")
    except Exception as e:
        return file_path.name, f"Error reading file: {e}"

# --- UI Rendering ---

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    initializer = get_system_initializer()

    if st.session_state.is_deployed:
        llm_options = list(st.secrets.llm.keys()) if 'llm' in st.secrets else []
        st.info("☁️ Cloud Mode: Config from Streamlit secrets.")
    else:
        config_data_local = load_config_data()
        llm_options = list(config_data_local.get("llm", {}).keys())
        st.info("💻 Local Mode: Config from `config.toml`.")

    if not llm_options:
        st.error("No LLM configurations found!")
        st.stop()

    active_llm = st.selectbox("Select Reasoning LLM", options=llm_options)
    st.markdown("---")
    st.caption("A Self-Evolving Meta-System for Agentic Healthcare AI.")
    st.markdown("For **COMP9501 Final Group Project**")
    st.markdown("[GitHub Repository](https://github.com/yhzhu99/COMP9501-Final-Project)") # <-- UPDATE YOUR REPO LINK

# --- Main Page ---
st.title("🌊 HealthFlow Agent")
st.markdown("An autonomous AI agent that formulates and executes plans to solve complex healthcare research tasks.")

# --- Input Area ---
st.subheader("1. Define Your Task")
task_input = st.text_area(
    "Describe the research task you want the agent to perform:",
    height=125,
    placeholder="e.g., Analyze the uploaded patient data to identify the top 3 risk factors for readmission.",
    disabled=st.session_state.running,
)

st.subheader("2. Upload Data (Optional)")
uploaded_files = st.file_uploader(
    "Upload CSV, JSON, or text files for the agent to use.",
    type=["csv", "json", "txt", "md"],
    accept_multiple_files=True,
    disabled=st.session_state.running,
)

st.subheader("3. Run Agent")
if st.button("🚀 Start Agent Execution", type="primary", disabled=st.session_state.running or not task_input):
    st.session_state.running = True
    st.session_state.task_result = None

    files_to_upload = {f.name: f.getvalue() for f in uploaded_files} if uploaded_files else {}

    with st.spinner("HealthFlow is orchestrating... This may take several minutes."):
        try:
            system = initializer(active_llm)
            result = run_healthflow_task_async(system, task_input, files_to_upload)
            st.session_state.task_result = result
        except Exception as e:
            st.session_state.task_result = {"error": str(e), "success": False}
            st.exception(e)

    st.session_state.running = False
    st.rerun()

# --- Output Area ---
if st.session_state.task_result:
    st.markdown("---")
    st.header("Execution Results")
    result = st.session_state.task_result

    if result.get("success"):
        st.success("**Task Completed Successfully!**", icon="✅")
    else:
        st.error(f"**Task Failed.**\n\nError: {result.get('error', result.get('final_summary', 'No specific error message.'))}", icon="❌")

    workspace = result.get("workspace_path", "")

    # Create tabs for organized output
    tab_answer, tab_log, tab_plan, tab_history, tab_summary = st.tabs([
        "✅ Final Answer", "📄 Execution Log", "🗺️ Final Plan", "🔍 Full History", "📝 Summary"
    ])

    with tab_answer:
        st.markdown(result.get("answer", "No answer was generated."))

    with tab_log:
        log_filename, log_content = read_file_from_workspace(workspace, "execution.log")
        if log_filename:
            st.code(log_content, language="log", line_numbers=True)
        else:
            st.warning(log_content)

    with tab_plan:
        plan_filename, plan_content = read_file_from_workspace(workspace, "task_list_v*.md")
        if plan_filename:
            st.markdown(f"**File:** `{plan_filename}`")
            st.markdown(plan_content)
        else:
            st.warning(plan_content)

    with tab_history:
        history_filename, history_content = read_file_from_workspace(workspace, "full_history.json")
        if history_filename:
            try:
                history_json = json.loads(history_content)
                st.json(history_json)
            except json.JSONDecodeError:
                st.error("Could not parse full_history.json")
                st.code(history_content, language="json")
        else:
            st.warning(history_content)

    with tab_summary:
        st.subheader("Final Outcome")
        st.write(result.get('final_summary', 'No summary available.'))
        st.metric(label="Execution Time", value=f"{result.get('execution_time', 0):.2f} seconds")
        st.info(f"All task artifacts are located in the following directory on the server:\n`{workspace}`")