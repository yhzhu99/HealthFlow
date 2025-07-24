import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from camel.interpreters import InternalPythonInterpreter
from camel.toolkits import FunctionTool
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

class MCPToolServer:
    """
    An MCP server that provides tools to HealthFlow agents.
    It is designed to be simple, extensible, and dynamically updatable,
    which is key to the system's self-evolving capability.
    """

    def __init__(self, host="127.0.0.1", port=8000, tools_dir: Path = Path("./data/tools")):
        self.host = host
        self.port = port
        self.tools_dir = tools_dir
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        self.mcp = FastMCP("HealthFlow ToolBank")
        self._setup_initial_tools()

    def _setup_initial_tools(self):
        """Sets up the default tools available at startup with enhanced healthcare and ML capabilities."""
        # 1. Advanced Code Interpreter Tool with ML Libraries
        interpreter = InternalPythonInterpreter()

        def execute_python_code(code: str) -> str:
            """
            Executes Python code with pre-loaded healthcare and ML libraries.
            Supports: PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn for complex data analysis.
            Ideal for time-series EHR data analysis, ML model training, and medical data processing.
            """
            try:
                # Pre-load common libraries for healthcare data analysis
                preload_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    pass

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:
    pass

try:
    from scipy import stats
    from scipy.stats import ttest_ind, chi2_contingency
except ImportError:
    pass
"""
                full_code = preload_code + "\n" + code
                result = interpreter.run(full_code, "python")
                return f"Execution successful.\nOutput:\n{result}"
            except Exception as e:
                return f"Execution failed.\nError: {e}\nSuggestion: Check imports, variable definitions, and data types."

        self.mcp.tool()(execute_python_code)

        # 2. Healthcare Data Analysis Tool
        def analyze_ehr_data(data_description: str, analysis_type: str) -> str:
            """
            Specialized tool for Electronic Health Record (EHR) data analysis.
            Supports time-series analysis, patient cohort studies, and clinical outcome prediction.
            
            Args:
                data_description: Description of the EHR dataset structure
                analysis_type: Type of analysis (time_series, cohort_analysis, risk_prediction, etc.)
            """
            try:
                analysis_code = f"""
# EHR Data Analysis Pipeline
print("Starting EHR Data Analysis...")
print(f"Data Description: {data_description}")
print(f"Analysis Type: {analysis_type}")

# Sample analysis framework
if '{analysis_type}' == 'time_series':
    print("Setting up time-series analysis for EHR data...")
    print("Recommended models: LSTM, GRU, Transformer")
    analysis_steps = [
        "1. Data preprocessing and temporal alignment",
        "2. Feature engineering for clinical variables",
        "3. Model selection (LSTM/GRU/Transformer)",
        "4. Training with temporal validation",
        "5. Performance evaluation and clinical interpretation"
    ]
else:
    analysis_steps = [
        "1. Data quality assessment",
        "2. Statistical analysis",
        "3. Feature selection",
        "4. Model training and validation",
        "5. Clinical significance testing"
    ]

for step in analysis_steps:
    print(step)

print("Analysis framework prepared. Ready for implementation.")
"""
                result = interpreter.run(analysis_code, "python")
                return f"EHR Analysis Framework:\n{result}"
            except Exception as e:
                return f"EHR Analysis failed: {e}"

        self.mcp.tool()(analyze_ehr_data)

        # 3. ML Model Comparison Tool
        def compare_ml_models(model_types: str, dataset_info: str) -> str:
            """
            Compare performance of different ML models (GRU, LSTM, Transformer) for healthcare data.
            Provides benchmarking framework for medical AI applications.
            
            Args:
                model_types: Comma-separated list of models to compare (e.g., "GRU,LSTM,Transformer")
                dataset_info: Information about the dataset characteristics
            """
            try:
                comparison_code = f"""
# ML Model Comparison Framework
print("Healthcare ML Model Comparison")
print(f"Models to compare: {model_types}")
print(f"Dataset info: {dataset_info}")

models = [model.strip() for model in '{model_types}'.split(',')]

print("\nModel Comparison Framework:")
for model in models:
    print(f"\n{model} Model:")
    if model.upper() == 'GRU':
        print("  - Good for: Sequential medical data, medication sequences")
        print("  - Complexity: Medium")
        print("  - Training time: Fast")
    elif model.upper() == 'LSTM':
        print("  - Good for: Long-term temporal patterns in EHR")
        print("  - Complexity: High")
        print("  - Training time: Medium")
    elif model.upper() == 'TRANSFORMER':
        print("  - Good for: Complex multi-modal medical data")
        print("  - Complexity: Very High")
        print("  - Training time: Slow")
    else:
        print(f"  - Custom model: {model}")

print("\nBenchmarking metrics to track:")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Clinical Significance"]
for metric in metrics:
    print(f"  - {metric}")

print("\nComparison framework ready for implementation.")
"""
                result = interpreter.run(comparison_code, "python")
                return f"Model Comparison Results:\n{result}"
            except Exception as e:
                return f"Model comparison failed: {e}"

        self.mcp.tool()(compare_ml_models)

        # 4. Data Probing Tool
        def probe_data_structure(file_path: str, file_type: str = "auto") -> str:
            """
            Probes and analyzes the structure of unknown data files.
            Supports pickle files, CSV, JSON, Excel, and other common formats.
            Automatically detects file type if not specified.
            
            Args:
                file_path: Path to the data file to probe
                file_type: File format hint ("pickle", "csv", "json", "excel", "auto")
            """
            try:
                probe_code = f"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

file_path = "{file_path}"
file_type = "{file_type}"

print(f"Probing data file: {{file_path}}")
print(f"File exists: {{os.path.exists(file_path)}}")

if not os.path.exists(file_path):
    print("ERROR: File does not exist")
else:
    file_size = os.path.getsize(file_path)
    print(f"File size: {{file_size:,}} bytes")
    
    # Auto-detect file type if needed
    if file_type == "auto":
        if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            file_type = "pickle"
        elif file_path.endswith('.csv'):
            file_type = "csv"
        elif file_path.endswith('.json'):
            file_type = "json"
        elif file_path.endswith(('.xlsx', '.xls')):
            file_type = "excel"
        else:
            print("Unknown file extension, trying multiple formats...")
    
    print(f"Detected/specified file type: {{file_type}}")
    
    try:
        if file_type == "pickle":
            print("\\n=== PICKLE FILE ANALYSIS ===")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Data type: {{type(data)}}")
            
            if isinstance(data, dict):
                print(f"Dictionary with {{len(data)}} keys:")
                for i, (key, value) in enumerate(list(data.items())[:5]):
                    print(f"  {{key}}: {{type(value)}} - {{str(value)[:50]}}...")
                    if i >= 4 and len(data) > 5:
                        print(f"  ... and {{len(data)-5}} more keys")
                        break
                        
            elif isinstance(data, (list, tuple)):
                print(f"{{type(data).__name__}} with {{len(data)}} elements")
                if len(data) > 0:
                    print(f"First element type: {{type(data[0])}}")
                    print(f"First element preview: {{str(data[0])[:100]}}...")
                    
            elif isinstance(data, np.ndarray):
                print(f"NumPy array shape: {{data.shape}}")
                print(f"Data type: {{data.dtype}}")
                print(f"Min value: {{data.min()}}, Max value: {{data.max()}}")
                print(f"Sample values: {{data.flat[:10]}}")
                
            elif hasattr(data, 'shape'):  # Tensor-like objects
                print(f"Tensor-like object shape: {{data.shape}}")
                print(f"Object type: {{type(data)}}")
                
            else:
                print(f"Data content preview: {{str(data)[:200]}}...")
                
        elif file_type == "csv":
            print("\\n=== CSV FILE ANALYSIS ===")
            # Read first few rows to understand structure
            df_preview = pd.read_csv(file_path, nrows=5)
            df_info = pd.read_csv(file_path)
            
            print(f"DataFrame shape: {{df_info.shape}}")
            print(f"Columns: {{list(df_info.columns)}}")
            print(f"Data types:\\n{{df_info.dtypes}}")
            print(f"\\nFirst 5 rows:\\n{{df_preview}}")
            print(f"\\nMissing values:\\n{{df_info.isnull().sum()}}")
            
        elif file_type == "json":
            print("\\n=== JSON FILE ANALYSIS ===")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"JSON structure type: {{type(data)}}")
            if isinstance(data, dict):
                print(f"Top-level keys: {{list(data.keys())}}")
                for key, value in list(data.items())[:3]:
                    print(f"  {{key}}: {{type(value)}} - {{str(value)[:50]}}...")
                    
        elif file_type == "excel":
            print("\\n=== EXCEL FILE ANALYSIS ===")
            excel_file = pd.ExcelFile(file_path)
            print(f"Sheet names: {{excel_file.sheet_names}}")
            
            for sheet in excel_file.sheet_names[:3]:  # First 3 sheets
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                print(f"\\nSheet '{{sheet}}' shape: {{df.shape}}")
                print(f"Columns: {{list(df.columns)}}")
                
    except Exception as probe_error:
        print(f"Error probing as {{file_type}}: {{probe_error}}")
        print("Trying alternative formats...")
        
        # Try alternative formats
        formats_to_try = ["pickle", "csv", "json"]
        for fmt in formats_to_try:
            if fmt != file_type:
                try:
                    print(f"\\nTrying {{fmt}} format...")
                    if fmt == "pickle":
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        print(f"SUCCESS: File is pickle format, type: {{type(data)}}")
                        break
                    elif fmt == "csv":
                        df = pd.read_csv(file_path, nrows=3)
                        print(f"SUCCESS: File is CSV format, shape: {{df.shape}}")
                        break
                    elif fmt == "json":
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        print(f"SUCCESS: File is JSON format, type: {{type(data)}}")
                        break
                except:
                    continue

print("\\nData probing complete.")
"""
                result = interpreter.run(probe_code, "python")
                return f"Data Structure Analysis:\n{result}"
            except Exception as e:
                return f"Data probing failed: {e}"

        self.mcp.tool()(probe_data_structure)

        # 5. Tool Management Tool (for self-evolution)
        def add_new_tool(name: str, code: str, description: str) -> str:
            """
            Dynamically creates and registers a new tool from Python code.
            Enhanced for healthcare-specific tool creation with validation.
            The code must define a single function with the same name as the 'name' parameter.
            """
            try:
                # Validate tool name for healthcare context
                if not name.replace('_', '').replace('-', '').isalnum():
                    return f"Error: Tool name '{name}' contains invalid characters. Use only letters, numbers, and underscores."
                
                tool_path = self.tools_dir / f"{name}.py"
                
                # Add healthcare-specific imports to the tool code
                enhanced_code = f"""
# Auto-generated healthcare tool: {name}
# Description: {description}

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Optional

# Healthcare-specific imports
try:
    import torch
    import torch.nn as nn
except ImportError:
    pass

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
except ImportError:
    pass

{code}
"""
                
                with tool_path.open("w") as f:
                    f.write(enhanced_code)

                # Dynamically import and register
                spec = __import__("importlib.util").util.spec_from_file_location(name, tool_path)
                module = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(module)

                new_func = getattr(module, name)
                new_func.__doc__ = description

                self.mcp.tool(new_func)
                logger.info(f"Successfully added new healthcare tool: {name}")
                return f"Healthcare tool '{name}' was added successfully with enhanced capabilities."
            except Exception as e:
                logger.error(f"Failed to add new tool '{name}': {e}", exc_info=True)
                return f"Error adding tool '{name}': {e}. Check function definition and imports."

        self.mcp.tool()(add_new_tool)

    def _load_dynamic_tools(self):
        """Load dynamically created tools from the tools directory."""
        for tool_file in self.tools_dir.glob('*.py'):
            if tool_file.name.startswith('_'):
                continue
            try:
                module_name = tool_file.stem
                spec = __import__("importlib.util").util.spec_from_file_location(module_name, tool_file)
                module = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, module_name)
                self.mcp.tool(func)
                logger.info(f"Loaded dynamic tool: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load tool from {tool_file}: {e}")

    async def start(self):
        """Starts the MCP server (simplified approach without subprocess)."""
        # Load any dynamically created tools
        self._load_dynamic_tools()
        logger.info("MCP Tool Server initialized (using direct integration)")

    async def stop(self):
        """Stops the MCP server (cleanup if needed)."""
        logger.info("MCP Tool Server stopped.")

    def as_camel_tool(self, include_management: bool = False) -> FunctionTool:
        """
        Creates a Camel AI FunctionTool that acts as a client to this MCP server.
        """
        from ..core.simple_interpreter import SimpleHealthcareInterpreter
        
        def execute_python_code(code: str) -> str:
            """
            Executes a given string of Python code and returns the output.
            This tool is powerful for data analysis, calculations, and dynamic tasks.
            Uses the enhanced SimpleHealthcareInterpreter with pre-loaded ML libraries.
            
            Args:
                code: The Python code to execute as a string.
            
            Returns:
                The result of the code execution or error message.
            """
            try:
                interpreter = SimpleHealthcareInterpreter()
                result = interpreter.run(code, "python")
                return f"Execution successful.\nOutput:\n{result}"
            except Exception as e:
                return f"Execution failed.\nError: {e}"
        
        return FunctionTool(execute_python_code)