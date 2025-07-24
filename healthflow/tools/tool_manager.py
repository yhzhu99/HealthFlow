import logging
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Callable

from camel.toolkits import FunctionTool
from healthflow.core.interpreter import HealthFlowInterpreter

logger = logging.getLogger(__name__)

class ToolManager:
    """
    A simple, in-process manager for agent tools.
    It supports dynamic loading and on-the-fly creation of new tools.
    """
    def __init__(self, tools_dir: Path):
        self.tools_dir = tools_dir
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.interpreter = HealthFlowInterpreter()
        self.tools: Dict[str, Callable] = {}
        self._register_initial_tools()

    def _register_initial_tools(self):
        """Registers the built-in tools available at startup using wrapper functions."""

        # BUG FIX: Define local wrapper functions to hold the docstrings,
        # as the __doc__ attribute of a bound method is not writable.

        def code_interpreter(code: str) -> str:
            """
            Executes a string of Python code in a secure environment with pre-loaded libraries
            (pandas, numpy, torch, scikit-learn).
            Args:
                code (str): The Python code to execute.
            Returns:
                str: The captured output (stdout) from the code execution or an error message.
            """
            return self.interpreter.run(code)

        def probe_data_structure(file_path: str) -> str:
            """
            Analyzes the structure of a data file (e.g., CSV, pickle) to understand its contents.
            Args:
                file_path (str): The path to the data file.
            Returns:
                str: A summary of the file's structure, columns, data types, and a preview.
            """
            return self._probe_data_structure(file_path)

        def add_new_tool(name: str, code: str, description: str) -> str:
            """
            Dynamically creates and registers a new tool for the agent to use.
            Args:
                name (str): The name for the new tool function. Must be a valid Python identifier.
                code (str): A string containing the complete Python code for the function definition.
                description (str): A clear docstring explaining what the tool does, its arguments, and what it returns.
            Returns:
                str: A confirmation message indicating success or failure.
            """
            # This wrapper calls the class method that contains the logic
            return self._add_new_tool_logic(name=name, code=code, description=description)

        self.tools = {
            "code_interpreter": code_interpreter,
            "probe_data_structure": probe_data_structure,
            "add_new_tool": add_new_tool,
        }

        logger.info(f"Initial tools registered: {', '.join(self.tools.keys())}")

    async def load_tools(self):
        """Loads all dynamically created tools from the tools directory."""
        self._register_initial_tools()  # Reset to defaults first
        for tool_file in self.tools_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue
            try:
                module_name = tool_file.stem
                spec = importlib.util.spec_from_file_location(module_name, tool_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                tool_func = getattr(module, module_name, None)
                if callable(tool_func):
                    self.tools[module_name] = tool_func
                    logger.info(f"Successfully loaded dynamic tool: {module_name}")
                else:
                    logger.warning(f"Could not find callable function '{module_name}' in {tool_file}")
            except Exception as e:
                logger.error(f"Failed to load dynamic tool from {tool_file}: {e}")
        logger.info(f"All tools loaded. Current toolset: {', '.join(self.tools.keys())}")

    def _add_new_tool_logic(self, name: str, code: str, description: str) -> str:
        """Internal logic for adding a new tool, called by the wrapper."""
        if not name.isidentifier():
            return f"Error: Tool name '{name}' is not a valid Python identifier."

        tool_path = self.tools_dir / f"{name}.py"

        if not code.strip().startswith('def '):
            return f"Error: The provided code for tool '{name}' does not start with 'def'."

        # Inject the docstring into the function code
        lines = code.strip().split('\n')
        first_line = lines[0]
        # Find the indentation of the function body
        indent = ""
        if len(lines) > 1:
            indent = ' ' * (len(lines[1]) - len(lines[1].lstrip(' ')))
        else: # single-line function like 'def my_func(x): return x*2'
             indent = ' ' * (len(first_line) - len(first_line.lstrip(' '))) + '    '

        docstring = f'{indent}"""{description}"""'

        # Find the position to insert the docstring (after the 'def' line)
        if ':' in first_line:
            lines.insert(1, docstring)
            full_code = '\n'.join(lines)
        else:
            return "Error: Could not properly format the new tool code. Ensure the 'def' line ends with a colon."

        try:
            with tool_path.open("w") as f:
                f.write(full_code)

            spec = importlib.util.spec_from_file_location(name, tool_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            new_func = getattr(module, name)
            self.tools[name] = new_func

            logger.info(f"Successfully created and registered new tool: '{name}'")
            return f"Tool '{name}' was created and is now available for use."
        except Exception as e:
            logger.error(f"Failed to create new tool '{name}': {e}", exc_info=True)
            return f"Error creating tool '{name}': {e}"

    def _probe_data_structure(self, file_path: str) -> str:
        """Implementation for the `probe_data_structure` tool."""
        # This implementation is correct and does not need changes.
        probe_code = f"""
import pandas as pd
import pickle
import os

file_path = r'{file_path}'
print(f"Probing file: {{file_path}}")

if not os.path.exists(file_path):
    print(f"Error: File not found at '{{file_path}}'")
else:
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=5)
            print("File type: CSV")
            # Get info by reading the file into memory (can be slow for large files)
            df_full = pd.read_csv(file_path)
            print(f"Shape: {{df_full.shape}}")
            print(f"Columns: {{list(df_full.columns)}}")
            print(f"Data types:\\n{{df_full.dtypes.to_string()}}")
            print(f"\\nFirst 5 rows:\\n{{df.to_string()}}")
        elif file_path.endswith(('.pkl', '.pickle')):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("File type: Pickle")
            print(f"Data type: {{type(data)}}")
            if isinstance(data, pd.DataFrame):
                print(f"Shape: {{data.shape}}")
                print(f"Columns: {{list(data.columns)}}")
                print(f"\\nFirst 5 rows:\\n{{data.head().to_string()}}")
            else:
                print(f"Data preview: {{str(data)[:500]}}...")
        else:
            print(f"Unsupported file type '{{file_path.split('.')[-1]}}' for probing. Please use 'code_interpreter' for custom loading.")
    except Exception as e:
        print(f"Error while probing file: {{e}}")
"""
        return self.interpreter.run(probe_code)

    def get_camel_tools(self) -> List[FunctionTool]:
        """
        Exposes the entire toolset as a list of Camel AI FunctionTools.
        """
        # I'm renaming `add_new_tool` in the function object to match the key for clarity
        # This helps Camel AI correctly identify the function name from the tool call
        tool_functions = []
        for name, func in self.tools.items():
            func.__name__ = name
            tool_functions.append(FunctionTool(func))
        return tool_functions

    async def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict[str, Any]]:
        """Executes a list of tool calls from a Camel message."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = f"Error: Invalid JSON arguments for tool '{tool_name}'."
                logger.error(result)
                results.append({"tool_call_id": tool_call.id, "name": tool_name, "result": result})
                continue

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](**tool_args)
                except Exception as e:
                    result = f"Error executing tool '{tool_name}': {e}"
                    logger.error(result, exc_info=True)
            else:
                result = f"Error: Tool '{tool_name}' not found."
                logger.error(result)

            results.append({"tool_call_id": tool_call.id, "name": tool_name, "result": str(result)})
        return results

    def get_tool_info(self) -> Dict[str, str]:
        """Returns a dictionary of tool names and their docstrings."""
        return {name: func.__doc__ or "No description available." for name, func in self.tools.items()}