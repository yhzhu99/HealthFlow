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
        """Sets up the default tools available at startup."""
        # 1. Code Interpreter Tool
        interpreter = InternalPythonInterpreter()

        def execute_python_code(code: str) -> str:
            """
            Executes a given string of Python code and returns the output.
            This tool is powerful for data analysis, calculations, and dynamic tasks.
            Only use it for trusted code.
            """
            try:
                result = interpreter.run(code, "python")
                return f"Execution successful.\nOutput:\n{result}"
            except Exception as e:
                return f"Execution failed.\nError: {e}"

        self.mcp.tool()(execute_python_code)

        # 2. Tool Management Tool (for self-evolution)
        def add_new_tool(name: str, code: str, description: str) -> str:
            """
            Dynamically creates and registers a new tool from a string of Python code.
            This is a meta-tool used by the system to evolve its capabilities.
            The code must define a single function with the same name as the 'name' parameter.
            """
            try:
                tool_path = self.tools_dir / f"{name}.py"
                with tool_path.open("w") as f:
                    f.write(code)

                # Dynamically import and register
                spec = __import__("importlib.util").util.spec_from_file_location(name, tool_path)
                module = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(module)

                new_func = getattr(module, name)
                new_func.__doc__ = description # Set the docstring for description

                self.mcp.tool(new_func)
                logger.info(f"Successfully added new tool: {name}")
                return f"Tool '{name}' was added successfully."
            except Exception as e:
                logger.error(f"Failed to add new tool '{name}': {e}", exc_info=True)
                return f"Error adding tool '{name}': {e}"

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
        
        def execute_python_code(code: str) -> str:
            """
            Executes a given string of Python code and returns the output.
            This tool is powerful for data analysis, calculations, and dynamic tasks.
            
            Args:
                code: The Python code to execute as a string.
            
            Returns:
                The result of the code execution or error message.
            """
            try:
                interpreter = InternalPythonInterpreter()
                result = interpreter.run(code, "python")
                return f"Execution successful.\nOutput:\n{result}"
            except Exception as e:
                return f"Execution failed.\nError: {e}"
        
        return FunctionTool(execute_python_code)