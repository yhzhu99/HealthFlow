import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from camel.interpreters import InternalPythonInterpreter
from fastmcp import FastMCP
from langchain_core.tools import BaseTool

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
        self.process = None
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

    async def start(self):
        """Starts the MCP server in a separate process."""
        # Save the server script to a temporary file to run it
        server_script_path = self.tools_dir / "_mcp_server_runner.py"
        with server_script_path.open("w") as f:
            f.write(f"""
from fastmcp import FastMCP
from pathlib import Path
import sys
# Add tools dir to path to import dynamic tools
sys.path.append(str(Path('{self.tools_dir}')))

from healthflow.tools.mcp_server import MCPToolServer

server_instance = MCPToolServer(host='{self.host}', port={self.port}, tools_dir=Path('{self.tools_dir}'))

# Load dynamically created tools
for tool_file in Path('{self.tools_dir}').glob('*.py'):
    if tool_file.name.startswith('_'):
        continue
    try:
        module_name = tool_file.stem
        spec = __import__("importlib.util").util.spec_from_file_location(module_name, tool_file)
        module = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, module_name)
        server_instance.mcp.tool(func)
    except Exception as e:
        print(f"Failed to load tool from {{tool_file}}: {{e}}")

server_instance.mcp.run(transport='streamable-http', host='{self.host}', port={self.port})
""")

        # Using subprocess.Popen to run in the background
        self.process = subprocess.Popen(
            [sys.executable, str(server_script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await asyncio.sleep(2)  # Give it a moment to start
        if self.process.poll() is not None:
             stdout, stderr = self.process.communicate()
             logger.error("MCP server failed to start.")
             logger.error(f"STDOUT: {stdout.decode()}")
             logger.error(f"STDERR: {stderr.decode()}")
        else:
            logger.info(f"MCP Tool Server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stops the MCP server process."""
        if self.process:
            self.process.terminate()
            await asyncio.sleep(1)
            if self.process.poll() is None: # still running
                self.process.kill()
            self.process.wait()
            logger.info("MCP Tool Server stopped.")

    def as_langchain_tool(self, include_management: bool = False) -> BaseTool:
        """
        Creates a LangChain-compatible tool that acts as a client to this MCP server.
        """
        from langchain_community.tools.mcp import MCPSearch

        client_url = f"http://{self.host}:{self.port}"

        exclude_tools = []
        if not include_management:
            exclude_tools.append("add_new_tool")

        return MCPSearch(
            name="mcp_tool_server",
            description=f"Client for the HealthFlow ToolBank. Use this to execute code or other specialized tools. Excluded tools: {exclude_tools}",
            url=client_url,
            exclude_tools=exclude_tools,
        )