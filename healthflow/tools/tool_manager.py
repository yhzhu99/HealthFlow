"""
Tool manager that works with the updated system.
"""
import logging
from typing import Dict, Any, List, Callable, Optional
import asyncio

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages available tools and their execution."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        from .code_interpreter import CodeInterpreterTool
        from .data_probe import DataProbeTool
        from .tool_creator import ToolCreatorTool

        # Register built-in tools
        self.register_tool("code_interpreter", CodeInterpreterTool())
        self.register_tool("probe_data_structure", DataProbeTool())

        # Register tool creator
        self.tool_creator = ToolCreatorTool()
        self.register_tool("add_new_tool", self.tool_creator)

        logger.info(f"Initial tools registered: {', '.join(self.tools.keys())}")

    def register_tool(self, name: str, tool_instance: Any):
        """Register a tool."""
        self.tools[name] = {
            "instance": tool_instance,
            "description": getattr(tool_instance, 'description', f"Tool: {name}"),
            "parameters": getattr(tool_instance, 'parameters', {})
        }

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        return self.tools.get(tool_name, {})

    async def execute_tool(self, tool_name: str, tool_input: str) -> Any:
        """Execute a tool asynchronously."""
        if tool_name not in self.tools:
            # Check if it's a dynamically created tool
            if hasattr(self, 'tool_creator'):
                created_tool = self.tool_creator.get_tool_instance(tool_name)
                if created_tool:
                    self.register_tool(tool_name, created_tool)
                else:
                    raise ValueError(f"Tool '{tool_name}' not found")
            else:
                raise ValueError(f"Tool '{tool_name}' not found")

        tool_instance = self.tools[tool_name]["instance"]

        try:
            # Check if tool has async execute method
            if hasattr(tool_instance, 'aexecute'):
                return await tool_instance.aexecute(tool_input)
            elif hasattr(tool_instance, 'execute'):
                # Run sync method in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, tool_instance.execute, tool_input)
            else:
                # Fallback: try calling the tool directly
                return await loop.run_in_executor(None, tool_instance, tool_input)
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def call_tool(self, tool_name: str, tool_input: str) -> Any:
        """Execute a tool synchronously (fallback)."""
        if tool_name not in self.tools:
            # Check if it's a dynamically created tool
            if hasattr(self, 'tool_creator'):
                created_tool = self.tool_creator.get_tool_instance(tool_name)
                if created_tool:
                    self.register_tool(tool_name, created_tool)
                else:
                    raise ValueError(f"Tool '{tool_name}' not found")
            else:
                raise ValueError(f"Tool '{tool_name}' not found")

        tool_instance = self.tools[tool_name]["instance"]

        try:
            if hasattr(tool_instance, 'execute'):
                return tool_instance.execute(tool_input)
            else:
                return tool_instance(tool_input)
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"
