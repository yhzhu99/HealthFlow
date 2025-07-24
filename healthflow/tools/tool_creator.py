"""
Tool creator for dynamically creating new tools.
"""
import logging
import os
import json
from typing import Any, Dict
import importlib.util
import tempfile

logger = logging.getLogger(__name__)

class ToolCreatorTool:
    """Tool for creating new tools dynamically."""

    def __init__(self):
        self.description = "Create new tools dynamically by providing tool code and specifications"
        self.parameters = {
            "tool_name": "Name of the new tool",
            "tool_code": "Python code for the tool implementation",
            "description": "Description of what the tool does",
            "parameters": "JSON object describing tool parameters"
        }
        self.created_tools = {}

    def execute(self, tool_input: str) -> str:
        """Execute tool creation."""
        try:
            # Parse input
            if isinstance(tool_input, str):
                if tool_input.strip().startswith('{'):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(tool_input)
                        tool_name = parsed.get('tool_name', '')
                        tool_code = parsed.get('tool_code', '')
                        description = parsed.get('description', '')
                        parameters = parsed.get('parameters', {})
                    except:
                        return f"Error: Invalid JSON format in tool input"
                else:
                    return f"Error: Tool creation requires JSON input with tool_name, tool_code, description, and parameters"
            else:
                return f"Error: Tool creation requires string input"

            if not tool_name or not tool_code:
                return f"Error: Both tool_name and tool_code are required"

            # Validate tool name
            if not tool_name.isidentifier():
                return f"Error: Tool name '{tool_name}' is not a valid Python identifier"

            # Create the tool
            result = self._create_tool(tool_name, tool_code, description, parameters)
            return result

        except Exception as e:
            logger.error(f"Tool creation error: {e}")
            return f"Error creating tool: {str(e)}"

    def _create_tool(self, tool_name: str, tool_code: str, description: str, parameters: Dict) -> str:
        """Create a new tool from code."""
        try:
            # Create a temporary file for the tool code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # Write the tool code with proper structure
                full_code = f"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class {tool_name.capitalize()}Tool:
    def __init__(self):
        self.description = "{description}"
        self.parameters = {parameters}

    def execute(self, tool_input: str) -> str:
        '''Execute the tool.'''
        try:
{self._indent_code(tool_code, 12)}
        except Exception as e:
            logger.error(f"Tool execution error: {{e}}")
            return f"Error: {{str(e)}}"

    async def aexecute(self, tool_input: str) -> str:
        '''Async version of execute.'''
        return self.execute(tool_input)
"""
                temp_file.write(full_code)
                temp_file_path = temp_file.name

            # Load the module
            spec = importlib.util.spec_from_file_location(f"{tool_name}_tool", temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the tool class
            tool_class = getattr(module, f"{tool_name.capitalize()}Tool")
            tool_instance = tool_class()

            # Store the created tool
            self.created_tools[tool_name] = {
                "instance": tool_instance,
                "code": tool_code,
                "description": description,
                "parameters": parameters
            }

            # Clean up temp file
            os.unlink(temp_file_path)

            return f"Successfully created tool '{tool_name}' with description: {description}"

        except Exception as e:
            logger.error(f"Error creating tool '{tool_name}': {e}")
            return f"Error creating tool '{tool_name}': {str(e)}"

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        lines = code.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():  # Only indent non-empty lines
                indented_lines.append(indent + line)
            else:
                indented_lines.append(line)
        return '\n'.join(indented_lines)

    def get_created_tools(self) -> Dict[str, Dict]:
        """Get all created tools."""
        return self.created_tools

    def get_tool_instance(self, tool_name: str) -> Any:
        """Get a specific tool instance."""
        return self.created_tools.get(tool_name, {}).get("instance")

    async def aexecute(self, tool_input: str) -> str:
        """Async version of execute."""
        return self.execute(tool_input)
