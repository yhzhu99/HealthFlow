"""
Code interpreter tool that executes Python code safely.
"""
import logging
from typing import Any, Dict
import sys
from io import StringIO
import traceback

logger = logging.getLogger(__name__)

class CodeInterpreterTool:
    """Tool for executing Python code safely."""

    def __init__(self):
        self.description = "Execute Python code and return the result"
        self.parameters = {
            "code": "Python code to execute"
        }
        # Safe namespace for code execution
        self.namespace = {
            '__builtins__': {
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'round': round, 'sorted': sorted, 'reversed': reversed
            },
            'math': None, 'numpy': None, 'pandas': None, 'torch': None
        }

        # Try to import common libraries
        try:
            import math
            import numpy as np
            import pandas as pd
            self.namespace['math'] = math
            self.namespace['np'] = np
            self.namespace['numpy'] = np
            self.namespace['pd'] = pd
            self.namespace['pandas'] = pd
        except ImportError:
            pass

        try:
            import torch
            self.namespace['torch'] = torch
        except ImportError:
            pass

    def execute(self, code_input: str) -> str:
        """Execute Python code and return the result."""
        try:
            # Parse the input - could be just code or formatted request
            if isinstance(code_input, str):
                if code_input.strip().startswith('{'):
                    # Try to parse as JSON
                    import json
                    try:
                        parsed = json.loads(code_input)
                        code = parsed.get('code', code_input)
                    except:
                        code = code_input
                else:
                    code = code_input
            else:
                code = str(code_input)

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            try:
                # Execute the code
                result = eval(code, self.namespace)
                output = captured_output.getvalue()

                # Return both output and result
                if output:
                    return f"{output}\nResult: {result}" if result is not None else output
                else:
                    return str(result) if result is not None else "Code executed successfully"

            except SyntaxError:
                # Try exec instead of eval for statements
                exec(code, self.namespace)
                output = captured_output.getvalue()
                return output if output else "Code executed successfully"

        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return f"Error: {str(e)}"

        finally:
            sys.stdout = old_stdout

    async def aexecute(self, code_input: str) -> str:
        """Async version of execute."""
        return self.execute(code_input)
