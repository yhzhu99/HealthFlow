import ast
import io
import logging
from contextlib import redirect_stdout
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HealthFlowInterpreter:
    """
    A secure and capable Python interpreter for HealthFlow agents.
    It pre-loads essential data science libraries for healthcare AI tasks.
    """
    def __init__(self):
        self.global_namespace = self._create_safe_namespace()

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Creates a namespace pre-loaded with safe, essential libraries."""
        namespace = {}

        # Pre-load libraries
        try:
            import numpy as np
            namespace['np'] = np
            import pandas as pd
            namespace['pd'] = pd
            import torch
            namespace['torch'] = torch
            import torch.nn as nn
            namespace['nn'] = nn
            from sklearn.model_selection import train_test_split
            namespace['train_test_split'] = train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            namespace['accuracy_score'] = accuracy_score
            namespace['precision_score'] = precision_score
            namespace['recall_score'] = recall_score
            namespace['f1_score'] = f1_score
            import math
            namespace['math'] = math
            import pickle
            namespace['pickle'] = pickle
            import os
            namespace['os'] = os # For file existence checks etc.

            logger.info("Interpreter namespace loaded with numpy, pandas, torch, sklearn.")
        except ImportError as e:
            logger.warning(f"Could not import a core library for the interpreter: {e}")

        return namespace

    def run(self, code: str) -> str:
        """
        Executes a string of Python code in a controlled environment.
        Args:
            code: The Python code to execute.
        Returns:
            The captured stdout and the value of the last expression, or an error message.
        """
        code = self._clean_code(code)
        if not code:
            return "Error: No code to execute."

        logger.debug(f"Executing code:\n{code}")

        # Use a local namespace for each run that inherits from the global one
        local_namespace = self.global_namespace.copy()

        output_buffer = io.StringIO()
        try:
            with redirect_stdout(output_buffer):
                # Compile the code
                compiled_code = compile(code, '<string>', 'exec')
                # Execute the code
                exec(compiled_code, local_namespace)

            # Get output from buffer
            stdout_result = output_buffer.getvalue()

            # The result of the last expression is not automatically captured by exec
            # This is a simplification; for true agentic coding, the agent should use `print()`
            # to explicitly signal what output is important.

            return f"Execution successful.\nOutput:\n{stdout_result}" if stdout_result else "Execution successful (no output)."
        except Exception as e:
            error_message = f"Execution failed.\nError: {type(e).__name__}: {e}"
            logger.error(f"Code execution failed: {error_message}")
            return error_message

    def _clean_code(self, code: str) -> str:
        """Removes markdown fences from code blocks."""
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()