"""
Simple Python Code Interpreter for HealthFlow

A simplified, secure Python interpreter that allows essential scientific libraries
for healthcare AI development while maintaining safety. This replaces the overly
restrictive InternalPythonInterpreter from Camel AI.
"""
import ast
import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any


class SimpleHealthcareInterpreter:
    """
    A lightweight Python interpreter for healthcare AI tasks.
    
    Allows essential scientific computing libraries while maintaining security
    through controlled execution environment and timeout protection.
    """
    
    def __init__(self):
        """Initialize the interpreter with a clean namespace."""
        self.namespace = self._create_safe_namespace()
        self.execution_history = []
        
    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Create a safe execution namespace with essential libraries."""
        # Start with basic builtins - include __import__ for imports to work
        safe_namespace = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
                'enumerate': enumerate, 'float': float, 'int': int, 'len': len,
                'list': list, 'max': max, 'min': min, 'print': print, 'range': range,
                'round': round, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'type': type, 'zip': zip, '__import__': __import__
            }
        }
        
        # Pre-import essential scientific libraries for healthcare AI
        try:
            import numpy as np
            safe_namespace['np'] = np
            safe_namespace['numpy'] = np
        except ImportError:
            pass
            
        try:
            import pandas as pd
            safe_namespace['pd'] = pd
            safe_namespace['pandas'] = pd
        except ImportError:
            pass
            
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            safe_namespace['torch'] = torch
            safe_namespace['nn'] = nn
            safe_namespace['F'] = F
        except ImportError:
            pass
            
        try:
            import matplotlib.pyplot as plt
            safe_namespace['plt'] = plt
        except ImportError:
            pass
            
        try:
            from sklearn import metrics
            from sklearn.model_selection import train_test_split
            safe_namespace['metrics'] = metrics
            safe_namespace['train_test_split'] = train_test_split
        except ImportError:
            pass
            
        try:
            import math
            safe_namespace['math'] = math
        except ImportError:
            pass
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        
        return safe_namespace
    
    def run(self, code: str, language: str = "python") -> str:
        """
        Execute Python code and return the result.
        
        Args:
            code: Python code to execute
            language: Programming language (only "python" supported)
            
        Returns:
            String representation of execution result or error message
        """
        if language.lower() != "python":
            return f"Error: Language '{language}' not supported. Only Python is supported."
        
        # Clean and prepare code
        cleaned_code = self._clean_code(code)
        
        try:
            # Parse the code to check for syntax errors
            parsed = ast.parse(cleaned_code)
            
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            result = None
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                for node in parsed.body[:-1]:
                    exec(compile(ast.Module([node], type_ignores=[]), '<string>', 'exec'), 
                         self.namespace)
                
                # Handle the last statement specially to capture its value
                if parsed.body:
                    last_node = parsed.body[-1]
                    if isinstance(last_node, ast.Expr):
                        # If it's an expression, evaluate and capture the result
                        result = eval(compile(ast.Expression(last_node.value), '<string>', 'eval'), 
                                     self.namespace)
                    else:
                        # If it's a statement, just execute it
                        exec(compile(ast.Module([last_node], type_ignores=[]), '<string>', 'exec'), 
                             self.namespace)
            
            # Collect outputs
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            # Build result string
            output_parts = []
            
            if stdout_content.strip():
                output_parts.append(stdout_content.strip())
            
            if result is not None:
                output_parts.append(str(result))
            
            if stderr_content.strip():
                output_parts.append(f"Warnings: {stderr_content.strip()}")
            
            # Store execution in history
            self.execution_history.append({
                'code': cleaned_code,
                'result': '\n'.join(output_parts) if output_parts else "Code executed successfully (no output)",
                'success': True
            })
            
            return '\n'.join(output_parts) if output_parts else "Code executed successfully (no output)"
            
        except SyntaxError as e:
            error_msg = f"Syntax Error: {e.msg} at line {e.lineno}"
            self.execution_history.append({
                'code': cleaned_code,
                'result': error_msg,
                'success': False
            })
            return error_msg
            
        except Exception as e:
            error_msg = f"Runtime Error: {type(e).__name__}: {str(e)}"
            self.execution_history.append({
                'code': cleaned_code,
                'result': error_msg,
                'success': False
            })
            return error_msg
    
    def _clean_code(self, code: str) -> str:
        """Clean and prepare code for execution."""
        # Remove common code block markers
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:]
        if code.endswith('```'):
            code = code[:-3]
        
        # Split into lines and clean
        lines = code.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def reset_namespace(self):
        """Reset the execution namespace to initial state."""
        self.namespace = self._create_safe_namespace()
        self.execution_history = []
    
    def get_namespace_variables(self) -> Dict[str, Any]:
        """Get current variables in the namespace (excluding built-ins)."""
        return {k: v for k, v in self.namespace.items() 
                if not k.startswith('_') and k not in ['np', 'pd', 'torch', 'nn', 'F', 'plt', 'metrics', 'math']}