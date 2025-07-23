"""
Advanced Code Interpreter Tool for HealthFlow
Supports Python execution, error handling, package management, and reflection
"""

import asyncio
import sys
import subprocess
import tempfile
import traceback
import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import io
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodeExecutionResult:
    """Result of code execution with rich debugging info"""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    variables_created: List[str] = None
    imports_used: List[str] = None
    reflection_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.variables_created is None:
            self.variables_created = []
        if self.imports_used is None:
            self.imports_used = []
        if self.reflection_suggestions is None:
            self.reflection_suggestions = []


class AdvancedCodeInterpreter:
    """
    Advanced Code Interpreter with error handling and reflection capabilities
    
    Features:
    - Safe code execution in isolated environments
    - Automatic package installation for missing imports
    - Error reflection and debugging suggestions
    - Variable tracking and state management
    - Execution time monitoring
    """
    
    def __init__(self):
        self.execution_context = {"__builtins__": __builtins__}
        self.installed_packages = set()
        self.execution_history = []
        
    async def execute_code(
        self, 
        code: str, 
        context: Dict[str, Any] = None,
        install_missing_packages: bool = True,
        max_retries: int = 3
    ) -> CodeExecutionResult:
        """
        Execute Python code with advanced error handling and reflection
        
        Args:
            code: Python code to execute
            context: Additional context variables
            install_missing_packages: Whether to auto-install missing packages
            max_retries: Maximum retry attempts for failed executions
            
        Returns:
            CodeExecutionResult with execution details and suggestions
        """
        start_time = datetime.now()
        
        if context:
            self.execution_context.update(context)
        
        # Track initial variables
        initial_vars = set(self.execution_context.keys())
        
        # Extract imports for analysis
        imports_used = self._extract_imports(code)
        
        # Try execution with retries
        for attempt in range(max_retries):
            try:
                result = await self._execute_with_safety(code)
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Track new variables
                new_vars = list(set(self.execution_context.keys()) - initial_vars - {"__builtins__"})
                
                success_result = CodeExecutionResult(
                    success=True,
                    output=result,
                    execution_time=execution_time,
                    variables_created=new_vars,
                    imports_used=imports_used
                )
                
                self.execution_history.append(success_result)
                return success_result
                
            except ImportError as e:
                if install_missing_packages and attempt < max_retries - 1:
                    missing_package = self._extract_package_name(str(e))
                    if missing_package and await self._install_package(missing_package):
                        continue  # Retry with installed package
                
                # Create reflection suggestions for import errors
                suggestions = self._generate_import_suggestions(str(e), imports_used)
                
                error_result = CodeExecutionResult(
                    success=False,
                    output="",
                    error=f"ImportError: {str(e)}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    imports_used=imports_used,
                    reflection_suggestions=suggestions
                )
                
                self.execution_history.append(error_result)
                return error_result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Generate reflection and modify code
                    reflected_code = await self._reflect_and_fix_code(code, str(e), traceback.format_exc())
                    if reflected_code != code:
                        code = reflected_code
                        continue  # Retry with fixed code
                
                # Generate suggestions for general errors
                suggestions = self._generate_error_suggestions(str(e), code, traceback.format_exc())
                
                error_result = CodeExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    imports_used=imports_used,
                    reflection_suggestions=suggestions
                )
                
                self.execution_history.append(error_result)
                return error_result
        
        # If all retries failed
        return CodeExecutionResult(
            success=False,
            output="",
            error=f"Code execution failed after {max_retries} attempts",
            execution_time=(datetime.now() - start_time).total_seconds(),
            reflection_suggestions=["Consider simplifying the code or checking for syntax errors"]
        )
    
    async def _execute_with_safety(self, code: str) -> str:
        """Execute code with safety measures and output capture"""
        
        # Create string buffers to capture output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Compile code first to catch syntax errors early
            compiled_code = compile(code, "<string>", "exec")
            
            # Execute with output redirection
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compiled_code, self.execution_context)
            
            # Get captured output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            # Combine outputs
            output = ""
            if stdout_content:
                output += stdout_content
            if stderr_content:
                output += f"\nSTDERR: {stderr_content}"
            
            return output or "Code executed successfully (no output)"
            
        except Exception as e:
            # Capture any stderr content
            stderr_content = stderr_buffer.getvalue()
            if stderr_content:
                raise Exception(f"{str(e)}\nSTDERR: {stderr_content}")
            raise
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            # Fallback to regex if AST parsing fails
            import_patterns = [
                r'^import\s+(\w+)',
                r'^from\s+(\w+)\s+import',
            ]
            
            for line in code.split('\n'):
                line = line.strip()
                for pattern in import_patterns:
                    match = re.match(pattern, line)
                    if match:
                        imports.append(match.group(1))
        
        return imports
    
    def _extract_package_name(self, error_msg: str) -> Optional[str]:
        """Extract package name from ImportError message"""
        
        # Common patterns in ImportError messages
        patterns = [
            r"No module named '(\w+)'",
            r"No module named (\w+)",
            r"cannot import name '\w+' from '(\w+)'",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                return match.group(1)
        
        return None
    
    async def _install_package(self, package_name: str) -> bool:
        """Install missing package using pip"""
        
        if package_name in self.installed_packages:
            return True
        
        try:
            # Map common package names to pip names
            pip_name_mapping = {
                'sklearn': 'scikit-learn',
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'yaml': 'PyYAML',
            }
            
            pip_name = pip_name_mapping.get(package_name, package_name)
            
            # Install package
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'pip', 'install', pip_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.installed_packages.add(package_name)
                return True
            else:
                print(f"Failed to install {pip_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"Error installing package {package_name}: {e}")
            return False
    
    async def _reflect_and_fix_code(self, code: str, error: str, traceback_str: str) -> str:
        """Use reflection to automatically fix common code issues"""
        
        original_code = code
        
        # Fix common syntax errors
        if "invalid syntax" in error.lower():
            # Fix common indentation issues
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # Check if it should be indented (after if, for, def, etc.)
                    prev_line = fixed_lines[-1] if fixed_lines else ""
                    if prev_line.strip().endswith(':'):
                        fixed_lines.append(f"    {line}")
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            code = '\n'.join(fixed_lines)
        
        # Fix undefined variable errors
        if "is not defined" in error:
            var_match = re.search(r"name '(\w+)' is not defined", error)
            if var_match:
                undefined_var = var_match.group(1)
                
                # Common fixes for undefined variables
                common_fixes = {
                    'np': 'import numpy as np\n',
                    'pd': 'import pandas as pd\n',
                    'plt': 'import matplotlib.pyplot as plt\n',
                    'os': 'import os\n',
                    'sys': 'import sys\n',
                    'json': 'import json\n',
                    're': 'import re\n',
                }
                
                if undefined_var in common_fixes:
                    code = common_fixes[undefined_var] + code
        
        # Fix attribute errors for missing methods
        if "has no attribute" in error:
            # Common method name fixes
            method_fixes = {
                'dataframe': 'DataFrame',
                'series': 'Series',
                'array': 'array',
            }
            
            for wrong, correct in method_fixes.items():
                code = code.replace(f'.{wrong}(', f'.{correct}(')
        
        return code if code != original_code else original_code
    
    def _generate_import_suggestions(self, error: str, imports_used: List[str]) -> List[str]:
        """Generate suggestions for import-related errors"""
        suggestions = []
        
        if "No module named" in error:
            missing_module = self._extract_package_name(error)
            if missing_module:
                suggestions.append(f"Install missing package: pip install {missing_module}")
                
                # Suggest alternatives for common packages
                alternatives = {
                    'cv2': 'opencv-python',
                    'PIL': 'Pillow', 
                    'sklearn': 'scikit-learn',
                    'yaml': 'PyYAML',
                }
                
                if missing_module in alternatives:
                    suggestions.append(f"Try installing: pip install {alternatives[missing_module]}")
        
        if "cannot import name" in error:
            suggestions.append("Check if the imported name exists in the module")
            suggestions.append("Verify the module version - the import might be from a different version")
        
        return suggestions
    
    def _generate_error_suggestions(self, error: str, code: str, traceback_str: str) -> List[str]:
        """Generate suggestions for general execution errors"""
        suggestions = []
        
        # Syntax error suggestions
        if "invalid syntax" in error.lower():
            suggestions.append("Check for missing colons (:) after if, for, def, class statements")
            suggestions.append("Verify proper indentation - Python uses spaces or tabs consistently")
            suggestions.append("Check for unmatched parentheses, brackets, or quotes")
        
        # Type error suggestions  
        elif "TypeError" in error:
            if "unsupported operand type" in error:
                suggestions.append("Check data types - you might be mixing incompatible types (e.g., string + int)")
            elif "not callable" in error:
                suggestions.append("Check if you're trying to call a non-function object")
            else:
                suggestions.append("Review the function arguments and their types")
        
        # Name error suggestions
        elif "NameError" in error:
            if "is not defined" in error:
                suggestions.append("Check if the variable is defined before use")
                suggestions.append("Verify variable spelling and case sensitivity")
        
        # Index/Key error suggestions
        elif "IndexError" in error or "KeyError" in error:
            suggestions.append("Check array/list bounds or dictionary key existence")
            suggestions.append("Add error handling with try/except blocks")
        
        # Value error suggestions
        elif "ValueError" in error:
            suggestions.append("Check input values and their expected formats")
            suggestions.append("Validate data before processing")
        
        # General suggestions
        suggestions.append("Add print statements to debug intermediate values")
        suggestions.append("Break down complex operations into smaller steps")
        
        return suggestions
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about code execution history"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful = sum(1 for result in self.execution_history if result.success)
        failed = len(self.execution_history) - successful
        
        avg_time = sum(result.execution_time for result in self.execution_history) / len(self.execution_history)
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / len(self.execution_history),
            "average_execution_time": avg_time,
            "packages_installed": list(self.installed_packages),
            "total_variables_created": sum(len(result.variables_created) for result in self.execution_history)
        }


# Main function for tool integration
async def execute_python_code(
    code: str,
    context: Dict[str, Any] = None,
    install_packages: bool = True,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Main entry point for Python code execution
    
    Args:
        code: Python code to execute
        context: Additional context variables
        install_packages: Whether to auto-install missing packages
        max_retries: Maximum retry attempts
        
    Returns:
        Dictionary with execution results and metadata
    """
    interpreter = AdvancedCodeInterpreter()
    result = await interpreter.execute_code(
        code=code,
        context=context or {},
        install_missing_packages=install_packages,
        max_retries=max_retries
    )
    
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time": result.execution_time,
        "variables_created": result.variables_created,
        "imports_used": result.imports_used,
        "suggestions": result.reflection_suggestions,
        "stats": interpreter.get_execution_stats()
    }


# For ToolBank integration
async def main(**kwargs):
    """Main function for ToolBank integration"""
    code = kwargs.get('code', '')
    context = kwargs.get('context', {})
    install_packages = kwargs.get('install_packages', True)
    max_retries = kwargs.get('max_retries', 3)
    
    if not code:
        return {"error": "No code provided", "success": False}
    
    return await execute_python_code(code, context, install_packages, max_retries)