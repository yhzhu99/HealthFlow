"""
ToolBank System for HealthFlow
Dynamic tool creation, management, and execution system
Supports MCP (Model Context Protocol) driven tool development
Uses jsonl, parquet, and pickle for persistence
"""

import json
import pickle
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import inspect
import importlib.util
import tempfile

import pandas as pd
from pydantic import BaseModel


class ToolType(Enum):
    """Types of tools in the ToolBank"""
    PYTHON_FUNCTION = "python_function"
    SHELL_COMMAND = "shell_command"
    API_CALL = "api_call"
    DATA_PROCESSOR = "data_processor"
    MEDICAL_ANALYZER = "medical_analyzer"
    CODE_GENERATOR = "code_generator"


@dataclass
class ToolMetadata:
    """Metadata for tools in the ToolBank"""
    tool_id: str
    name: str
    description: str
    tool_type: ToolType
    version: str
    author: str
    created_at: datetime
    last_used: datetime
    usage_count: int
    success_rate: float
    parameters: Dict[str, Any]
    return_type: str
    tags: List[str]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'tool_id': self.tool_id,
            'name': self.name,
            'description': self.description,
            'tool_type': self.tool_type.value,
            'version': self.version,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat(),
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'tags': self.tags,
            'dependencies': self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolMetadata':
        """Create from dictionary"""
        return cls(
            tool_id=data['tool_id'],
            name=data['name'],
            description=data['description'],
            tool_type=ToolType(data['tool_type']),
            version=data['version'],
            author=data['author'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_used=datetime.fromisoformat(data['last_used']),
            usage_count=data['usage_count'],
            success_rate=data['success_rate'],
            parameters=data['parameters'],
            return_type=data['return_type'],
            tags=data['tags'],
            dependencies=data['dependencies']
        )


class Tool(BaseModel):
    """Represents a tool in the ToolBank"""
    metadata: ToolMetadata
    implementation: str  # Source code or command
    test_cases: List[Dict[str, Any]]
    documentation: str
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'metadata': self.metadata.to_dict(),
            'implementation': self.implementation,
            'test_cases': self.test_cases,
            'documentation': self.documentation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Create from dictionary"""
        return cls(
            metadata=ToolMetadata.from_dict(data['metadata']),
            implementation=data['implementation'],
            test_cases=data['test_cases'],
            documentation=data['documentation']
        )


class ToolExecutionResult(BaseModel):
    """Result of tool execution"""
    tool_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float
    timestamp: datetime
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'tool_id': self.tool_id,
            'success': self.success,
            'output': str(self.output),  # Convert to string for serialization
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }


class ToolBank:
    """Dynamic tool creation and management system using file-based persistence"""
    
    def __init__(self, tools_dir: Path):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.tools_registry_path = self.tools_dir / "tools_registry.jsonl"
        self.tools_metadata_path = self.tools_dir / "tools_metadata.pkl"
        self.execution_history_path = self.tools_dir / "execution_history.parquet"
        self.tools_code_dir = self.tools_dir / "code"
        self.tools_tests_dir = self.tools_dir / "tests"
        self.mcps_dir = self.tools_dir / "mcps"
        
        # Create subdirectories
        self.tools_code_dir.mkdir(exist_ok=True)
        self.tools_tests_dir.mkdir(exist_ok=True)
        self.mcps_dir.mkdir(exist_ok=True)
        
        # In-memory registry
        self.tools_registry: Dict[str, Tool] = {}
        self.execution_history: List[ToolExecutionResult] = []
        
    async def initialize(self):
        """Initialize and load existing tools"""
        await self._load_tools()
    
    async def _load_tools(self):
        """Load tools from persistent storage"""
        # Load from JSONL
        if self.tools_registry_path.exists():
            try:
                with open(self.tools_registry_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            tool_data = json.loads(line)
                            tool = Tool.from_dict(tool_data)
                            self.tools_registry[tool.metadata.tool_id] = tool
            except Exception as e:
                print(f"Error loading tools from JSONL: {e}")
        
        # Load from pickle as backup
        if self.tools_metadata_path.exists() and not self.tools_registry:
            try:
                with open(self.tools_metadata_path, 'rb') as f:
                    tools_data = pickle.load(f)
                    for tool_data in tools_data:
                        tool = Tool.from_dict(tool_data)
                        self.tools_registry[tool.metadata.tool_id] = tool
            except Exception as e:
                print(f"Error loading tools from pickle: {e}")
        
        # Load execution history from parquet
        if self.execution_history_path.exists():
            try:
                df = pd.read_parquet(self.execution_history_path)
                # Convert to ToolExecutionResult objects if needed for analysis
            except Exception as e:
                print(f"Error loading execution history: {e}")
    
    async def _save_tools(self):
        """Save tools to persistent storage"""
        # Save to JSONL
        try:
            with open(self.tools_registry_path, 'w') as f:
                for tool in self.tools_registry.values():
                    f.write(json.dumps(tool.to_dict()) + '\n')
        except Exception as e:
            print(f"Error saving tools to JSONL: {e}")
        
        # Save to pickle as backup
        try:
            with open(self.tools_metadata_path, 'wb') as f:
                tools_data = [tool.to_dict() for tool in self.tools_registry.values()]
                pickle.dump(tools_data, f)
        except Exception as e:
            print(f"Error saving tools to pickle: {e}")
    
    async def _save_execution_history(self):
        """Save execution history to parquet"""
        if not self.execution_history:
            return
            
        try:
            # Convert to DataFrame
            data = [result.to_dict() for result in self.execution_history]
            df = pd.DataFrame(data)
            
            # Append to existing or create new
            if self.execution_history_path.exists():
                existing_df = pd.read_parquet(self.execution_history_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(self.execution_history_path, index=False)
            
            # Clear in-memory history to save memory
            self.execution_history = []
            
        except Exception as e:
            print(f"Error saving execution history: {e}")
    
    async def create_python_tool(
        self,
        name: str,
        description: str,
        implementation: str,
        parameters: Dict[str, Any],
        return_type: str = "Any",
        tags: List[str] = None,
        test_cases: List[Dict[str, Any]] = None,
        author: str = "HealthFlow Agent"
    ) -> str:
        """Create a new Python function tool"""
        
        tool_id = str(uuid.uuid4())
        tags = tags or []
        test_cases = test_cases or []
        
        # Create metadata
        metadata = ToolMetadata(
            tool_id=tool_id,
            name=name,
            description=description,
            tool_type=ToolType.PYTHON_FUNCTION,
            version="1.0.0",
            author=author,
            created_at=datetime.now(),
            last_used=datetime.now(),
            usage_count=0,
            success_rate=1.0,
            parameters=parameters,
            return_type=return_type,
            tags=tags,
            dependencies=self._extract_dependencies(implementation)
        )
        
        # Create tool object
        tool = Tool(
            metadata=metadata,
            implementation=implementation,
            test_cases=test_cases,
            documentation=f"# {name}\n\n{description}\n\n## Parameters\n{json.dumps(parameters, indent=2)}"
        )
        
        # Validate the implementation
        if await self._validate_python_tool(tool):
            self.tools_registry[tool_id] = tool
            
            # Save implementation to file
            tool_file = self.tools_code_dir / f"{tool_id}.py"
            with open(tool_file, 'w') as f:
                f.write(implementation)
            
            # Generate and save test file
            test_code = await self._generate_test_code(tool)
            test_file = self.tools_tests_dir / f"test_{tool_id}.py"
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            await self._save_tools()
            return tool_id
        else:
            raise ValueError(f"Tool validation failed for: {name}")
    
    async def create_medical_analyzer_tool(
        self,
        name: str,
        medical_domain: str,
        analysis_type: str,
        implementation: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any]
    ) -> str:
        """Create a specialized medical analysis tool"""
        
        return await self.create_python_tool(
            name=name,
            description=f"Medical analyzer for {medical_domain}: {analysis_type}",
            implementation=implementation,
            parameters={
                "input_schema": input_schema,
                "output_schema": output_schema,
                "medical_domain": medical_domain,
                "analysis_type": analysis_type
            },
            return_type="Dict[str, Any]",
            tags=["medical", "analyzer", medical_domain, analysis_type]
        )
    
    async def create_code_generator_tool(
        self,
        name: str,
        target_language: str,
        code_template: str,
        generation_logic: str
    ) -> str:
        """Create a code generation tool"""
        
        implementation = f"""
import string
import json

def generate_code(template_vars: dict, target_language: str = "{target_language}") -> str:
    \"\"\"
    Generate code using template and variables
    \"\"\"
    template = '''
{code_template}
    '''
    
    # Custom generation logic
{generation_logic}
    
    # Apply template variables
    return string.Template(template).safe_substitute(template_vars)

def main(**kwargs):
    \"\"\"Main entry point for the tool\"\"\"
    return generate_code(kwargs.get('template_vars', {{}}), kwargs.get('target_language', '{target_language}'))
"""
        
        return await self.create_python_tool(
            name=name,
            description=f"Code generator for {target_language}",
            implementation=implementation,
            parameters={
                "template_vars": "Dict[str, Any]",
                "target_language": target_language
            },
            return_type="str",
            tags=["code_generator", target_language, "automation"]
        )
    
    async def _validate_python_tool(self, tool: Tool) -> bool:
        """Validate Python tool implementation"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(tool.implementation)
                temp_path = f.name
            
            # Try to compile the code
            with open(temp_path, 'r') as f:
                code = f.read()
            
            compile(code, temp_path, 'exec')
            
            # Clean up
            Path(temp_path).unlink()
            
            return True
        except Exception as e:
            print(f"Tool validation error: {e}")
            return False
    
    async def _generate_test_code(self, tool: Tool) -> str:
        """Generate basic test code for a tool"""
        return f'''
import pytest
import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

# Import the tool module
import {tool.metadata.tool_id} as tool_module

def test_{tool.metadata.name.lower().replace(" ", "_")}_basic():
    """Basic test for {tool.metadata.name}"""
    # Test that the main function exists and is callable
    assert hasattr(tool_module, 'main')
    assert callable(tool_module.main)

def test_{tool.metadata.name.lower().replace(" ", "_")}_with_empty_input():
    """Test {tool.metadata.name} with empty input"""
    try:
        result = tool_module.main()
        assert result is not None
    except Exception as e:
        # It's ok if the tool requires specific inputs
        assert "required" in str(e).lower() or "missing" in str(e).lower()

# Add more specific tests based on tool functionality
'''
    
    def _extract_dependencies(self, implementation: str) -> List[str]:
        """Extract dependencies from implementation code"""
        dependencies = []
        
        for line in implementation.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                dep = line.replace('import ', '').split()[0].split('.')[0]
                dependencies.append(dep)
            elif line.startswith('from '):
                dep = line.split()[1].split('.')[0]
                dependencies.append(dep)
        
        return list(set(dependencies))
    
    async def execute_tool(
        self,
        tool_id: str,
        inputs: Dict[str, Any],
        timeout: int = 30
    ) -> ToolExecutionResult:
        """Execute a tool with given inputs"""
        
        if tool_id not in self.tools_registry:
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                output=None,
                error="Tool not found in registry",
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        tool = self.tools_registry[tool_id]
        start_time = datetime.now()
        
        try:
            if tool.metadata.tool_type == ToolType.PYTHON_FUNCTION:
                result = await self._execute_python_tool(tool, inputs, timeout)
            elif tool.metadata.tool_type == ToolType.SHELL_COMMAND:
                result = await self._execute_shell_tool(tool, inputs, timeout)
            else:
                raise NotImplementedError(f"Tool type {tool.metadata.tool_type} not implemented")
            
            # Update usage statistics
            tool.metadata.usage_count += 1
            tool.metadata.last_used = datetime.now()
            
            if result.success:
                # Update success rate using exponential moving average
                alpha = 0.1
                tool.metadata.success_rate = (1 - alpha) * tool.metadata.success_rate + alpha * 1.0
            else:
                tool.metadata.success_rate = (1 - alpha) * tool.metadata.success_rate + alpha * 0.0
            
            # Add to execution history
            self.execution_history.append(result)
            
            # Save periodically
            if len(self.execution_history) > 100:
                await self._save_execution_history()
            
            await self._save_tools()
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            self.execution_history.append(result)
            return result
    
    async def _execute_python_tool(
        self,
        tool: Tool,
        inputs: Dict[str, Any],
        timeout: int
    ) -> ToolExecutionResult:
        """Execute Python function tool"""
        
        start_time = datetime.now()
        
        try:
            # Load the tool code
            tool_file = self.tools_code_dir / f"{tool.metadata.tool_id}.py"
            
            spec = importlib.util.spec_from_file_location("tool_module", tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the main function (first defined function or 'main' function)
            main_function = None
            if hasattr(module, 'main'):
                main_function = module.main
            else:
                functions = [obj for name, obj in inspect.getmembers(module) 
                            if inspect.isfunction(obj) and obj.__module__ == module.__name__]
                if functions:
                    main_function = functions[0]
            
            if not main_function:
                raise ValueError("No function found in tool implementation")
            
            # Execute with timeout
            try:
                if asyncio.iscoroutinefunction(main_function):
                    output = await asyncio.wait_for(main_function(**inputs), timeout=timeout)
                else:
                    output = main_function(**inputs)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ToolExecutionResult(
                    tool_id=tool.metadata.tool_id,
                    success=True,
                    output=output,
                    error=None,
                    execution_time=execution_time,
                    timestamp=datetime.now()
                )
                
            except asyncio.TimeoutError:
                raise TimeoutError(f"Tool execution timeout after {timeout} seconds")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolExecutionResult(
                tool_id=tool.metadata.tool_id,
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _execute_shell_tool(
        self,
        tool: Tool,
        inputs: Dict[str, Any],
        timeout: int
    ) -> ToolExecutionResult:
        """Execute shell command tool"""
        
        start_time = datetime.now()
        
        try:
            # Format command with inputs
            command = tool.implementation.format(**inputs)
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            success = process.returncode == 0
            output = stdout.decode() if success else stderr.decode()
            
            return ToolExecutionResult(
                tool_id=tool.metadata.tool_id,
                success=success,
                output=output,
                error=stderr.decode() if not success else None,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolExecutionResult(
                tool_id=tool.metadata.tool_id,
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def search_tools(
        self,
        query: str = None,
        tool_type: ToolType = None,
        tags: List[str] = None,
        min_success_rate: float = 0.0
    ) -> List[Tool]:
        """Search for tools in the registry"""
        
        results = []
        
        for tool in self.tools_registry.values():
            # Filter by query
            if query and query.lower() not in tool.metadata.name.lower() and \
               query.lower() not in tool.metadata.description.lower():
                continue
            
            # Filter by tool type
            if tool_type and tool.metadata.tool_type != tool_type:
                continue
            
            # Filter by tags
            if tags and not any(tag in tool.metadata.tags for tag in tags):
                continue
            
            # Filter by success rate
            if tool.metadata.success_rate < min_success_rate:
                continue
            
            results.append(tool)
        
        # Sort by success rate and usage count
        results.sort(
            key=lambda t: (t.metadata.success_rate, t.metadata.usage_count),
            reverse=True
        )
        
        return results
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get ToolBank statistics"""
        
        if not self.tools_registry:
            return {}
        
        total_tools = len(self.tools_registry)
        
        # Count by type
        type_counts = {}
        for tool in self.tools_registry.values():
            tool_type = tool.metadata.tool_type.value
            type_counts[tool_type] = type_counts.get(tool_type, 0) + 1
        
        # Calculate average success rate
        avg_success_rate = sum(t.metadata.success_rate for t in self.tools_registry.values()) / total_tools
        
        # Most used tools
        most_used = sorted(
            self.tools_registry.values(),
            key=lambda t: t.metadata.usage_count,
            reverse=True
        )[:5]
        
        return {
            "total_tools": total_tools,
            "tool_type_distribution": type_counts,
            "average_success_rate": avg_success_rate,
            "most_used_tools": [
                {
                    "name": tool.metadata.name,
                    "usage_count": tool.metadata.usage_count,
                    "success_rate": tool.metadata.success_rate
                }
                for tool in most_used
            ],
            "total_executions": len(self.execution_history)
        }
    
    async def backup_tools(self, backup_path: Path):
        """Backup all tools to specified path"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save registry as both JSONL and pickle
        with open(backup_path / "tools_registry.jsonl", 'w') as f:
            for tool in self.tools_registry.values():
                f.write(json.dumps(tool.to_dict()) + '\n')
        
        with open(backup_path / "tools_metadata.pkl", 'wb') as f:
            tools_data = [tool.to_dict() for tool in self.tools_registry.values()]
            pickle.dump(tools_data, f)
        
        # Save execution history
        if self.execution_history:
            await self._save_execution_history()
        
        # Copy execution history parquet
        if self.execution_history_path.exists():
            import shutil
            shutil.copy2(self.execution_history_path, backup_path / "execution_history.parquet")
        
        # Copy code and test files
        import shutil
        if self.tools_code_dir.exists():
            shutil.copytree(self.tools_code_dir, backup_path / "code", dirs_exist_ok=True)
        if self.tools_tests_dir.exists():
            shutil.copytree(self.tools_tests_dir, backup_path / "tests", dirs_exist_ok=True)
    
    async def cleanup_unused_tools(self, days_unused: int = 30):
        """Remove tools that haven't been used for specified days"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_unused)
        
        tools_to_remove = []
        for tool_id, tool in self.tools_registry.items():
            if tool.metadata.last_used < cutoff_date and tool.metadata.usage_count == 0:
                tools_to_remove.append(tool_id)
        
        for tool_id in tools_to_remove:
            tool = self.tools_registry[tool_id]
            
            # Remove from registry
            del self.tools_registry[tool_id]
            
            # Remove code files
            tool_file = self.tools_code_dir / f"{tool_id}.py"
            if tool_file.exists():
                tool_file.unlink()
            
            test_file = self.tools_tests_dir / f"test_{tool_id}.py"
            if test_file.exists():
                test_file.unlink()
        
        await self._save_tools()
        return len(tools_to_remove)