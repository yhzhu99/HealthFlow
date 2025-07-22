"""
MCP (Model Context Protocol) Compatible Tool Interface for HealthFlow

Provides compatibility layer for integrating external MCP tools into the HealthFlow ecosystem.
Supports tool discovery, registration, and execution with proper error handling and security.
"""

import json
import asyncio
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..core.security import DataProtector, ProtectionConfig


class MCPToolType(Enum):
    """Types of MCP tools"""
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"
    TOOL = "tool"


class MCPExecutionMode(Enum):
    """MCP tool execution modes"""
    SUBPROCESS = "subprocess"  # Run as separate process
    INLINE = "inline"         # Run within current process
    SANDBOXED = "sandboxed"   # Run in sandboxed environment


@dataclass
class MCPToolSpec:
    """Specification for an MCP tool"""
    tool_id: str
    name: str
    description: str
    tool_type: MCPToolType
    execution_mode: MCPExecutionMode
    schema: Dict[str, Any]  # Input/output schema
    implementation: Optional[str] = None  # For inline tools
    executable_path: Optional[str] = None  # For subprocess tools
    capabilities: List[str] = None  # List of capabilities
    security_level: str = "medium"  # low, medium, high
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['tool_type'] = self.tool_type.value
        data['execution_mode'] = self.execution_mode.value
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPToolSpec':
        """Create from dictionary"""
        data = data.copy()
        data['tool_type'] = MCPToolType(data['tool_type'])
        data['execution_mode'] = MCPExecutionMode(data['execution_mode'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class MCPExecutionResult:
    """Result of MCP tool execution"""
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    security_warnings: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MCPToolManager:
    """
    Manager for MCP-compatible tools with security and extensibility features
    
    Features:
    - Tool registration and discovery
    - Secure execution with data protection
    - Schema validation
    - Performance monitoring
    - Error handling and logging
    """
    
    def __init__(
        self, 
        tools_dir: Path,
        data_protector: Optional[DataProtector] = None
    ):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Security and data protection
        self.data_protector = data_protector or DataProtector(
            config=ProtectionConfig(
                anonymization_level="medium",
                schema_only_mode=True,
                generate_mock_data=True
            )
        )
        
        # Storage paths - all JSON format
        self.tools_registry_path = self.tools_dir / "mcp_tools_registry.json"
        self.execution_log_path = self.tools_dir / "mcp_execution_log.json"
        self.performance_metrics_path = self.tools_dir / "mcp_performance_metrics.json"
        
        # In-memory stores
        self.registered_tools: Dict[str, MCPToolSpec] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Execution limits
        self.max_execution_time = 60  # seconds
        self.max_memory_usage = 512  # MB
        
        # Logger
        self.logger = logging.getLogger("MCPToolManager")
        
    async def initialize(self):
        """Initialize the MCP tool manager"""
        await self._load_registered_tools()
        await self._discover_available_tools()
        
    async def _load_registered_tools(self):
        """Load registered tools from JSON"""
        if not self.tools_registry_path.exists():
            return
            
        try:
            with open(self.tools_registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
                
            for tool_id, tool_data in registry_data.get("tools", {}).items():
                try:
                    tool_spec = MCPToolSpec.from_dict(tool_data)
                    self.registered_tools[tool_id] = tool_spec
                except Exception as e:
                    self.logger.error(f"Error loading tool {tool_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading tools registry: {e}")
    
    async def _save_registered_tools(self):
        """Save registered tools to JSON"""
        try:
            registry_data = {
                "tools": {
                    tool_id: tool_spec.to_dict() 
                    for tool_id, tool_spec in self.registered_tools.items()
                },
                "last_updated": datetime.now().isoformat(),
                "total_tools": len(self.registered_tools)
            }
            
            with open(self.tools_registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving tools registry: {e}")
    
    async def _discover_available_tools(self):
        """Auto-discover MCP tools in the tools directory"""
        
        # Look for MCP tool definition files
        for tool_file in self.tools_dir.glob("*.mcp.json"):
            try:
                with open(tool_file, 'r', encoding='utf-8') as f:
                    tool_definition = json.load(f)
                    
                # Register discovered tool
                await self._register_discovered_tool(tool_definition, tool_file)
                
            except Exception as e:
                self.logger.error(f"Error discovering tool from {tool_file}: {e}")
    
    async def _register_discovered_tool(self, tool_definition: Dict[str, Any], source_file: Path):
        """Register a discovered tool"""
        
        try:
            # Create tool spec from definition
            tool_spec = MCPToolSpec(
                tool_id=tool_definition.get("id", str(uuid.uuid4())),
                name=tool_definition["name"],
                description=tool_definition["description"],
                tool_type=MCPToolType(tool_definition.get("type", "function")),
                execution_mode=MCPExecutionMode(tool_definition.get("execution_mode", "subprocess")),
                schema=tool_definition.get("schema", {}),
                implementation=tool_definition.get("implementation"),
                executable_path=tool_definition.get("executable_path"),
                capabilities=tool_definition.get("capabilities", []),
                security_level=tool_definition.get("security_level", "medium"),
                tags=tool_definition.get("tags", []),
                metadata={
                    "source_file": str(source_file),
                    "discovered": True,
                    **tool_definition.get("metadata", {})
                }
            )
            
            await self.register_tool(tool_spec)
            self.logger.info(f"Auto-registered MCP tool: {tool_spec.name}")
            
        except Exception as e:
            self.logger.error(f"Error registering discovered tool: {e}")
    
    async def register_tool(self, tool_spec: MCPToolSpec) -> str:
        """
        Register a new MCP tool
        
        Args:
            tool_spec: Tool specification
            
        Returns:
            Tool ID
        """
        
        # Validate tool specification
        validation_result = await self._validate_tool_spec(tool_spec)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid tool specification: {validation_result['errors']}")
        
        # Security check
        security_check = await self._security_check_tool(tool_spec)
        if not security_check["approved"]:
            raise SecurityError(f"Tool failed security check: {security_check['issues']}")
        
        # Register the tool
        self.registered_tools[tool_spec.tool_id] = tool_spec
        
        # Save to persistent storage
        await self._save_registered_tools()
        
        self.logger.info(f"Registered MCP tool: {tool_spec.name} ({tool_spec.tool_id})")
        return tool_spec.tool_id
    
    async def _validate_tool_spec(self, tool_spec: MCPToolSpec) -> Dict[str, Any]:
        """Validate tool specification"""
        
        errors = []
        
        # Required fields validation
        required_fields = ["name", "description", "tool_type", "schema"]
        for field in required_fields:
            if not getattr(tool_spec, field):
                errors.append(f"Missing required field: {field}")
        
        # Schema validation
        if tool_spec.schema:
            if "input" not in tool_spec.schema:
                errors.append("Tool schema must include 'input' definition")
            if "output" not in tool_spec.schema:
                errors.append("Tool schema must include 'output' definition")
        
        # Execution mode validation
        if tool_spec.execution_mode == MCPExecutionMode.SUBPROCESS:
            if not tool_spec.executable_path:
                errors.append("Subprocess tools must specify executable_path")
        elif tool_spec.execution_mode == MCPExecutionMode.INLINE:
            if not tool_spec.implementation:
                errors.append("Inline tools must include implementation code")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _security_check_tool(self, tool_spec: MCPToolSpec) -> Dict[str, Any]:
        """Perform security check on tool"""
        
        issues = []
        
        # Check for suspicious patterns in implementation
        if tool_spec.implementation:
            suspicious_patterns = [
                "import os", "subprocess", "eval", "exec", "__import__",
                "open(", "file(", "input(", "raw_input"
            ]
            
            for pattern in suspicious_patterns:
                if pattern in tool_spec.implementation:
                    if tool_spec.security_level != "high":
                        issues.append(f"Suspicious pattern '{pattern}' requires high security level")
        
        # Check executable permissions
        if tool_spec.executable_path:
            executable_path = Path(tool_spec.executable_path)
            if executable_path.exists():
                # Basic permission check (in real implementation, would be more thorough)
                if not executable_path.is_file():
                    issues.append("Executable path must point to a file")
            else:
                issues.append("Executable path does not exist")
        
        # Security level requirements
        if tool_spec.security_level == "low" and "network" in tool_spec.capabilities:
            issues.append("Network capabilities require at least medium security level")
        
        return {
            "approved": len(issues) == 0,
            "issues": issues
        }
    
    async def execute_tool(
        self,
        tool_id: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> MCPExecutionResult:
        """
        Execute an MCP tool with security and data protection
        
        Args:
            tool_id: ID of the tool to execute
            input_data: Input data for the tool
            context: Additional execution context
            
        Returns:
            Execution result
        """
        
        if tool_id not in self.registered_tools:
            return MCPExecutionResult(
                success=False,
                result=None,
                error=f"Tool {tool_id} not found",
                execution_time=0.0,
                security_warnings=[],
                metadata={}
            )
        
        tool_spec = self.registered_tools[tool_id]
        context = context or {}
        
        start_time = datetime.now()
        security_warnings = []
        
        try:
            # Data protection for sensitive inputs
            protected_input = await self._protect_input_data(input_data, tool_spec)
            if protected_input.get("protection_applied"):
                security_warnings.append("Input data was protected for privacy")
            
            # Validate input against schema
            validation_result = await self._validate_input(
                protected_input.get("protected_data", input_data), 
                tool_spec.schema.get("input", {})
            )
            
            if not validation_result["valid"]:
                return MCPExecutionResult(
                    success=False,
                    result=None,
                    error=f"Input validation failed: {validation_result['errors']}",
                    execution_time=0.0,
                    security_warnings=security_warnings,
                    metadata={}
                )
            
            # Execute based on execution mode
            if tool_spec.execution_mode == MCPExecutionMode.INLINE:
                result = await self._execute_inline_tool(tool_spec, protected_input.get("protected_data", input_data), context)
            elif tool_spec.execution_mode == MCPExecutionMode.SUBPROCESS:
                result = await self._execute_subprocess_tool(tool_spec, protected_input.get("protected_data", input_data), context)
            elif tool_spec.execution_mode == MCPExecutionMode.SANDBOXED:
                result = await self._execute_sandboxed_tool(tool_spec, protected_input.get("protected_data", input_data), context)
            else:
                raise ValueError(f"Unsupported execution mode: {tool_spec.execution_mode}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validate output
            output_validation = await self._validate_output(result, tool_spec.schema.get("output", {}))
            if not output_validation["valid"]:
                security_warnings.append(f"Output validation issues: {output_validation['errors']}")
            
            # Create execution result
            execution_result = MCPExecutionResult(
                success=True,
                result=result,
                error=None,
                execution_time=execution_time,
                security_warnings=security_warnings,
                metadata={
                    "tool_id": tool_id,
                    "tool_name": tool_spec.name,
                    "execution_mode": tool_spec.execution_mode.value,
                    "input_protected": protected_input.get("protection_applied", False)
                }
            )
            
            # Log execution
            await self._log_execution(tool_spec, execution_result, input_data)
            
            # Update performance metrics
            await self._update_performance_metrics(tool_id, execution_time, True)
            
            return execution_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            execution_result = MCPExecutionResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                security_warnings=security_warnings,
                metadata={
                    "tool_id": tool_id,
                    "tool_name": tool_spec.name,
                    "execution_mode": tool_spec.execution_mode.value
                }
            )
            
            await self._log_execution(tool_spec, execution_result, input_data)
            await self._update_performance_metrics(tool_id, execution_time, False)
            
            return execution_result
    
    async def _protect_input_data(self, input_data: Dict[str, Any], tool_spec: MCPToolSpec) -> Dict[str, Any]:
        """Apply data protection to input data"""
        
        # Only apply protection if tool requires it or data is sensitive
        if tool_spec.security_level == "high" or any(
            keyword in str(input_data).lower() 
            for keyword in ["patient", "medical", "health", "diagnosis"]
        ):
            return await self.data_protector.protect_data(input_data)
        
        return {"protected_data": input_data, "protection_applied": False}
    
    async def _validate_input(self, input_data: Any, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against schema"""
        
        errors = []
        
        # Basic schema validation (in real implementation, would use jsonschema)
        if input_schema.get("required"):
            if not isinstance(input_data, dict):
                errors.append("Input must be an object when required fields are specified")
            else:
                for required_field in input_schema["required"]:
                    if required_field not in input_data:
                        errors.append(f"Missing required field: {required_field}")
        
        # Type validation for key fields
        if input_schema.get("properties") and isinstance(input_data, dict):
            for field, field_schema in input_schema["properties"].items():
                if field in input_data:
                    expected_type = field_schema.get("type")
                    if expected_type == "string" and not isinstance(input_data[field], str):
                        errors.append(f"Field '{field}' must be a string")
                    elif expected_type == "number" and not isinstance(input_data[field], (int, float)):
                        errors.append(f"Field '{field}' must be a number")
                    elif expected_type == "boolean" and not isinstance(input_data[field], bool):
                        errors.append(f"Field '{field}' must be a boolean")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _validate_output(self, output_data: Any, output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output data against schema"""
        
        # Similar validation logic as input validation
        return await self._validate_input(output_data, output_schema)
    
    async def _execute_inline_tool(
        self, 
        tool_spec: MCPToolSpec, 
        input_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        """Execute an inline tool"""
        
        if not tool_spec.implementation:
            raise ValueError("Inline tool must have implementation code")
        
        # Create safe execution environment
        safe_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "range": range,
                "zip": zip,
                "print": lambda *args: None,  # Disable print for security
            },
            "json": json,
            "datetime": datetime,
        }
        
        safe_locals = {
            "input_data": input_data,
            "context": context,
            "result": None
        }
        
        try:
            # Execute the tool implementation
            exec(tool_spec.implementation, safe_globals, safe_locals)
            return safe_locals.get("result")
            
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}")
    
    async def _execute_subprocess_tool(
        self, 
        tool_spec: MCPToolSpec, 
        input_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        """Execute a subprocess tool"""
        
        if not tool_spec.executable_path:
            raise ValueError("Subprocess tool must have executable_path")
        
        # Create temporary file for input data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump({
                "input_data": input_data,
                "context": context
            }, temp_file, ensure_ascii=False, indent=2)
            temp_input_path = temp_file.name
        
        try:
            # Execute subprocess with timeout
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    tool_spec.executable_path,
                    temp_input_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=self.max_execution_time
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"Tool execution failed with code {result.returncode}: {stderr.decode()}")
            
            # Parse JSON output
            try:
                return json.loads(stdout.decode())
            except json.JSONDecodeError:
                return stdout.decode()  # Return raw output if not JSON
            
        finally:
            # Clean up temporary file
            Path(temp_input_path).unlink(missing_ok=True)
    
    async def _execute_sandboxed_tool(
        self, 
        tool_spec: MCPToolSpec, 
        input_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        """Execute a tool in sandboxed environment"""
        
        # For now, use inline execution with additional restrictions
        # In a full implementation, this would use containers or other sandboxing
        return await self._execute_inline_tool(tool_spec, input_data, context)
    
    async def _log_execution(
        self, 
        tool_spec: MCPToolSpec, 
        result: MCPExecutionResult, 
        original_input: Dict[str, Any]
    ):
        """Log tool execution for monitoring and debugging"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_id": tool_spec.tool_id,
            "tool_name": tool_spec.name,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error,
            "security_warnings": result.security_warnings,
            "input_summary": str(original_input)[:200],  # Truncated for privacy
            "metadata": result.metadata
        }
        
        self.execution_history.append(log_entry)
        
        # Keep log size manageable
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
        
        # Save to persistent storage periodically
        try:
            log_data = {
                "executions": self.execution_history,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.execution_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving execution log: {e}")
    
    async def _update_performance_metrics(self, tool_id: str, execution_time: float, success: bool):
        """Update performance metrics for tool"""
        
        if tool_id not in self.performance_metrics:
            self.performance_metrics[tool_id] = []
        
        # Store execution time and success/failure
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "success": success
        }
        
        self.performance_metrics[tool_id].append(metric_entry)
        
        # Keep metrics manageable
        if len(self.performance_metrics[tool_id]) > 100:
            self.performance_metrics[tool_id] = self.performance_metrics[tool_id][-50:]
    
    async def get_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool"""
        
        if tool_id not in self.registered_tools:
            return None
        
        tool_spec = self.registered_tools[tool_id]
        
        # Calculate performance metrics
        metrics = self.performance_metrics.get(tool_id, [])
        if metrics:
            success_rate = sum(1 for m in metrics if m["success"]) / len(metrics)
            avg_execution_time = sum(m["execution_time"] for m in metrics) / len(metrics)
        else:
            success_rate = 0.0
            avg_execution_time = 0.0
        
        return {
            "tool_spec": tool_spec.to_dict(),
            "performance": {
                "total_executions": len(metrics),
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time
            },
            "recent_executions": [
                log for log in self.execution_history 
                if log.get("tool_id") == tool_id
            ][-10:]  # Last 10 executions
        }
    
    async def list_tools(
        self, 
        tool_type: Optional[MCPToolType] = None,
        security_level: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List registered tools with optional filtering"""
        
        tools = []
        
        for tool_id, tool_spec in self.registered_tools.items():
            # Apply filters
            if tool_type and tool_spec.tool_type != tool_type:
                continue
            if security_level and tool_spec.security_level != security_level:
                continue
            if tags and not any(tag in tool_spec.tags for tag in tags):
                continue
            
            # Get basic info
            tool_info = {
                "tool_id": tool_id,
                "name": tool_spec.name,
                "description": tool_spec.description,
                "tool_type": tool_spec.tool_type.value,
                "execution_mode": tool_spec.execution_mode.value,
                "security_level": tool_spec.security_level,
                "tags": tool_spec.tags,
                "capabilities": tool_spec.capabilities
            }
            
            tools.append(tool_info)
        
        return tools
    
    async def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool"""
        
        if tool_id not in self.registered_tools:
            return False
        
        del self.registered_tools[tool_id]
        await self._save_registered_tools()
        
        # Clean up metrics
        if tool_id in self.performance_metrics:
            del self.performance_metrics[tool_id]
        
        self.logger.info(f"Unregistered tool: {tool_id}")
        return True
    
    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for log in self.execution_history if log["success"])
        
        # Tool usage statistics
        tool_usage = {}
        for log in self.execution_history:
            tool_name = log.get("tool_name", "unknown")
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        # Security warnings analysis
        total_warnings = sum(len(log.get("security_warnings", [])) for log in self.execution_history)
        
        # Performance metrics
        execution_times = [log.get("execution_time", 0) for log in self.execution_history if log.get("execution_time")]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "total_registered_tools": len(self.registered_tools),
            "tool_usage_distribution": tool_usage,
            "total_security_warnings": total_warnings,
            "average_execution_time": avg_execution_time,
            "execution_time_range": {
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0
            }
        }


class SecurityError(Exception):
    """Exception raised for security-related issues"""
    pass


# Example MCP tool creation helper
def create_example_mcp_tool() -> MCPToolSpec:
    """Create an example MCP tool for demonstration"""
    
    implementation = """
# Example medical data analysis tool
import json

def analyze_patient_vitals(vitals_data):
    \"\"\"Analyze patient vital signs\"\"\"
    results = {}
    
    if 'blood_pressure' in vitals_data:
        bp = vitals_data['blood_pressure']
        if '/' in str(bp):
            systolic, diastolic = map(int, str(bp).split('/'))
            if systolic > 140 or diastolic > 90:
                results['hypertension_risk'] = 'high'
            else:
                results['hypertension_risk'] = 'normal'
    
    if 'heart_rate' in vitals_data:
        hr = vitals_data['heart_rate']
        if hr > 100:
            results['tachycardia'] = True
        elif hr < 60:
            results['bradycardia'] = True
        else:
            results['heart_rate_status'] = 'normal'
    
    return results

# Main execution
try:
    result = analyze_patient_vitals(input_data.get('vitals', {}))
    result = {
        'analysis': result,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }
except Exception as e:
    result = {
        'error': str(e),
        'status': 'failed'
    }
"""
    
    return MCPToolSpec(
        tool_id="example_vitals_analyzer",
        name="Patient Vitals Analyzer",
        description="Analyzes patient vital signs and identifies potential health concerns",
        tool_type=MCPToolType.FUNCTION,
        execution_mode=MCPExecutionMode.INLINE,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "vitals": {
                        "type": "object",
                        "properties": {
                            "blood_pressure": {"type": "string"},
                            "heart_rate": {"type": "number"},
                            "temperature": {"type": "number"},
                            "respiratory_rate": {"type": "number"}
                        }
                    }
                },
                "required": ["vitals"]
            },
            "output": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "object"},
                    "timestamp": {"type": "string"},
                    "status": {"type": "string"}
                }
            }
        },
        implementation=implementation,
        capabilities=["medical_analysis", "vitals_processing"],
        security_level="medium",
        tags=["medical", "analysis", "vitals", "healthcare"]
    )