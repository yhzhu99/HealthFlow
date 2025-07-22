"""
ToolBank: Dynamic Tool Creation and Management System

This module implements the core tool creation and management capabilities:
- MCP-driven tool generation
- Automatic tool discovery from requirements
- Code generation for specialized healthcare tools
- Tool validation and testing
- Tool registry and versioning
- Reusable component library
"""

import json
import sqlite3
import hashlib
import subprocess
import tempfile
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import ast
import inspect

from openai import AsyncOpenAI


@dataclass
class ToolSpec:
    """Specification for a tool to be created"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any] 
    task_type: str
    domain: str
    complexity: str  # simple, medium, complex
    dependencies: List[str]
    examples: List[Dict[str, Any]]


@dataclass
class ToolInfo:
    """Information about a created tool"""
    tool_id: str
    name: str
    description: str
    version: str
    code: str
    test_code: str
    dependencies: List[str]
    performance_metrics: Dict[str, float]
    usage_count: int
    success_rate: float
    created_at: datetime
    last_used: datetime
    creator_agent: str


@dataclass
class MCPDefinition:
    """Model Context Protocol definition for a tool"""
    name: str
    version: str
    description: str
    tools: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    prompts: List[Dict[str, Any]]
    server_info: Dict[str, Any]


class ToolBank:
    """
    Dynamic tool creation and management system.
    
    Features:
    - Automatic tool generation from specifications
    - MCP creation and management
    - Tool validation and testing
    - Performance tracking
    - Version control for tools
    - Dependency management
    """
    
    def __init__(self, openai_api_key: str, toolbank_dir: str = "toolbank"):
        self.openai_api_key = openai_api_key
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        self.toolbank_dir = Path(toolbank_dir)
        self.toolbank_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.toolbank_dir / "toolbank.db"
        self._init_database()
        
        # Tool storage
        self.tools_dir = self.toolbank_dir / "tools"
        self.tools_dir.mkdir(exist_ok=True)
        
        self.mcps_dir = self.toolbank_dir / "mcps"
        self.mcps_dir.mkdir(exist_ok=True)
        
        self.tests_dir = self.toolbank_dir / "tests"
        self.tests_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("ToolBank")
        self.logger.info("ToolBank initialized")
        
        # Load existing tools
        self.tools: Dict[str, ToolInfo] = {}
        self._load_existing_tools()
    
    def _init_database(self):
        """Initialize SQLite database for tool storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    tool_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    version TEXT NOT NULL,
                    code TEXT NOT NULL,
                    test_code TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    creator_agent TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_msg TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (tool_id) REFERENCES tools (tool_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mcps (
                    mcp_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    version TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    tools TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_name ON tools (name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_domain ON tools (creator_agent)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_tool ON tool_executions (tool_id)")
    
    def _load_existing_tools(self):
        """Load existing tools from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT tool_id, name, description, version, code, test_code,
                       dependencies, performance_metrics, usage_count, success_rate,
                       created_at, last_used, creator_agent
                FROM tools
            """)
            
            for row in cursor.fetchall():
                tool_info = ToolInfo(
                    tool_id=row[0],
                    name=row[1],
                    description=row[2],
                    version=row[3],
                    code=row[4],
                    test_code=row[5],
                    dependencies=json.loads(row[6]),
                    performance_metrics=json.loads(row[7]),
                    usage_count=row[8],
                    success_rate=row[9],
                    created_at=datetime.fromisoformat(row[10]),
                    last_used=datetime.fromisoformat(row[11]),
                    creator_agent=row[12]
                )
                self.tools[tool_info.name] = tool_info
        
        self.logger.info(f"Loaded {len(self.tools)} existing tools")
    
    async def identify_required_tools(
        self,
        task_description: str,
        task_type: str,
        experience_insights: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify tools required for a task"""
        
        # Check existing tools first
        existing_tools = await self._find_existing_tools(task_description, task_type)
        
        # Use LLM to identify additional tools needed
        required_tools = await self._llm_identify_tools(
            task_description, task_type, experience_insights, existing_tools
        )
        
        return required_tools
    
    async def _find_existing_tools(
        self,
        task_description: str,
        task_type: str
    ) -> List[Dict[str, Any]]:
        """Find existing tools that match the task requirements"""
        
        existing_tools = []
        task_keywords = self._extract_keywords(task_description + " " + task_type)
        
        for tool_name, tool_info in self.tools.items():
            tool_keywords = self._extract_keywords(tool_info.description)
            
            # Simple keyword matching (could be improved with embeddings)
            overlap = len(set(task_keywords) & set(tool_keywords))
            if overlap > 0:
                existing_tools.append({
                    "name": tool_name,
                    "description": tool_info.description,
                    "relevance_score": overlap / len(task_keywords),
                    "existing": True,
                    "usage_count": tool_info.usage_count,
                    "success_rate": tool_info.success_rate
                })
        
        # Sort by relevance
        existing_tools.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return existing_tools[:5]  # Return top 5 most relevant
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        # Simple keyword extraction (could be improved with NLP)
        words = text.lower().replace(',', ' ').replace('.', ' ').split()
        
        # Healthcare-specific keywords
        healthcare_keywords = {
            'diagnosis', 'treatment', 'patient', 'medical', 'clinical', 'health',
            'disease', 'symptom', 'medication', 'therapy', 'lab', 'test',
            'analysis', 'prediction', 'classification', 'detection', 'screening'
        }
        
        # Filter for healthcare keywords and longer words
        keywords = [
            word for word in words 
            if len(word) > 3 and (word in healthcare_keywords or len(word) > 6)
        ]
        
        return list(set(keywords))  # Remove duplicates
    
    async def _llm_identify_tools(
        self,
        task_description: str,
        task_type: str,
        experience_insights: List[str],
        existing_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to identify required tools"""
        
        system_prompt = """
        You are a healthcare AI tool analyst. Given a task description, identify the specific tools needed.
        
        For each tool, provide:
        - name: Short, descriptive name
        - description: What the tool does
        - complexity: simple/medium/complex
        - input_schema: Expected input format
        - output_schema: Expected output format
        - dependencies: Required libraries/packages
        
        Focus on healthcare-specific tools like:
        - Medical data analyzers
        - Diagnostic assistants
        - Drug interaction checkers
        - Clinical decision support
        - EHR processors
        - Medical literature searchers
        
        Return a JSON list of tool specifications.
        """
        
        user_prompt = f"""
        Task: {task_description}
        Task Type: {task_type}
        
        Experience Insights:
        {chr(10).join(experience_insights)}
        
        Existing Available Tools:
        {json.dumps([{'name': t['name'], 'description': t['description']} for t in existing_tools], indent=2)}
        
        Identify 1-3 specific tools needed for this task that are not already available.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                tools_data = json.loads(content)
                return tools_data
            except json.JSONDecodeError:
                # Extract JSON from response if wrapped in text
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    tools_data = json.loads(json_match.group())
                    return tools_data
                else:
                    self.logger.warning("Could not parse LLM response for tool identification")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error in LLM tool identification: {e}")
            return []
    
    async def create_tool(self, tool_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new tool from specification"""
        
        tool_name = tool_spec['name']
        
        self.logger.info(f"Creating tool: {tool_name}")
        
        # Generate tool code
        tool_code = await self._generate_tool_code(tool_spec)
        
        # Generate test code
        test_code = await self._generate_test_code(tool_spec, tool_code)
        
        # Validate the generated code
        is_valid, validation_errors = await self._validate_tool_code(tool_code, test_code)
        
        if not is_valid:
            self.logger.error(f"Tool validation failed: {validation_errors}")
            return {
                "success": False,
                "error": "Tool validation failed",
                "details": validation_errors
            }
        
        # Create tool info
        tool_id = hashlib.md5(
            f"{tool_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        tool_info = ToolInfo(
            tool_id=tool_id,
            name=tool_name,
            description=tool_spec['description'],
            version="1.0.0",
            code=tool_code,
            test_code=test_code,
            dependencies=tool_spec.get('dependencies', []),
            performance_metrics={},
            usage_count=0,
            success_rate=1.0,
            created_at=datetime.now(),
            last_used=datetime.now(),
            creator_agent="system"  # Will be updated with actual agent ID
        )
        
        # Store in database
        await self._store_tool(tool_info)
        
        # Add to in-memory tools
        self.tools[tool_name] = tool_info
        
        # Save code files
        tool_file = self.tools_dir / f"{tool_name}.py"
        with open(tool_file, 'w') as f:
            f.write(tool_code)
        
        test_file = self.tests_dir / f"test_{tool_name}.py"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        self.logger.info(f"Successfully created tool: {tool_name}")
        
        return {
            "success": True,
            "tool_id": tool_id,
            "name": tool_name,
            "description": tool_info.description,
            "file_path": str(tool_file)
        }
    
    async def _generate_tool_code(self, tool_spec: Dict[str, Any]) -> str:
        """Generate Python code for a tool"""
        
        system_prompt = """
        You are an expert Python programmer specializing in healthcare applications.
        Generate complete, production-ready Python code for a healthcare tool.
        
        Requirements:
        - Include comprehensive docstrings
        - Add type hints
        - Include error handling
        - Follow Python best practices
        - Add logging where appropriate
        - Make the code modular and testable
        - Include input validation
        - Handle edge cases
        
        The generated code should be a complete Python module with a main function
        that implements the tool's functionality.
        """
        
        user_prompt = f"""
        Create a Python tool with the following specification:
        
        Name: {tool_spec['name']}
        Description: {tool_spec['description']}
        Input Schema: {json.dumps(tool_spec.get('input_schema', {}), indent=2)}
        Output Schema: {json.dumps(tool_spec.get('output_schema', {}), indent=2)}
        Dependencies: {tool_spec.get('dependencies', [])}
        Complexity: {tool_spec.get('complexity', 'medium')}
        
        Generate complete Python code that implements this tool.
        Include a main function that takes the specified inputs and returns the specified outputs.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating tool code: {e}")
            return f"# Error generating code for {tool_spec['name']}\n# {str(e)}"
    
    async def _generate_test_code(self, tool_spec: Dict[str, Any], tool_code: str) -> str:
        """Generate test code for a tool"""
        
        system_prompt = """
        You are an expert in Python testing and healthcare applications.
        Generate comprehensive unit tests for the provided tool code.
        
        Requirements:
        - Use pytest framework
        - Test normal cases and edge cases
        - Test error conditions
        - Include fixtures if needed
        - Test input validation
        - Mock external dependencies if needed
        - Add descriptive test names and docstrings
        """
        
        user_prompt = f"""
        Generate comprehensive unit tests for this tool:
        
        Tool Name: {tool_spec['name']}
        Tool Description: {tool_spec['description']}
        
        Tool Code:
        ```python
        {tool_code}
        ```
        
        Create complete pytest test cases that thoroughly test this tool.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating test code: {e}")
            return f"# Error generating test code for {tool_spec['name']}\n# {str(e)}"
    
    async def _validate_tool_code(self, tool_code: str, test_code: str) -> Tuple[bool, List[str]]:
        """Validate generated tool code"""
        
        errors = []
        
        # Basic syntax validation
        try:
            ast.parse(tool_code)
        except SyntaxError as e:
            errors.append(f"Tool code syntax error: {e}")
        
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            errors.append(f"Test code syntax error: {e}")
        
        # Try to compile and import the tool code
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(tool_code)
                temp_file = f.name
            
            spec = importlib.util.spec_from_file_location("temp_tool", temp_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                errors.append("Could not create module spec")
                
        except Exception as e:
            errors.append(f"Tool code compilation error: {e}")
        finally:
            # Clean up temp file
            try:
                Path(temp_file).unlink()
            except:
                pass
        
        return len(errors) == 0, errors
    
    async def _store_tool(self, tool_info: ToolInfo):
        """Store tool information in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tools
                (tool_id, name, description, version, code, test_code,
                 dependencies, performance_metrics, usage_count, success_rate,
                 created_at, last_used, creator_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tool_info.tool_id,
                tool_info.name,
                tool_info.description,
                tool_info.version,
                tool_info.code,
                tool_info.test_code,
                json.dumps(tool_info.dependencies),
                json.dumps(tool_info.performance_metrics),
                tool_info.usage_count,
                tool_info.success_rate,
                tool_info.created_at.isoformat(),
                tool_info.last_used.isoformat(),
                tool_info.creator_agent
            ))
    
    async def tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists in the toolbank"""
        return tool_name in self.tools
    
    async def get_tool(self, tool_name: str) -> Optional[ToolInfo]:
        """Get tool information"""
        return self.tools.get(tool_name)
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        
        tools_list = []
        for tool_name, tool_info in self.tools.items():
            tools_list.append({
                "name": tool_name,
                "description": tool_info.description,
                "version": tool_info.version,
                "usage_count": tool_info.usage_count,
                "success_rate": tool_info.success_rate,
                "dependencies": tool_info.dependencies
            })
        
        return tools_list
    
    async def execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given inputs"""
        
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found"
            }
        
        tool_info = self.tools[tool_name]
        start_time = datetime.now()
        
        try:
            # Load and execute tool
            tool_file = self.tools_dir / f"{tool_name}.py"
            
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load tool module {tool_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Execute main function
            if hasattr(module, 'main'):
                result = module.main(**inputs)
            else:
                raise AttributeError(f"Tool {tool_name} has no main function")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update tool statistics
            await self._update_tool_stats(tool_name, execution_time, True)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update tool statistics for failure
            await self._update_tool_stats(tool_name, execution_time, False, str(e))
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _update_tool_stats(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        error_msg: Optional[str] = None
    ):
        """Update tool usage statistics"""
        
        tool_info = self.tools[tool_name]
        
        # Update in-memory stats
        tool_info.usage_count += 1
        tool_info.last_used = datetime.now()
        
        # Calculate new success rate
        with sqlite3.connect(self.db_path) as conn:
            # Record execution
            conn.execute("""
                INSERT INTO tool_executions
                (tool_id, execution_time, success, error_msg, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                tool_info.tool_id,
                execution_time,
                success,
                error_msg,
                datetime.now().isoformat()
            ))
            
            # Calculate success rate
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
                FROM tool_executions
                WHERE tool_id = ?
            """, (tool_info.tool_id,))
            
            total, successes = cursor.fetchone()
            tool_info.success_rate = successes / total if total > 0 else 1.0
            
            # Update database
            conn.execute("""
                UPDATE tools
                SET usage_count = ?, success_rate = ?, last_used = ?
                WHERE tool_id = ?
            """, (
                tool_info.usage_count,
                tool_info.success_rate,
                tool_info.last_used.isoformat(),
                tool_info.tool_id
            ))
    
    async def create_mcp(self, tools: List[str]) -> Dict[str, Any]:
        """Create Model Context Protocol for a set of tools"""
        
        mcp_name = f"healthflow_mcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get tool definitions
        mcp_tools = []
        for tool_name in tools:
            if tool_name in self.tools:
                tool_info = self.tools[tool_name]
                mcp_tools.append({
                    "name": tool_name,
                    "description": tool_info.description,
                    "inputSchema": self._extract_input_schema(tool_info.code),
                    "outputSchema": self._extract_output_schema(tool_info.code)
                })
        
        # Create MCP definition
        mcp_definition = MCPDefinition(
            name=mcp_name,
            version="1.0.0",
            description=f"HealthFlow MCP with {len(mcp_tools)} specialized healthcare tools",
            tools=mcp_tools,
            resources=[],
            prompts=[],
            server_info={
                "name": mcp_name,
                "version": "1.0.0",
                "description": "Auto-generated HealthFlow MCP server"
            }
        )
        
        # Store MCP
        mcp_id = hashlib.md5(f"{mcp_name}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO mcps (mcp_id, name, version, definition, tools, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                mcp_id,
                mcp_name,
                "1.0.0",
                json.dumps(asdict(mcp_definition)),
                json.dumps(tools),
                datetime.now().isoformat()
            ))
        
        # Save MCP file
        mcp_file = self.mcps_dir / f"{mcp_name}.json"
        with open(mcp_file, 'w') as f:
            json.dump(asdict(mcp_definition), f, indent=2)
        
        self.logger.info(f"Created MCP: {mcp_name} with {len(mcp_tools)} tools")
        
        return {
            "success": True,
            "mcp_id": mcp_id,
            "name": mcp_name,
            "tools_count": len(mcp_tools),
            "file_path": str(mcp_file)
        }
    
    def _extract_input_schema(self, code: str) -> Dict[str, Any]:
        """Extract input schema from tool code"""
        
        # Simple extraction - in production would use AST parsing
        # For now, return a generic schema
        return {
            "type": "object",
            "properties": {
                "data": {"type": "any", "description": "Input data for the tool"}
            },
            "required": ["data"]
        }
    
    def _extract_output_schema(self, code: str) -> Dict[str, Any]:
        """Extract output schema from tool code"""
        
        # Simple extraction - in production would use AST parsing
        # For now, return a generic schema
        return {
            "type": "object",
            "properties": {
                "result": {"type": "any", "description": "Output result from the tool"}
            }
        }
    
    def get_toolbank_statistics(self) -> Dict[str, Any]:
        """Get comprehensive toolbank statistics"""
        
        total_tools = len(self.tools)
        
        if total_tools == 0:
            return {
                "total_tools": 0,
                "message": "No tools in toolbank"
            }
        
        # Calculate statistics
        avg_usage = sum(tool.usage_count for tool in self.tools.values()) / total_tools
        avg_success_rate = sum(tool.success_rate for tool in self.tools.values()) / total_tools
        
        # Most used tools
        most_used = sorted(
            [(name, info.usage_count) for name, info in self.tools.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Tools by success rate
        best_tools = sorted(
            [(name, info.success_rate) for name, info in self.tools.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_tools": total_tools,
            "average_usage_count": avg_usage,
            "average_success_rate": avg_success_rate,
            "most_used_tools": most_used,
            "best_performing_tools": best_tools,
            "tools_directory": str(self.tools_dir),
            "mcps_directory": str(self.mcps_dir)
        }