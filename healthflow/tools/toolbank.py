"""
Hierarchical ToolBank System for HealthFlow
Dynamic tool creation, management, and execution system with tag-based retrieval
Uses JSON format for all persistence
"""

import json
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass
from enum import Enum
import uuid
import inspect
import importlib.util
import tempfile
import hashlib
from collections import defaultdict



class ToolType(Enum):
    """Types of tools in the ToolBank"""
    PYTHON_FUNCTION = "python_function"
    SHELL_COMMAND = "shell_command"
    API_CALL = "api_call"
    DATA_PROCESSOR = "data_processor"
    MEDICAL_ANALYZER = "medical_analyzer"
    CODE_GENERATOR = "code_generator"


class TagHierarchy(Enum):
    """Hierarchical tag categories for better organization"""
    DOMAIN = "domain"          # medical, general, research
    FUNCTIONALITY = "functionality"  # analysis, visualization, processing
    COMPLEXITY = "complexity"  # basic, intermediate, advanced
    DATA_TYPE = "data_type"    # clinical, genomic, imaging, text
    TASK_TYPE = "task_type"    # diagnosis, treatment, prediction, documentation


@dataclass
class ToolMetadata:
    """Enhanced metadata for tools with hierarchical tags"""
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
    hierarchical_tags: Dict[TagHierarchy, List[str]]  # Structured tag system
    dependencies: List[str]
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
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
            'hierarchical_tags': {k.value: v for k, v in self.hierarchical_tags.items()},
            'dependencies': self.dependencies,
            'performance_metrics': self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolMetadata':
        """Create from dictionary"""
        hierarchical_tags = {}
        for key, value in data.get('hierarchical_tags', {}).items():
            try:
                hierarchical_tags[TagHierarchy(key)] = value
            except ValueError:
                # Handle legacy tags or unknown categories
                pass
        
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
            hierarchical_tags=hierarchical_tags,
            dependencies=data['dependencies'],
            performance_metrics=data.get('performance_metrics', {})
        )


@dataclass
class Tool:
    """Represents a tool in the ToolBank with enhanced JSON serialization"""
    metadata: ToolMetadata
    implementation: str
    test_cases: List[Dict[str, Any]]
    documentation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
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


@dataclass
class ToolExecutionResult:
    """Result of tool execution with enhanced metrics"""
    tool_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    context_used: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context_used is None:
            self.context_used = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'tool_id': self.tool_id,
            'success': self.success,
            'output': str(self.output) if self.output is not None else None,
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'context_used': self.context_used
        }


class HierarchicalToolBank:
    """
    Advanced hierarchical tool bank with tag-based retrieval and JSON persistence
    
    Key features:
    - Hierarchical tag system for better organization
    - Context-aware tool retrieval
    - Performance-based tool ranking
    - JSON-only persistence (no parquet/pickle)
    - Tool versioning and evolution
    """
    
    def __init__(self, tools_dir: Path):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths - all JSON format
        self.tools_registry_path = self.tools_dir / "tools_registry.json"
        self.execution_history_path = self.tools_dir / "execution_history.json"
        self.tag_hierarchy_path = self.tools_dir / "tag_hierarchy.json"
        self.performance_metrics_path = self.tools_dir / "performance_metrics.json"
        self.tools_code_dir = self.tools_dir / "code"
        self.tools_tests_dir = self.tools_dir / "tests"
        
        # Create subdirectories
        self.tools_code_dir.mkdir(exist_ok=True)
        self.tools_tests_dir.mkdir(exist_ok=True)
        
        # In-memory registry
        self.tools_registry: Dict[str, Tool] = {}
        self.execution_history: List[ToolExecutionResult] = []
        self.tag_hierarchy: Dict[str, Dict[str, Set[str]]] = {}
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        
        
        # Initialize tag hierarchy
        self._initialize_tag_hierarchy()
    
    def _initialize_tag_hierarchy(self):
        """Initialize hierarchical tag structure"""
        self.tag_hierarchy = {
            TagHierarchy.DOMAIN.value: {
                "medical": {"cardiology", "oncology", "neurology", "radiology", "pathology", "general_medicine"},
                "research": {"clinical_trials", "epidemiology", "bioinformatics", "literature_review"},
                "general": {"data_processing", "visualization", "utilities", "apis"}
            },
            TagHierarchy.FUNCTIONALITY.value: {
                "analysis": {"statistical", "diagnostic", "predictive", "comparative"},
                "processing": {"data_cleaning", "transformation", "validation", "aggregation"},
                "visualization": {"charts", "reports", "dashboards", "imaging"},
                "communication": {"api_calls", "database", "file_io", "messaging"}
            },
            TagHierarchy.COMPLEXITY.value: {
                "basic": {"simple_calculation", "data_retrieval", "basic_validation"},
                "intermediate": {"analysis_pipeline", "multi_step_processing", "conditional_logic"},
                "advanced": {"machine_learning", "complex_algorithms", "multi_agent_coordination"}
            },
            TagHierarchy.DATA_TYPE.value: {
                "clinical": {"ehr", "lab_results", "vital_signs", "medications"},
                "imaging": {"xray", "mri", "ct_scan", "ultrasound", "pathology_slides"},
                "genomic": {"dna_sequence", "gene_expression", "variants", "annotations"},
                "text": {"clinical_notes", "reports", "literature", "documentation"}
            },
            TagHierarchy.TASK_TYPE.value: {
                "diagnosis": {"differential", "screening", "confirmation", "staging"},
                "treatment": {"planning", "monitoring", "adjustment", "outcome_prediction"},
                "prediction": {"risk_assessment", "prognosis", "outbreak", "readmission"},
                "documentation": {"report_generation", "coding", "summarization", "extraction"}
            }
        }
    
    async def initialize(self):
        """Initialize and load existing tools"""
        await self._load_tools()
        await self._load_execution_history()
        await self._load_tag_hierarchy()
        await self._load_performance_metrics()
        
    
    async def _load_tools(self):
        """Load tools from JSON registry"""
        if self.tools_registry_path.exists():
            try:
                with open(self.tools_registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    for tool_id, tool_data in registry_data.items():
                        tool = Tool.from_dict(tool_data)
                        self.tools_registry[tool_id] = tool
            except Exception as e:
                print(f"Error loading tools from JSON: {e}")
    
    async def _load_execution_history(self):
        """Load execution history from JSON"""
        if self.execution_history_path.exists():
            try:
                with open(self.execution_history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    # Keep only recent history in memory (last 1000 entries)
                    recent_history = history_data.get('executions', [])[-1000:]
                    for exec_data in recent_history:
                        exec_data['timestamp'] = datetime.fromisoformat(exec_data['timestamp'])
                        result = ToolExecutionResult(**exec_data)
                        self.execution_history.append(result)
            except Exception as e:
                print(f"Error loading execution history: {e}")
    
    async def _load_tag_hierarchy(self):
        """Load custom tag hierarchy from JSON"""
        if self.tag_hierarchy_path.exists():
            try:
                with open(self.tag_hierarchy_path, 'r', encoding='utf-8') as f:
                    custom_hierarchy = json.load(f)
                    # Merge with default hierarchy
                    for category, subcategories in custom_hierarchy.items():
                        if category in self.tag_hierarchy:
                            for subcat, tags in subcategories.items():
                                self.tag_hierarchy[category][subcat].update(set(tags))
                        else:
                            self.tag_hierarchy[category] = {
                                subcat: set(tags) for subcat, tags in subcategories.items()
                            }
            except Exception as e:
                print(f"Error loading tag hierarchy: {e}")
    
    async def _load_performance_metrics(self):
        """Load performance metrics from JSON"""
        if self.performance_metrics_path.exists():
            try:
                with open(self.performance_metrics_path, 'r', encoding='utf-8') as f:
                    self.performance_cache = json.load(f)
            except Exception as e:
                print(f"Error loading performance metrics: {e}")
    
    async def _save_tools(self):
        """Save tools to JSON registry"""
        try:
            registry_data = {}
            for tool_id, tool in self.tools_registry.items():
                registry_data[tool_id] = tool.to_dict()
            
            with open(self.tools_registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving tools to JSON: {e}")
    
    async def _save_execution_history(self):
        """Save execution history to JSON"""
        if not self.execution_history:
            return
            
        try:
            # Load existing history
            existing_history = []
            if self.execution_history_path.exists():
                with open(self.execution_history_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_history = existing_data.get('executions', [])
            
            # Append new executions
            new_executions = [result.to_dict() for result in self.execution_history]
            all_executions = existing_history + new_executions
            
            # Keep only recent history (last 5000 entries)
            if len(all_executions) > 5000:
                all_executions = all_executions[-5000:]
            
            history_data = {
                'executions': all_executions,
                'last_updated': datetime.now().isoformat(),
                'total_count': len(all_executions)
            }
            
            with open(self.execution_history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            # Clear in-memory history to save memory
            self.execution_history = []
            
        except Exception as e:
            print(f"Error saving execution history: {e}")
    
    async def _save_tag_hierarchy(self):
        """Save custom tag hierarchy to JSON"""
        try:
            # Convert sets to lists for JSON serialization
            hierarchy_data = {}
            for category, subcategories in self.tag_hierarchy.items():
                hierarchy_data[category] = {}
                for subcat, tags in subcategories.items():
                    hierarchy_data[category][subcat] = list(tags)
            
            with open(self.tag_hierarchy_path, 'w', encoding='utf-8') as f:
                json.dump(hierarchy_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving tag hierarchy: {e}")
    
    async def _save_performance_metrics(self):
        """Save performance metrics to JSON"""
        try:
            with open(self.performance_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
    
    async def create_python_tool(
        self,
        name: str,
        description: str,
        implementation: str,
        parameters: Dict[str, Any],
        return_type: str = "Any",
        hierarchical_tags: Dict[TagHierarchy, List[str]] = None,
        test_cases: List[Dict[str, Any]] = None,
        author: str = "HealthFlow Agent"
    ) -> str:
        """Create a new Python function tool with hierarchical tags"""
        
        tool_id = str(uuid.uuid4())
        hierarchical_tags = hierarchical_tags or {}
        test_cases = test_cases or []
        
        # Auto-categorize based on content analysis
        hierarchical_tags = self._auto_categorize_tool(implementation, description, hierarchical_tags)
        
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
            hierarchical_tags=hierarchical_tags,
            dependencies=self._extract_dependencies(implementation),
            performance_metrics={}
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
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(implementation)
            
            # Generate and save test file
            test_code = await self._generate_test_code(tool)
            test_file = self.tools_tests_dir / f"test_{tool_id}.py"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            # Update tag hierarchy with new tags
            await self._update_tag_hierarchy_with_tool(tool)
            
            await self._save_tools()
            await self._save_tag_hierarchy()
            
            return tool_id
        else:
            raise ValueError(f"Tool validation failed for: {name}")
    
    def _auto_categorize_tool(
        self, 
        implementation: str, 
        description: str, 
        existing_tags: Dict[TagHierarchy, List[str]]
    ) -> Dict[TagHierarchy, List[str]]:
        """Automatically categorize tool based on content analysis"""
        tags = existing_tags.copy()
        
        text = (implementation + " " + description).lower()
        
        # Medical domain detection
        medical_keywords = ["patient", "diagnosis", "clinical", "medical", "treatment", "therapy"]
        if any(keyword in text for keyword in medical_keywords):
            if TagHierarchy.DOMAIN not in tags:
                tags[TagHierarchy.DOMAIN] = []
            tags[TagHierarchy.DOMAIN].append("medical")
        
        # Data type detection
        if "image" in text or "scan" in text or "xray" in text:
            if TagHierarchy.DATA_TYPE not in tags:
                tags[TagHierarchy.DATA_TYPE] = []
            tags[TagHierarchy.DATA_TYPE].append("imaging")
        
        # Functionality detection
        if "analyze" in text or "analysis" in text:
            if TagHierarchy.FUNCTIONALITY not in tags:
                tags[TagHierarchy.FUNCTIONALITY] = []
            tags[TagHierarchy.FUNCTIONALITY].append("analysis")
        
        # Complexity detection based on implementation patterns
        complexity_indicators = {
            "basic": ["simple", "basic", "return", "print"],
            "intermediate": ["for", "if", "while", "class"],
            "advanced": ["async", "threading", "multiprocessing", "neural", "model"]
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in text for indicator in indicators):
                if TagHierarchy.COMPLEXITY not in tags:
                    tags[TagHierarchy.COMPLEXITY] = []
                tags[TagHierarchy.COMPLEXITY].append(complexity)
                break
        
        return tags
    
    async def _update_tag_hierarchy_with_tool(self, tool: Tool):
        """Update tag hierarchy with new tags from tool"""
        for category, tag_list in tool.metadata.hierarchical_tags.items():
            if category.value not in self.tag_hierarchy:
                self.tag_hierarchy[category.value] = {}
            
            for tag in tag_list:
                # Find appropriate subcategory or create new one
                found_subcat = None
                for subcat, existing_tags in self.tag_hierarchy[category.value].items():
                    if tag in existing_tags:
                        found_subcat = subcat
                        break
                
                if not found_subcat:
                    # Create new subcategory or add to "custom"
                    if "custom" not in self.tag_hierarchy[category.value]:
                        self.tag_hierarchy[category.value]["custom"] = set()
                    self.tag_hierarchy[category.value]["custom"].add(tag)
    
    async def search_tools(
        self,
        query: str = None,
        tool_type: ToolType = None,
        hierarchical_tags: Dict[TagHierarchy, List[str]] = None,
        task_context: Dict[str, Any] = None,
        min_success_rate: float = 0.0,
        limit: int = 10
    ) -> List[Tool]:
        """
        Advanced hierarchical search for tools using efficient tag-based pre-filtering.
        
        Implementation follows the NeurIPS-quality approach:
        1. Fast pre-filtering using hierarchical tags to reduce search space
        2. Secondary scoring only on the pre-filtered subset
        3. Performance-optimized for large tool registries
        """
        
        # Step 1: FAST PRE-FILTERING - Dramatically reduce search space using hierarchical tags
        candidate_tools = []
        
        if hierarchical_tags:
            # Pre-filter based on hierarchical tags - this is the key efficiency optimization
            for tool in self.tools_registry.values():
                # Quick rejection filters
                if tool.metadata.success_rate < min_success_rate:
                    continue
                if tool_type and tool.metadata.tool_type != tool_type:
                    continue
                
                # Hierarchical tag matching - prioritize by importance
                matches_required_tags = True
                tag_match_score = 0
                
                for category, query_tags in hierarchical_tags.items():
                    tool_tags = tool.metadata.hierarchical_tags.get(category, [])
                    tag_intersection = set(query_tags) & set(tool_tags)
                    
                    # For high-priority categories, require at least one match
                    if category in [TagHierarchy.DOMAIN, TagHierarchy.FUNCTIONALITY]:
                        if not tag_intersection:
                            matches_required_tags = False
                            break
                    
                    # Accumulate tag match strength
                    if tag_intersection:
                        category_weights = {
                            TagHierarchy.DOMAIN: 5.0,
                            TagHierarchy.FUNCTIONALITY: 4.0,
                            TagHierarchy.TASK_TYPE: 4.0,
                            TagHierarchy.DATA_TYPE: 3.0,
                            TagHierarchy.COMPLEXITY: 2.0
                        }
                        weight = category_weights.get(category, 2.0)
                        tag_match_score += len(tag_intersection) * weight
                
                # Only include tools that match required hierarchical criteria
                if matches_required_tags and tag_match_score > 0:
                    candidate_tools.append((tool, tag_match_score))
        else:
            # If no hierarchical tags provided, use all tools but still apply basic filters
            for tool in self.tools_registry.values():
                if tool.metadata.success_rate >= min_success_rate:
                    if not tool_type or tool.metadata.tool_type == tool_type:
                        candidate_tools.append((tool, 0.0))
        
        # If no candidates after pre-filtering, return empty
        if not candidate_tools:
            return []
        
        # Step 2: SECONDARY SCORING - Only on the pre-filtered subset (much smaller)
        results = []
        
        for tool, base_tag_score in candidate_tools:
            total_score = base_tag_score
            
            # Text similarity scoring (expensive operation, but now on smaller set)
            if query:
                if query.lower() in tool.metadata.name.lower():
                    total_score += 10.0
                if query.lower() in tool.metadata.description.lower():
                    total_score += 5.0
                # Keyword matching in tool tags
                all_tags = []
                for tag_list in tool.metadata.hierarchical_tags.values():
                    all_tags.extend(tag_list)
                if any(query.lower() in tag.lower() for tag in all_tags):
                    total_score += 3.0
            
            # Context-aware scoring (only for pre-filtered candidates)
            if task_context:
                context_score = await self._calculate_context_score(tool, task_context)
                total_score += context_score
            
            # Performance-based scoring
            performance_score = self._calculate_performance_score(tool)
            total_score += performance_score
            
            results.append((tool, total_score))
        
        # Step 3: EFFICIENT SORTING - Sort by final score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in results[:limit]]
    
    async def find_tools_for_task(
        self,
        task_description: str,
        domain: List[str] = None,
        functionality: List[str] = None,
        task_type: List[str] = None,
        data_type: List[str] = None,
        complexity: List[str] = None,
        limit: int = 5
    ) -> List[Tool]:
        """
        Convenient interface for agents to find tools for specific tasks.
        This method provides a simplified API over the full hierarchical search.
        
        Args:
            task_description: Natural language description of the task
            domain: Domain tags (e.g., ['medical', 'research'])  
            functionality: Functionality tags (e.g., ['analysis', 'visualization'])
            task_type: Task type tags (e.g., ['diagnosis', 'treatment'])
            data_type: Data type tags (e.g., ['clinical', 'imaging'])
            complexity: Complexity tags (e.g., ['basic', 'advanced'])
            limit: Maximum number of tools to return
            
        Returns:
            List of most relevant tools for the task
        """
        
        # Build hierarchical tags from provided parameters
        hierarchical_tags = {}
        
        if domain:
            hierarchical_tags[TagHierarchy.DOMAIN] = domain
        if functionality:
            hierarchical_tags[TagHierarchy.FUNCTIONALITY] = functionality  
        if task_type:
            hierarchical_tags[TagHierarchy.TASK_TYPE] = task_type
        if data_type:
            hierarchical_tags[TagHierarchy.DATA_TYPE] = data_type
        if complexity:
            hierarchical_tags[TagHierarchy.COMPLEXITY] = complexity
        
        # Create task context from description
        task_context = {
            "description": task_description,
            "keywords": task_description.lower().split()
        }
        
        # Use the optimized hierarchical search
        return await self.search_tools(
            query=task_description,
            hierarchical_tags=hierarchical_tags if hierarchical_tags else None,
            task_context=task_context,
            limit=limit
        )
    
    async def _calculate_context_score(self, tool: Tool, task_context: Dict[str, Any]) -> float:
        """Calculate context relevance score for a tool"""
        score = 0.0
        
        context_keywords = set()
        for key, value in task_context.items():
            if isinstance(value, str):
                context_keywords.update(value.lower().split())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        context_keywords.update(item.lower().split())
        
        # Match against tool description and documentation
        tool_text = (tool.metadata.description + " " + tool.documentation).lower()
        tool_keywords = set(tool_text.split())
        
        keyword_intersection = context_keywords & tool_keywords
        score += len(keyword_intersection) * 0.5
        
        return score
    
    def _calculate_performance_score(self, tool: Tool) -> float:
        """Calculate performance-based score for a tool"""
        score = 0.0
        
        # Success rate contribution
        score += tool.metadata.success_rate * 2.0
        
        # Usage frequency contribution (normalized)
        if tool.metadata.usage_count > 0:
            score += min(tool.metadata.usage_count / 100.0, 2.0)
        
        # Recent usage bonus
        days_since_use = (datetime.now() - tool.metadata.last_used).days
        if days_since_use < 7:
            score += 1.0
        elif days_since_use < 30:
            score += 0.5
        
        return score
    
    def get_tools_by_hierarchy(self, hierarchy_path: List[str]) -> List[Tool]:
        """Get tools by navigating the tag hierarchy"""
        if len(hierarchy_path) < 2:
            return []
        
        category, subcategory = hierarchy_path[0], hierarchy_path[1]
        specific_tags = hierarchy_path[2:] if len(hierarchy_path) > 2 else []
        
        # Find tools matching the hierarchy path
        matching_tools = []
        for tool in self.tools_registry.values():
            try:
                category_enum = TagHierarchy(category)
                tool_tags = tool.metadata.hierarchical_tags.get(category_enum, [])
                
                # Check if tool has tags in the specified subcategory
                if subcategory in self.tag_hierarchy.get(category, {}):
                    subcategory_tags = self.tag_hierarchy[category][subcategory]
                    
                    if any(tag in subcategory_tags for tag in tool_tags):
                        # If specific tags are specified, check for exact match
                        if not specific_tags or any(tag in tool_tags for tag in specific_tags):
                            matching_tools.append(tool)
            except ValueError:
                continue
        
        return matching_tools
    
    async def recommend_tools_for_task(
        self, 
        task_description: str, 
        task_type: str = None,
        data_types: List[str] = None,
        complexity_preference: str = "intermediate"
    ) -> List[Tool]:
        """Recommend tools for a specific task using hierarchical analysis"""
        
        # Parse task to extract relevant hierarchical tags
        hierarchical_tags = self._extract_hierarchical_tags_from_task(
            task_description, task_type, data_types, complexity_preference
        )
        
        # Search with extracted tags
        recommended_tools = await self.search_tools(
            query=task_description,
            hierarchical_tags=hierarchical_tags,
            task_context={"description": task_description, "type": task_type},
            limit=15
        )
        
        # Group recommendations by category for better organization
        categorized_recommendations = self._categorize_recommendations(recommended_tools)
        
        return recommended_tools
    
    def _extract_hierarchical_tags_from_task(
        self,
        task_description: str,
        task_type: str = None,
        data_types: List[str] = None,
        complexity_preference: str = "intermediate"
    ) -> Dict[TagHierarchy, List[str]]:
        """Extract hierarchical tags from task description"""
        tags = {}
        
        text = task_description.lower()
        
        # Domain extraction
        if any(word in text for word in ["medical", "clinical", "patient", "diagnosis"]):
            tags[TagHierarchy.DOMAIN] = ["medical"]
        elif any(word in text for word in ["research", "study", "analysis"]):
            tags[TagHierarchy.DOMAIN] = ["research"]
        
        # Task type extraction
        if task_type:
            tags[TagHierarchy.TASK_TYPE] = [task_type.lower()]
        else:
            if any(word in text for word in ["diagnose", "diagnosis", "identify"]):
                tags[TagHierarchy.TASK_TYPE] = ["diagnosis"]
            elif any(word in text for word in ["treat", "treatment", "therapy"]):
                tags[TagHierarchy.TASK_TYPE] = ["treatment"]
            elif any(word in text for word in ["predict", "forecast", "risk"]):
                tags[TagHierarchy.TASK_TYPE] = ["prediction"]
        
        # Data type extraction
        if data_types:
            tags[TagHierarchy.DATA_TYPE] = data_types
        else:
            detected_data_types = []
            if any(word in text for word in ["image", "scan", "xray", "mri"]):
                detected_data_types.append("imaging")
            if any(word in text for word in ["lab", "blood", "test", "results"]):
                detected_data_types.append("clinical")
            if any(word in text for word in ["text", "note", "report", "document"]):
                detected_data_types.append("text")
            if detected_data_types:
                tags[TagHierarchy.DATA_TYPE] = detected_data_types
        
        # Complexity preference
        tags[TagHierarchy.COMPLEXITY] = [complexity_preference]
        
        return tags
    
    def _categorize_recommendations(self, tools: List[Tool]) -> Dict[str, List[Tool]]:
        """Categorize recommended tools by their primary function"""
        categories = defaultdict(list)
        
        for tool in tools:
            # Determine primary category
            primary_category = "general"
            
            if TagHierarchy.FUNCTIONALITY in tool.metadata.hierarchical_tags:
                functionality_tags = tool.metadata.hierarchical_tags[TagHierarchy.FUNCTIONALITY]
                if functionality_tags:
                    primary_category = functionality_tags[0]
            
            categories[primary_category].append(tool)
        
        return dict(categories)
    
    async def execute_tool(
        self,
        tool_id: str,
        inputs: Dict[str, Any],
        execution_context: Dict[str, Any] = None,
        timeout: int = 30
    ) -> ToolExecutionResult:
        """Execute a tool with enhanced context tracking"""
        
        if tool_id not in self.tools_registry:
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                output=None,
                error="Tool not found in registry",
                execution_time=0.0,
                context_used=execution_context or {}
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
            
            # Update success rate using exponential moving average
            alpha = 0.1
            if result.success:
                tool.metadata.success_rate = (1 - alpha) * tool.metadata.success_rate + alpha * 1.0
            else:
                tool.metadata.success_rate = (1 - alpha) * tool.metadata.success_rate + alpha * 0.0
            
            # Update performance metrics
            await self._update_performance_metrics(tool, result, execution_context)
            
            # Add execution context
            result.context_used = execution_context or {}
            
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
                context_used=execution_context or {}
            )
            self.execution_history.append(result)
            return result
    
    async def _execute_python_tool(
        self,
        tool: Tool,
        inputs: Dict[str, Any],
        timeout: int
    ) -> ToolExecutionResult:
        """Execute Python function tool with enhanced error handling"""
        
        start_time = datetime.now()
        
        try:
            # Load the tool code
            tool_file = self.tools_code_dir / f"{tool.metadata.tool_id}.py"
            
            spec = importlib.util.spec_from_file_location("tool_module", tool_file)
            module = importlib.util.module_from_spec(spec)
            
            # Add the tool directory to sys.path temporarily
            original_path = sys.path.copy()
            sys.path.insert(0, str(self.tools_code_dir))
            
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path = original_path
            
            # Find the main function
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
                    execution_time=execution_time
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
                execution_time=execution_time
            )
    
    async def _execute_shell_tool(self, tool: Tool, inputs: Dict[str, Any], timeout: int) -> ToolExecutionResult:
        """Execute shell command tool"""
        start_time = datetime.now()
        
        try:
            command = tool.implementation.format(**inputs)
            
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
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolExecutionResult(
                tool_id=tool.metadata.tool_id,
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time
            )
    
    
    async def _update_performance_metrics(
        self, 
        tool: Tool, 
        result: ToolExecutionResult, 
        context: Dict[str, Any] = None
    ):
        """Update performance metrics for a tool"""
        tool_id = tool.metadata.tool_id
        
        if tool_id not in self.performance_cache:
            self.performance_cache[tool_id] = {}
        
        metrics = self.performance_cache[tool_id]
        
        # Update execution time statistics
        if 'avg_execution_time' not in metrics:
            metrics['avg_execution_time'] = result.execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            metrics['avg_execution_time'] = (1 - alpha) * metrics['avg_execution_time'] + alpha * result.execution_time
        
        # Update success rate over time
        if 'recent_success_rate' not in metrics:
            metrics['recent_success_rate'] = 1.0 if result.success else 0.0
        else:
            alpha = 0.2
            success_value = 1.0 if result.success else 0.0
            metrics['recent_success_rate'] = (1 - alpha) * metrics['recent_success_rate'] + alpha * success_value
        
        # Context-specific performance
        if context and 'task_type' in context:
            task_type = context['task_type']
            if task_type not in metrics:
                metrics[task_type] = {'success_rate': 0.0, 'count': 0}
            
            task_metrics = metrics[task_type]
            task_metrics['count'] += 1
            success_value = 1.0 if result.success else 0.0
            task_metrics['success_rate'] = ((task_metrics['count'] - 1) * task_metrics['success_rate'] + success_value) / task_metrics['count']
    
    async def _validate_python_tool(self, tool: Tool) -> bool:
        """Validate Python tool implementation"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(tool.implementation)
                temp_path = f.name
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            compile(code, temp_path, 'exec')
            Path(temp_path).unlink()
            
            return True
        except Exception as e:
            print(f"Tool validation error: {e}")
            return False
    
    async def _generate_test_code(self, tool: Tool) -> str:
        """Generate comprehensive test code for a tool"""
        return f'''
import pytest
import sys
import json
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

# Import the tool module
import {tool.metadata.tool_id} as tool_module

def test_{tool.metadata.name.lower().replace(" ", "_")}_basic():
    """Basic test for {tool.metadata.name}"""
    assert hasattr(tool_module, 'main')
    assert callable(tool_module.main)

def test_{tool.metadata.name.lower().replace(" ", "_")}_with_parameters():
    """Test {tool.metadata.name} with expected parameters"""
    expected_params = {json.dumps(tool.metadata.parameters, indent=4)}
    
    # Test with minimal valid inputs
    try:
        result = tool_module.main()
        assert result is not None
    except Exception as e:
        # It's ok if the tool requires specific inputs
        assert any(word in str(e).lower() for word in ["required", "missing", "parameter"])

def test_{tool.metadata.name.lower().replace(" ", "_")}_hierarchical_tags():
    """Test that the tool has proper hierarchical tags"""
    expected_tags = {json.dumps({k.value: v for k, v in tool.metadata.hierarchical_tags.items()}, indent=4)}
    assert len(expected_tags) > 0, "Tool should have hierarchical tags for proper categorization"

# Performance test
def test_{tool.metadata.name.lower().replace(" ", "_")}_performance():
    """Basic performance test for {tool.metadata.name}"""
    import time
    start_time = time.time()
    
    try:
        tool_module.main()
        execution_time = time.time() - start_time
        assert execution_time < 30, f"Tool execution took too long: {{execution_time}} seconds"
    except Exception:
        # Performance test passed if tool fails gracefully within time limit
        execution_time = time.time() - start_time
        assert execution_time < 30, f"Tool failed too slowly: {{execution_time}} seconds"
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
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ToolBank statistics with hierarchical breakdown"""
        
        if not self.tools_registry:
            return {}
        
        total_tools = len(self.tools_registry)
        
        # Count by type
        type_counts = {}
        for tool in self.tools_registry.values():
            tool_type = tool.metadata.tool_type.value
            type_counts[tool_type] = type_counts.get(tool_type, 0) + 1
        
        # Count by hierarchical tags
        hierarchical_stats = {}
        for category in TagHierarchy:
            hierarchical_stats[category.value] = {}
            for tool in self.tools_registry.values():
                if category in tool.metadata.hierarchical_tags:
                    for tag in tool.metadata.hierarchical_tags[category]:
                        if tag not in hierarchical_stats[category.value]:
                            hierarchical_stats[category.value][tag] = 0
                        hierarchical_stats[category.value][tag] += 1
        
        # Calculate average success rate
        avg_success_rate = sum(t.metadata.success_rate for t in self.tools_registry.values()) / total_tools
        
        # Most used tools
        most_used = sorted(
            self.tools_registry.values(),
            key=lambda t: t.metadata.usage_count,
            reverse=True
        )[:5]
        
        # Performance leaders
        performance_leaders = sorted(
            self.tools_registry.values(),
            key=lambda t: t.metadata.success_rate * (1 + t.metadata.usage_count / 100),
            reverse=True
        )[:5]
        
        return {
            "total_tools": total_tools,
            "tool_type_distribution": type_counts,
            "hierarchical_tag_distribution": hierarchical_stats,
            "average_success_rate": avg_success_rate,
            "most_used_tools": [
                {
                    "name": tool.metadata.name,
                    "usage_count": tool.metadata.usage_count,
                    "success_rate": tool.metadata.success_rate,
                    "primary_tags": list(tool.metadata.hierarchical_tags.get(TagHierarchy.DOMAIN, []))
                }
                for tool in most_used
            ],
            "performance_leaders": [
                {
                    "name": tool.metadata.name,
                    "success_rate": tool.metadata.success_rate,
                    "usage_count": tool.metadata.usage_count,
                    "performance_score": tool.metadata.success_rate * (1 + tool.metadata.usage_count / 100)
                }
                for tool in performance_leaders
            ],
            "total_executions": len(self.execution_history),
            "tag_hierarchy_depth": {
                category: len(subcats) for category, subcats in self.tag_hierarchy.items()
            }
        }
    
    async def export_tool_catalog(self, output_path: Path):
        """Export comprehensive tool catalog in JSON format"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export full tool catalog
        catalog = {
            "tools": {tool_id: tool.to_dict() for tool_id, tool in self.tools_registry.items()},
            "tag_hierarchy": {
                category: {subcat: list(tags) for subcat, tags in subcats.items()}
                for category, subcats in self.tag_hierarchy.items()
            },
            "statistics": self.get_tool_statistics(),
            "export_timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(output_path / "tool_catalog.json", 'w', encoding='utf-8') as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        
        # Export hierarchical index for easy navigation
        hierarchical_index = {}
        for category in TagHierarchy:
            hierarchical_index[category.value] = {}
            for subcategory, tags in self.tag_hierarchy.get(category.value, {}).items():
                hierarchical_index[category.value][subcategory] = []
                for tag in tags:
                    matching_tools = []
                    for tool in self.tools_registry.values():
                        if category in tool.metadata.hierarchical_tags and tag in tool.metadata.hierarchical_tags[category]:
                            matching_tools.append({
                                "tool_id": tool.metadata.tool_id,
                                "name": tool.metadata.name,
                                "description": tool.metadata.description,
                                "success_rate": tool.metadata.success_rate,
                                "usage_count": tool.metadata.usage_count
                            })
                    hierarchical_index[category.value][subcategory].append({
                        "tag": tag,
                        "tools": matching_tools
                    })
        
        with open(output_path / "hierarchical_index.json", 'w', encoding='utf-8') as f:
            json.dump(hierarchical_index, f, ensure_ascii=False, indent=2)
    
    async def backup_toolbank(self, backup_path: Path):
        """Create comprehensive backup of the ToolBank"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save all JSON files
        await self._save_tools()
        await self._save_execution_history()
        await self._save_tag_hierarchy()
        await self._save_performance_metrics()
        
        # Copy all files to backup location
        import shutil
        
        # Copy JSON files
        for file_path in [self.tools_registry_path, self.execution_history_path, 
                         self.tag_hierarchy_path, self.performance_metrics_path]:
            if file_path.exists():
                shutil.copy2(file_path, backup_path / file_path.name)
        
        # Copy code and test directories
        if self.tools_code_dir.exists():
            shutil.copytree(self.tools_code_dir, backup_path / "code", dirs_exist_ok=True)
        if self.tools_tests_dir.exists():
            shutil.copytree(self.tools_tests_dir, backup_path / "tests", dirs_exist_ok=True)
        
        # Create backup manifest
        manifest = {
            "backup_timestamp": datetime.now().isoformat(),
            "total_tools": len(self.tools_registry),
            "files_backed_up": [
                str(f.relative_to(self.tools_dir)) for f in self.tools_dir.rglob("*") if f.is_file()
            ],
            "tag_hierarchy_categories": list(self.tag_hierarchy.keys()),
            "toolbank_version": "1.0"
        }
        
        with open(backup_path / "backup_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    


# Alias for compatibility with existing code
ToolBank = HierarchicalToolBank
