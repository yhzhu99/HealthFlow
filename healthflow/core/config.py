"""
HealthFlow Configuration Management
Handles environment variables and configuration settings
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


@dataclass
class HealthFlowConfig:
    """Configuration settings for HealthFlow"""
    
    # LLM Configuration
    base_url: str
    api_key: str
    model_name: str
    max_tokens: int
    temperature: float
    
    # Data Storage Configuration
    data_dir: Path
    memory_dir: Path
    tools_dir: Path
    cache_dir: Path
    evaluation_dir: Path
    
    # Agent Configuration
    max_iterations: int
    max_agents: int
    memory_window: int
    tool_timeout: int
    
    # Logging Configuration
    log_level: str
    log_file: Optional[Path]
    
    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> 'HealthFlowConfig':
        """Load configuration from environment variables and .env file"""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        # LLM Configuration
        base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('API_KEY', '')
        model_name = os.getenv('MODEL_NAME', 'gpt-4-turbo-preview')
        max_tokens = int(os.getenv('MAX_TOKENS', '4096'))
        temperature = float(os.getenv('TEMPERATURE', '0.7'))
        
        # Data Storage Configuration
        data_dir = Path(os.getenv('DATA_DIR', './data'))
        memory_dir = Path(os.getenv('MEMORY_DIR', './data/memory'))
        tools_dir = Path(os.getenv('TOOLS_DIR', './data/tools'))
        cache_dir = Path(os.getenv('CACHE_DIR', './data/cache'))
        evaluation_dir = Path(os.getenv('EVALUATION_DIR', './data/evaluation'))
        
        # Agent Configuration
        max_iterations = int(os.getenv('MAX_ITERATIONS', '10'))
        max_agents = int(os.getenv('MAX_AGENTS', '5'))
        memory_window = int(os.getenv('MEMORY_WINDOW', '1000'))
        tool_timeout = int(os.getenv('TOOL_TIMEOUT', '30'))
        
        # Logging Configuration
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file_str = os.getenv('LOG_FILE')
        log_file = Path(log_file_str) if log_file_str else None
        
        # Create directories
        for dir_path in [data_dir, memory_dir, tools_dir, cache_dir, evaluation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        return cls(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            data_dir=data_dir,
            memory_dir=memory_dir,
            tools_dir=tools_dir,
            cache_dir=cache_dir,
            evaluation_dir=evaluation_dir,
            max_iterations=max_iterations,
            max_agents=max_agents,
            memory_window=memory_window,
            tool_timeout=tool_timeout,
            log_level=log_level,
            log_file=log_file
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.api_key:
            raise ValueError("API_KEY must be provided")
        
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be positive")
            
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")
            
        if self.max_iterations <= 0:
            raise ValueError("MAX_ITERATIONS must be positive")
            
        return True