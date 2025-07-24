"""
ReAct (Reasoning and Acting) Agent Implementation

A simple, LLM-driven agent that can reason about problems and take actions
in a loop until the goal is achieved or max rounds reached.
"""
import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

from camel.agents import ChatAgent
from camel.messages import BaseMessage

from .simple_interpreter import SimpleHealthcareInterpreter
from .enhanced_logging import get_enhanced_logger

logger = logging.getLogger(__name__)
enhanced_logger = get_enhanced_logger()


@dataclass
class ReactStep:
    """Represents a single step in the ReAct loop."""
    thought: str
    action: str
    observation: str
    success: bool
    timestamp: datetime


class ReactAgent:
    """
    A ReAct (Reasoning and Acting) agent that can iteratively work towards goals.
    
    The agent follows a simple loop:
    1. Think about the current situation
    2. Decide on an action to take  
    3. Execute the action and observe results
    4. Repeat until goal achieved or max rounds reached
    """
    
    def __init__(self, chat_agent: ChatAgent, max_rounds: int = 3):
        """
        Initialize the ReAct agent.
        
        Args:
            chat_agent: The underlying chat agent for reasoning
            max_rounds: Maximum number of reasoning/action loops
        """
        self.chat_agent = chat_agent
        self.max_rounds = max_rounds
        self.interpreter = SimpleHealthcareInterpreter()
        self.available_actions = {
            'execute_code': self._execute_code,
            'analyze_error': self._analyze_error,
            'fix_code': self._fix_code,
            'complete_task': self._complete_task
        }
    
    def solve_task(self, task_description: str, context: str = "") -> Dict[str, Any]:
        """
        Solve a task using the ReAct approach.
        
        Args:
            task_description: Description of the task to solve
            context: Additional context or constraints
            
        Returns:
            Dictionary containing the solution and execution trace
        """
        logger.info(f"🤖 ReactAgent starting task: {task_description}")
        
        steps = []
        current_round = 0
        task_completed = False
        final_result = None
        
        # Initial context
        full_context = f"""
Task: {task_description}
{f"Context: {context}" if context else ""}

You are a ReAct agent. CRITICAL RULE: For ANY mathematical calculation, numbers, equations, or computational task - you MUST use execute_code as your FIRST action. Do NOT attempt manual calculations.

For each round:
1. THINK: Analyze the current situation and decide what to do next
2. ACT: Choose an action and execute it  
3. OBSERVE: Analyze the results

Available actions:
- execute_code: Run Python code to solve computational tasks (USE THIS FIRST for any math!)
- analyze_error: Analyze error messages to understand what went wrong
- fix_code: Fix broken code based on error analysis
- complete_task: Mark the task as completed with final answer

MANDATORY: If you see numbers, calculations, equations, or math - use execute_code immediately.

Format your response as:
THINK: [your reasoning]
ACT: [action_name] [action_parameters]
"""
        
        while current_round < self.max_rounds and not task_completed:
            current_round += 1
            logger.info(f"🔄 Round {current_round}/{self.max_rounds}")
            
            # Add step history to context
            step_history = ""
            if steps:
                step_history = "\n\nPrevious steps:\n"
                for i, step in enumerate(steps, 1):
                    step_history += f"Step {i}:\n"
                    step_history += f"  THOUGHT: {step.thought}\n"
                    step_history += f"  ACTION: {step.action}\n"
                    step_history += f"  RESULT: {step.observation}\n"
            
            current_context = full_context + step_history + f"\n\nRound {current_round}:"
            
            try:
                # Get reasoning and action from LLM
                user_message = BaseMessage.make_user_message(
                    role_name="User",
                    content=current_context
                )
                
                response = self.chat_agent.step(user_message)
                if not response or not hasattr(response, 'msg') or not response.msg:
                    raise Exception("No response from chat agent")
                
                # Parse the response
                response_content = response.msg.content
                thought, action, action_params = self._parse_response(response_content)
                
                logger.info(f"💭 THINK: {thought}")
                logger.info(f"🎯 ACT: {action} {action_params}")
                
                # Execute the action
                observation, success = self._execute_action(action, action_params)
                
                logger.info(f"👁️ OBSERVE: {observation}")
                
                # Record the step
                step = ReactStep(
                    thought=thought,
                    action=f"{action} {action_params}",
                    observation=observation,
                    success=success,
                    timestamp=datetime.now()
                )
                steps.append(step)
                
                # Check if task is completed
                if action == 'complete_task' and success:
                    task_completed = True
                    final_result = observation
                    
            except Exception as e:
                logger.error(f"❌ Error in round {current_round}: {e}")
                error_step = ReactStep(
                    thought=f"Error occurred: {str(e)}",
                    action="error_handling",
                    observation=f"Round {current_round} failed: {str(e)}",
                    success=False,
                    timestamp=datetime.now()
                )
                steps.append(error_step)
        
        # Compile final result
        if not task_completed and steps:
            # Use the last successful observation as result
            successful_steps = [s for s in steps if s.success]
            if successful_steps:
                final_result = successful_steps[-1].observation
            else:
                final_result = f"Task not completed after {self.max_rounds} rounds. Last attempt: {steps[-1].observation if steps else 'No steps completed'}"
        elif not final_result:
            final_result = "Task could not be completed"
        
        result = {
            'task_description': task_description,
            'completed': task_completed,
            'rounds_used': current_round,
            'max_rounds': self.max_rounds,
            'final_result': final_result,
            'steps': steps,
            'success': task_completed
        }
        
        logger.info(f"✅ ReactAgent completed: {task_completed} in {current_round} rounds")
        return result
    
    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """Parse the LLM response to extract thought, action, and parameters."""
        lines = response.strip().split('\n')
        thought = ""
        action = ""
        action_params = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('THINK:'):
                thought = line[6:].strip()
            elif line.startswith('ACT:'):
                act_content = line[4:].strip()
                parts = act_content.split(' ', 1)
                action = parts[0]
                action_params = parts[1] if len(parts) > 1 else ""
        
        # Special handling for code execution
        if action == 'execute_code' and '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                action_params = response[start:end].strip()
        
        # Fallback parsing if no action was found or action_params is empty
        if not action or (action == 'execute_code' and not action_params.strip()):
            # Simple extraction with preference for code execution
            if 'execute_code' in response or '```python' in response:
                action = 'execute_code'
                # Extract code block
                if '```python' in response:
                    start = response.find('```python') + 9
                    end = response.find('```', start)
                    if end > start:
                        action_params = response[start:end].strip()
                    else:
                        action_params = "# No code block found"
                else:
                    # If no code block, try to extract from action parameters or anywhere in response
                    import re
                    match = re.search(r'execute_code\s+(.+)', response, re.DOTALL)
                    if match:
                        action_params = match.group(1).strip()
                    elif 'execute_code' in response:
                        # Extract everything after execute_code
                        code_start = response.find('execute_code') + len('execute_code')
                        action_params = response[code_start:].strip()
                    else:
                        action_params = "# No code provided"
            elif 'complete_task' in response or 'completed' in response.lower():
                action = 'complete_task'
                action_params = response
            else:
                # For computational tasks, default to execute_code if no clear action
                if any(char.isdigit() for char in response):
                    action = 'execute_code' 
                    action_params = f"# Task: {response[:100]}\nresult = 'Please provide Python code for this calculation'"
                else:
                    action = 'analyze_error'
                    action_params = response
            
            thought = thought or (response[:100] + "..." if len(response) > 100 else response)
        
        return thought, action, action_params
    
    def _execute_action(self, action: str, params: str) -> tuple[str, bool]:
        """Execute an action and return observation and success status."""
        if action not in self.available_actions:
            return f"Unknown action: {action}", False
        
        try:
            return self.available_actions[action](params)
        except Exception as e:
            return f"Action execution failed: {str(e)}", False
    
    def _execute_code(self, code: str) -> tuple[str, bool]:
        """Execute Python code and return result."""
        if not code.strip():
            return "No code provided", False
        
        # Show code execution in logs
        code_preview = code.replace('\n', '; ')[:50] + "..." if len(code) > 50 else code.replace('\n', '; ')
        enhanced_logger.code_execution(code_preview, True, "")  # Will update success after execution
        
        result = self.interpreter.run(code, "python")
        success = "Error" not in result and "Traceback" not in result
        
        # Update the log with actual result
        result_preview = result[:50] + "..." if len(result) > 50 else result
        enhanced_logger.code_execution(code_preview, success, result_preview)
        
        return result, success
    
    def _analyze_error(self, error_info: str) -> tuple[str, bool]:
        """Analyze error information to provide insights."""
        analysis = f"Error analysis: {error_info}"
        
        # Simple error pattern detection
        if "import" in error_info.lower() and "not permitted" in error_info.lower():
            analysis += "\nSuggestion: The required module is not in the whitelist. Try using pre-loaded libraries (numpy, pandas, torch, matplotlib)."
        elif "undefined" in error_info.lower() or "not defined" in error_info.lower():
            analysis += "\nSuggestion: Variable or function not defined. Check variable names and ensure proper initialization."
        elif "syntax" in error_info.lower():
            analysis += "\nSuggestion: Syntax error detected. Check for missing colons, parentheses, or indentation issues."
        
        return analysis, True
    
    def _fix_code(self, broken_code: str) -> tuple[str, bool]:
        """Attempt to fix broken code automatically."""
        # Simple code fixes
        fixed_code = broken_code
        
        # Fix common import issues
        if "import numpy" in fixed_code and "np" not in fixed_code:
            fixed_code = fixed_code.replace("import numpy", "# numpy available as np")
        if "import torch" in fixed_code:
            fixed_code = fixed_code.replace("import torch", "# torch available globally")
        
        # Remove prohibited imports
        prohibited_imports = ["import os", "import sys", "import subprocess"]
        for imp in prohibited_imports:
            if imp in fixed_code:
                fixed_code = fixed_code.replace(imp, f"# {imp} - not available")
        
        result = self.interpreter.run(fixed_code, "python")
        success = "Error" not in result
        
        return f"Fixed code executed: {result}", success
    
    def _complete_task(self, final_answer: str) -> tuple[str, bool]:
        """Mark task as completed with final answer."""
        return final_answer, True