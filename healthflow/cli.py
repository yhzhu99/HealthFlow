"""
HealthFlow Command Line Interface
Main entry point for the HealthFlow healthcare agent system
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import sys

from .core.config import HealthFlowConfig
from .core.agent import HealthFlowAgent, AgentRole
from .core.llm_provider import create_llm_provider
from .evaluation.evaluator import TaskEvaluator


class HealthFlowCLI:
    """Command Line Interface for HealthFlow"""

    def __init__(self):
        self.config: Optional[HealthFlowConfig] = None
        self.agents: Dict[str, HealthFlowAgent] = {}
        self.coordinator_agent: Optional[HealthFlowAgent] = None
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("HealthFlowCLI")

    async def initialize(self, config_path: str = "config.toml"):
        """Initialize HealthFlow system"""
        self.logger.info("Initializing HealthFlow system...")

        # Load configuration
        try:
            self.config = HealthFlowConfig.from_toml(config_path)
            self.config.validate()

            # Update logging level
            logging.getLogger().setLevel(self.config.log_level)

        except Exception as e:
            self.logger.error(f"Configuration error: {e}")
            sys.exit(1)

        # Create multi-agent system
        await self._create_agent_network()

        self.logger.info("HealthFlow system initialized successfully!")

    async def _create_agent_network(self):
        """Create and connect the multi-agent network"""

        # Define agent configurations
        agent_configs = [
            ("coordinator", AgentRole.COORDINATOR),
            ("medical_expert", AgentRole.MEDICAL_EXPERT),
            ("data_analyst", AgentRole.DATA_ANALYST),
            ("researcher", AgentRole.RESEARCHER),
            ("diagnosis_specialist", AgentRole.DIAGNOSIS_SPECIALIST),
            ("treatment_planner", AgentRole.TREATMENT_PLANNER),
            ("code_executor", AgentRole.CODE_EXECUTOR)
        ]

        # Create agents
        for agent_id, role in agent_configs:
            try:
                agent = HealthFlowAgent(
                    agent_id=agent_id,
                    role=role,
                    config=self.config
                )
                await agent.initialize()
                self.agents[agent_id] = agent

                if role == AgentRole.COORDINATOR:
                    self.coordinator_agent = agent

            except Exception as e:
                self.logger.error(f"Failed to create agent {agent_id}: {e}")

        # Connect agents in collaboration network
        for agent in self.agents.values():
            for other_agent in self.agents.values():
                if agent.agent_id != other_agent.agent_id:
                    agent.add_collaborator(other_agent)

        self.logger.info(f"Created {len(self.agents)} agents in collaboration network")

    async def execute_task(
        self,
        task_description: str,
        task_context: Dict[str, Any] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a healthcare task using the agent system"""

        if not self.agents:
            raise RuntimeError("Agents not initialized. Call initialize() first.")

        # Use coordinator agent by default
        executing_agent = self.agents.get(agent_id, self.coordinator_agent)
        if not executing_agent:
            raise ValueError(f"Agent {agent_id} not found")

        self.logger.info(f"Executing task with {executing_agent.role.value}: {task_description[:50]}...")

        # Execute task
        task_result = await executing_agent.execute_task(
            task_description=task_description,
            task_context=task_context or {}
        )

        # Format response
        response = {
            "task_id": task_result.task_id,
            "success": task_result.success,
            "result": task_result.result,
            "agent_used": executing_agent.agent_id,
            "agent_role": executing_agent.role.value,
            "execution_time": task_result.execution_time,
            "tools_used": task_result.tools_used,
            "evaluation": task_result.evaluation
        }

        if task_result.error:
            response["error"] = task_result.error

        self.logger.info(f"Task {'completed' if task_result.success else 'failed'} in {task_result.execution_time:.2f}s")

        return response

    async def load_tasks_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load tasks from JSONL file"""
        tasks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            task = json.loads(line)
                            tasks.append(task)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                            continue
        except Exception as e:
            self.logger.error(f"Error loading tasks from {file_path}: {e}")
            return []

        self.logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
        return tasks

    async def run_task_file(self, file_path: Path, max_tasks: Optional[int] = None):
        """Run tasks from a JSONL file"""

        tasks = await self.load_tasks_from_file(file_path)

        if max_tasks:
            tasks = tasks[:max_tasks]

        results = []
        successful = 0

        for i, task in enumerate(tasks, 1):
            self.logger.info(f"Processing task {i}/{len(tasks)}")

            try:
                # Extract task information
                task_description = task.get('task', task.get('description', ''))
                task_context = task.get('context', {})

                if not task_description:
                    self.logger.warning(f"Task {i} has no description, skipping")
                    continue

                # Execute task
                result = await self.execute_task(task_description, task_context)
                results.append(result)

                if result['success']:
                    successful += 1

            except Exception as e:
                self.logger.error(f"Error processing task {i}: {e}")
                results.append({
                    "task_id": f"error_{i}",
                    "success": False,
                    "error": str(e),
                    "task_description": task.get('task', 'Unknown')
                })

        success_rate = successful / len(results) if results else 0
        self.logger.info(f"Completed {len(results)} tasks. Success rate: {success_rate:.2%}")

        return results

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        if not self.agents:
            return {"status": "not_initialized", "agents": 0}

        agent_statuses = {}
        total_tasks = 0
        total_successful = 0

        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            agent_statuses[agent_id] = status
            total_tasks += status['total_tasks']
            total_successful += status['successful_tasks']

        # Get evaluator statistics
        evaluator = TaskEvaluator()
        eval_stats = evaluator.get_evaluation_statistics()

        return {
            "status": "active",
            "total_agents": len(self.agents),
            "agent_statuses": agent_statuses,
            "system_performance": {
                "total_tasks": total_tasks,
                "successful_tasks": total_successful,
                "success_rate": total_successful / total_tasks if total_tasks > 0 else 0
            },
            "evaluation_statistics": eval_stats
        }

    async def interactive_mode(self):
        """Run HealthFlow in interactive mode"""

        self.logger.info("Starting HealthFlow interactive mode...")
        self.logger.info("Type 'help' for commands, 'exit' to quit")

        while True:
            try:
                user_input = input("\nHealthFlow> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break

                elif user_input.lower() == 'help':
                    self._print_help()

                elif user_input.lower() == 'status':
                    status = await self.get_system_status()
                    print(json.dumps(status, indent=2))

                elif user_input.lower().startswith('agent '):
                    # Switch to specific agent
                    agent_name = user_input[6:].strip()
                    if agent_name in self.agents:
                        print(f"Switched to agent: {agent_name}")
                        # Could implement agent-specific interaction here
                    else:
                        print(f"Agent '{agent_name}' not found")

                else:
                    # Execute as task
                    print("Processing your request...")
                    result = await self.execute_task(user_input)

                    print(f"\nTask {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
                    print(f"Agent: {result['agent_role']}")
                    print(f"Time: {result['execution_time']:.2f}s")

                    if result.get('tools_used'):
                        print(f"Tools: {', '.join(result['tools_used'])}")

                    print(f"\nResult:\n{result['result']}")

                    if result.get('error'):
                        print(f"\nError: {result['error']}")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _print_help(self):
        """Print help message"""
        print("""
HealthFlow Commands:
  help           - Show this help message
  status         - Show system status
  agent <name>   - Switch to specific agent
  exit/quit      - Exit HealthFlow

  Or simply type any medical question or task to execute it.

Available agents:
  - coordinator         - Orchestrates multi-agent tasks
  - medical_expert     - Provides medical expertise
  - data_analyst       - Analyzes medical data
  - researcher         - Conducts literature research
  - diagnosis_specialist - Specializes in diagnosis
  - treatment_planner  - Plans treatments
  - code_executor      - Executes code and analysis
""")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HealthFlow Healthcare Agent System")
    parser.add_argument('--config', '-c', default="config.toml", help='Path to configuration file')
    parser.add_argument('--task', '-t', help='Single task to execute')
    parser.add_argument('--file', '-f', help='JSONL file with tasks to execute')
    parser.add_argument('--max-tasks', type=int, help='Maximum tasks to process from file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--status', '-s', action='store_true', help='Show system status')

    args = parser.parse_args()

    # Create CLI instance
    cli = HealthFlowCLI()

    try:
        # Initialize system
        await cli.initialize(args.config)

        # Execute based on arguments
        if args.status:
            status = await cli.get_system_status()
            print(json.dumps(status, indent=2))

        elif args.task:
            result = await cli.execute_task(args.task)
            print(json.dumps(result, indent=2))

        elif args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File {file_path} not found")
                return

            results = await cli.run_task_file(file_path, args.max_tasks)

            # Save results
            output_file = file_path.parent / f"{file_path.stem}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Results saved to: {output_file}")

        elif args.interactive:
            await cli.interactive_mode()

        else:
            # Default to interactive mode
            await cli.interactive_mode()

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())