"""
Command Line Interface for HealthFlow

Provides command-line tools for interacting with the HealthFlow system.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .core.agent import HealthFlowAgent


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HealthFlow: Self-Evolving LLM Agent for Healthcare"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create agent command
    create_parser = subparsers.add_parser('create', help='Create a new HealthFlow agent')
    create_parser.add_argument('--agent-id', required=True, help='Unique agent identifier')
    create_parser.add_argument('--domains', nargs='*', help='Specialized domains')
    create_parser.add_argument('--config', help='Configuration file path')
    
    # Execute task command
    execute_parser = subparsers.add_parser('execute', help='Execute a healthcare task')
    execute_parser.add_argument('--agent-id', required=True, help='Agent identifier')
    execute_parser.add_argument('--task', required=True, help='Task description')
    execute_parser.add_argument('--type', required=True, help='Task type')
    execute_parser.add_argument('--data', help='Input data file (JSON)')
    execute_parser.add_argument('--output', help='Output file path')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show agent statistics')
    stats_parser.add_argument('--agent-id', required=True, help='Agent identifier')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration tasks')
    demo_parser.add_argument('--scenario', choices=['diagnosis', 'analysis', 'all'], 
                           default='all', help='Demo scenario to run')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    if args.command == 'create':
        asyncio.run(create_agent(args, api_key))
    elif args.command == 'execute':
        asyncio.run(execute_task(args, api_key))
    elif args.command == 'stats':
        asyncio.run(show_stats(args, api_key))
    elif args.command == 'demo':
        asyncio.run(run_demo(args, api_key))


async def create_agent(args, api_key: str):
    """Create a new HealthFlow agent"""
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    agent = HealthFlowAgent(
        agent_id=args.agent_id,
        openai_api_key=api_key,
        specialized_domains=args.domains,
        config=config
    )
    
    print(f"✓ Created HealthFlow agent: {args.agent_id}")
    if args.domains:
        print("  Specialized domains: " + ', '.join(args.domains))
    
    # Save agent configuration
    agent_config = {
        "agent_id": args.agent_id,
        "specialized_domains": args.domains or [],
        "config": config,
        "created_at": agent.state.timestamp.isoformat() if hasattr(agent.state, 'timestamp') else None
    }
    
    config_dir = Path("agents")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / f"{args.agent_id}.json"
    with open(config_file, 'w') as f:
        json.dump(agent_config, f, indent=2)
    
    print(f"  Configuration saved to: {config_file}")


async def execute_task(args, api_key: str):
    """Execute a healthcare task"""
    
    # Load agent configuration
    config_file = Path("agents") / f"{args.agent_id}.json"
    if not config_file.exists():
        print(f"Error: Agent {args.agent_id} not found. Create it first with 'healthflow create'")
        return
    
    with open(config_file, 'r') as f:
        agent_config = json.load(f)
    
    # Create agent
    agent = HealthFlowAgent(
        agent_id=args.agent_id,
        openai_api_key=api_key,
        specialized_domains=agent_config.get('specialized_domains', []),
        config=agent_config.get('config', {})
    )
    
    # Load input data if provided
    data = None
    if args.data and Path(args.data).exists():
        with open(args.data, 'r') as f:
            data = json.load(f)
    
    print(f"🚀 Executing task with agent {args.agent_id}")
    print(f"   Task: {args.task}")
    print(f"   Type: {args.type}")
    
    # Execute task
    try:
        result = await agent.execute_task(
            task_description=args.task,
            task_type=args.type,
            data=data
        )
        
        print(f"\n✓ Task completed successfully!")
        print(f"   Success: {result.success}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        print(f"   Tools used: {len(result.tools_used)}")
        
        if result.feedback:
            print(f"   Feedback: {result.feedback}")
        
        # Save output if requested
        if args.output:
            output_data = {
                "task_result": {
                    "task_id": result.task_id,
                    "success": result.success,
                    "output": result.output,
                    "feedback": result.feedback,
                    "execution_time": result.execution_time,
                    "tools_used": result.tools_used,
                    "timestamp": result.timestamp.isoformat()
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"   Output saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Task execution failed: {e}")


async def show_stats(args, api_key: str):
    """Show agent statistics"""
    
    config_file = Path("agents") / f"{args.agent_id}.json"
    if not config_file.exists():
        print(f"Error: Agent {args.agent_id} not found")
        return
    
    with open(config_file, 'r') as f:
        agent_config = json.load(f)
    
    agent = HealthFlowAgent(
        agent_id=args.agent_id,
        openai_api_key=api_key,
        specialized_domains=agent_config.get('specialized_domains', []),
        config=agent_config.get('config', {})
    )
    
    stats = agent.get_agent_statistics()
    
    print(f"📊 Statistics for agent {args.agent_id}")
    print("=" * 50)
    
    # Agent state
    state = stats['agent_state']
    print(f"Status: {'Active' if state['active'] else 'Inactive'}")
    print(f"Experience Level: {state['experience_level']}")
    print(f"Specialized Domains: {', '.join(state['specialized_domains']) or 'None'}")
    
    # Task history
    history = stats['task_history_summary']
    print(f"\nTask History:")
    print(f"  Total tasks: {history['total_tasks']}")
    print(f"  Successful tasks: {history['successful_tasks']}")
    if history['total_tasks'] > 0:
        success_rate = history['successful_tasks'] / history['total_tasks']
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average execution time: {history['average_execution_time']:.2f}s")
    
    # Performance metrics
    if state['performance_metrics']:
        print(f"\nPerformance Metrics:")
        for metric, value in state['performance_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Memory status
    memory_status = stats['memory_status']
    print(f"\nMemory System:")
    print(f"  Total memories: {memory_status.get('total_memories', 0)}")
    print(f"  Average importance: {memory_status.get('average_importance', 0):.3f}")
    
    # ToolBank status
    toolbank_status = stats['toolbank_status']
    print(f"\nToolBank:")
    print(f"  Available tools: {toolbank_status.get('total_tools', 0)}")
    print(f"  Custom tools created: {toolbank_status.get('custom_tools_created', 0)}")


async def run_demo(args, api_key: str):
    """Run demonstration scenarios"""
    
    agent_id = "demo_agent"
    
    # Create demo agent
    agent = HealthFlowAgent(
        agent_id=agent_id,
        openai_api_key=api_key,
        specialized_domains=["diagnosis", "analysis", "clinical_decision_support"]
    )
    
    print("🔬 HealthFlow Demonstration")
    print("=" * 50)
    
    demo_tasks = []
    
    if args.scenario in ['diagnosis', 'all']:
        demo_tasks.extend([
            {
                "description": "Analyze symptoms: fever, cough, shortness of breath, and provide differential diagnosis",
                "type": "diagnosis",
                "data": {
                    "symptoms": ["fever", "cough", "shortness_of_breath"],
                    "patient_age": 45,
                    "patient_sex": "M"
                }
            },
            {
                "description": "Review lab results and suggest follow-up tests",
                "type": "analysis", 
                "data": {
                    "lab_results": {
                        "WBC": 12000,
                        "hemoglobin": 10.5,
                        "glucose": 180
                    }
                }
            }
        ])
    
    if args.scenario in ['analysis', 'all']:
        demo_tasks.extend([
            {
                "description": "Analyze medication interactions for a patient on multiple drugs",
                "type": "drug_interaction",
                "data": {
                    "medications": ["warfarin", "aspirin", "lisinopril", "metformin"]
                }
            }
        ])
    
    for i, task in enumerate(demo_tasks, 1):
        print(f"\n📋 Demo Task {i}: {task['type']}")
        print(f"   {task['description']}")
        
        try:
            result = await agent.execute_task(
                task_description=task['description'],
                task_type=task['type'],
                data=task.get('data')
            )
            
            print(f"   ✓ Success: {result.success}")
            print(f"   ⏱️  Time: {result.execution_time:.2f}s")
            
            if isinstance(result.output, dict) and 'response' in result.output:
                response = result.output['response'][:200]
                print(f"   💬 Response: {response}...")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Show final statistics
    print(f"\n📊 Demo Complete - Agent Statistics:")
    stats = agent.get_agent_statistics()
    history = stats['task_history_summary']
    print(f"   Tasks completed: {history['total_tasks']}")
    print(f"   Success rate: {history['successful_tasks']}/{history['total_tasks']}")
    
    print(f"\n🎉 HealthFlow demonstration completed!")
    print(f"   Agent '{agent_id}' is ready for more tasks.")


if __name__ == "__main__":
    main()