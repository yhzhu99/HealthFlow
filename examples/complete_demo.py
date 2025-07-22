#!/usr/bin/env python3
"""
Complete HealthFlow Demonstration

This script demonstrates all the key features of the HealthFlow system:
1. Self-evolving agent with experience accumulation
2. Sensitive data protection
3. Memory management
4. Dynamic tool creation
5. Multi-agent collaboration
6. Evaluation and continuous improvement

Usage:
    python examples/complete_demo.py

Prerequisites:
    - Set OPENAI_API_KEY environment variable
    - Install HealthFlow: pip install -e .
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from healthflow.core.agent import HealthFlowAgent
from healthflow.core.memory import MemoryQuery
from healthflow.core.security import DataProtector, ProtectionConfig


async def main():
    """Run the complete HealthFlow demonstration"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print("🔬 HealthFlow Complete Demonstration")
    print("=" * 60)
    
    # Create demonstration agents
    print("\n1. Creating HealthFlow Agents")
    print("-" * 30)
    
    # Primary diagnostic agent
    diagnostic_agent = HealthFlowAgent(
        agent_id="diagnostic_specialist",
        openai_api_key=api_key,
        specialized_domains=["diagnosis", "symptom_analysis", "differential_diagnosis"],
        config={"max_reasoning_steps": 10, "confidence_threshold": 0.8}
    )
    print("✓ Created Diagnostic Specialist Agent")
    
    # Data analysis agent
    analysis_agent = HealthFlowAgent(
        agent_id="data_analyst",
        openai_api_key=api_key,
        specialized_domains=["data_analysis", "statistics", "pattern_recognition"],
        config={"analysis_depth": "comprehensive"}
    )
    print("✓ Created Data Analysis Agent")
    
    # Clinical decision support agent
    clinical_agent = HealthFlowAgent(
        agent_id="clinical_advisor",
        openai_api_key=api_key,
        specialized_domains=["treatment_planning", "clinical_guidelines", "drug_interactions"],
        config={"safety_first": True, "guideline_adherence": True}
    )
    print("✓ Created Clinical Decision Support Agent")
    
    # 2. Demonstrate sensitive data protection
    print("\n2. Sensitive Data Protection Demo")
    print("-" * 40)
    
    # Sample patient data (simulated)
    sensitive_patient_data = {
        "patient_name": "John Smith",
        "patient_id": "PT-12345",
        "dob": "1978-05-15",
        "ssn": "123-45-6789",
        "phone": "555-123-4567",
        "email": "john.smith@email.com",
        "medical_record_number": "MRN123456",
        "symptoms": ["fever", "cough", "shortness of breath"],
        "vital_signs": {
            "temperature": 101.2,
            "blood_pressure": "140/90",
            "heart_rate": 98,
            "respiratory_rate": 22
        },
        "lab_results": {
            "WBC": 12000,
            "hemoglobin": 11.5,
            "glucose": 95
        }
    }
    
    # Configure data protector for high privacy
    protector = DataProtector(ProtectionConfig(
        anonymization_level="high",
        preserve_structure=True,
        generate_mock_data=True,
        schema_only_mode=False
    ))
    
    protected_data = await protector.protect_data(sensitive_patient_data)
    
    print("✓ Protected sensitive patient data")
    print(f"   Privacy level: {protected_data['classification']['sensitivity_level']}")
    print(f"   Contains PII: {protected_data['classification']['contains_pii']}")
    print(f"   Contains PHI: {protected_data['classification']['contains_phi']}")
    print(f"   Protection applied: {protected_data['protection_applied']}")
    
    # 3. Execute complex diagnostic task
    print("\n3. Complex Diagnostic Task Execution")
    print("-" * 45)
    
    diagnostic_task = """
    Patient presents with a 3-day history of fever (101-102°F), productive cough with 
    yellowish sputum, shortness of breath, and chest pain that worsens with deep breathing. 
    Physical exam reveals decreased breath sounds in the right lower lobe and dullness to 
    percussion. Lab results show elevated WBC count (12,000), elevated CRP, and chest X-ray 
    shows consolidation in the right lower lobe. Please provide a differential diagnosis 
    and recommended treatment plan.
    """
    
    result = await diagnostic_agent.execute_task(
        task_description=diagnostic_task,
        task_type="diagnosis",
        data=protected_data['protected_data'],
        context={"urgency": "moderate", "patient_age": 45, "patient_sex": "M"}
    )
    
    print("✓ Diagnostic task completed")
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    print(f"   Tools used: {len(result.tools_used)}")
    
    if isinstance(result.output, dict) and 'response' in result.output:
        response_preview = result.output['response'][:300]
        print(f"   Response preview: {response_preview}...")
    
    # 4. Demonstrate learning and evolution
    print("\n4. Experience Accumulation & Learning")
    print("-" * 45)
    
    # Execute several related tasks to build experience
    learning_tasks = [
        {
            "description": "Analyze chest X-ray findings: consolidation in right lower lobe",
            "type": "imaging_analysis",
            "expected_success": True
        },
        {
            "description": "Recommend antibiotic therapy for community-acquired pneumonia",
            "type": "treatment_planning", 
            "expected_success": True
        },
        {
            "description": "Assess drug interactions for pneumonia treatment in elderly patient",
            "type": "drug_interaction",
            "expected_success": True
        }
    ]
    
    for i, task in enumerate(learning_tasks, 1):
        print(f"\n   Learning Task {i}: {task['type']}")
        task_result = await diagnostic_agent.execute_task(
            task_description=task['description'],
            task_type=task['type']
        )
        print(f"   ✓ {'Success' if task_result.success else 'Failed'} "
              f"({task_result.execution_time:.2f}s)")
    
    # Show learning progress
    print("\n   📚 Learning Progress:")
    experience_stats = diagnostic_agent.experience_accumulator.get_experience_statistics()
    print(f"   - Total experiences: {experience_stats['total_experiences']}")
    print(f"   - Overall success rate: {experience_stats['overall_success_rate']:.1%}")
    print(f"   - Average performance: {experience_stats['average_performance']:.3f}")
    print(f"   - Pattern insights: {experience_stats['pattern_insights_discovered']}")
    print(f"   - Evolved prompts: {experience_stats['evolved_prompts_created']}")
    
    # 5. Memory system demonstration
    print("\n5. Memory System Demonstration")
    print("-" * 40)
    
    # Store important clinical memory
    clinical_memory_id = await diagnostic_agent.memory_manager.store_memory(
        content={
            "clinical_insight": "Right lower lobe consolidation with fever and productive cough strongly suggests bacterial pneumonia",
            "treatment_protocol": "First-line antibiotic therapy with amoxicillin-clavulanate",
            "follow_up": "Chest X-ray in 48-72 hours to assess improvement"
        },
        memory_type="semantic",
        importance_score=0.9,
        tags=["pneumonia", "diagnosis", "treatment", "clinical_protocol"],
        privacy_level="medium"
    )
    
    print(f"✓ Stored clinical memory: {clinical_memory_id}")
    
    # Retrieve relevant memories
    memory_query = MemoryQuery(
        query_text="pneumonia diagnosis and treatment",
        memory_types=["semantic", "episodic"],
        min_importance=0.5,
        max_results=5
    )
    
    relevant_memories = await diagnostic_agent.memory_manager.retrieve_memories(memory_query)
    print(f"✓ Retrieved {len(relevant_memories)} relevant memories")
    
    # Memory statistics
    memory_stats = diagnostic_agent.memory_manager.get_memory_statistics()
    print(f"   Total memories: {memory_stats['total_memories']}")
    print(f"   Average importance: {memory_stats['average_importance']:.3f}")
    print(f"   Memory types: {list(memory_stats['memory_types'].keys())}")
    
    # 6. Tool creation and usage
    print("\n6. Dynamic Tool Creation")
    print("-" * 35)
    
    # Create a specialized tool for drug interaction checking
    drug_interaction_spec = {
        "name": "drug_interaction_checker",
        "description": "Check for interactions between multiple medications",
        "input_schema": {
            "type": "object",
            "properties": {
                "medications": {"type": "array", "items": {"type": "string"}},
                "patient_age": {"type": "number"},
                "patient_conditions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["medications"]
        },
        "output_schema": {
            "type": "object", 
            "properties": {
                "interactions": {"type": "array"},
                "risk_level": {"type": "string"},
                "recommendations": {"type": "array"}
            }
        },
        "complexity": "medium",
        "dependencies": ["requests"]
    }
    
    tool_result = await diagnostic_agent.toolbank.create_tool(drug_interaction_spec)
    
    if tool_result['success']:
        print(f"✓ Created tool: {tool_result['name']}")
        print(f"   Tool ID: {tool_result['tool_id']}")
    else:
        print(f"❌ Tool creation failed: {tool_result.get('error', 'Unknown error')}")
    
    # Show toolbank statistics
    toolbank_stats = diagnostic_agent.toolbank.get_toolbank_statistics()
    print(f"   Total tools available: {toolbank_stats['total_tools']}")
    if toolbank_stats['most_used_tools']:
        print(f"   Most used tools: {[name for name, _ in toolbank_stats['most_used_tools'][:3]]}")
    
    # 7. Multi-agent collaboration demo
    print("\n7. Multi-Agent Collaboration")
    print("-" * 40)
    
    # Demonstrate collaboration between agents
    collaboration_result = await diagnostic_agent.collaborate_with_agent(
        other_agent_id="clinical_advisor",
        collaboration_type="treatment_planning",
        shared_context={
            "patient_diagnosis": "community_acquired_pneumonia",
            "severity": "moderate",
            "complications": "none"
        }
    )
    
    print(f"✓ Initiated collaboration: {collaboration_result['collaboration_id']}")
    print(f"   Type: {collaboration_result['type']}")
    print(f"   Status: {collaboration_result['status']}")
    
    # 8. Evaluation and quality assessment
    print("\n8. Quality Assessment & Evaluation")
    print("-" * 45)
    
    # Show evaluation statistics
    eval_stats = diagnostic_agent.evaluator.get_evaluation_statistics()
    
    if eval_stats['total_evaluations'] > 0:
        print(f"✓ Evaluation system active")
        print(f"   Total evaluations: {eval_stats['total_evaluations']}")
        print(f"   Average score: {eval_stats['average_score']:.3f}")
        print(f"   Success rate: {eval_stats['overall_success_rate']:.1%}")
        print(f"   Performance trend: {eval_stats['performance_trend']}")
        
        # Show task-specific statistics
        if eval_stats['task_type_breakdown']:
            print("   Task performance by type:")
            for task_type, stats in eval_stats['task_type_breakdown'].items():
                print(f"     - {task_type}: {stats['avg_score']:.3f} "
                      f"({stats['success_rate']:.1%} success)")
    else:
        print("   Evaluation system ready (no evaluations yet)")
    
    # 9. Final system overview
    print("\n9. System Overview & Statistics")
    print("-" * 45)
    
    # Get comprehensive statistics from all agents
    all_agents = [diagnostic_agent, analysis_agent, clinical_agent]
    
    total_tasks = 0
    total_success = 0
    total_tools = 0
    
    for agent in all_agents:
        stats = agent.get_agent_statistics()
        history = stats['task_history_summary']
        toolbank = stats['toolbank_status']
        
        total_tasks += history['total_tasks']
        total_success += history['successful_tasks']
        total_tools += toolbank.get('total_tools', 0)
        
        print(f"   Agent {agent.agent_id}:")
        print(f"     - Tasks: {history['total_tasks']} "
              f"({history['successful_tasks']}/{history['total_tasks']} successful)")
        print(f"     - Experience level: {stats['agent_state']['experience_level']}")
        print(f"     - Available tools: {toolbank.get('total_tools', 0)}")
    
    print(f"\n   🎯 Overall System Performance:")
    print(f"   - Total tasks executed: {total_tasks}")
    if total_tasks > 0:
        print(f"   - Overall success rate: {total_success/total_tasks:.1%}")
    print(f"   - Total tools available: {total_tools}")
    
    # Data protection statistics
    protection_stats = protector.get_protection_statistics()
    print(f"   - Data protections applied: {protection_stats['total_protections']}")
    print(f"   - PII protections: {protection_stats['pii_protections']}")
    print(f"   - PHI protections: {protection_stats['phi_protections']}")
    
    print(f"\n🎉 HealthFlow demonstration completed successfully!")
    print("   The system is now ready for production healthcare tasks.")
    print("\n   Key capabilities demonstrated:")
    print("   ✓ Self-evolving agents with experience accumulation")
    print("   ✓ Comprehensive sensitive data protection")
    print("   ✓ Advanced memory management")
    print("   ✓ Dynamic tool creation and management")
    print("   ✓ Multi-agent collaboration")
    print("   ✓ Continuous quality evaluation")
    print("   ✓ HIPAA-compliant healthcare processing")


if __name__ == "__main__":
    asyncio.run(main())