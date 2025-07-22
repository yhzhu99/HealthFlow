"""
Example usage of MCP (Model Context Protocol) integration in HealthFlow

This script demonstrates how to use MCP tools with the HealthFlow framework,
including tool discovery, registration, and execution.
"""

import asyncio
import json
from pathlib import Path

# Import HealthFlow components
from healthflow.tools.toolbank import HierarchicalToolBank
from healthflow.tools.mcp_interface import MCPToolSpec, MCPToolType, MCPExecutionMode, create_example_mcp_tool
from healthflow.core.security import DataProtector, ProtectionConfig


async def main():
    """Main demonstration function"""
    
    print("🏥 HealthFlow MCP Integration Demo")
    print("=" * 50)
    
    # Create ToolBank with MCP support
    tools_dir = Path("./demo_tools")
    toolbank = HierarchicalToolBank(tools_dir)
    
    # Initialize the ToolBank (this will auto-discover MCP tools)
    print("\n📚 Initializing ToolBank with MCP support...")
    await toolbank.initialize()
    
    # Auto-discover MCP tools from the mcps directory
    print("\n🔍 Discovering MCP tools...")
    discovered_tools = await toolbank.discover_mcp_tools()
    print(f"Discovered {len(discovered_tools)} MCP tools: {discovered_tools}")
    
    # Manually register the example MCP tool
    print("\n📝 Registering example MCP tool...")
    example_tool = create_example_mcp_tool()
    tool_id = await toolbank.register_mcp_tool(example_tool)
    print(f"Registered example tool with ID: {tool_id}")
    
    # List all MCP tools
    print("\n📋 Listing all MCP tools...")
    mcp_tools = await toolbank.list_mcp_tools()
    for tool in mcp_tools:
        print(f"  • {tool['name']}: {tool['description']}")
    
    # Test tool execution with example medical data
    print("\n🧪 Testing MCP tool execution...")
    
    # Example 1: Medical data analysis
    if discovered_tools:
        test_data = {
            "data_type": "vitals",
            "data": {
                "blood_pressure": "140/90",
                "heart_rate": 85,
                "temperature": 98.6,
                "respiratory_rate": 16
            },
            "analysis_type": "comprehensive"
        }
        
        print(f"Testing medical data analyzer with: {test_data}")
        try:
            result = await toolbank.execute_mcp_tool(
                tool_id=discovered_tools[0],  # Use first discovered tool
                input_data=test_data,
                context={"patient_id": "demo_patient_001"}
            )
            
            print(f"✅ Analysis Result:")
            print(f"  Success: {result.success}")
            print(f"  Execution Time: {result.execution_time:.3f}s")
            if result.success and result.result:
                analysis = result.result
                print(f"  Risk Indicators: {len(analysis.get('risk_indicators', []))}")
                print(f"  Recommendations: {len(analysis.get('recommendations', []))}")
                print(f"  Confidence Score: {analysis.get('confidence_score', 0):.2f}")
            if result.security_warnings:
                print(f"  Security Warnings: {len(result.security_warnings)}")
                
        except Exception as e:
            print(f"❌ Error executing MCP tool: {e}")
    
    # Example 2: Example vitals analyzer
    print(f"\n🫀 Testing example vitals analyzer...")
    vitals_data = {
        "vitals": {
            "blood_pressure": "160/100",  # High BP
            "heart_rate": 110,  # Tachycardia
            "temperature": 99.8,
            "respiratory_rate": 22
        }
    }
    
    try:
        result = await toolbank.execute_mcp_tool(
            tool_id=tool_id,  # Use the manually registered example tool
            input_data=vitals_data,
            context={"urgency": "high"}
        )
        
        print(f"✅ Vitals Analysis Result:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Analysis: {result.result}")
        else:
            print(f"  Error: {result.error}")
            
    except Exception as e:
        print(f"❌ Error executing vitals analyzer: {e}")
    
    # Search for tools including MCP tools
    print(f"\n🔎 Searching for medical analysis tools...")
    search_results = await toolbank.search_tools_with_mcp(
        query="medical",
        tags=["analysis"],
        include_mcp=True
    )
    
    print(f"Found {len(search_results)} tools:")
    for tool in search_results[:3]:  # Show first 3
        print(f"  • {tool.metadata.name} ({tool.metadata.tool_type.value})")
        print(f"    Success Rate: {tool.metadata.success_rate:.1%}")
        print(f"    Usage Count: {tool.metadata.usage_count}")
    
    # Get MCP statistics
    print(f"\n📊 MCP Tool Statistics:")
    mcp_stats = await toolbank.get_mcp_statistics()
    if "message" not in mcp_stats:
        print(f"  Total Executions: {mcp_stats.get('total_executions', 0)}")
        print(f"  Success Rate: {mcp_stats.get('success_rate', 0):.1%}")
        print(f"  Registered Tools: {mcp_stats.get('total_registered_tools', 0)}")
        print(f"  Average Execution Time: {mcp_stats.get('average_execution_time', 0):.3f}s")
    else:
        print(f"  {mcp_stats['message']}")
    
    # Demonstrate data protection with MCP tools
    print(f"\n🔒 Testing data protection with sensitive medical data...")
    sensitive_data = {
        "data_type": "lab_results",
        "data": {
            "patient_id": "P12345",
            "name": "John Doe",
            "ssn": "123-45-6789",
            "glucose": 180,  # High glucose
            "cholesterol": 250,  # High cholesterol
            "diagnosis": "Type 2 Diabetes"
        }
    }
    
    if discovered_tools:
        try:
            result = await toolbank.execute_mcp_tool(
                tool_id=discovered_tools[0],
                input_data=sensitive_data,
                context={"protect_sensitive_data": True}
            )
            
            print(f"✅ Protected Analysis Result:")
            print(f"  Success: {result.success}")
            print(f"  Security Warnings: {len(result.security_warnings)}")
            for warning in result.security_warnings:
                print(f"    ⚠️  {warning}")
                
        except Exception as e:
            print(f"❌ Error with sensitive data: {e}")
    
    print(f"\n🎉 MCP Integration Demo Complete!")
    print(f"The HealthFlow framework now supports:")
    print(f"  • Model Context Protocol (MCP) tool integration")
    print(f"  • Automatic tool discovery and registration")
    print(f"  • Secure execution with data protection")
    print(f"  • Performance monitoring and statistics")
    print(f"  • Unified search across all tool types")


if __name__ == "__main__":
    asyncio.run(main())