#!/usr/bin/env python
"""Test script for InferenceMAX MCP Server."""

import asyncio
import sys
from pathlib import Path
import importlib.util

# Load the server module directly
server_path = Path(__file__).parent / '.claude' / 'mcp' / 'server.py'
spec = importlib.util.spec_from_file_location('mcp_server', server_path)
mcp_server = importlib.util.module_from_spec(spec)
sys.modules['mcp_server'] = mcp_server
spec.loader.exec_module(mcp_server)

InferenceMAXMCPServer = mcp_server.InferenceMAXMCPServer

async def test_initialization():
    """Test MCP server initialization."""
    print("=" * 60)
    print("Testing InferenceMAX MCP Server")
    print("=" * 60)

    try:
        print("\n1. Creating server instance...")
        server = InferenceMAXMCPServer()

        print("2. Initializing (this will clone repos, may take 2-3 minutes)...")
        await server.initialize()

        print("\n" + "=" * 60)
        print("✓ Initialization complete!")
        print("=" * 60)

        print(f"\nvLLM resources: {len(server.resource_cache.get('vllm', []))} files")
        print(f"SGLang resources: {len(server.resource_cache.get('sglang', []))} files")

        # Show sample resources
        print("\nSample vLLM files:")
        vllm_files = server.resource_cache.get('vllm', [])
        for file in vllm_files[:5]:
            print(f"  - {file.relative_to(file.parent.parent.parent)}")

        print("\nSample SGLang files:")
        sglang_files = server.resource_cache.get('sglang', [])
        for file in sglang_files[:5]:
            print(f"  - {file.relative_to(file.parent.parent.parent)}")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_initialization())
