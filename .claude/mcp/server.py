"""MCP Server for vLLM and SGLang repository access."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import timezone
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent

# Import utility functions - use absolute import path
try:
    # Try relative import first (when run as module)
    from .mcp_utils import (
        get_config_paths,
        get_clone_dir,
        detect_versions,
        select_primary_version,
        initialize_repo,
        checkout_version,
        list_filtered_files,
        REPO_URLS,
    )
except ImportError:
    # Fall back to direct import (when run as script)
    import importlib.util
    mcp_utils_path = Path(__file__).parent / 'mcp_utils.py'
    spec = importlib.util.spec_from_file_location('mcp_utils', mcp_utils_path)
    mcp_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcp_utils)

    get_config_paths = mcp_utils.get_config_paths
    get_clone_dir = mcp_utils.get_clone_dir
    detect_versions = mcp_utils.detect_versions
    select_primary_version = mcp_utils.select_primary_version
    initialize_repo = mcp_utils.initialize_repo
    checkout_version = mcp_utils.checkout_version
    list_filtered_files = mcp_utils.list_filtered_files
    REPO_URLS = mcp_utils.REPO_URLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,  # Log to stderr so it doesn't interfere with stdio MCP communication
)
logger = logging.getLogger(__name__)


class InferenceXMCPServer:
    """MCP Server for vLLM and SGLang source code access."""

    def __init__(self):
        self.server = Server("inferencemax-repos")
        self.repos = {}  # {framework: git.Repo}
        self.resource_cache = {}  # {framework: List[Path]}

        # Register MCP handlers
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return await self.list_resources()

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            return await self.read_resource(uri)

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return await self.list_tools()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
            return await self.call_tool(name, arguments)

    async def initialize(self):
        """Initialize repositories and detect versions."""
        logger.info("Initializing InferenceX MCP Server...")

        try:
            # Detect versions from config files
            config_paths = get_config_paths()
            logger.info(f"Reading configs from: {config_paths}")
            versions = detect_versions(config_paths)
            logger.info(f"Detected versions: {versions}")

            # Initialize repositories
            clone_dir = get_clone_dir()
            clone_dir.mkdir(parents=True, exist_ok=True)

            for framework, url in REPO_URLS.items():
                repo_path = clone_dir / framework
                logger.info(f"Initializing {framework} repository...")

                # Clone or update repository
                repo = initialize_repo(framework, url, repo_path)
                self.repos[framework] = repo

                # Checkout appropriate version
                framework_versions = versions.get(framework, set())
                if framework_versions:
                    primary_version = select_primary_version(framework_versions)
                    logger.info(f"Checking out {framework} version {primary_version}")
                    checkout_version(repo, framework, primary_version)
                else:
                    logger.warning(f"No versions detected for {framework}, using default branch")

                # Build resource cache
                logger.info(f"Building resource cache for {framework}...")
                self.resource_cache[framework] = list_filtered_files(repo_path)
                logger.info(f"Found {len(self.resource_cache[framework])} files for {framework}")

            logger.info("Initialization complete!")

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

    async def list_resources(self) -> List[Resource]:
        """List all available MCP resources."""
        resources = []

        for framework, files in self.resource_cache.items():
            repo_path = get_clone_dir() / framework

            for file_path in files:
                try:
                    rel_path = file_path.relative_to(repo_path)
                    uri = f"{framework}:///{rel_path}"

                    resources.append(Resource(
                        uri=uri,
                        name=str(rel_path),
                        mimeType="text/plain",
                        description=f"{framework} source file: {rel_path}",
                    ))
                except Exception as e:
                    logger.warning(f"Error creating resource for {file_path}: {e}")

        logger.info(f"Listed {len(resources)} total resources")
        return resources

    async def read_resource(self, uri: str) -> str:
        """Read a specific resource by URI."""
        # The MCP SDK may pass `uri` as a pydantic AnyUrl in v2 (not a str), so normalize to str
        uri = str(uri)
        logger.info(f"Reading resource: {uri}")

        # Parse URI: vllm:///path/to/file.py
        if uri.startswith('vllm:///'):
            framework = 'vllm'
            rel_path = uri[8:]  # Remove 'vllm:///'
        elif uri.startswith('sglang:///'):
            framework = 'sglang'
            rel_path = uri[10:]  # Remove 'sglang:///'
        else:
            error_msg = f"Unknown URI scheme: {uri}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if framework not in self.repos:
            error_msg = f"Framework not available: {framework}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Construct file path
        repo_path = get_clone_dir() / framework
        file_path = repo_path / rel_path

        # Validate file exists and is in cache
        if not file_path.exists():
            error_msg = f"File not found: {rel_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if file_path not in self.resource_cache.get(framework, []):
            error_msg = f"File is filtered out: {rel_path}"
            logger.error(error_msg)
            raise PermissionError(error_msg)

        # Read and return file contents
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully read {len(content)} bytes from {uri}")
            return content
        except Exception as e:
            error_msg = f"Error reading file {rel_path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg)

    async def list_tools(self) -> List[Tool]:
        """List available MCP tools."""
        return [
            Tool(
                name="switch_version",
                description="Switch to a different version of vLLM or SGLang",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "framework": {
                            "type": "string",
                            "enum": ["vllm", "sglang"],
                            "description": "Framework to switch version for"
                        },
                        "version": {
                            "type": "string",
                            "description": "Version to switch to (e.g., '0.5.7', 'v0.13.0')"
                        }
                    },
                    "required": ["framework", "version"]
                }
            ),
            Tool(
                name="list_versions",
                description="List all detected versions from InferenceX configs",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="list_tags",
                description=(
                    "List git tags for the cloned vLLM or SGLang repo. "
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "framework": {
                            "type": "string",
                            "enum": ["vllm", "sglang"],
                            "description": "Repository: vllm or sglang"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                            "description": "Maximum number of tags to return (default 20)"
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional substring filter for tag names"
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["time", "name", "semver"],
                            "default": "time",
                            "description": "Sort order: by commit time, name, or semantic version"
                        }
                    },
                    "required": ["framework"]
                }
            ),
        ]

    async def call_tool(self, name: str, arguments: dict) -> List[TextContent]:
        """Handle tool calls."""
        logger.info(f"Tool call: {name} with args: {arguments}")

        if name == "switch_version":
            framework = arguments["framework"]
            version = arguments["version"]

            if framework not in self.repos:
                return [TextContent(
                    type="text",
                    text=f"Error: Framework '{framework}' not available"
                )]

            # Checkout version
            repo = self.repos[framework]
            success = checkout_version(repo, framework, version)

            if success:
                # Rebuild resource cache
                repo_path = get_clone_dir() / framework
                logger.info(f"Rebuilding resource cache for {framework}...")
                self.resource_cache[framework] = list_filtered_files(repo_path)
                logger.info(f"Found {len(self.resource_cache[framework])} files after version switch")

                return [TextContent(
                    type="text",
                    text=f"Successfully switched {framework} to version {version}\n"
                         f"Found {len(self.resource_cache[framework])} source files"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Failed to switch {framework} to version {version}\n"
                         f"Check logs for details"
                )]

        elif name == "list_versions":
            config_paths = get_config_paths()
            versions = detect_versions(config_paths)

            result_lines = ["Detected versions from InferenceX configs:", ""]
            for framework, version_set in versions.items():
                if version_set:
                    versions_str = ', '.join(sorted(version_set))
                    result_lines.append(f"  {framework}: {versions_str}")
                else:
                    result_lines.append(f"  {framework}: (none detected)")

            return [TextContent(
                type="text",
                text="\n".join(result_lines)
            )]

        elif name == "list_tags":
            framework = arguments.get("framework")
            if framework not in self.repos:
                return [TextContent(type="text", text=f"Error: Framework '{framework}' not available")]

            repo = self.repos[framework]
            limit = int(arguments.get("limit", 100))
            query = arguments.get("query")
            sort = arguments.get("sort", "time")

            # Gather tag data (name, commit, time)
            tags = []
            try:
                for t in repo.tags:
                    try:
                        commit = t.commit
                        # Normalize timezone to UTC ISO8601 for display
                        dt = commit.committed_datetime
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        tags.append({
                            "name": t.name,
                            "sha": commit.hexsha[:12],
                            "time": dt,
                        })
                    except Exception:
                        # Fallback if commit lookup fails
                        tags.append({"name": t.name, "sha": "", "time": None})
            except Exception as e:
                return [TextContent(type="text", text=f"Error reading tags for {framework}: {e}")]

            # Optional filter
            if query:
                q = str(query).lower()
                tags = [t for t in tags if q in t["name"].lower()]

            # Sorting strategies
            def semver_key(name: str):
                v = name.lstrip('v')
                v = v.replace('rc', '.').replace('.post', '.')
                parts = [p for p in v.split('.') if p]
                key = []
                for p in parts:
                    try:
                        key.append(int(p))
                    except ValueError:
                        key.append(0)
                # Pad to fixed length to avoid tuple length effects
                while len(key) < 5:
                    key.append(0)
                return tuple(key)

            if sort == "time":
                tags.sort(key=lambda t: (t["time"] or 0), reverse=True)
            elif sort == "name":
                tags.sort(key=lambda t: t["name"], reverse=False)
            else:  # semver
                tags.sort(key=lambda t: semver_key(t["name"]), reverse=True)

            # Limit
            tags = tags[:limit]

            # Render response
            lines = [f"Tags for {framework} (sorted by {sort}, limit {limit}):", ""]
            for t in tags:
                time_str = t["time"].isoformat() if t["time"] else ""
                sha = t["sha"]
                lines.append(f"  {t['name']:<20} {sha:<12} {time_str}")

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]

    async def run(self):
        """Run the MCP server."""
        # Initialize repositories
        await self.initialize()

        # Start stdio server
        logger.info("Starting MCP server on stdio...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    try:
        server = InferenceXMCPServer()
        await server.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())