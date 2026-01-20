"""Utility functions for MCP server - version detection, repo management, and filtering."""

import os
import re
import logging
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml
import git

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

def get_config_paths() -> List[str]:
    """Get paths to InferenceMAX config files."""
    root = Path(os.getenv('INFERENCEMAX_ROOT', Path.cwd()))
    return [
        str(root / '.github/configs/amd-master.yaml'),
        str(root / '.github/configs/nvidia-master.yaml'),
    ]


def get_clone_dir() -> Path:
    """Get directory for repository clones."""
    return Path('/tmp/inferencemax-mcp')


# ============================================================================
# Version Detection
# ============================================================================

# Regex patterns for version extraction
VLLM_PATTERNS = [
    r'vllm/vllm-openai:v?([\d.]+(?:rc\d+)?)',  # vllm/vllm-openai:v0.13.0
    r'vllm_([\d.]+)',                           # vllm_0.10.1
]

SGLANG_PATTERNS = [
    r'lmsysorg/sglang:v?([\d.]+(?:\.post\d+)?)',   # lmsysorg/sglang:v0.5.7
    r'sglang[:-](\d+\.\d+\.\d+(?:\.post\d+)?)',    # sglang-0.5.6.post1
]


def extract_version(image: str, patterns: List[str]) -> Optional[str]:
    """
    Extract version from Docker image tag using regex patterns.

    Args:
        image: Docker image string (e.g., "vllm/vllm-openai:v0.13.0")
        patterns: List of regex patterns to try

    Returns:
        Extracted version string or None if not found
    """
    for pattern in patterns:
        match = re.search(pattern, image, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def select_primary_version(versions: Set[str]) -> str:
    """
    Select primary version from a set using semantic versioning.

    Args:
        versions: Set of version strings

    Returns:
        Latest version by semantic versioning, or 'main' if empty
    """
    if not versions:
        return 'main'

    def parse_version(v: str) -> tuple:
        """Parse version string into tuple for comparison."""
        v_clean = v.lstrip('v')
        # Handle post-releases and rc versions
        v_clean = v_clean.replace('.post', '.').replace('rc', '.')
        parts = v_clean.split('.')
        return tuple(int(p) if p.isdigit() else 0 for p in parts)

    return max(versions, key=parse_version)


def detect_versions(config_paths: List[str]) -> Dict[str, Set[str]]:
    """
    Parse config files and extract vLLM and SGLang versions.

    Args:
        config_paths: Paths to YAML config files

    Returns:
        Dictionary mapping framework names to sets of detected versions:
        {'vllm': {'0.13.0', '0.10.1'}, 'sglang': {'0.5.7', ...}}
    """
    versions = {'vllm': set(), 'sglang': set()}

    for config_path in config_paths:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            continue

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                continue

            for config_name, config_data in config.items():
                if not isinstance(config_data, dict) or 'image' not in config_data:
                    continue

                image = config_data['image']
                framework = config_data.get('framework', '')

                # Detect vLLM version
                if 'vllm' in framework.lower() or 'vllm' in image.lower():
                    version = extract_version(image, VLLM_PATTERNS)
                    if version:
                        versions['vllm'].add(version)
                        logger.debug(f"Detected vLLM version {version} from {config_name}")

                # Detect SGLang version
                if 'sglang' in framework.lower() or 'sglang' in image.lower():
                    version = extract_version(image, SGLANG_PATTERNS)
                    if version:
                        versions['sglang'].add(version)
                        logger.debug(f"Detected SGLang version {version} from {config_name}")

        except Exception as e:
            logger.error(f"Error parsing config file {config_path}: {e}")

    return versions


# ============================================================================
# Repository Management
# ============================================================================

REPO_URLS = {
    'vllm': 'https://github.com/vllm-project/vllm.git',
    'sglang': 'https://github.com/sgl-project/sglang.git',
}


def initialize_repo(name: str, url: str, path: Path) -> git.Repo:
    """
    Clone or update a repository.

    Args:
        name: Repository name (for logging)
        url: Git repository URL
        path: Local path for repository

    Returns:
        GitPython Repo object
    """
    if path.exists():
        # Repository exists, update it
        logger.info(f"Updating {name} repository at {path}")
        repo = git.Repo(path)
        try:
            origin = repo.remotes.origin
            origin.fetch()
        except Exception as e:
            logger.warning(f"Failed to fetch updates for {name}: {e}")
    else:
        # Clone repository
        logger.info(f"Cloning {name} from {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        repo = git.Repo.clone_from(url, path)

    return repo


def fuzzy_match_tag(repo: git.Repo, version: str) -> Optional[str]:
    """
    Find best matching tag for a version using fuzzy matching.

    Args:
        repo: GitPython Repo object
        version: Version string to match

    Returns:
        Matching tag name or None
    """
    all_tags = [tag.name for tag in repo.tags]

    # Try exact matches with different prefixes
    tag_candidates = [
        f'v{version}',      # v0.5.7
        version,            # 0.5.7
        f'{version}.0',     # 0.5.7.0
    ]

    for candidate in tag_candidates:
        if candidate in all_tags:
            return candidate

    # Fuzzy match: find tags containing version digits
    version_digits = version.replace('.', '')
    matching_tags = [t for t in all_tags if version_digits in t.replace('.', '')]

    if matching_tags:
        # Return the first match (usually the most specific)
        return matching_tags[0]

    return None


def checkout_version(repo: git.Repo, framework: str, version: str) -> bool:
    """
    Checkout a specific version with fuzzy matching.

    Args:
        repo: GitPython Repo object
        framework: Framework name (for logging)
        version: Version string to checkout

    Returns:
        True if checkout successful, False otherwise
    """
    # Try to find matching tag
    tag = fuzzy_match_tag(repo, version)

    if tag:
        try:
            repo.git.checkout(tag)
            logger.info(f"Checked out {framework} at {tag}")
            return True
        except git.GitCommandError as e:
            logger.warning(f"Failed to checkout {tag} for {framework}: {e}")
    else:
        logger.warning(f"Could not find tag for version {version} in {framework}")

    # Fallback to main/master
    for branch in ['main', 'master']:
        try:
            repo.git.checkout(branch)
            logger.info(f"Checked out {framework} at {branch} (fallback)")
            return True
        except git.GitCommandError as e:
            logger.debug(f"Branch {branch} not available: {e}")

    logger.error(f"Failed to checkout any branch for {framework}")
    return False


# ============================================================================
# Path Filtering
# ============================================================================

EXCLUDE_PATTERNS = [
    # Test directories
    '**/tests/**',
    '**/test/**',
    '**/*_test.py',
    '**/testing/**',

    # Build artifacts
    '**/build/**',
    '**/dist/**',
    '**/__pycache__/**',
    '**/*.pyc',
    '**/*.pyo',
    '**/.git/**',

    # IDE and config
    '**/.vscode/**',
    '**/.idea/**',
    '**/.github/**',

    # Large binary/data files
    '**/*.so',
    '**/*.whl',
    '**/*.egg-info/**',
]

INCLUDE_PATTERNS = [
    # Core Python source
    '**/*.py',

    # Configuration files
    '**/pyproject.toml',
    '**/setup.py',
    '**/requirements*.txt',
]

# Special case: allow root-level README.md
ROOT_ALLOWED_FILES = ['README.md', 'LICENSE']


def should_include_file(file_path: Path, repo_path: Path) -> bool:
    """
    Determine if a file should be included in MCP resources.

    Args:
        file_path: Absolute path to file
        repo_path: Repository root path

    Returns:
        True if file should be included
    """
    try:
        rel_path = file_path.relative_to(repo_path)
    except ValueError:
        # File is not inside repo_path
        return False

    rel_path_str = str(rel_path)

    # Check if it's a root-level allowed file
    if rel_path.name in ROOT_ALLOWED_FILES and len(rel_path.parts) == 1:
        return True

    # Check exclude patterns first
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(rel_path_str, pattern):
            return False

    # Check include patterns
    for pattern in INCLUDE_PATTERNS:
        if fnmatch.fnmatch(rel_path_str, pattern):
            return True

    return False


def list_filtered_files(repo_path: Path) -> List[Path]:
    """
    List all files in repository that pass the filter.

    Args:
        repo_path: Repository root path

    Returns:
        List of absolute file paths that should be exposed
    """
    if not repo_path.exists():
        logger.warning(f"Repository path does not exist: {repo_path}")
        return []

    filtered_files = []

    try:
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and should_include_file(file_path, repo_path):
                filtered_files.append(file_path)
    except Exception as e:
        logger.error(f"Error listing files in {repo_path}: {e}")

    return filtered_files