import asyncio
import re
import shutil
from datetime import datetime
from pathlib import Path

import git


def sanitize_repo_name(repo_url):
    """
    Extract and sanitize repository name from URL

    Args:
        repo_url: Git repository URL

    Returns:
        Sanitized repository name
    """
    # Extract repo name from URL
    # https://github.com/mrdoob/three.js.git -> three.js
    # https://github.com/mrdoob/three.js -> three.js
    match = re.search(r"/([^/]+?)(\.git)?$", repo_url)
    if match:
        return match.group(1)
    return "repo"


async def clone_repository(repo_url, temp_dir="temp/repos"):
    """
    Clone a git repository (async)

    Args:
        repo_url: Git repository URL
        temp_dir: Temporary directory for cloning

    Returns:
        Path to cloned repository or None if failed
    """
    print(f"📦 Cloning repository: {repo_url}")

    # Create temp directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Generate clone path
    repo_name = sanitize_repo_name(repo_url)
    clone_path = temp_path / repo_name

    # Remove if already exists
    if clone_path.exists():
        print(f"   Removing existing clone...")
        shutil.rmtree(clone_path)

    # Clone in executor (blocking operation)
    loop = asyncio.get_event_loop()

    def _clone():
        try:
            # Shallow clone (only latest commit, much faster)
            git.Repo.clone_from(
                repo_url,
                clone_path,
                depth=1,  # Shallow clone
                single_branch=True,  # Only default branch
            )
            return clone_path
        except git.GitCommandError as e:
            print(f"✗ Git clone failed: {e}")
            return None
        except Exception as e:
            print(f"✗ Error cloning repository: {e}")
            return None

    result = await loop.run_in_executor(None, _clone)

    if result:
        print(f"✓ Repository cloned to: {clone_path}")

    return result


def find_files_by_pattern(directory, patterns):
    """
    Find files matching patterns in a directory

    Args:
        directory: Directory to search in
        patterns: List of glob patterns (e.g., ['*.js', '*.html'])

    Returns:
        List of matching file paths
    """
    path = Path(directory)
    files = []

    for pattern in patterns:
        files.extend(path.rglob(pattern))

    return files


def should_exclude_path(file_path, exclude_patterns):
    """
    Check if a file path should be excluded

    Args:
        file_path: Path object to check
        exclude_patterns: List of patterns to exclude (e.g., ['node_modules', 'test'])

    Returns:
        True if should be excluded
    """
    path_str = str(file_path)

    for pattern in exclude_patterns:
        if pattern in path_str:
            return True

    return False


async def extract_files(
    clone_path,
    folders_to_extract=None,
    file_patterns=None,
    exclude_patterns=None,
    output_dir="output/repos",
):
    """
    Extract specific files from cloned repository

    Args:
        clone_path: Path to cloned repository
        folders_to_extract: List of folder paths to extract (e.g., ['examples', 'docs'])
        file_patterns: List of file patterns to match (e.g., ['*.js', '*.md'])
        exclude_patterns: List of patterns to exclude (e.g., ['node_modules', 'test'])
        output_dir: Output directory

    Returns:
        Dict with extraction results
    """
    if not clone_path or not clone_path.exists():
        return {
            "success": False,
            "error": "Clone path does not exist",
            "files_extracted": 0,
        }

    print(f"📂 Extracting files from: {clone_path.name}")

    # Default values
    if file_patterns is None:
        file_patterns = ["*.js", "*.ts", "*.html", "*.md", "*.json"]

    if exclude_patterns is None:
        exclude_patterns = [
            "node_modules",
            ".git",
            "test",
            "tests",
            "__pycache__",
            "dist",
            "build",
        ]

    # Create output directory
    repo_name = clone_path.name
    output_path = Path(output_dir) / repo_name
    output_path.mkdir(parents=True, exist_ok=True)

    files_extracted = 0

    # Run in executor (file operations)
    loop = asyncio.get_event_loop()

    def _extract():
        nonlocal files_extracted

        # If specific folders are specified
        if folders_to_extract:
            for folder in folders_to_extract:
                folder_path = clone_path / folder

                if not folder_path.exists():
                    print(f"   ⚠️  Folder not found: {folder}")
                    continue

                print(f"   Extracting: {folder}/")

                # Find matching files in this folder
                for pattern in file_patterns:
                    for file_path in folder_path.rglob(pattern):
                        # Skip if matches exclude patterns
                        if should_exclude_path(file_path, exclude_patterns):
                            continue

                        # Calculate relative path
                        rel_path = file_path.relative_to(clone_path)
                        dest_path = output_path / rel_path

                        # Create parent directories
                        dest_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy file
                        shutil.copy2(file_path, dest_path)
                        files_extracted += 1
        else:
            # Extract all matching files from entire repo
            print(f"   Extracting all matching files...")

            for pattern in file_patterns:
                for file_path in clone_path.rglob(pattern):
                    # Skip if matches exclude patterns
                    if should_exclude_path(file_path, exclude_patterns):
                        continue

                    # Calculate relative path
                    rel_path = file_path.relative_to(clone_path)
                    dest_path = output_path / rel_path

                    # Create parent directories
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    shutil.copy2(file_path, dest_path)
                    files_extracted += 1

        return files_extracted

    files_count = await loop.run_in_executor(None, _extract)

    print(f"✓ Extracted {files_count} files to: {output_path}")

    return {
        "success": True,
        "files_extracted": files_count,
        "output_path": str(output_path),
    }


async def cleanup_clone(clone_path):
    """
    Remove cloned repository (async)

    Args:
        clone_path: Path to cloned repository
    """
    if not clone_path or not clone_path.exists():
        return

    print(f"🧹 Cleaning up: {clone_path.name}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: shutil.rmtree(clone_path))

    print(f"✓ Cleanup complete")


async def process_repository(
    repo_url,
    folders_to_extract=None,
    file_patterns=None,
    exclude_patterns=None,
    output_dir="output/repos",
    keep_clone=False,
):
    """
    Process a git repository: clone, extract files, cleanup

    Args:
        repo_url: Git repository URL
        folders_to_extract: List of folders to extract (None = all)
        file_patterns: List of file patterns to match
        exclude_patterns: List of patterns to exclude
        output_dir: Output directory
        keep_clone: If True, don't delete the cloned repo

    Returns:
        Dict with operation result
    """
    print(f"🚀 Processing repository: {repo_url}")

    try:
        # Clone repository
        clone_path = await clone_repository(repo_url)

        if not clone_path:
            return {
                "repo_url": repo_url,
                "success": False,
                "error": "Failed to clone repository",
            }

        # Extract files
        extraction_result = await extract_files(
            clone_path, folders_to_extract, file_patterns, exclude_patterns, output_dir
        )

        # Cleanup (unless keep_clone=True)
        if not keep_clone:
            await cleanup_clone(clone_path)

        if extraction_result["success"]:
            return {
                "repo_url": repo_url,
                "success": True,
                "files_extracted": extraction_result["files_extracted"],
                "output_path": extraction_result["output_path"],
            }
        else:
            return {
                "repo_url": repo_url,
                "success": False,
                "error": extraction_result.get("error", "Unknown error"),
            }

    except Exception as e:
        print(f"✗ Error processing repository: {e}")
        return {"repo_url": repo_url, "success": False, "error": str(e)}


async def process_repositories(
    repo_urls,
    folders_to_extract=None,
    file_patterns=None,
    exclude_patterns=None,
    output_dir="output/repos",
    keep_clones=False,
    max_concurrent=2,
):
    """
    Process multiple repositories concurrently

    Args:
        repo_urls: List of git repository URLs
        folders_to_extract: List of folders to extract
        file_patterns: List of file patterns to match
        exclude_patterns: List of patterns to exclude
        output_dir: Output directory
        keep_clones: If True, don't delete cloned repos
        max_concurrent: Maximum concurrent repository processing

    Returns:
        List of results
    """
    print(f"🚀 Starting repository processing...")
    print(f"   Repositories: {len(repo_urls)}")
    print(f"   Folders to extract: {folders_to_extract or 'all'}")
    print(f"   File patterns: {file_patterns or 'default'}")
    print(f"   Max concurrent: {max_concurrent}")
    print(f"   Output: {output_dir}\n")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(repo_url):
        async with semaphore:
            result = await process_repository(
                repo_url,
                folders_to_extract,
                file_patterns,
                exclude_patterns,
                output_dir,
                keep_clones,
            )
            # Small delay between repos
            await asyncio.sleep(0.5)
            return result

    # Process all repos concurrently
    tasks = [process_with_semaphore(url) for url in repo_urls]
    results = await asyncio.gather(*tasks)

    return results


async def load_urls_async(filepath):
    """Load URLs from a text file (async)"""
    loop = asyncio.get_event_loop()

    def _load():
        with open(filepath, "r") as f:
            urls = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        return urls

    return await loop.run_in_executor(None, _load)
