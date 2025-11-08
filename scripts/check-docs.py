#!/usr/bin/env python3
"""Local documentation build checker.

This script builds the documentation locally and reports any warnings or errors.
Use this to catch documentation issues before committing.

Usage:
    python scripts/check-docs.py

Or with uv:
    uv run python scripts/check-docs.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


def main() -> int:
    """Main function to check documentation build."""
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    build_dir = docs_dir / "_build" / "html"

    print("🔍 Checking documentation build...")
    print(f"📁 Project root: {project_root}")
    print(f"📖 Docs directory: {docs_dir}")

    # Clean previous build
    if build_dir.exists():
        print("🧹 Cleaning previous build...")
        subprocess.run(["rm", "-rf", str(build_dir)], check=False)

    # Build documentation with warnings as errors
    print("🏗️  Building documentation...")
    cmd = [
        "uv",
        "run",
        "sphinx-build",
        "-b",
        "html",
        ".",
        "_build/html",
        "-W",  # Treat warnings as errors
        "--keep-going",  # Continue on errors to see all issues
        "-v",  # Verbose output
    ]

    exit_code, stdout, stderr = run_command(cmd, docs_dir)

    # Print output
    if stdout:
        print("📋 Build output:")
        print(stdout)

    if stderr:
        print("⚠️  Build warnings/errors:")
        print(stderr)

    # Check results
    if exit_code == 0:
        print("✅ Documentation build successful!")
        print(
            f"📖 Documentation available at: file://{build_dir.absolute()}/index.html"
        )
        return 0
    else:
        print(f"❌ Documentation build failed with exit code {exit_code}")
        print("\n💡 Common fixes:")
        print("  - Check RST syntax in docstrings")
        print("  - Ensure code blocks use '.. code-block:: python'")
        print("  - Check for malformed inline literals (use double backticks)")
        print("  - Verify all imports and references are correct")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
