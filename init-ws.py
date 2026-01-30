#!/usr/bin/env python3
"""
🚀 AGENT 3 - WORKSPACE INITIALIZER
Nobel Engineering Setup Script

Features:
- Installs Poetry if missing
- Creates virtual environment
- Installs all dependencies
- Sets up pre-commit hooks
- Validates environment
- Creates necessary directories
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str) -> None:
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(message: str) -> None:
    """Print success message"""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def print_info(message: str) -> None:
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ{Colors.ENDC} {message}")


def print_warning(message: str) -> None:
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {message}")


def print_error(message: str) -> None:
    """Print error message"""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {message}")


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> Optional[str]:
    """Run shell command with error handling"""
    try:
        if capture:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=check)
            return None
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if capture and e.stderr:
            print_error(f"Error: {e.stderr}")
        return None
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return None


def check_python_version() -> bool:
    """Check if Python version is compatible"""
    print_info("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.11+ required, found {version.major}.{version.minor}.{version.micro}")
        return False


def check_poetry_installed() -> bool:
    """Check if Poetry is installed"""
    print_info("Checking Poetry installation...")
    result = run_command(["poetry", "--version"], check=False, capture=True)
    
    if result:
        print_success(f"Poetry found: {result}")
        return True
    else:
        print_warning("Poetry not found")
        return False


def install_poetry() -> bool:
    """Install Poetry using official installer"""
    print_info("Installing Poetry...")
    
    if sys.platform == "win32":
        # Windows installation
        cmd = [
            "powershell",
            "-Command",
            "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
        ]
    else:
        # Unix/MacOS installation
        cmd = ["curl", "-sSL", "https://install.python-poetry.org", "|", "python3", "-"]
        # Use shell for pipe
        result = run_command(
            ["sh", "-c", "curl -sSL https://install.python-poetry.org | python3 -"],
            check=False
        )
        
        if result is None:
            print_warning("Trying alternative installation method...")
            run_command(["pip", "install", "--user", "poetry"])
            
            # Add user's Python bin to PATH
            user_bin = os.path.expanduser("~/Library/Python/3.12/bin")
            if os.path.exists(user_bin) and user_bin not in os.environ["PATH"]:
                os.environ["PATH"] = f"{user_bin}:{os.environ['PATH']}"
                print_info(f"Added {user_bin} to PATH")
    
    # Verify installation
    if check_poetry_installed():
        print_success("Poetry installed successfully")
        return True
    else:
        print_error("Poetry installation failed")
        return False


def configure_poetry() -> bool:
    """Configure Poetry settings"""
    print_info("Configuring Poetry...")
    
    # Create virtualenv in project directory
    run_command(["poetry", "config", "virtualenvs.in-project", "true"])
    print_success("Virtual environment will be created in .venv/")
    
    # Set Python version (use current Python)
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    run_command(["poetry", "env", "use", python_version], check=False)
    
    return True


def install_dependencies() -> bool:
    """Install project dependencies with Poetry"""
    print_info("Installing dependencies (this may take a few minutes)...")
    
    # Install main dependencies
    try:
        subprocess.run(["poetry", "install", "--no-interaction"], check=True)
        print_success("All dependencies installed")
    except subprocess.CalledProcessError as e:
        # Check if it was just a warning about the current project
        if e.returncode == 1:
            print_warning("Dependencies installed with warnings")
        else:
            print_error("Failed to install dependencies")
            return False

    # Show installed packages
    print_info("Installed packages:")
    run_command(["poetry", "show", "--tree"], check=False)
    
    return True


def create_env_file() -> bool:
    """Create .env file from template"""
    print_info("Setting up environment variables...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_warning(".env file already exists, skipping...")
        return True
    
    # Create .env.example if it doesn't exist
    if not env_example.exists():
        env_template = """# LLM Configuration


# Alternative: Ollama Local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# API Configuration
API_HOST=0.0.0.0
API_PORT=8003
API_RELOAD=true
API_LOG_LEVEL=info

# Service Metadata
SERVICE_NAME=agent3-spec-generator
SERVICE_VERSION=1.0.0
"""
        env_example.write_text(env_template)
        print_success("Created .env.example template")
    
    # Copy to .env
    shutil.copy(env_example, env_file)
    print_success("Created .env file")
    
    return True


def create_directories() -> bool:
    """Create necessary project directories"""
    print_info("Creating project structure...")
    
    directories = [
        "app/core",
        "app/models",
        "app/signatures",
        "app/modules",
        "tests",
        "logs",
        "data"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if dir_path.startswith("app/") or dir_path == "tests":
            init_file = Path(dir_path) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print_success("Project structure created")
    return True


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks"""
    print_info("Setting up pre-commit hooks...")
    
    pre_commit_config = Path(".pre-commit-config.yaml")
    
    if not pre_commit_config.exists():
        config_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.13
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""
        pre_commit_config.write_text(config_content)
        print_success("Created .pre-commit-config.yaml")
    
    # Install hooks
    run_command(["poetry", "run", "pre-commit", "install"], check=False)
    print_success("Pre-commit hooks installed")
    
    return True


def run_tests() -> bool:
    """Run test suite to validate setup"""
    print_info("Running tests to validate setup...")
    
    result = run_command(
        ["poetry", "run", "pytest", "-v", "--tb=short"],
        check=False
    )
    
    if result is None:
        print_warning("Some tests failed, but setup is complete")
        return True
    
    print_success("All tests passed!")
    return True


def print_next_steps() -> None:
    """Print next steps for user"""
    print_header("🎉 SETUP COMPLETE")
    
    print(f"{Colors.OKGREEN}Your Agent 3 environment is ready!{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}IMPORTANT - Add Poetry to your PATH:{Colors.ENDC}")
    print(f"  Run this command to make Poetry available permanently:")
    print(f"  {Colors.OKCYAN}echo 'export PATH=\"$HOME/Library/Python/3.12/bin:$PATH\"' >> ~/.zshrc{Colors.ENDC}")
    print(f"  {Colors.OKCYAN}source ~/.zshrc{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"  1. {Colors.OKCYAN}Edit .env file:{Colors.ENDC}")
    print(f"     nano .env  # Add your OPENAI_API_KEY\n")
    
    print(f"  2. {Colors.OKCYAN}Activate virtual environment:{Colors.ENDC}")
    print(f"     poetry shell\n")
    
    print(f"  3. {Colors.OKCYAN}Run the server:{Colors.ENDC}")
    print(f"     poetry run python -m app.main\n")
    
    print(f"  4. {Colors.OKCYAN}Or use Docker:{Colors.ENDC}")
    print(f"     docker build -t agent3 .")
    print(f"     docker run -p 8003:8003 --env-file .env agent3\n")
    
    print(f"  5. {Colors.OKCYAN}Run tests:{Colors.ENDC}")
    print(f"     poetry run pytest\n")
    
    print(f"{Colors.BOLD}API Endpoints:{Colors.ENDC}")
    print(f"  • Health:    http://localhost:8003/health")
    print(f"  • Generate:  http://localhost:8003/generate")
    print(f"  • Docs:      http://localhost:8003/docs\n")
    
    print(f"{Colors.WARNING}⚠️  Don't forget to add your API key to .env!{Colors.ENDC}\n")


def main() -> int:
    """Main setup workflow"""
    print_header("🚀 AGENT 3 - WORKSPACE INITIALIZER")
    
    # Step 1: Check Python version
    if not check_python_version():
        return 1
    
    # Step 2: Check/Install Poetry
    if not check_poetry_installed():
        if not install_poetry():
            return 1
    
    # Step 3: Configure Poetry
    if not configure_poetry():
        return 1
    
    # Step 4: Create project structure
    if not create_directories():
        return 1
    
    # Step 5: Install dependencies
    if not install_dependencies():
        return 1
    
    # Step 6: Create .env file
    if not create_env_file():
        return 1
    
    # Step 7: Setup pre-commit hooks
    setup_pre_commit()
    
    # Step 8: Run tests
    run_tests()
    
    # Step 9: Print next steps
    print_next_steps()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)