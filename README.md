# ING3513 - Introduction to artificial intelligence and machine learning

Lab material for [ING3513](https://www.forsvaret.no/utdanning/emner/ING3513).

## Setup Instructions

### Prerequisites

Before starting, ensure you have:
- A GitHub account
- Access to the `ing3513/course-materials` repository

### Installation Steps

1. **Install WSL (Windows Subsystem for Linux)**
   - Open PowerShell as Administrator
   - Run: `wsl --install`
   - Restart your computer when prompted

2. **Install Visual Studio Code**
   - Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Run the installer and follow the setup wizard

3. **Install WSL Extension**
   - Download from [https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)
   - Or in VS Code: Go to Extensions (Ctrl+Shift+X), search for "WSL" and install the official Microsoft WSL extension

4. **Connect to WSL**
   - Press F1 to open the Command Palette
   - Type and select: `WSL: Connect to WSL`
   - VS Code will reopen in your Linux home directory (~ or /home/<username>)

5. **Login with GitHub**
   - Open the Accounts menu in VS Code (bottom left corner)
   - Select "Sign in with GitHub"
   - Follow the authentication process

6. **Clone Repository**
   - Open the Source Control view (Ctrl+Shift+G)
   - Click "Clone Repository" or press F1 and type `Git: Clone`
   - Enter the repository URL: `https://github.com/ing3513/course-materials.git`
   - Select a folder location and open the cloned repository

7. **Setup Python Environment**
   - Open a terminal in VS Code (Ctrl+`)
   - Run the setup script: `./scripts/setupPythonEnvironment.sh`
   - This will install uv, Python 3.13, and all project dependencies
   - If this is your first run, restart your terminal or run: `source ~/.bashrc`

## About the Python Environment

This course uses [uv](https://docs.astral.sh/uv/), a modern Python package and project manager. uv is significantly faster than pip and provides better dependency resolution.

### Key Features

- Automatic Python version management
- Fast dependency resolution and installation
- Reproducible environments via lockfiles
- All-in-one tool replacing pip, venv, and virtualenv

### Common Commands

- `uv run <command>` - Run a command in the project's virtual environment
- `uv add <package>` - Add a new dependency
- `uv sync --locked` - Sync dependencies from lockfile (use `--locked` to ensure exact versions match)
- `uv python install` - Install Python versions
