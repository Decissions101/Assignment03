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