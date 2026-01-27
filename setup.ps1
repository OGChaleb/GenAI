# Setup Script for ProfessorGPT Project
# This script sets up the project from scratch on a Windows machine.

# Step 1: Check if Python is installed
Write-Host "Checking if Python is installed..."
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-Host "Python is already installed: $pythonVersion"
    } else {
        throw
    }
} catch {
    Write-Host "Python not found. Installing Python 3.11.8..."
    # Download Python installer
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe" -OutFile "$env:TEMP\python-installer.exe"
    # Install silently
    Start-Process -FilePath "$env:TEMP\python-installer.exe" -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1" -Wait
    Write-Host "Python installed."
}

# Step 2: Navigate to project directory
# Assume the script is run from the project root (GenAI-main)
Set-Location -Path "GenAI-main"

# Step 3: Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# Step 4: Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Step 5: Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Step 6: Install dependencies
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 7: Note about PDF folder
Write-Host "Note: The PDF folder 'D:\GenAI\Haystack2\Haystack RAG' does not exist. Please create it and add PDF files if needed."

# Step 8: Note about Ollama
Write-Host "Note: Ollama must be installed and running on localhost:11434 for the LLM to work. Download from https://ollama.ai/"

# Step 9: Start the server
Write-Host "Starting the server..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000