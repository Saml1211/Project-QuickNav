# PowerShell script to build standalone QuickNav EXE with PyInstaller

param (
    [string]$Version = (Get-Content ../VERSION.txt).Trim()
)

$ErrorActionPreference = 'Stop'

Write-Host "Building QuickNav EXE, version $Version"

# Ensure venv is active, install requirements
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Installing pyinstaller into current Python environment..."
    pip install pyinstaller
}

# Ensure project is installed in editable mode (for import resolution)
pip install -e .. 

# Clean old build
Remove-Item -Recurse -Force ../dist,../build -ErrorAction SilentlyContinue

# Build (single exe)
pyinstaller --onefile --name "quicknav-$Version-win64" --console ../quicknav/cli.py

Write-Host "Copying lld_navigator.ahk next to EXE..."
Copy-Item ../lld_navigator.ahk ../dist/ -Force

Write-Host "EXE build complete. Output in ../dist/"