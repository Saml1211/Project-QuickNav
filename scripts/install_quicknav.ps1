# PowerShell script to install QuickNav & dependencies on Windows

$ErrorActionPreference = "Stop"

Write-Host "=== QuickNav Automated Installer ==="

# Check for admin
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "It's recommended to run this script as Administrator for full install."
}

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Python not found. Installing via winget..."
    winget install --id Python.Python.3.10 -e --source winget
}
python --version

# Ensure pip is present (Python >=3.4+)
$pip = Get-Command pip -ErrorAction SilentlyContinue
if (-not $pip) {
    Write-Host "Pip not found. Attempting ensurepip..."
    python -m ensurepip --upgrade
}
pip --version

# Install QuickNav
Write-Host "Installing QuickNav..."
pip install quicknav

# Confirm install
Write-Host ""
Write-Host "QuickNav version:"
quicknav --version

Write-Host ""
Write-Host "If you want the GUI, please install AutoHotkey v2 from https://www.autohotkey.com/"
Write-Host "Then run or compile 'lld_navigator.ahk'."
Write-Host ""
Write-Host "Install complete. Add Python\\Scripts to PATH if needed, then re-open terminal."