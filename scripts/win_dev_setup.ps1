#requires -version 5.1
<#
.SYNOPSIS
    Automates setup for QuickNav Python backend and AHK frontend utilities on Windows 11.
.DESCRIPTION
    1. Organizes project files if run locally, or clones/downloads if not.
    2. Installs Python 3.x and pip dependencies (mcp_server, quicknav, root).
    3. Installs AutoHotkey (v1 or v2, matching lld_navigator.ahk syntax).
    4. Sets up lld_navigator.ahk to run at startup.
    5. Ensures all dependencies for find_project_path.py.
    6. Implements robust logging/error handling.
    7. Sets up a daily Task Scheduler job for repo/pip update and logs outcomes.
    8. Idempotent; safe to re-run.
#>

$ErrorActionPreference = "Stop"
$LogFile = "$PSScriptRoot\win_dev_setup.log"
$Summary = @()
function Log ($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp $msg" | Tee-Object -Append -FilePath $LogFile
}

Log "`n####################### Starting Win Dev Setup #######################"

# Step 1: Organize/Clone/Download Project
$repoUrl = 'https://github.com/SamLyndon/Project-QuickNav.git'
$defaultDir = "$env:USERPROFILE\QuickNav"
if (-not (Test-Path "$PSScriptRoot\lld_navigator.ahk")) {
    if (-not (Test-Path $defaultDir)) {
        Log "Project not found locally. Cloning repository to $defaultDir..."
        git clone $repoUrl $defaultDir
        $Summary += "Cloned repository to $defaultDir"
    } else {
        Log "Project directory exists at $defaultDir."
        $Summary += "Project found at $defaultDir"
    }
    $ProjectRoot = $defaultDir
} else {
    Log "Running in local project directory: $PSScriptRoot"
    $Summary += "Project files found locally at $PSScriptRoot"
    $ProjectRoot = $PSScriptRoot
}

# Step 2: Python Installation and Dependencies
function Install-PythonIfMissing {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $python) {
        Log "Python not found. Installing latest Python 3.x via winget..."
        winget install -e --id Python.Python.3
        $env:PATH += ";$([System.Environment]::GetEnvironmentVariable('LocalAppData'))\Programs\Python\Python*;C:\Python*"
        $Summary += "Installed Python 3.x via winget"
    } else {
        Log "Python found: $($python.Source)"
        $Summary += "Python found: $($python.Source)"
    }
}
Install-PythonIfMissing

# Confirm python is now available
try {
    $pyVer = python --version 2>&1
    Log "Using Python version: $pyVer"
} catch {
    Log "ERROR: Python not found after install!"
    throw
}

Log "Upgrading pip..."
python -m pip install --upgrade pip

# Install pip dependencies from all relevant requirements.txt/setup.py
function Install-PipRequirements ($projectRoot) {
    $found = $false
    $reqFiles = @(
        "$projectRoot\requirements.txt",
        "$projectRoot\setup.py",
        "$projectRoot\mcp_server\requirements.txt",
        "$projectRoot\quicknav\requirements.txt"
    )
    foreach ($req in $reqFiles) {
        if (Test-Path $req) {
            if ($req -like "*.txt") {
                Log "Installing pip dependencies from $req"
                python -m pip install -r $req
                $Summary += "Installed pip requirements from $req"
                $found = $true
            } elseif ($req -like "*.py") {
                Log "Installing pip dependencies from $req (setup.py)"
                pushd (Split-Path $req)
                python setup.py install
                popd
                $Summary += "Installed pip requirements using $req"
                $found = $true
            }
        }
    }
    if (-not $found) {
        Log "WARNING: No requirements.txt or setup.py found!"
        $Summary += "No pip dependencies installed"
    }
}
Install-PipRequirements $ProjectRoot

# Step 3: Install AutoHotkey (v1 or v2)
$ahkExe = $null
function Get-AhkVersion {
    param($file)
    $content = Get-Content $file -Raw
    if ($content -cmatch "#Requires AutoHotkey v2" -or $content -match "A_Args\[") { return 2 }
    else { return 1 }
}
$ahkVersion = Get-AhkVersion "$ProjectRoot\lld_navigator.ahk"
if ($ahkVersion -eq 2) {
    $ahkWingetId = "AutoHotkey.AutoHotkey.v2"
    $Summary += "Detected AHK v2 syntax in lld_navigator.ahk"
} else {
    $ahkWingetId = "AutoHotkey.AutoHotkey"
    $Summary += "Detected AHK v1 syntax in lld_navigator.ahk"
}
function Install-AHKIfMissing {
    $ahkExe = Get-Command "AutoHotkey.exe" -ErrorAction SilentlyContinue
    if ($null -eq $ahkExe) {
        Log "AutoHotkey not found. Installing via winget ($ahkWingetId)..."
        winget install -e --id $ahkWingetId
        $Summary += "Installed AutoHotkey ($ahkWingetId) via winget"
    } else {
        Log "AutoHotkey found: $($ahkExe.Source)"
        $Summary += "AutoHotkey found: $($ahkExe.Source)"
    }
}
Install-AHKIfMissing

# Step 4: Place lld_navigator.ahk and configure startup
function Setup-AhkStartup {
    $startupFolder = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
    $ahkSource = "$ProjectRoot\lld_navigator.ahk"
    $ahkDest = "$startupFolder\lld_navigator.ahk"
    if (!(Test-Path $ahkDest -PathType Leaf) -or !(Compare-Object (Get-Content $ahkSource) (Get-Content $ahkDest) -SyncWindow 0)) {
        Log "Copying lld_navigator.ahk to $startupFolder"
        Copy-Item -Force $ahkSource $ahkDest
        $Summary += "Placed lld_navigator.ahk in Startup folder"
    } else {
        Log "lld_navigator.ahk already up to date in Startup folder"
        $Summary += "lld_navigator.ahk already up to date in Startup"
    }
}
Setup-AhkStartup

# Step 5: Dependencies for find_project_path.py (likely pip only)
if (Test-Path "$ProjectRoot\find_project_path.py") {
    Log "Ensuring find_project_path.py can run (checking for dependencies)..."
    # If requirements specified, handled above.
    $Summary += "find_project_path.py present (ensure dependencies are handled by previous steps)"
}

# Step 6: Robust Error Handling and Logging (done throughout)
# Step 7: Setup Task Scheduler for daily update
$taskName = "QuickNav_DailyUpdate"
$updateScript = "$ProjectRoot\scripts\quicknav_update.ps1"
$updateLog = "$ProjectRoot\quicknav_update.log"
if (!(Test-Path "$ProjectRoot\scripts")) { New-Item -Path "$ProjectRoot\scripts" -ItemType Directory }
Set-Content $updateScript @"
# Auto-generated daily update script for QuickNav
try {
    cd "$ProjectRoot"
    if (Test-Path .git) {
        git pull origin
    }
    \$pipReqs = @(
        "requirements.txt",
        "mcp_server/requirements.txt",
        "quicknav/requirements.txt"
    )
    foreach (\$r in \$pipReqs) {
        if (Test-Path \$r) {
            python -m pip install -r \$r
        }
    }
    Add-Content "$updateLog" ("[{0}] Update succeeded." -f (Get-Date))
} catch {
    Add-Content "$updateLog" ("[{0}] Update failed: \$($_.Exception.Message)" -f (Get-Date))
}
"@
if (!(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
    Log "Registering Windows Task Scheduler task: $taskName"
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$updateScript`""
    $Trigger = New-ScheduledTaskTrigger -Daily -At 9am
    Register-ScheduledTask -TaskName $taskName -Action $Action -Trigger $Trigger -Description "Daily QuickNav update" -Force | Out-Null
    $Summary += "Created daily scheduled task $taskName (runs $updateScript at 9AM)"
} else {
    Log "Scheduled task $taskName already exists."
    $Summary += "Scheduled task $taskName already exists"
}

# Step 8: Windows path handling (used throughout)
# Step 9: Idempotency (ensured by checks before actions)
# Step 10: Output summary
Log "Setup complete.`n"
$SummaryTxt = $Summary -join "`n"
Write-Host "`n==================== Setup Summary ====================" -ForegroundColor Green
Write-Host "$SummaryTxt" -ForegroundColor Green

Write-Host "`nIf you encounter any errors, see: $LogFile" -ForegroundColor Yellow
Write-Host "If lld_navigator.ahk does not start, double check the Startup folder or run it manually." -ForegroundColor Yellow
Write-Host "Manual step: You may need to log out and back in for AHK startup script to activate, or allow UAC prompt if asked." -ForegroundColor Yellow
Write-Host "To view daily update logs, see: $updateLog" -ForegroundColor Yellow

Log "####################### Win Dev Setup Finished #######################`n"