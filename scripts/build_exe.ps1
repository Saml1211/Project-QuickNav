# PowerShell script to build standalone QuickNav EXE with AHK2EXE (AutoHotkey v2)

param (
    [string]$Version
)

$ErrorActionPreference = 'Stop'

# Get the script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Get version from VERSION.txt if not provided
if (-not $Version) {
    $VersionFile = Join-Path $ProjectRoot "VERSION.txt"
    if (Test-Path $VersionFile) {
        $Version = (Get-Content $VersionFile).Trim()
    } else {
        Write-Error "VERSION.txt not found at $VersionFile"
        exit 1
    }
}

Write-Host "Building QuickNav EXE with AutoHotkey v2, version $Version"

# Check for AutoHotkey installation
$AhkPath = Get-Command "AutoHotkey64.exe" -ErrorAction SilentlyContinue
if (-not $AhkPath) {
    # Try common installation paths
    $CommonPaths = @(
        "C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe",
        "C:\Program Files (x86)\AutoHotkey\v2\AutoHotkey64.exe",
        "$env:LOCALAPPDATA\Programs\AutoHotkey\v2\AutoHotkey64.exe"
    )
    
    foreach ($Path in $CommonPaths) {
        if (Test-Path $Path) {
            $AhkPath = @{ Source = $Path }
            break
        }
    }
    
    if (-not $AhkPath) {
        Write-Error "AutoHotkey v2 not found. Please install AutoHotkey v2 from https://www.autohotkey.com/"
        exit 1
    }
}

$AutoHotkey = $AhkPath.Source

# Check for Ahk2Exe compiler
$Ahk2ExePath = Split-Path $AutoHotkey -Parent
$PossibleCompilerPaths = @(
    (Join-Path $Ahk2ExePath "Compiler\Ahk2Exe.exe"),
    (Join-Path $Ahk2ExePath "Ahk2Exe.exe"),
    "C:\Program Files\AutoHotkey\Compiler\Ahk2Exe.exe",
    "C:\Program Files (x86)\AutoHotkey\Compiler\Ahk2Exe.exe",
    "$env:LOCALAPPDATA\Programs\AutoHotkey\Compiler\Ahk2Exe.exe"
)

$Ahk2Exe = $null
foreach ($Path in $PossibleCompilerPaths) {
    if (Test-Path $Path) {
        $Ahk2Exe = $Path
        break
    }
}

if (-not $Ahk2Exe) {
    Write-Host "Searched for Ahk2Exe.exe in these locations:"
    foreach ($Path in $PossibleCompilerPaths) {
        Write-Host "  - $Path"
    }
    Write-Error "Ahk2Exe.exe not found. Please ensure AutoHotkey v2 is properly installed with the compiler, or install it separately."
    exit 1
}

# Locate the main AHK script
$AhkSource = Join-Path $ProjectRoot "src\lld_navigator.ahk"
if (-not (Test-Path $AhkSource)) {
    Write-Error "Main AutoHotkey script not found at $AhkSource"
    exit 1
}

# Create dist directory
$DistDir = Join-Path $ProjectRoot "dist"
if (-not (Test-Path $DistDir)) {
    New-Item -ItemType Directory -Path $DistDir -Force | Out-Null
}

# Output EXE path
$OutputExe = Join-Path $DistDir "quicknav-$Version-win64.exe"

Write-Host "Compiling AutoHotkey script..."
Write-Host "  Source: $AhkSource"
Write-Host "  Output: $OutputExe"
Write-Host "  Compiler: $Ahk2Exe"

# Compile the AHK script to EXE
& $Ahk2Exe /in $AhkSource /out $OutputExe /base $AutoHotkey

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile AutoHotkey script. Exit code: $LASTEXITCODE"
    exit 1
}

if (Test-Path $OutputExe) {
    Write-Host "SUCCESS: EXE build completed successfully!"
    Write-Host "Output: $OutputExe"
    
    # Show file size
    $FileSize = (Get-Item $OutputExe).Length
    $FileSizeKB = [math]::Round($FileSize / 1KB, 2)
    Write-Host "File size: $FileSizeKB KB"
} else {
    Write-Error "Build failed - output EXE not found at $OutputExe"
    exit 1
}

Write-Host ""
Write-Host "QuickNav EXE build complete!"