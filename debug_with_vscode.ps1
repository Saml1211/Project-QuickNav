# PowerShell script to launch AutoHotkey with VSCode debugger

# Configuration
$ahkPath = "c:\Users\SamLyndon\AppData\Local\Programs\AutoHotkey\v2\AutoHotkey64.exe"
$scriptPath = "lld_navigator.ahk"
$debugPort = 9002

# Kill any existing instances
Get-Process -Name "AutoHotkey*" -ErrorAction SilentlyContinue | Stop-Process -Force

# Launch with debugger
Write-Host "Launching AutoHotkey with debugger on port $debugPort..."
Start-Process -FilePath $ahkPath -ArgumentList "/Debug=127.0.0.1:$debugPort", $scriptPath -NoNewWindow

Write-Host "Script launched. Connect VSCode debugger to port $debugPort."
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
