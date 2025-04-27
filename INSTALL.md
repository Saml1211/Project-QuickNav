# Project QuickNav – Installation Guide

## Overview

Project QuickNav simplifies navigation among project directories and provides an optional GUI launcher.  
This guide explains how to install and configure QuickNav **v1.0.0** on Windows 10/11.

---

## Prerequisites

| Component | Minimum Version | Notes |
|-----------|-----------------|-------|
| Windows   | 10 (build 19041) | Tested on 10 & 11 |
| PowerShell| 5.1             | For convenience scripts |
| Python    | 3.8 x64         | Required only for *pip* installation |
| AutoHotkey| v2              | Needed for GUI launcher `lld_navigator.ahk` |

> If you plan to use the standalone **EXE**, Python is *not* required.

---

## Quick Start

Choose one of the installation methods below.

### Option A – Install via **pip**

1. Ensure Python 3.8+ is in your `PATH`.  
   ```powershell
   python --version
   ```  
   Install if missing:  
   ```powershell
   winget install --id Python.Python.3.10
   ```
2. Open PowerShell and run:
   ```powershell
   pip install quicknav
   ```
3. Verify:
   ```powershell
   quicknav --version
   ```

### Option B – Download standalone EXE

1. Navigate to the latest [GitHub Releases](https://github.com/your-org/quicknav/releases).
2. Download `quicknav-<version>-win64.exe`.
3. Place it somewhere in your `PATH` (e.g. `C:\Tools`).
4. Test:
   ```powershell
   quicknav.exe --version
   ```

---

## Installing the GUI Launcher (AutoHotkey)

1. Install AutoHotkey v2 from <https://www.autohotkey.com>.  
2. Copy `lld_navigator.ahk` to any folder.  
3. Double-click the script to launch.
4. (Optional) Compile to EXE: right-click → “Compile”.

> **Tip:** After running the script, you can open or focus the QuickNav GUI window at any time by pressing **Ctrl+Alt+Q** (global shortcut).

The script internally invokes `quicknav` or `quicknav.exe` that must be reachable via `PATH`.

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `UserProfile` | inherited | Used to locate OneDrive folder |
| `QUICKNAV_PROJECT_ROOT` | *(auto)* | Override default OneDrive path |

Example (PowerShell profile):
```powershell
$Env:QUICKNAV_PROJECT_ROOT = "D:\Projects"
```

---

## Versioning

The project follows **SemVer 2.0**.  
`quicknav --version` prints the current version, loaded from `VERSION.txt` embedded into the wheel and EXE.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| ERROR:UserProfile environment variable not found | Running under non-interactive service account | Set `UserProfile` or use `QUICKNAV_PROJECT_ROOT` |
| `quicknav` not recognized | Python/Scripts not in PATH | Re-open terminal or add `%APPDATA%\Python\Scripts` to PATH |
| AutoHotkey script opens but searches forever | `quicknav` not detected | Ensure EXE or pip script is in PATH |

---

## Uninstall

```powershell
pip uninstall quicknav        # pip install users
Remove-Item quicknav*.exe     # EXE users
```

---

© 2025 Pro AV Solutions