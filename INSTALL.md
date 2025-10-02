# Project QuickNav – Installation Guide

## Overview

Project QuickNav is an intelligent project navigation assistant with machine learning capabilities that transforms how you interact with project directories. This comprehensive guide explains how to install and configure QuickNav **v3.0** with full ML and analytics features on Windows, macOS, and Linux.

🚀 **New in v3.0**: ML-powered recommendations, real-time analytics dashboard, smart navigation, and AI-enhanced project assistance.

---

## Prerequisites

### Core Requirements
| Component | Minimum Version | Notes |
|-----------|-----------------|-------|
| **Operating System** | Windows 10+, macOS 10.14+, Linux | Full cross-platform support |
| **Python** | 3.8+ (64-bit recommended) | Required for all ML features |
| **Memory** | 4GB RAM minimum, 8GB recommended | For ML processing and analytics |
| **Storage** | 500MB free space | For database, models, and cache |

### ML & Analytics Dependencies (Auto-installed)
| Component | Version | Purpose |
|-----------|---------|---------|
| **scikit-learn** | Latest | Machine learning algorithms |
| **pandas** | Latest | Data processing and analytics |
| **numpy** | Latest | Numerical computations |
| **matplotlib** | Latest | Analytics visualizations |
| **watchdog** | Latest | Real-time file monitoring |

### Optional Components
| Component | Purpose | Installation |
|-----------|---------|-------------|
| **LiteLLM** | AI chat assistant | `pip install litellm` |
| **keyboard** | Global hotkeys | `pip install keyboard` |
| **pystray** | System tray | `pip install pystray` |
| **AutoHotkey v2** | Legacy Windows GUI | Download from autohotkey.com |

---

## 🚀 Quick Start Installation

### Recommended: Full ML Installation

```bash
# 1. Clone the repository
git clone https://github.com/Saml1211/Project-QuickNav.git
cd Project-QuickNav

# 2. Install with ML dependencies
pip install -e .
pip install scikit-learn pandas numpy matplotlib watchdog

# 3. Launch the intelligent GUI
python quicknav/gui_launcher.py
```

### Alternative: Basic Installation (Legacy)

#### Option A – Install via pip (Limited Features)

```bash
# Basic installation without ML features
pip install quicknav
quicknav --version
```

#### Option B – Standalone EXE (Windows Only)

1. Download from [GitHub Releases](https://github.com/Saml1211/Project-QuickNav/releases)
2. Download `quicknav-<version>-win64.exe`
3. Place in your `PATH` (e.g. `C:\Tools`)
4. Test: `quicknav.exe --version`

**⚠️ Note**: Standalone EXE does not include ML features. Use full installation for complete functionality.

---

## 🖥️ GUI Installation Options

### Primary: Modern Tkinter GUI (Recommended)

```bash
# Launch the enhanced GUI with ML features
python quicknav/gui_launcher.py

# Or use the module approach
python -m quicknav.gui
```

**Features**:
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ ML-powered recommendations and analytics
- ✅ AI chat assistant integration
- ✅ Modern theming and accessibility
- ✅ Smart navigation with autocomplete

### Legacy: AutoHotkey GUI (Windows Only)

1. Install AutoHotkey v2 from <https://www.autohotkey.com>
2. Copy `lld_navigator.ahk` to any folder
3. Double-click the script to launch
4. (Optional) Compile to EXE: right-click → "Compile"

> **Note**: AutoHotkey GUI is legacy and lacks ML features. Use Tkinter GUI for full functionality.

### Global Hotkey

**Ctrl+Alt+Q**: Opens/focuses QuickNav window from anywhere (configurable in settings)

---

## ⚙️ Configuration & Setup

### First-Time Setup

1. **Launch GUI**: `python quicknav/gui_launcher.py`
2. **Configure Project Paths**: Settings → General → Custom Project Roots
3. **Initialize ML Components**: Automatic on first launch (30-60 seconds)
4. **Optional AI Setup**: Settings → AI → Configure API keys

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `QUICKNAV_PROJECT_ROOT` | Override OneDrive path | `D:\Projects` |
| `OPENAI_API_KEY` | AI functionality | `sk-...` |
| `ANTHROPIC_API_KEY` | AI functionality | `sk-ant-...` |
| `AZURE_API_KEY` | AI functionality | `your-azure-key` |

### ML Configuration Files

**Automatic Creation**:
- **Database**: User data directory (`AppData`/`Library`/`.config`)
- **ML Models**: `training_data/models/` (auto-created)
- **Settings**: Platform-appropriate locations with backup
- **Cache**: Temporary directory with configurable retention

### Performance Tuning

**Small Environments (<1000 documents)**:
```python
# Default settings are optimal
```

**Large Environments (10,000+ documents)**:
```python
# In settings.json:
{
  "ml": {
    "batch_size": 100,
    "max_workers": 8,
    "use_duckdb": true,
    "cache_timeout_minutes": 10
  }
}
```

---

## Versioning

The project follows **SemVer 2.0**.  
`quicknav --version` prints the current version, loaded from `VERSION.txt` embedded into the wheel and EXE.

---

## 🔧 Troubleshooting

### Common Installation Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **ML features not working** | Missing dependencies | Run `pip install scikit-learn pandas numpy matplotlib` |
| **Analytics dashboard empty** | Initial ML training | Wait 5-10 minutes after first launch |
| **GUI won't start** | Tkinter missing | Install `python3-tk` (Linux) or reinstall Python |
| **Database errors** | Permissions | Check write access to user data directory |
| **Memory issues** | Large datasets | Reduce ML model dimensions in settings |
| **Slow performance** | Cache disabled | Enable caching in Settings → Advanced |

### ML-Specific Issues

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| **No recommendations** | `RecommendationEngine not initialized` | Restart app, check logs |
| **Analytics not loading** | Database connection failed | Check database permissions |
| **File monitoring disabled** | Watchdog not installed | `pip install watchdog` |
| **Poor recommendation accuracy** | Insufficient training data | Use for 1-2 weeks to improve |

### Legacy Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `quicknav` not recognized | Python/Scripts not in PATH | Add `%APPDATA%\Python\Scripts` to PATH |
| AutoHotkey script issues | `quicknav` not detected | Use Tkinter GUI instead |
| UserProfile errors | Service account | Set `QUICKNAV_PROJECT_ROOT` |

### Performance Diagnostics

```bash
# Check ML component health
python -c "from src.ml.recommendation_engine import RecommendationEngine; print('ML Status:', RecommendationEngine().validate_models())"

# Check database connectivity
python -c "from src.database.database_manager import DatabaseManager; print('DB Status:', DatabaseManager().health_check())"

# Run performance benchmarks
python tests/test_performance_benchmarks.py
```

---

## 🗑️ Uninstall

### Complete Removal

```bash
# Uninstall Python package
pip uninstall quicknav

# Remove ML dependencies (optional)
pip uninstall scikit-learn pandas numpy matplotlib watchdog

# Remove optional components
pip uninstall litellm keyboard pystray
```

### Data Cleanup

```bash
# Remove user data (Windows)
Remove-Item -Recurse "$env:APPDATA\QuickNav"

# Remove user data (macOS)
rm -rf "~/Library/Application Support/QuickNav"

# Remove user data (Linux)
rm -rf "~/.config/QuickNav"

# Remove training data and models
rm -rf "training_data/models"

# Remove database files
rm -f "*.db" "*.duckdb"
```

### Legacy Cleanup

```bash
# Remove standalone EXE
Remove-Item quicknav*.exe

# Remove AutoHotkey script
# (manually delete lld_navigator.ahk)
```

## 📊 Verification & Testing

### Installation Verification

```bash
# Test basic functionality
python -c "import quicknav; print('✅ QuickNav imported successfully')"

# Test ML components
python -c "from src.ml.recommendation_engine import RecommendationEngine; print('✅ ML components working')"

# Test GUI launch
python quicknav/gui_launcher.py --test

# Run comprehensive tests
python -m pytest tests/ -v
```

### Performance Benchmarks

After installation, expect these performance metrics:
- **ML Initialization**: 30-60 seconds (first launch)
- **Recommendation Generation**: <500ms for 10 suggestions
- **Analytics Dashboard**: <2s load time
- **Smart Autocomplete**: <100ms response
- **Document Processing**: 50+ docs/second

## 🚀 Next Steps

1. **Launch GUI**: `python quicknav/gui_launcher.py`
2. **Configure Project Paths**: Settings → General
3. **Enable AI Features**: Settings → AI (optional)
4. **Explore Analytics**: Main GUI → Analytics tab
5. **Start Navigating**: Enter project codes or search terms

**Pro Tip**: The system learns from your usage patterns. After 1-2 weeks of use, you'll see significantly improved recommendations and predictions.

---

**Project QuickNav v3.0** - Intelligent Project Navigation with Machine Learning

© 2025 Pro AV Solutions - Enhanced with Data-Driven Intelligence