# Project QuickNav - File Organization Plan

## Current State Analysis

Based on the project structure review and Memory Bank documentation, the following files in the root directory need to be organized into appropriate subdirectories:

### Unclassified Files Requiring Organization:

#### Analysis & Intelligence Scripts (should move to `analysis/`)
- `ai_agent_training_generator.py` - AI agent training data and insights generator
- `hybrid_training_analyzer.py` - Hybrid rule-based + AI training data processor  
- `project_extractor.py` - Comprehensive project-specific information extractor
- `scope_generator.py` - Real scope content extraction from project documents

#### Core Application Files (should move to `src/` or stay organized)
- `find_project_path.py` - Core Python backend for navigation
- `lld_navigator.ahk` - AutoHotkey GUI frontend
- `test_find_project_path.py` - Unit tests for core functionality

#### Analysis Results & Outputs (should move to `analysis-results/`)
- `ai_agent_training_dataset.json` - Generated training dataset
- `ai_agent_training_report.txt` - Analysis report
- `ai_analysis_results.json` - AI analysis outputs
- `hybrid_analysis_results.json` - Hybrid analysis results
- `project_profile_17741.json` - Project profile for 17741
- `project_profile_20692.json` - Project profile for 20692

#### Project Scope Documents (should move to `project-scopes/`)
- `17741_project_scope.md` - Project 17741 scope
- `17742_detailed_scope.md` - Detailed scope for 17742  
- `17742_project_scope.md` - Project 17742 scope

#### Documentation (should move to `docs/`)
- `NEW_SCRIPTS_DOCUMENTATION.md` - Documentation for analysis scripts
- `AHK_TEST_PLAN.md` - AutoHotkey testing plan

#### Logs & Debug (should move to `logs/`)
- `error_log.txt` - Error logging output

#### Build & Setup (should move to `build/` or `setup/`)
- `setup.py` - Python package setup

## Proposed Directory Structure

```
Project-QuickNav-MVP/
├── analysis/                    # Analysis & intelligence scripts
│   ├── ai_agent_training_generator.py
│   ├── hybrid_training_analyzer.py
│   ├── project_extractor.py
│   └── scope_generator.py
│
├── analysis-results/            # Generated analysis outputs
│   ├── ai_agent_training_dataset.json
│   ├── ai_agent_training_report.txt
│   ├── ai_analysis_results.json
│   ├── hybrid_analysis_results.json
│   ├── project_profile_17741.json
│   └── project_profile_20692.json
│
├── build/                       # Build and setup files
│   └── setup.py
│
├── docs/                        # Additional documentation
│   ├── AHK_TEST_PLAN.md
│   └── NEW_SCRIPTS_DOCUMENTATION.md
│
├── logs/                        # Log files and debug output
│   └── error_log.txt
│
├── project-scopes/              # Generated project scope documents
│   ├── 17741_project_scope.md
│   ├── 17742_detailed_scope.md
│   └── 17742_project_scope.md
│
├── src/                         # Core application source
│   ├── find_project_path.py
│   ├── lld_navigator.ahk
│   └── test_find_project_path.py
│
├── memory-bank/                 # (existing) Memory bank documentation
├── mcp_server/                  # (existing) MCP server
├── quicknav/                    # (existing) QuickNav CLI
├── scripts/                     # (existing) Build & utility scripts
├── training_data/               # (existing) Training data files
└── <standard project files>    # README.md, VERSION.txt, etc.
```

## Organization Strategy

### Phase 1: Create New Directories
- Create: `analysis/`, `analysis-results/`, `build/`, `docs/`, `logs/`, `project-scopes/`, `src/`

### Phase 2: Move Files by Category
1. **Analysis Scripts** → `analysis/`
2. **Analysis Results** → `analysis-results/`
3. **Core Source Files** → `src/`
4. **Documentation** → `docs/`
5. **Build Files** → `build/`
6. **Logs** → `logs/`
7. **Project Scopes** → `project-scopes/`

### Phase 3: Update Path References
- Update import statements and file paths in scripts
- Update documentation references
- Verify MCP server and other integrations still work
- Update any hardcoded paths in AutoHotkey script

### Phase 4: Update Memory Bank Documentation
- Update activeContext.md and systemPatterns.md with new structure
- Document the improved organization in progress.md

## Benefits of This Organization

1. **Clear Separation of Concerns**: Each directory has a specific purpose
2. **Scalability**: Easy to add new analysis scripts or results without cluttering root
3. **Professional Structure**: Follows standard software project conventions
4. **Easier Navigation**: Developers can quickly find relevant files
5. **Better Maintenance**: Related files are grouped together
6. **Cleaner Root**: Only essential project files remain in root directory

## Implementation Notes

- Maintain backward compatibility where possible
- Update any import statements that reference moved files
- Ensure the MCP server and AutoHotkey GUI continue to work with new paths
- Update .gitignore if needed for new directories
- Consider symlinks for critical files if path changes break integrations 