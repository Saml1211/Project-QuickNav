# Active Context – Project QuickNav

## Current Focus

All major components of Project QuickNav have been implemented and enhanced:
- Python backend for folder lookup with document discovery capabilities
- AutoHotkey GUI for user interaction with training data generation option
- MCP server for AI/automation integration

**NEW: Document Training Data System** - Recently added comprehensive document discovery and training data generation functionality for AI/ML workflows.

Current focus has shifted to feature enhancement, with the addition of training data capabilities for AI document processing workflows.

## Recent Changes

- **MAJOR ADDITION: Training Data System** - Added comprehensive document discovery functionality:
  - `discover_documents()` function for recursive PDF, Word, and RTF document discovery
  - `training_script()` function for bulk training data generation across all projects
  - AHK GUI toggle for "Generate Training Data" option
  - Unique filename generation with project number suffixes
  - Organized storage in dedicated `training_data/` directory
- Completed initial implementation of all three core components.
- Integrated backend with both frontend and MCP server.
- Validated end-to-end operation for user and AI workflows.
- Fixed issue with AHK script not starting the Python backend correctly by removing the initialization code and only running the backend when needed with the proper job number argument.

## Training Data Features

- **Document Types**: Discovers .pdf, .docx, .doc, .rtf files recursively
- **Smart Filename Generation**: 
  - Single projects: `training_data_17741.json`
  - Search results: `training_data_search_YYYYMMDD_HHMMSS.json`
- **No Overwriting**: Each run creates unique files preserving all previous data
- **JSON Structure**: Project folder, document path, document name, and empty extracted_info field
- **GUI Integration**: Toggle checkbox in AHK interface for optional training data generation

## Potential Next Steps

- **AI Integration Enhancement:** Develop document content extraction for the extracted_info field
- **Training Data Management:** Create utilities for managing, merging, or analyzing training data files
- **Comprehensive Testing:** Expand test coverage for new training data functionality
- **Packaging & Distribution:** Update installation to include training data capabilities
- **User Documentation:** Update guides to include training data workflow instructions
- **User Feedback:** Solicit feedback on training data usefulness and workflow integration
- **Cross-platform Investigation:** Assess feasibility of replacing/upgrading the frontend for non-Windows environments

## Active Decisions & Considerations

- Decision to keep training data generation optional via GUI toggle for workflow flexibility
- Chose to generate training data only for actively accessed projects, not bulk processing by default
- Decision to organize training files in dedicated directory with intelligent naming for easy management
- Monitor for performance impact of document discovery on large project folders
- Decision to keep the inter-process protocol simple and based on stdout for maximum reliability
- Decision to run the Python backend only when needed rather than as a persistent process, improving reliability and simplifying the workflow
- Consideration for future enhancements such as content extraction, search history, favorites, or direct integration with code editors