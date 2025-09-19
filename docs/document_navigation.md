# Document Navigation Feature

The Project QuickNav enhanced document navigation feature enables intelligent discovery and selection of various project artifacts including design documents, change orders, reports, floor plans, site photos, and more.

## Features

### Document Types Supported

- **Low-Level Design (LLD)** - Visio and PDF design documents
- **High-Level Design (HLD)** - System architecture documents
- **Change Orders** - Project modification documents
- **Sales & PO Reports** - Financial and procurement reports
- **Floor Plans** - Architectural drawings and layouts
- **Scope Documents** - Project handover and proposal documents
- **QA/ITP Reports** - Quality assurance and inspection reports
- **SWMS** - Safe Work Method Statements
- **Supplier Quotes** - Vendor quotations and pricing
- **Site Photos** - Progress and installation images

### Intelligent Version Selection

The system automatically parses and ranks document versions using multiple patterns:

- **REV-based**: `REV 100`, `REV 101`, `Rev 200` (integer comparison)
- **Period-based**: `1.03`, `2.01` (major.minor tuple comparison)
- **Letter-based**: `Rev A`, `Rev D`, `(C)` (alphabetical ordering)
- **Status Tags**: `AS-BUILT` > `SIGNED` > `FINAL` > `APPROVED` > `DRAFT`

### Advanced Filtering

- **Room-specific**: Filter by room number (e.g., `Room 5`, `Room01`)
- **Change Order**: Filter by CO number (e.g., `CO123`, `Change Order 5`)
- **Date range**: Parse dates from filenames with multiple formats
- **Archive exclusion**: Automatically exclude `ARCHIVE` and `OLD DRAWINGS` folders
- **Project code matching**: Boost documents with matching 5-digit project codes

### High-DPI Scaling Support

The AutoHotkey GUI automatically scales for high-resolution displays:
- Dynamic DPI detection using `A_ScreenDPI / 96.0`
- Proportional scaling of all controls and text
- `+DPIScale` option for consistent rendering
- Tested on 2K/4K displays without overlap or cutoff

## Usage

### Command Line Interface

```bash
# Find latest LLD documents for project 17741
quicknav doc 17741 --type lld

# Find change orders for specific CO number
quicknav doc 17741 --type change_order --co 5

# Find room-specific QA reports including archived
quicknav doc 17741 --type qa_itp --room 12 --include-archive

# Show all available options for user selection
quicknav doc 17741 --type floor_plans --choose

# Use custom root directory
quicknav doc 17741 --type lld --root "\\fileserver\projects"
```

### AutoHotkey GUI (Enhanced)

1. **Launch**: Press `Ctrl+Alt+Q` or run `lld_navigator.ahk`
2. **Select Mode**: Choose "Find Documents" radio button
3. **Enter Project**: Input 5-digit project number or search term
4. **Document Type**: Select from dropdown (LLD, Change Orders, etc.)
5. **Filters**: Optionally specify room number, CO number, or version filter
6. **Execute**: Click "Find Documents" for auto-selection or "Choose From List" for manual selection

### Python API

```python
from quicknav.doc_navigator import navigate_to_document

# Auto-select best document
result = navigate_to_document(
    project_path="/path/to/project",
    doc_type="lld",
    selection_mode="auto",
    project_code="17741"
)

# Get latest version of each series
result = navigate_to_document(
    project_path="/path/to/project",
    doc_type="lld",
    selection_mode="latest"
)

# Filter by room and exclude archive
result = navigate_to_document(
    project_path="/path/to/project",
    doc_type="qa_itp",
    room_filter=5,
    exclude_archive=True
)
```

## Configuration

### Custom Root Directories

Add alternative project root paths in `%APPDATA%\QuickNav\settings.json`:

```json
{
  "custom_roots": [
    "%USERPROFILE%\\OneDrive - Pro AV Solutions\\Project Files",
    "\\\\fileserver\\Share\\Projects",
    "C:\\Local\\Projects"
  ]
}
```

### Document Type Configuration

Customize document types and their classification rules:

```json
{
  "doc_types": [
    {
      "type": "lld",
      "label": "Low-Level Design",
      "folders": ["4. System Designs"],
      "exts": [".vsdx", ".vsd", ".pdf"],
      "name_includes_any": ["LLD", "Low Level"],
      "exclude_folders": ["OLD DRAWINGS", "ARCHIVE"],
      "version_patterns": ["REV\\s*(\\d+(?:\\.\\d+)?)", "\\((\\d+)\\)$"],
      "status_tags": ["AS-BUILT", "DRAFT", "FINAL"],
      "priority": 100
    }
  ]
}
```

### Ranking Weights

Adjust scoring weights for document selection:

```json
{
  "ranking_weights": {
    "match_project_code": 5.0,
    "in_preferred_folder": 2.0,
    "status_as_built": 2.0,
    "status_signed": 1.5,
    "newer_version": 3.0,
    "archive_penalty": -2.0
  }
}
```

## Examples from Real Projects

### Version Progression Example
```
17741 - Project LLD Level 1 - REV 100.vsdx          # Initial
17741 - Project LLD Level 1 - REV 101.vsdx          # Revision
17741 - Project LLD Level 1 - REV 102 AS-BUILT.vsdx # Final
```
**Result**: Auto-selects REV 102 AS-BUILT (highest version + AS-BUILT status)

### Sales Report Example
```
17741 - QPS MTR Room Upgrades - Sales & PO Report 23.09.22.pdf
```
**Parsed**: Project code `17741`, Date `2022-09-23`, Type `sales_po`

### Change Order Example
```
17741 - Change Order 5 - Additional Work APPROVED.pdf
```
**Parsed**: Project code `17741`, CO number `5`, Status `APPROVED`

### Room-Specific Example
```
17741 - Room 12 QA Report SIGNED.pdf
```
**Filters**: `--room 12` returns only this document

## Implementation Notes

### Performance Optimizations

- **Recursive scanning**: Limited to 5 levels deep by default
- **Extension filtering**: Only scan relevant file types per document type
- **Archive exclusion**: Skip `ARCHIVE`/`OLD DRAWINGS` folders unless explicitly included
- **Caching**: Results cached for 60 minutes (configurable)

### Error Handling

- **Graceful fallback**: If OneDrive unavailable, try custom roots then test environment
- **Partial matches**: Return parent folder if no documents found
- **Multiple matches**: Present selection dialog with metadata columns
- **Invalid input**: Clear error messages with suggestions

### Corpus Validation

The implementation is validated against real training data examples:
- **REV patterns**: `REV 100`, `REV 101`, `Rev.D`, `(A)`
- **Status combinations**: `AS-BUILT`, `DRAFT`, `SIGNED`
- **Date formats**: `23.09.22`, `240606`, `17-12-2024`
- **Project codes**: All examples include valid 5-digit codes

## Testing

Run the comprehensive test suite:

```bash
# Run all document navigation tests
python -m pytest tests/test_doc_navigator.py -v

# Test specific functionality
python -m pytest tests/test_doc_navigator.py::TestDocumentParser::test_rev_numeric_parsing -v

# Test with real training data
python -m pytest tests/test_doc_navigator.py::TestIntegration::test_corpus_validation_rev_patterns -v
```

## Migration from Legacy

The enhanced document navigation is fully backward compatible:

- **Existing hotkey**: `Ctrl+Alt+Q` still works
- **Folder navigation**: Original subfolder selection remains available
- **CLI compatibility**: `quicknav 17741` works as before (legacy mode)
- **Settings preservation**: Existing settings.json files are automatically migrated

## Troubleshooting

### Common Issues

1. **No documents found**: Check project folder structure and document naming conventions
2. **DPI scaling issues**: Verify `A_ScreenDPI` detection and `+DPIScale` option
3. **Custom roots not working**: Ensure environment variables are expanded and paths exist
4. **Version parsing fails**: Check filename patterns against supported regex patterns

### Debug Mode

Enable debug output to troubleshoot:
- **CLI**: Add `--debug` flag to any command
- **GUI**: Check "Show Debug Output" option
- **Logs**: View Python backend command and output details

### Performance Issues

If scanning is slow:
- Reduce `max_scan_depth` in settings
- Enable `cache_enabled` for repeated scans
- Use more specific document type filters
- Exclude network drives from custom_roots if latency is high