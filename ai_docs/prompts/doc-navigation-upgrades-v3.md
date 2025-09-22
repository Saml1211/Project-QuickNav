You are an expert in the Project QuickNav repository. Implement an extended document navigation feature that enables opening various project artifacts (Visio LLD/HLD, change orders, sales/PO reports, floor plans, scope docs, site photos, QA-ITP reports, SWMS, supplier quotes, etc.) directly from the tool. Include intelligent version/revision selection, filtering, previews, high-DPI GUI scaling, and support for alternative file roots (e.g., network shares for users without OneDrive shortcuts). Maintain seamless UX without disrupting existing functionality. Work exclusively within this repo; do not modify external projects.

**Repository Context**

* **Core Components**:
  - Python Backend: `src/find_project_path.py` (path resolution: `get_onedrive_folder()`, `get_project_folders()`, `get_range_folder()`, `search_project_dirs()`, `discover_documents()` for scanning; CLI in `quicknav/cli.py` outputs SUCCESS:path|ERROR:msg|SELECT:path1|path2).
  - AHK Frontend: `src/lld_navigator.ahk` (GUI with radios for folders like "4. System Designs"; hotkey ^!q; tray menu). Controller: `lld_navigator_controller.ahk` (business logic: `Controller_OpenProject()`, `AsyncProcessCallback` for async Python calls via temp files; JSON persistence in %AppData%\QuickNav; logging via `LogError()`; validation in `ValidateAndNormalizeInputs()`).
  - Config: Settings in %AppData%\QuickNav\settings.json (load/save via `Controller_LoadSettings()`); recents/favorites in recent.json.
  - Tests: Python in `tests/test_find_project_path.py`; AHK suite in `tests/ahk/` (extend with new test_*.ahk for doc nav).
  - Data: Training samples in `training_data/` (e.g., filenames with REV patterns); use for parser validation.
* **Constraints**: No new deps (keep Python stdlib + existing; AHK v2). Incremental commits. Non-destructive (read/list/open only). Graceful errors (e.g., offline roots via try-catch in path funcs).

**Data-Informed Patterns from Real-World Repositories**

* **Project Identity and Foldering**:
  - Project root folders follow "[5-digit code] - [Client or Location] - [Project Type or Phase]".
  - Common phase directories: 1. Sales Handover, 2. BOM & Orders, 3. PMO, 4. System Designs, 5. Floor Plans, 6. Site Photos, ENG INFO, OLD DRAWINGS, ARCHIVE, Pre Sales, Received [date], SWM for compliance.
  - Hierarchical depth: 3-5 levels with room-specific branching (e.g., RoomNN subfolders).
* **File Naming and Extensions**:
  - Predominantly PDF with DOCX/RTF for registers and scopes, VSDX for designs, images (.jpg/.png/.heic) for photos.
  - Names include 5-digit project code prefix, document type, description, variant/revision, optional date.
  - Status tags: DRAFT, AS-BUILT, APPROVED, SIGNED, FINAL.
* **Versioning and Dates**:
  - Revisions: Rev 100/200 (int compare), 1.03/2.01 (major.minor tuple), parenthetical (2), lettered (A=1, B=2, Rev D < numeric).
  - Dates: DD.MM.YY, YYMMDD, Received dd-mm-yyyy; normalize with day_first=True, handle MM/DD vs DD/MM via context.
* **Archive and Templates**:
  - OLD DRAWINGS/ARCHIVE: Valid but deprioritized; exclude unless necessary.
  - Non-prefixed templates: Global, deprioritize.

**Scope & Goals**

* Extend to support multiple doc types in relevant folders (map type → folder: LLD/HLD → "4. System Designs", Change Orders/Sales PO → "2. BOM & Orders", Floor Plans → "5. Floor Plans", Scope Docs → "1. Sales Handover", Site Photos → "6. Site Photos", QA-ITP → "3. PMO/QA-ITP Reports" (room-scoped), SWMS → "3. PMO" (v[version]), Supplier Quotes → "1. Sales Handover").
* After resolving project path (reuse existing), scan for docs; auto-open latest/relevant or provide filtered picker with previews (e.g., photo thumbnails).
* **Custom Roots**: Allow fallback/configurable roots (e.g., env var QUICKNAV_ROOT or settings.custom_roots:[] array). Default: OneDrive. If primary fails (e.g., !os.path.isdir()), try alternatives silently. No UX change: Auto-detect/fallback; optional CLI flag `--root /path` or GUI settings dialog.
* **High-DPI Scaling**: Fix GUI for 4K (use `Gui +DPIScale`; scale fonts/controls with `A_ScreenDPI / 96`; dynamic y-positions: base + (spacing * scale)). Reference `lld_navigator_analysis.md` (DPI inconsistencies, fixed sizes); test: Elements resize proportionally without overlap on 2K/4K.
* Design modularly: Easy to add types (config-driven patterns/folders).

**Root Resolution and Project Discovery**

* Root resolution order: CLI override first, configured custom roots next, default OneDrive last. Fast checks, silent IO errors, INFO logging.
* Project identity: Prefer 5-digit code from folder/filename prefix. Require strong signal.
* If zero candidates: Return parent phase folder/project root + message.

**Config Schema for Doc Types and Ranking**

Extend settings.json with detailed schema for robustness:

```json
{
  "custom_roots": [
    "%OneDrive%",
    "%USERPROFILE%/Documents/Projects",
    "\\\\fileserver\\Share\\Projects"
  ],
  "doc_types": [
    {
      "type": "lld",
      "label": "Low-Level Design",
      "folders": ["4. System Designs"],
      "exts": [".vsdx", ".pdf"],
      "name_includes_any": ["LLD"],
      "exclude_folders": ["OLD DRAWINGS", "ARCHIVE"],
      "version_patterns": ["Rev\\s*(\\d+(?:\\.\\d+)?)", "\\((\\d+)\\)$", "_\\(([A-Z])\\)"],
      "status_tags": ["AS-BUILT", "DRAFT"],
      "date_patterns": ["(\\d{2}\\.\\d{2}\\.\\d{2})", "(\\d{6})", "Received\\s+(\\d{2}-\\d{2}-\\d{4})"],
      "room_key_regex": "Room\\d+",
      "priority": 100
    },
    {
      "type": "change_order",
      "label": "Change Orders",
      "folders": ["2. BOM & Orders", "2. BOM & Orders/Sales & Change Orders", "2. BOM & Orders/Sales & Change Orders/All COs"],
      "exts": [".pdf"],
      "name_includes_any": ["Change Order", "CO"],
      "co_patterns": ["\\bCO\\s*\\d+\\b", "\\bChange Order\\s*\\d+\\b"],
      "status_tags": ["APPROVED", "DRAFT"],
      "date_patterns": ["(\\d{2}\\.\\d{2}\\.\\d{2})", "(\\d{6})"],
      "exclude_folders": ["ARCHIVE"],
      "priority": 90
    },
    {
      "type": "sales_po",
      "label": "Sales and PO Reports",
      "folders": ["2. BOM & Orders"],
      "exts": [".pdf"],
      "name_includes_any": ["Sales & PO Report"],
      "date_patterns": ["(\\d{2}\\.\\d{2}\\.\\d{2})", "(\\d{6})"],
      "priority": 80
    },
    {
      "type": "floor_plans",
      "label": "Floor Plans",
      "folders": ["5. Floor Plans", "1. Sales Handover/Floor Plans"],
      "exts": [".pdf"],
      "name_includes_any": ["GA PLAN", "SHEET", "A-"],
      "version_patterns": ["_\\(([A-Z])\\)"],
      "room_key_regex": "Room\\d+",
      "status_tags": ["AS-BUILT", "OLD"],
      "exclude_folders": ["OLD DRAWINGS", "ARCHIVE"],
      "priority": 85
    },
    {
      "type": "scope",
      "label": "Scope and Handover",
      "folders": ["1. Sales Handover"],
      "exts": [".pdf", ".docx"],
      "name_includes_any": ["Project Handover Document", "Scope", "Proposal"],
      "status_tags": ["FINAL", "DRAFT"],
      "priority": 70
    },
    {
      "type": "qa_itp",
      "label": "QA and ITP Reports",
      "folders": ["3. PMO", "QA-ITP Reports"],
      "exts": [".pdf"],
      "room_key_regex": "Room\\d+",
      "status_tags": ["SIGNED", "FINAL"],
      "priority": 75
    },
    {
      "type": "swms",
      "label": "SWMS",
      "folders": ["3. PMO"],
      "exts": [".pdf"],
      "name_includes_any": ["SWMS"],
      "version_patterns": ["v(\\d+(?:\\.\\d+)?)"],
      "status_tags": ["SIGNED", "DRAFT"],
      "priority": 78
    },
    {
      "type": "supplier_quotes",
      "label": "Supplier Quotes",
      "folders": ["1. Sales Handover"],
      "exts": [".pdf", ".xlsx"],
      "name_includes_any": ["Quote", "Supplier"],
      "status_tags": ["APPROVED", "FINAL"],
      "priority": 65
    },
    {
      "type": "photos",
      "label": "Site Photos",
      "folders": ["6. Site Photos"],
      "exts": [".jpg", ".jpeg", ".png", ".heic"],
      "thumbs": true,
      "priority": 60
    }
  ],
  "date_parse_defaults": {
    "day_first": true,
    "y2k_window": 1970
  },
  "ranking_weights": {
    "match_project_code": 5.0,
    "in_preferred_folder": 2.0,
    "not_in_archive_or_old": 1.5,
    "status_as_built": 2.0,
    "status_signed": 1.5,
    "status_draft": -0.5,
    "archive_penalty": -2.0,
    "newer_version": 3.0,
    "newer_date": 1.0,
    "room_match": 1.5,
    "ext_preference": 0.5
  }
}
```

**Version/Revision Parsing Model**

* **Primary (REV-based)**: `REV \d{3}` (e.g., REV 100 → 100, REV 109 → 109, REV 200 → 200). Case-insensitive; variants: rev/Rev, spaces/underscores (REV_100, rev 100). Parse as int; higher number wins (109 < 200).
* **Alternative (Period-based)**: `(\d+)\.(\d+)` after prefix (e.g., "1.00" → (1,0), "1.09" → (1,9), "2.01" → (2,1)). Case-insensitive; variants: Rev.1.00, v1_09. Compare: Major desc, then minor desc (1.09 < 2.01). Fallback if no REV match.
* **Other**: Parenthetical (2), lettered (A=1, B=2, Rev D=0.D < numeric). Prioritize REV > dotted > parenthetical/lettered.
* **Tags**: `AS BUILT`/`AS-BUILT`/`ASBUILT` (case-insens., ±hyphens); prioritize as "final" (boost sort weight). SIGNED > FINAL > APPROVED > DRAFT.
* **Dates**: Normalize with day_first=True; handle DD.MM.YY, YYMMDD, Received [date].
* **Project Code**: Regex for 5-digit prefix (e.g., 17741).
* **CO Numbers**: CO[Number] patterns.
* **Sheet Numbers**: SHEET [Number] for plans.
* **Grouping**: Strip version/tags for series (e.g., "Project LLD – Level 1" from "Project LLD – Level 1 – REV 100 AS-BUILT.vsdx"). Multi-series support (floors/variants).
* **Corpus Alignment**: Validate against `training_data/` samples:
  - REV examples: `17741 - R 51 - ... - REV 100.pdf`, `... - REV 101 AS-BUILT.pdf`, `... - REV 100 AS-BUILT.pdf`.
  - Period examples: `E 403_ ... Rev.D markup.pdf` (D=0.D?), `CD-...-Rev.0.pdf`, `12074_..._Rev 2 240606_DRAFT.pdf` (hybrid: Rev 2 + date).
  - Note discrepancies in comments (e.g., "Corpus uses PDF not VSDX; adapt parser"; "DRAFT tag: deprioritize unless filtered").
* **Parser**: Flexible regex chain (try REV → period → date → time); return {series:str, version:(int|float|date), tags:set, is_as_built:bool, is_initial:bool (100/1.00), mod_time:datetime}. Handle ambiguities (log INFO).
* **Tiebreak**: Status (AS-BUILT > FINAL > DRAFT), then date, then mod time. Deprioritize non-prefixed templates.

**Scanning and Classification**

* Traverse recursively but exclude deep nests unless room-matched.
* Include by extension, name hints; exclude ARCHIVE/OLD unless necessary.
* Extract metadata: Project code, CO number, version/tags, date (normalized), room key, mod time, path flags (in_preferred_folder, in_archive).

**Ranking and Selection**

* Weighted score: Boost for project_code_match, in_preferred_folder, room_match; penalty for archive (-2.0); status boosts (SIGNED +1.5, AS-BUILT +2.0, DRAFT -0.5); newer version/date.
* Auto-open if top score >30% over next; else picker with columns: Type, Version, Status, Date, Room, Path.
* For photos: Enable thumbs; for plans: Detect GA PLAN or A-[Number].

**User Actions (GUI + CLI)**

* **Auto-Open Latest**: Default; per type/series: Highest version (REV > period); primary series tiebreaker: Recent mod time → contains type keyword (e.g., "LLD") → shortest name. Comment rules explicitly.
* **Choose Doc**: Picker with:
  - Type selector (dropdown: LLD, HLD, Change Order, etc.).
  - Toggles/Filters: Latest/series only | All; As Built/Initial/Latest/No filter; Date range; Room [NN]; Exclude Archive.
  - Sort: Series alpha → version desc (custom cmp: REV int > period tuple > date) → mod time desc.
  - Previews: Photos (AHK Picture control, 100x100 thumbs); docs (icon or first-page via temp PDF render if feasible).
* **Open**: Default app (Visio for .vsdx, Acrobat/Word for PDF/DOCX, image viewer for JPG). Reuse AHK `Run` or Python `os.startfile()`.
* **CLI**: `quicknav doc --type lld|hld|changeorder|floorplan|scope|photos|qa_itp|swms|supplier_quotes --project 12345 [--latest|--choose] [--filter asbuilt|initial|latest|date:YYYY-MM-DD] [--root /custom/path] [--co [number]] [--room [NN]] [--exclude-archive]`. Output: SUCCESS:path|SELECT:path1|...|ERROR:msg (extend cli.py subparser).
* **AHK Enhancements**: Add room dropdown, CO input; high-DPI scales thumbnails.

**Implementation Notes**

* **Python (`quicknav/doc_navigator.py`)**:
  - Functions: `get_project_docs_path(project_path:str, doc_type:str) → str` (map type→folder, append to project).
  - `scan_docs(folder:str, doc_type:str) → list[dict]` (os.walk; filter extensions: .vsdx/.vsd for Visio, .pdf/.docx for reports/scope, .dwg/.pdf for plans, .jpg/.png for photos).
  - `parse_doc_metadata(fn:str, doc_type:str) → dict` (regex for versions/tags; os.path.getmtime()).
  - `select_docs(candidates:list, mode:str, filter:str, primary_series_rule:bool) → path_or_list` (group by series; sort custom; apply filters; log decisions).
  - CLI Integration: Extend `quicknav/cli.py` (argparse subcommand 'doc'; call navigator; format output like main()).
  - Roots: Try settings['custom_roots'] or os.environ['QUICKNAV_ROOT']; chain with OneDrive (first success).
  - Edge: No docs → return parent path + msg; ties → log + pick by time; offline → try next root or error.
* **AHK GUI (`src/lld_navigator.ahk` + controller)**:
  - Add: DocType dropdown (map to folders); integrate with existing radios (disable if type-specific).
  - Picker: Extend search ListView (columns: Series|Version|Tags|ModTime|Preview); for photos, AddPicture w/ thumb (Gdip lib if needed, but keep simple).
  - Scaling: Global scale := A_ScreenDPI / 96; positions: x += scale*10, etc. Set `Gui +DPIScale +Resize`; OnSize handler (reuse SearchGuiSize).
  - Backend: Extend `Controller_OpenProject` (add doc_type param; cmd += f" --type {doc_type}"); handle new SELECT for lists.
  - Settings: Add 'custom_roots' array to settings.json; GUI dialog for adding paths (extend prefs if exists).
  - Previews: For images, temp thumb: Run PowerShell resize; display in ListView subitem or tooltip.
* **Extensibility**: Config in settings.json: doc_types: [{type:'lld', folder:'4. System Designs', patterns:['REV \d{3}', '\d+\.\d+'], exts:['.vsdx']}, ...]. Load in Python/AHK.
* **Logging**: Extend `LogError` (INFO for parses/selections); respect debug mode.
* **Non-Destructive**: Scan only; open via shell.

**Deliverables**

* Modules: `quicknav/doc_navigator.py` (parsers/scanners); update cli.py.
* AHK: Extended lld_navigator.ahk (type selector, scaled layout, previews); controller updates (doc_type handling, root fallback).
* Config: settings.json schema/docs for custom_roots/doc_types.
* README: "Document Navigation" section (usage per type, CLI flags, scaling notes, custom roots setup).
* Tests: Python (parse REV/period/tags; selection logic; root fallback) in tests/test_doc_navigator.py. AHK: test_doc_types.ahk, test_scaling.ahk (simulate DPI).
* Changelog: vX.Y - Doc nav extension, scaling fixes, custom roots.

**Acceptance Criteria**

* **Parsing**: Handles REV 100/200, 1.09<2.01; corpus examples (e.g., Rev.D=0.D); tags prioritize As Built/Signed.
* **Doc Types**: LLD opens latest Visio; photos preview thumbs + open full; reports filter by date; QA-ITP room-scoped; SWMS v[version]; Supplier Quotes under Sales.
* **Ranking**: Project code match +5.0, archive -2.0, SIGNED +1.5; auto-open >30% over next.
* **Corpus Examples**: "17741 - QPS MTR Room Upgrades - Bulk Order - Sales & PO Report 23.09.22.pdf" → code 17741, date 2022-09-23; Rev 1.03 < Rev 2.01; Room01 filters correctly; fallback to parent if no candidates.
* **Scaling**: GUI adapts to 4K (controls 2x size, no cutoff); 2K unchanged.
* **Roots**: --root /network works; falls back if OneDrive offline; GUI settings add paths.
* **UX**: Incremental (existing project open unchanged); choose mode paginates large lists; all CLI/GUI.
* No regressions: Core nav works; tests pass.

**Milestones**

1. Parser module with regex for project code, CO, sheet numbers, date normalization.
2. CLI extension with --co, --room, --exclude-archive flags.
3. AHK scaling + type selector, room dropdown, CO input, thumbnail scaling.
4. Full integration with config schema, ranking weights, scanning exclusions.
5. Tests for corpus alignment, acceptance criteria.

**Non-Goals**

No OCR, no renaming, no file modification.

Implement step-by-step: 1) Parser module. 2) CLI extension. 3) AHK scaling + type selector. 4) Full integration/tests. Small commits.