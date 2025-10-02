You are an expert in the Project QuickNav repository. Implement an extended document navigation feature that enables opening various project artifacts (Visio LLD/HLD, change orders, sales/PO reports, floor plans, scope docs, site photos, etc.) directly from the tool. Include intelligent version/revision selection, filtering, previews, high-DPI GUI scaling, and support for alternative file roots (e.g., network shares for users without OneDrive shortcuts). Maintain seamless UX without disrupting existing functionality. Work exclusively within this repo; do not modify external projects.

**Repository Context**

* **Core Components**:
  - Python Backend: `src/find_project_path.py` (path resolution: `get_onedrive_folder()`, `get_project_folders()`, `get_range_folder()`, `search_project_dirs()`, `discover_documents()` for scanning; CLI in `quicknav/cli.py` outputs SUCCESS:path|ERROR:msg|SELECT:path1|path2).
  - AHK Frontend: `src/lld_navigator.ahk` (GUI with radios for folders like "4. System Designs"; hotkey ^!q; tray menu). Controller: `lld_navigator_controller.ahk` (business logic: `Controller_OpenProject()`, `AsyncProcessCallback` for async Python calls via temp files; JSON persistence in %AppData%\QuickNav; logging via `LogError()`; validation in `ValidateAndNormalizeInputs()`).
  - Config: Settings in %AppData%\QuickNav\settings.json (load/save via `Controller_LoadSettings()`); recents/favorites in recent.json.
  - Tests: Python in `tests/test_find_project_path.py`; AHK suite in `tests/ahk/` (extend with new test_*.ahk for doc nav).
  - Data: Training samples in `training_data/` (e.g., filenames with REV patterns); use for parser validation.
* **Constraints**: No new deps (keep Python stdlib + existing; AHK v2). Incremental commits. Non-destructive (read/list/open only). Graceful errors (e.g., offline roots via try-catch in path funcs).

**Scope & Goals**

* Extend to support multiple doc types in relevant folders (map type → folder: LLD/HLD → "4. System Designs", Change Orders/Sales PO → "2. BOM & Orders", Floor Plans → "5. Floor Plans", Scope Docs → "1. Sales Handover", Site Photos → "6. Site Photos").
* After resolving project path (reuse existing), scan for docs; auto-open latest/relevant or provide filtered picker with previews (e.g., photo thumbnails).
* **Custom Roots**: Allow fallback/configurable roots (e.g., env var QUICKNAV_ROOT or settings.custom_roots:[] array). Default: OneDrive. If primary fails (e.g., !os.path.isdir()), try alternatives silently. No UX change: Auto-detect/fallback; optional CLI flag `--root /path` or GUI settings dialog.
* **High-DPI Scaling**: Fix GUI for 4K (use `Gui +DPIScale`; scale fonts/controls with `A_ScreenDPI / 96`; dynamic y-positions: base + (spacing * scale)). Reference `lld_navigator_analysis.md` (DPI inconsistencies, fixed sizes); test: Elements resize proportionally without overlap on 2K/4K.
* Design modularly: Easy to add types (config-driven patterns/folders).

**Version/Revision Parsing Model**

* **Primary (REV-based)**: `REV \d{3}` (e.g., REV 100 → 100, REV 109 → 109, REV 200 → 200). Case-insensitive; variants: rev/Rev, spaces/underscores (REV_100, rev 100). Parse as int; higher number wins (109 < 200).
* **Alternative (Period-based)**: `(\d+)\.(\d+)` after prefix (e.g., "1.00" → (1,0), "1.09" → (1,9), "2.01" → (2,1)). Case-insensitive; variants: Rev.1.00, v1_09. Compare: Major desc, then minor desc (1.09 < 2.01). Fallback if no REV match.
* **Tags**: `AS BUILT`/`AS-BUILT`/`ASBUILT` (case-insens., ±hyphens); prioritize as "final" (boost sort weight).
* **Other**: Dates (YYYYMMDD in filename, e.g., Rev2_240606 → sort by date); numeric prefixes for photos (001.jpg > 002.jpg) or mod time.
* **Grouping**: Strip version/tags for series (e.g., "Project LLD – Level 1" from "Project LLD – Level 1 – REV 100 AS-BUILT.vsdx"). Multi-series support (floors/variants).
* **Corpus Alignment**: Validate against `training_data/` samples:
  - REV examples: `17741 - R 51 - ... - REV 100.pdf`, `... - REV 101 AS-BUILT.pdf`, `... - REV 100 AS-BUILT.pdf`.
  - Period examples: `E 403_ ... Rev.D markup.pdf` (D=0.D?), `CD-...-Rev.0.pdf`, `12074_..._Rev 2 240606_DRAFT.pdf` (hybrid: Rev 2 + date).
  - Note discrepancies in comments (e.g., "Corpus uses PDF not VSDX; adapt parser"; "DRAFT tag: deprioritize unless filtered").
* **Parser**: Flexible regex chain (try REV → period → date → time); return {series:str, version:(int|float|date), tags:set, is_as_built:bool, is_initial:bool (100/1.00), mod_time:datetime}. Handle ambiguities (log INFO).

**User Actions (GUI + CLI)**

* **Auto-Open Latest**: Default; per type/series: Highest version (REV > period); primary series tiebreaker: Recent mod time → contains type keyword (e.g., "LLD") → shortest name. Comment rules explicitly.
* **Choose Doc**: Picker with:
  - Type selector (dropdown: LLD, HLD, Change Order, etc.).
  - Toggles/Filters: Latest/series only | All; As Built/Initial/Latest/No filter; Date range.
  - Sort: Series alpha → version desc (custom cmp: REV int > period tuple > date) → mod time desc.
  - Previews: Photos (AHK Picture control, 100x100 thumbs); docs (icon or first-page via temp PDF render if feasible).
* **Open**: Default app (Visio for .vsdx, Acrobat/Word for PDF/DOCX, image viewer for JPG). Reuse AHK `Run` or Python `os.startfile()`.
* **CLI**: `quicknav doc --type lld|hld|changeorder|floorplan|scope|photos --project 12345 [--latest|--choose] [--filter asbuilt|initial|latest|date:YYYY-MM-DD] [--view series|all] [--root /custom/path]`. Output: SUCCESS:path|SELECT:path1|...|ERROR:msg (extend cli.py subparser).

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

* **Parsing**: Handles REV 100/200, 1.09<2.01; corpus examples (e.g., Rev.D=0.D); tags prioritize As Built.
* **Doc Types**: LLD opens latest Visio; photos preview thumbs + open full; reports filter by date.
* **Scaling**: GUI adapts to 4K (controls 2x size, no cutoff); 2K unchanged.
* **Roots**: --root /network works; falls back if OneDrive offline; GUI settings add paths.
* **UX**: Incremental (existing project open unchanged); choose mode paginates large lists; all CLI/GUI.
* No regressions: Core nav works; tests pass.

Implement step-by-step: 1) Parser module. 2) CLI extension. 3) AHK scaling + type selector. 4) Full integration/tests. Small commits.