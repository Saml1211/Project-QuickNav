Title: Document Navigation Upgrades — Data-Informed Logical Framework (Enhanced)

Target file: [ai_docs/prompts/doc-navigation-upgrades.md](ai_docs/prompts/doc-navigation-upgrades.md)

Objective
- Implement a robust, data-informed document navigation system for project-based repositories that:
  - Resolves project roots reliably across multiple storage locations
  - Scans and classifies documents by type using real-world naming and location patterns
  - Extracts and normalizes versions, dates, and status tags
  - Ranks and opens the most relevant or most recent artifacts or presents a filtered, high-DPI-aware picker
  - Operates non-destructively and is fully configurable via settings
  - Aligns tightly with Pro AV style repository conventions surfaced by training data

Repository Context
- Core components:
  - Python backend: @src/find_project_path.py and CLI entry in @quicknav/cli.py with SUCCESS and SELECT outputs
  - AHK frontend: @src/lld_navigator.ahk GUI with radios and tray menu; controller and logging lives in AHK files at repo root
  - Configuration persistence lives in user AppData settings json noted in existing docs
  - Tests: Python under @tests and AHK under @tests/ahk
  - Training samples: referenced as training_data for parser validation
- Constraints:
  - Ideally no new dependencies beyond Python standard library and AHK v2
  - Incremental, non-destructive, graceful errors
  - Keep UX seamless with existing flows

Data-Informed Patterns from Real-World Repositories
- Project identity and foldering
  - Project root folders follow “[5-digit code] - [Client or Location] - [Project Type or Phase]”
  - Common phase directories:
    - 1. Sales Handover
    - 2. BOM & Orders
    - 3. PMO
    - 4. System Designs
    - Additional patterns: 5. Floor Plans, 6. Site Photos, ENG INFO, OLD DRAWINGS, ARCHIVE, Received [date]
- File naming and extensions
  - Predominantly PDF with DOCX or RTF for registers and scopes, VSDX for LLD, image files for site photos
  - Names commonly include the 5-digit project code prefix, document type, description, variant or revision, and an optional date
  - Status tags like AS-BUILT, DRAFT, FINAL, APPROVED are common
  - Room or location scoping appears as RoomNN in folder names and files, especially for QA-ITP and plans
- Versioning and dates
  - Revision forms include Rev 100 or Rev100 or Rev 1.03 or lettered variants like _(A) and parenthetical counts like (2)
  - Dates often appear as DD.MM.YY or compact YYMMDD six-digit tokens; dates may also appear in path segments such as Received dd-mm-yyyy
- Archive and old
  - OLD DRAWINGS or ARCHIVE folders contain historic versions; treat them as valid but de-prioritized unless needed

Scope and Goals
- Support multiple document types with type-to-folder mapping:
  - LLD or HLD to 4. System Designs
  - Change Orders and Sales or PO Reports to 2. BOM & Orders
  - Floor Plans to 5. Floor Plans or 1. Sales Handover or Floor Plans
  - Scope and Handover to 1. Sales Handover
  - Site Photos to 6. Site Photos
- After resolving the project path, scan for documents and either:
  - Auto-open the most relevant item, or
  - Provide a filtered, high-DPI-aware picker with previews for images
- Custom roots and fallback:
  - Allow configurable roots and environment overrides
  - Fall back silently when a root is unavailable without degrading UX
- High-DPI scaling:
  - Ensure the AHK GUI scales correctly on 2K and 4K displays, using DPIScale and proportional control sizing
- Modular and configurable:
  - Centralize patterns and mappings through a settings schema for easy extension

Root Resolution and Project Discovery
- Root resolution order
  - CLI override via flag first
  - Configured custom roots next
  - Default OneDrive last
  - Use fast checks and continue silently on IO errors, logging at INFO
- Project identity extraction
  - Prefer folder name prefix of a 5-digit project code
  - Fallback to filename prefix if folder signal is weak
  - Require at least one strong identity signal before proceeding
- If zero candidates are found
  - Return a sensible parent phase folder or project root along with an explanatory message

Config Schema for Doc Types and Ranking
- Extend settings to define doc types, folders, extensions, parsing patterns, exclusion rules, and ranking weights
- Example schema fragment
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
        "version_patterns": [
          "Rev\\s*(\\d+(?:\\.\\d+)?)",
          "\\((\\d+)\\)$",
          "_\\(([A-Z])\\)"
        ],
        "status_tags": ["AS-BUILT", "DRAFT"],
        "date_patterns": [
          "(\\d{2}\\.\\d{2}\\.\\d{2})",
          "(\\d{6})",
          "Received\\s+(\\d{2}-\\d{2}-\\d{4})"
        ],
        "room_key_regex": "Room\\d+",
        "priority": 100
      },
      {
        "type": "change_order",
        "label": "Change Orders",
        "folders": [
          "2. BOM & Orders",
          "2. BOM & Orders/Sales & Change Orders",
          "2. BOM & Orders/Sales & Change Orders/All COs"
        ],
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
      "status_draft": -0.5,
      "newer_version": 3.0,
      "newer_date": 1.0,
      "room_match": 1.5,
      "ext_preference": 0.5
    }
  }

Version and Tag Parsing
- Primary patterns
  - REV nnn where nnn is integer such as 100 or 200 and variants of Rev with spaces or underscores
  - Dotted semver-like numbers like 1.09 or 2.01
  - Parenthetical iteration such as (2) and lettered variants like _(A)
  - Date stamps as DD.MM.YY such as 23.09.22 or YYMMDD such as 240606
  - Path-derived dates such as in Received dd-mm-yyyy
- Normalization and comparison
  - Prefer numeric Rev when present
  - If dotted numbers are present, compare by major then minor as integers
  - Parenthetical iterations compare numerically within same series and are lower precedence than explicit Rev
  - Lettered variants map to 0.A through 0.Z within a series when no numeric Rev exists
  - Rev D is treated as 0.D and ranks below numeric Rev
  - Break ties by status then explicit date then file modification time
- Status tags
  - AS-BUILT boosts ranking as a final deliverable
  - DRAFT is de-prioritized unless explicitly requested

Scanning and Classification
- For a selected doc type, derive target folders from config and traverse recursively
- Include by extension and name-based hints and exclude by ARCHIVE or OLD categories unless no other options are found
- Exclude global templates such as Master Template or Template when no project code is present
- Extract metadata for each candidate
  - Project code from filename or ancestor folder prefix
  - Change Order number from CO patterns when applicable
  - Normalized version and tags
  - Normalized date fields with day-first default
  - Room key inferred from path if present
  - File modification time and path flags such as in_preferred_folder and in_archive_or_old

Ranking and Selection
- Compute a weighted score using ranking_weights from configuration
  - Strong preference for project code match and preferred folders
  - De-prioritize archive or old locations
  - Prioritize AS-BUILT, newer version, and newer date
  - Prefer type-preferred extensions such as VSDX for LLD
  - Boost room matches when a room is specified
- Selection modes
  - Auto-open the top candidate when its score is 30 percent higher than the next candidate
  - Otherwise present a picker with the top N candidates
  - Room-filtered selection when a room is specified

User Actions and UX
- CLI in [quicknav/cli.py](quicknav/cli.py)
  - Subcommand doc with flags
    - --type lld or hld or changeorder or floorplan or scope or photos
    - --root PATH to override roots
    - --room RoomNN to scope to a room
    - --list to force chooser
    - --open to force auto-open best
    - --limit N to bound candidate count
  - Output format remains SUCCESS path or SELECT path1 or path2 or ERROR msg
- AHK GUI in [src/lld_navigator.ahk](src/lld_navigator.ahk)
  - Add Doc Type dropdown and optional Room filter
  - Picker shows columns such as Type and Version and Status and Date and Folder and Name
  - For photos display thumbnails using a lightweight approach
  - High-DPI behavior
    - Use DPIScale and Resize flags
    - Compute scale factor from A ScreenDPI over 96 and apply to control dimensions and spacing
    - Maintain layout with resize handler
- Non-destructive behavior
  - Read only scanning with shell-open actions
- Logging
  - INFO for root resolution and selection outcomes
  - DEBUG for parse details and scoring breakdown
  - WARN or ERROR for missing roots or candidates or parse anomalies
  - Respect existing debug toggles

Folder-to-Type Guidance
- 4. System Designs holds LLD and HLD with preference to VSDX and fall back to PDF when needed
- 2. BOM & Orders contains Change Orders and Sales or PO Reports
- 1. Sales Handover holds Scope and Handover and Proposals
- 5. Floor Plans and 1. Sales Handover or Floor Plans contain plan sheets with lettered revisions and room scoping
- QA-ITP Reports are under PMO and are strongly room-scoped
- 6. Site Photos contains image formats with previews

Acceptance Criteria
- Root resolution and fallback
  - When OneDrive is offline the system silently iterates custom roots and succeeds, and CLI root overrides behave as expected
- Parsing correctness using realistic examples
  - File “17741 - QPS MTR Room Upgrades - Bulk Order - Sales & PO Report 23.09.22.pdf” yields code 17741 and date 2022-09-23
  - Plan “A-1101_(A) - GA PLAN GROUND - SHEET 1.pdf” detects lettered version A where B outranks A
  - “Change Order 12 - 17741 - .pdf” extracts CO number 12 and code 17741
  - “LLD … Rev 1.03.pdf” is outranked by “LLD … Rev 2.01.pdf”
  - “Rev.D” is treated as the lowest in its family
- Ranking and selection behavior
  - AS-BUILT outranks DRAFT unless configured otherwise or massive date gap outweighs status
  - Files under OLD DRAWINGS or ARCHIVE lose to primary folders unless they are the only candidates
  - With room filter Room01 any Room01 candidates outrank other rooms
  - Auto-open triggers only when the top candidate exceeds the runner-up by at least 30 percent
- UX and performance
  - CLI doc subcommand supports all flags and returns SUCCESS or SELECT or ERROR format
  - AHK picker scales properly at 200 percent on 4K without visual overlap and photos display thumbnails
  - Large projects paginate lists and remain responsive
  - Existing QuickNav behaviors remain unchanged when users do not engage the doc navigation
- Tests
  - Unit tests cover date parsing, version precedence, status priority, project code extraction, folder priority, change order detection, room ranking, and root fallback
  - AHK tests cover Doc Type dropdown behavior and DPI scaling on simulated resolutions
  - All new tests pass with no regressions

Deliverables
- Code
  - New module described as doc navigator in Python such as quicknav or a similar path to be introduced
  - CLI updates in @quicknav/cli.py
  - AHK UI and controller updates in @src/lld_navigator.ahk
- Config and documentation
  - Settings schema extended for custom_roots and doc_types
  - Document Navigation section in @README.md covering usage, flags, custom types, DPI notes, and roots configuration
  - Changelog entry in @RELEASE.md and a version bump in @VERSION.txt
- Tests
  - Python tests to be added such as @tests/test_doc_navigator.py
  - AHK tests such as @tests/ahk/test_doc_types.ahk and @tests/ahk/test_scaling.ahk

Milestones
1. Parser and scorer with extraction and normalization and ranking including unit tests
2. CLI integration with a doc subcommand and formatted outputs
3. AHK UI with Doc Type dropdown and room filter and DPI scaling and a responsive picker
4. Integration of root fallback and room filters and change order and plan handling with acceptance tests
5. Documentation and changelog and version update

Non-Goals
- No file renaming or automatic archival management
- No content OCR or PDF page rendering beyond simple thumbnails for images