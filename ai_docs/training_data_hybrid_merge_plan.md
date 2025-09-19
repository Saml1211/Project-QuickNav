## Hybrid Training Data Update Plan

### Overview
Implement a hybrid generation/merge process for per-project training data JSON so repeated runs:
- Preserve curated fields while refreshing the index of on-disk files
- Detect removals, renames/moves, and content changes
- Produce deterministic, stable output
- Ensure data security and integrity

Scope: single-project runs produced by the AHK GUI with "Generate Training Data". Multi-project search output remains timestamped and unchanged.

### Goals
- Preserve `extracted_info` without manual curation required on subsequent runs
- Mark disappeared files as removed
- Detect renames/moves via content hashes with collision-resistant hashing
- Track basic file metadata for change detection
- Maintain backward compatibility with current consumers
- Ensure PII detection and data security compliance
- Provide atomic operations and robust error handling

### Data model (per entry)
- `project_folder` (string)
- `document_path` (string, absolute, normalized)
- `document_name` (string)
- `extracted_info` (object)
- `file_size` (int)
- `modified_time` (ISO 8601 string)
- `content_hash` (sha256, lowercase hex)
- `removed` (bool)
- `renamed` (bool)
- `renamed_from` (string|null)
- `content_changed` (bool)
- `pii_detected` (bool, flags if PII scanning found sensitive data)
- `error_info` (object|null, contains error details if processing failed)

Deterministic ordering: sort by `document_path` ascending (case-insensitive on Windows).

### Identity and normalization
- Primary identity for merge: normalized absolute path (Windows case-insensitive)
- Secondary identity for rename detection: `content_hash` (SHA256)
- Normalization uses `os.path.abspath` + `os.path.normpath`; compare paths with `.lower()` on Windows
- File processing includes PII detection before content hashing

### Merge algorithm (single project)
1) Scan filesystem under selected project folder → build `scanned_entries` with:
   - path, name, size, modified_time; `extracted_info = {}`; placeholders for flags
   - Handle file access errors gracefully (log warning, skip file, continue)
2) Load existing JSON (if present) → `existing_entries`
3) Create backup of existing JSON before modifications
4) Index existing by:
   - `path_index[path_norm] = entry`
   - `hash_index[content_hash] = entry` (skip if missing)
5) For each `scanned` entry:
   - If path exists in `path_index`:
     - If size/mtime unchanged: copy `extracted_info`, keep hash if present; `content_changed = false`
     - If size/mtime changed: compute hash with PII detection; if equals existing `content_hash` → `content_changed = false` else `true`
   - Else (new-by-path): compute hash with PII detection
     - If hash in `hash_index` at different path → this is a rename/move
       - `renamed = true`, `renamed_from = old_path`, `removed = false`, carry `extracted_info`, `content_changed = false`
     - Else → brand new file
       - `renamed = false`, `removed = false`, `content_changed = false`, empty `extracted_info`
6) For any existing entries not matched by path or rename:
   - Keep them and set `removed = true`, `renamed = false`, `content_changed = false`
7) Output: union of updated/matched scanned entries plus removed ones; sort; write atomically with temp file + rename.

Notes:
- Hashing (SHA256) is done selectively: for new-by-path or size/mtime-changed files.
- PII detection runs during hash computation to flag sensitive content.
- On the very first run after deploying this feature, legacy files without `content_hash` cannot retro-detect past renames; after that, hashes enable rename detection.
- Error handling: file access failures are logged but don't halt processing; affected entries get `error_info` populated.

### Multi-project runs
- Keep current behavior: write a combined timestamped file. No per-project updates are performed in this mode.

### Implementation steps (in `src/find_project_path.py`)
1) Utilities
   - `normalize_path(path) -> str`
   - `get_file_metadata_safe(path) -> Optional[Tuple[int, str]]` (with error handling)
   - `compute_hash_with_pii_check(path) -> Tuple[str, bool]` (streaming SHA256 + PII detection)
   - `detect_pii(content: str) -> bool` (basic PII patterns: emails, SSNs, etc.)
   - `load_existing_training_data(file) -> list[dict]` (with schema validation)
   - `backup_training_data(file) -> Optional[str]`
   - `write_training_data_atomic(data, filepath)` (temp file + rename)
   - `index_by_path(entries) -> dict[str, dict]`
   - `index_by_hash(entries) -> dict[str, dict]`
   - `add_schema_metadata(entries) -> dict` (version, timestamp, entries)
2) Generation flow (single project)
   - Build `scanned_entries` with metadata (handle I/O errors gracefully)
   - Load existing (if exists) and build indices
   - Create backup before modifications
   - Merge using algorithm above with PII detection
   - Sort by `document_path`; write JSON atomically with schema versioning
3) Leave multi-project branch unchanged

### Edge cases
- Legacy JSON without new fields: treat missing fields as defaults; populate on write with schema migration
- Permission or transient I/O errors: log warning with structured error info; skip file, continue; populate `error_info` field
- Extremely large files: hashing is streaming SHA256; only computed when needed; memory-efficient processing
- Case-only path changes on Windows: normalization makes them equivalent; no rename event
- PII detection failures: log warning; continue processing with `pii_detected = false`
- Concurrent file access: retry with exponential backoff; fail gracefully if unable to access
- Schema version mismatches: handle backward compatibility; warn on unknown future versions

### Testing plan
- Preserve `extracted_info`: create existing JSON with edits; rerun; verify fields persisted
- Removed files: delete a file; rerun; entry marked `removed = true`
- Renamed files: rename a file; rerun twice (to ensure hash present); verify `renamed=true`, `renamed_from` set
- Modified files: change content; rerun; `content_changed = true`
- Determinism: run twice without changes; outputs identical and sorted
- PII detection: create files with mock PII (emails, SSNs); verify `pii_detected = true`
- Error handling: test with permission-denied files; verify graceful handling and `error_info` populated
- Atomic writes: simulate interruption during write; verify no corruption, backup exists
- Schema versioning: test with legacy JSON; verify proper migration and backward compatibility
- Performance: test with large codebases (1000+ files); verify memory usage and processing time
- Concurrent access: test with files being modified during scan; verify retry logic

### Migration notes
- First write after deploy will add schema version and new fields to all entries
- Past renames cannot be inferred for legacy entries lacking hashes; future runs will track correctly
- SHA1 → SHA256 migration: existing SHA1 hashes will be recomputed on next content change
- Backup files created during migration for safety; can be cleaned up after verification
- PII detection will run on all files during first scan after deployment

### Acceptance criteria
- No manual curation required; re-runs preserve `extracted_info`
- Disappeared files are retained in JSON with `removed=true`
- Renames/moves detected and annotated via `renamed=true` and `renamed_from` using SHA256 hashes
- Content changes flagged; metadata updated with collision-resistant hashing
- Output stable and deterministic with proper sorting
- PII detection flags sensitive content appropriately
- Error handling is robust with structured logging and graceful degradation
- Atomic writes prevent data corruption during interruptions
- Schema versioning enables backward compatibility
- Performance scales to large codebases (1000+ files) with reasonable memory usage

### Rollout
- Phase 1: Implement core utilities (hashing, PII detection, error handling, atomic writes)
- Phase 2: Implement merge logic with schema versioning and backup functionality
- Phase 3: Update single-project generation paths (both exact-number and single search match)
- Phase 4: Add comprehensive unit and integration tests under `tests/`
- Phase 5: Performance testing with large codebases and memory profiling
- Phase 6: Deploy with monitoring and rollback capability
- No AHK/UI changes necessary; backward compatible


