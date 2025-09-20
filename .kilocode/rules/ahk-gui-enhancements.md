## Brief overview
- Project-specific guidelines for developing and enhancing AutoHotkey v2 GUIs in Project QuickNav, focusing on UI stability, flicker prevention, and responsive sizing. Derived from iterative improvements to src/lld_navigator.ahk, emphasizing idempotent changes and v2 compatibility.

## Communication style
- Provide technical summaries in attempt_completion with precise change descriptions and markdown links to code locations, e.g., [function](src/lld_navigator.ahk:line).
- Use detailed but concise explanations of fixes, referencing line numbers and tool outcomes.
- End responses with clear outcome statements, avoiding conversational closers like questions.

## Development workflow
- Apply changes via apply_diff for surgical edits, ensuring idempotency (no duplicates on re-run).
- Validate AHK v2 syntax before completion; fix issues like unsupported destructuring with indexed arrays.
- Use read_file to confirm file state after modifications; iterate with tools like update_todo_list for multi-step tasks.
- Switch modes (e.g., code to ask) only when needed for validation; prefer step-by-step tool chaining.

## Coding best practices
- Prioritize UI anti-flicker: Wrap resize handlers (e.g., MainGuiSize, SearchGuiSize) with SetRedraw(false) ... finally SetRedraw(true), including RedrawWindow for immediate repaints.
- Implement dynamic min-size: Use CalcMinClient with DPI-scaled padding and mode-specific extra controls (options, status, buttons) to prevent overlap.
- Maintain AHK v2 compatibility: Place helpers after #Requires; avoid tuple destructuring, use array indexing like minSize[1].
- Preserve existing layout math; extend only for visibility checks and redraw control.

## Project context
- Focus on QuickNav GUI: Folder/document modes with exclusive visibility toggling in SwitchToFolderMode/SwitchToDocumentMode.
- Integrate DPI scaling throughout: Apply to margins, paddings, and extents in CalcMinClient.
- Ensure mode-specific MinSize recalculation at startup and switches, including bottom UI elements to tighten min-height.