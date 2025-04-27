# Project QuickNav – Release Preparation Guide

This checklist ensures every release of QuickNav is tested, built, and published consistently.

*Versioning Scheme*: **Semantic Versioning 2.0**  
Artifacts produced:  
1. **Wheel** – `quicknav-<ver>-py3-none-any.whl`  
2. **Source distribution** – `quicknav-<ver>.tar.gz`  
3. **Standalone Win-x64 EXE** – `quicknav-<ver>-win64.exe`  

---

## 1. Pre-flight

| Step | Command / Action |
|------|------------------|
| Confirm latest `main` merged | `git pull origin main` |
| Review open issues / PRs | triage for next milestone |
| Ensure local environment is clean | `git status` |

---

## 2. Bump Version

1. Edit `VERSION.txt` with the new **X.Y.Z**.  
2. Update **CHANGELOG.md** (or release notes section in `README.md`).  
3. Commit:  

   ```bash
   git add VERSION.txt CHANGELOG.md
   git commit -m "chore: bump version to vX.Y.Z"
   ```

---

## 3. Run Tests

```bash
python -m venv .venv && . .venv/Scripts/Activate
pip install -e .[dev]      # includes pytest
pytest                     # unit tests
```

Also run AutoHotkey test plan:

```powershell
& "AutoHotkey.exe" .\lld_navigator.ahk /Test
```

All tests must pass.

---

## 4. Build Artifacts

```bash
python -m build            # produces dist/*.whl and dist/*.tar.gz
```

### Build PyInstaller EXE

```powershell
.\scripts\build_exe.ps1
# output => dist\quicknav-<ver>-win64.exe
```

Verify the EXE:

```powershell
.\dist\quicknav-<ver>-win64.exe --version
```

---

## 5. Smoke Test (Wheel)

```bash
pipx uninstall quicknav || true
pipx install dist/quicknav-<ver>-py3-none-any.whl
quicknav --version
quicknav 12345             # expect SUCCESS/SELECT/ERROR
```

---

## 6. Tag & Push

```bash
git tag -a vX.Y.Z -m "QuickNav vX.Y.Z"
git push origin vX.Y.Z
git push origin main
```

---

## 7. Create GitHub Release

1. Draft a new release for tag `vX.Y.Z`.  
2. Upload the following files from `dist\`:
   * `.whl`
   * `.tar.gz`
   * `quicknav-<ver>-win64.exe`
3. Paste release notes (change log).

---

## 8. Publish to PyPI

```bash
twine upload dist/quicknav-*.{whl,tar.gz}
```

---

## 9. Post-release

* Verify `pip install quicknav==X.Y.Z` works in a fresh environment.  
* Close milestone in issue tracker.  
* Update project roadmap & documentation links.

---

**Enjoy your freshly shipped QuickNav release!**