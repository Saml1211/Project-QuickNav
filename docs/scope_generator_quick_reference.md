# Scope Generator - Quick Reference

## 🚀 Quick Start

```bash
# Basic usage - extract scope for project 21620
python analysis/scope_generator.py 21620

# With AI enhancement (recommended)
export OPENROUTER_API_KEY='your_api_key_here'
python analysis/scope_generator.py 21620
```

## 📋 Command Line Options

```bash
# Basic syntax
python analysis/scope_generator.py [PROJECT_ID]

# Examples
python analysis/scope_generator.py 21620    # Project 21620
python analysis/scope_generator.py 17741    # Project 17741
python analysis/scope_generator.py 22010    # Project 22010
```

## 🔧 Setup Requirements

### **Essential Libraries**
```bash
# Install required libraries for full functionality
pip install striprtf python-docx PyPDF2 pdfplumber requests
```

### **Optional AI Enhancement**
```bash
# Get OpenRouter API key from: https://openrouter.ai/
export OPENROUTER_API_KEY='your_api_key_here'
```

## 📁 File Types Supported

| Format | Extension | Library Used | Notes |
|--------|-----------|--------------|-------|
| RTF | `.rtf` | `striprtf` | Enhanced parsing with fallback |
| Word | `.docx`, `.doc` | `python-docx` | Full text extraction |
| PDF | `.pdf` | `PyPDF2`, `pdfplumber` | May need OCR for scanned docs |
| Text | `.txt` | Built-in | UTF-8 with encoding fallback |

## 📂 Project Structure Expected

```
OneDrive/
├── Project Files/
│   ├── 21600-21699/              # Range folder
│   │   ├── 21620 - Project Name/ # Project folder
│   │   │   ├── 1. Sales Handover/     # PRIMARY search location
│   │   │   │   ├── ScopeOfWork.rtf    # High priority
│   │   │   │   ├── Project Brief.docx
│   │   │   │   └── Requirements.pdf
│   │   │   ├── 2. Design/
│   │   │   └── 3. Installation/
```

## 🎯 Output Format

### **Generated File**
- **Filename**: `[PROJECT_ID]_project_scope.md`
- **Location**: Current working directory
- **Format**: Professional markdown with headers, lists, and source tracking

### **Content Structure**
```markdown
# Scope of Work - 21620
**Project:** Project Name
*Generated: 2025-06-10 14:38:50*

## Overview
Summary and document count

## Scope of Work
### Project Objective
### Equipment and Materials
### Project Activities
### Deliverables
### Exclusions

## Source Documents
- Document tracking and analysis methods
```

## ⚡ Processing Behavior

### **Document Prioritization**
1. **Scope-named files first** (filename contains "scope")
2. **Sales Handover folder preferred** (25 doc limit)
3. **Other project folders** (10 doc limit)
4. **Pattern-based relevance scoring**

### **Analysis Methods**
- **🤖 AI Mode**: Claude Haiku via OpenRouter (when API key set)
- **📊 Pattern Mode**: Regex-based scope detection (fallback)
- **🔄 Hybrid**: AI primary, pattern fallback for validation

## 🛠️ Troubleshooting

### **Common Issues**

| Problem | Solution |
|---------|----------|
| "No project folder found" | Check project ID exists in OneDrive structure |
| "No scope content found" | Verify scope documents in Sales Handover folder |
| RTF parsing errors | Install: `pip install striprtf` |
| PDF extraction failed | Install: `pip install PyPDF2 pdfplumber` |
| AI timeout errors | Check internet connection and API key |

### **Debug Output**
The script provides detailed console output:
- ✅ Success indicators
- ❌ Error messages with explanations
- 📁 Folder discovery process
- 📄 Document processing status
- 🔍 Pattern matching results

## 📊 Performance Metrics

| Metric | Sales Handover | Other Folders |
|--------|----------------|---------------|
| Max Documents | 25 | 10 |
| AI Timeout | 45 seconds | 45 seconds |
| Scope Priority | Files with "scope" in name | Pattern-based ranking |

## 🎨 Formatting Features

### **Automatic Formatting**
- **Headers**: Section titles → `### Header`
- **Sub-sections**: Location/equipment → `#### Sub-header`
- **Lists**: Action items → `- Bullet point`
- **Structure**: Professional markdown hierarchy

### **Content Recognition**
- **Sections**: Project Objective, Deliverables, etc.
- **Actions**: Arrange, Provide, Install, Mount, etc.
- **Locations**: Brisbane, Curtis Island, etc.
- **Equipment**: Displays, Brackets, Systems, etc.

## 🚦 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - scope extracted and saved |
| 1 | Error - no project found or processing failed |

## 📝 Example Workflow

```bash
# 1. Set up environment
pip install striprtf python-docx PyPDF2 pdfplumber
export OPENROUTER_API_KEY='your_key'

# 2. Run extraction
python analysis/scope_generator.py 21620

# 3. Check output
ls -la 21620_project_scope.md

# 4. Review generated document
cat 21620_project_scope.md
```

## 🔗 Related Tools

- **`find_project_path.py`**: Project discovery logic (embedded)
- **OpenRouter API**: AI analysis service
- **striprtf**: RTF text extraction
- **python-docx**: Word document processing

---

For detailed information, see: [`docs/scope_generator_documentation.md`](scope_generator_documentation.md) 