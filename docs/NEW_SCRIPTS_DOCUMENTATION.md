# New Analysis & Extraction Scripts Documentation

## Overview

Four powerful new scripts have been developed to enhance Project QuickNav's AI/ML capabilities and provide comprehensive project analysis:

1. **`hybrid_training_analyzer.py`** - Hybrid rule-based + AI training data processor
2. **`ai_agent_training_generator.py`** - AI agent training data and insights generator  
3. **`project_extractor.py`** - Comprehensive project-specific information extractor
4. **`scope_generator.py`** - Project scope document generator in markdown format

These scripts work together to transform raw project training data into actionable intelligence for AV project management and AI agent improvement.

---

## 1. Hybrid Training Analyzer (`hybrid_training_analyzer.py`)

### Purpose
Processes AV project training data using a hybrid approach that combines deterministic rule-based processing (90%) with strategic AI enhancement (10%), solving reliability and performance issues of pure AI approaches.

### Key Features

#### Rule-Based Processing (90% of work)
- **Document Classification**: Categorizes by folder structure and file patterns
- **Quality Filtering**: Removes duplicates, templates, archived files, low-value documents
- **Metadata Extraction**: Project IDs, client names, revision numbers using regex
- **Revision Management**: Automatically keeps latest revisions (Rev 102 over Rev 101)
- **Category Organization**: Groups documents into standardized AV project categories

#### Strategic AI Enhancement (10% of work)
- **Project Summaries**: AI-generated summaries for projects with 3+ documents
- **Scope Analysis**: AI-powered analysis of technical scope and systems
- **Focused API Calls**: Uses Claude Haiku for fast, cost-effective insights

### Usage
```bash
# Set API key
export OPENROUTER_API_KEY='your_key_here'

# Run with default training_data directory
python hybrid_training_analyzer.py

# Specify custom directory
python hybrid_training_analyzer.py /path/to/training/data
```

### Output
- **File**: `hybrid_analysis_results.json`
- **Structure**: Processed projects with filtered documents, AI summaries, and categorized information
- **Performance**: Completes in under 2 minutes vs. slow pure AI approaches

### Technical Implementation
- **Client Mappings**: Expands abbreviations (DWNR → Downer)
- **Document Types**: 11 different AV document classifications
- **Quality Patterns**: High/medium/low value document identification
- **Folder Categories**: 6 standard AV project categories
- **Error Handling**: Robust fallbacks and graceful degradation

---

## 2. AI Agent Training Generator (`ai_agent_training_generator.py`)

### Purpose
Generates structured training data and actionable insights specifically for improving AV project AI agents. Focuses on creating training examples and identifying improvement opportunities.

### Key Features

#### Training Data Generation
- **Project Identification Examples**: Train agents to extract metadata from folder names
- **Document Classification Examples**: Classify documents by type and category
- **Scope Assessment Examples**: Assess project complexity and scope
- **Quality Assessment Examples**: Evaluate documentation completeness

#### AI-Powered Insights
- **Pattern Analysis**: Identifies common documentation patterns across projects
- **Training Data Gaps**: Highlights where agents need more examples
- **Performance Metrics**: Suggests specific metrics to track agent improvement
- **Automation Opportunities**: Identifies tasks that could be automated

#### Structured Training Examples
Each example includes:
- **Input Context**: What the agent receives
- **Expected Output**: What the agent should produce
- **Example Type**: Category of training
- **Confidence Score**: Reliability of the example
- **Metadata**: Additional context for training

### Usage
```bash
# Must run hybrid analyzer first
python hybrid_training_analyzer.py

# Then generate AI training data
export OPENROUTER_API_KEY='your_key_here'
python ai_agent_training_generator.py
```

### Output Files
1. **`ai_agent_training_dataset.json`**: Complete training dataset with examples
2. **`ai_agent_training_report.txt`**: Human-readable summary and recommendations

### Training Categories Generated
- **Project Identification** (10 examples): Metadata extraction from folder names
- **Document Classification** (up to 75 examples): Document type and category classification
- **Scope Assessment** (8 examples): Project complexity and phase identification
- **Quality Assessment** (6 examples): Documentation quality scoring

### AI Insights Provided
- **Training Data Gaps**: Specific areas needing more examples
- **Agent Improvement Opportunities**: Actionable enhancement strategies
- **Automation Potential**: Tasks ready for automation
- **Quality Metrics**: KPIs for measuring agent performance
- **Training Priorities**: Ranked list of focus areas

---

## 3. Project Extractor (`project_extractor.py`)

### Purpose
Extracts and structures comprehensive project-specific information based on a 5-digit project number. Provides complete project analysis including documents, metadata, scope, and health assessment.

### Key Features

#### Comprehensive Project Analysis
- **Project Metadata**: Client info, project names, structured identifiers
- **Document Classification**: Categorizes and types all project documents
- **Scope Analysis**: Assesses complexity, technology areas, and phases
- **Quality Assessment**: Evaluates documentation completeness and quality
- **Timeline Generation**: Creates document revision timeline
- **Deliverables Tracking**: Identifies key deliverables and their status
- **Health Assessment**: Provides project health score and recommendations

#### Technology Detection
- **AV Keywords**: Analyzes content for audio, video, control, conferencing systems
- **Brand Recognition**: Identifies major AV brands and technologies
- **Complexity Assessment**: Evaluates technical complexity from documentation patterns

#### Quality Assessment
- **Completeness Score**: Percentage of expected documentation categories
- **Documentation Quality**: High/Medium/Low based on document analysis
- **Risk Indicators**: Missing critical documentation, quality issues
- **Recommendations**: Actionable suggestions for improvement

### Usage
```bash
# Basic extraction for project 20692
python project_extractor.py 20692

# Extract with summary report
python project_extractor.py 20692 --summary

# Custom output location
python project_extractor.py 20692 --output custom_profile.json

# Help
python project_extractor.py --help
```

### Output Files
1. **`project_profile_[project_id].json`**: Complete structured project data
2. **`project_summary_[project_id].txt`**: Human-readable summary (with --summary)

### Analysis Categories
- **Project Metadata**: ID, client, names, folder structure
- **Scope Analysis**: Technology areas, phases, complexity indicators
- **Document Summary**: Categories, types, quality distribution, file types
- **Project Timeline**: Document revision chronology with phase estimation
- **Key Deliverables**: Design, technical, handover, and support deliverables
- **Health Assessment**: Completeness, quality, risks, recommendations

---

## 4. Scope Generator (`scope_generator.py`)

### Purpose
Extracts actual Scope of Work content from project documents and formats into professional markdown. Integrates with `find_project_path.py` to locate project folders and intelligently extract scope content from various document types.

### Key Features

#### Document Discovery & Analysis
- **Intelligent Document Detection**: Finds scope-related documents using filename patterns
- **Multi-Format Support**: Extracts text from PDF, Word, TXT, and RTF documents
- **Content Pattern Recognition**: Identifies scope sections using smart content analysis
- **Project Integration**: Seamlessly integrates with existing project folder structure

#### Scope Content Extraction
- **Section Identification**: Automatically identifies scope sections ("scope of work", "deliverables", "objectives")
- **Content Cleaning**: Removes page numbers, headers, and formatting artifacts
- **Smart Boundaries**: Detects scope section start/end points (exclusions, assumptions, budget)
- **Multiple Sources**: Combines scope content from multiple documents when found

#### Professional Output
- **Markdown Formatting**: Clean, professional markdown with proper structure
- **Source Attribution**: Clear references to source documents and file paths
- **Graceful Fallbacks**: Professional handling when no scope content is found
- **Client-Ready**: Formatted for immediate use in client communications

### Usage
```bash
# Extract scope from project documents
python scope_generator.py 17741

# Custom output filename
python scope_generator.py 17741 --output custom_scope.md

# Help and dependencies info
python scope_generator.py --help
```

### Dependencies for Full Functionality
```bash
# Optional but recommended for best results
pip install PyPDF2 python-docx pdfplumber
```

### Document Detection Patterns
- **Filenames**: scope, SOW, statement of work, requirements, specifications, tender, proposal, brief
- **Content Sections**: "scope of work", "project scope", "deliverables", "objectives", "requirements"
- **End Markers**: exclusions, assumptions, limitations, schedule, budget, pricing

### Output
- **Default**: `[project_id]_project_scope.md`
- **Format**: Professional markdown document with extracted scope content
- **Structure**: Overview, extracted scope sections, source document references

### Example Generated Document
```markdown
# Scope of Work - 17741
**Project:** 17741 - Test Project
*Generated: 2025-06-10 05:33:01*

## Overview
This document contains the Scope of Work extracted from project documentation.
**Source Documents:** 1 documents analyzed

## Scope of Work
SCOPE OF WORK
Project Overview
This project involves the design, supply, installation, and commissioning 
of a complete audio visual system for the main conference room...

Project Requirements
The client requires a state-of-the-art presentation and video conferencing 
solution capable of supporting:
- Local presentations from laptops and mobile devices
- Video conferencing with remote participants
- High-quality audio capture and reproduction
- Automated system control via touch panel interface

## Source Documents
- **Project Scope Document.txt**
  - Path: `Project Scope Document.txt`
```

### Processing Intelligence
- **Text Extraction**: Handles various document formats with fallback options
- **Content Analysis**: Uses regex patterns to identify scope-related content
- **Section Boundaries**: Intelligently determines where scope content begins and ends
- **Quality Formatting**: Cleans and formats extracted text for professional presentation

---

## Integration & Workflow

### Sequential Processing Pipeline
```
1. QuickNav generates training_data_[project].json
2. hybrid_training_analyzer.py processes all training files
3. ai_agent_training_generator.py creates training examples
4. project_extractor.py analyzes specific projects
5. scope_generator.py extracts scope content from actual documents
```

### File Dependencies
- **Input**: `training_data/training_data_[project].json` (from QuickNav)
- **Intermediate**: `hybrid_analysis_results.json` (from hybrid analyzer)
- **Output**: Multiple analysis, training, and documentation files

### Common Patterns
All scripts share:
- **AV Project Structure**: 6 standard categories (Sales, BOM, PMO, Designs, Handover, Support)
- **Document Types**: 15+ AV-specific document classifications
- **Quality Patterns**: Consistent high/medium/low value assessment
- **Client Mappings**: Standard abbreviation expansions
- **Error Handling**: Robust fallbacks and user feedback

---

## Benefits & Use Cases

### For Project Management
- **Project Health Monitoring**: Quick assessment of documentation completeness
- **Quality Audits**: Identify gaps and improvement areas
- **Client Reporting**: Professional project summaries and scope documents
- **Risk Management**: Early identification of project risks

### For AI Agent Development
- **Training Data Generation**: Structured examples for agent improvement
- **Performance Metrics**: Clear KPIs for measuring agent effectiveness
- **Pattern Recognition**: Understanding of successful project patterns
- **Automation Roadmap**: Identification of automation opportunities

### For Business Intelligence
- **Portfolio Analysis**: Understanding of project patterns across the business
- **Technology Trends**: Identification of common technology implementations
- **Process Optimization**: Data-driven insights for workflow improvement
- **Resource Planning**: Better understanding of project complexity and requirements

### For Client Communication
- **Professional Documentation**: Well-formatted scope documents for client sharing
- **Standardized Reporting**: Consistent project scope presentation across all projects
- **Easy Distribution**: Markdown format compatible with various platforms and tools
- **Quick Generation**: Rapid creation of scope documents for client meetings and proposals

---

## Technical Architecture

### Design Principles
- **Deterministic Processing**: Rule-based approach for reliability and speed
- **Strategic AI Use**: AI only where it adds genuine value
- **Modular Design**: Each script serves a specific purpose
- **Error Resilience**: Graceful handling of missing or malformed data
- **Performance Focus**: Optimized for speed and efficiency

### Dependencies
- **Python 3.7+**: Core language requirements
- **requests**: For OpenRouter API calls (hybrid and AI training scripts)
- **json, re, time**: Standard library modules
- **argparse**: Command-line interface
- **pathlib**: Modern path handling
- **collections**: Data structure utilities

### Error Handling
- **Missing Files**: Graceful degradation with informative messages
- **API Failures**: Fallback responses and retry logic (where applicable)
- **Data Quality Issues**: Filtering and validation with user feedback
- **Processing Errors**: Comprehensive logging and recovery

This comprehensive suite of analysis tools transforms Project QuickNav from a simple navigation utility into a powerful platform for AV project intelligence and AI agent development, now including professional documentation generation capabilities. 