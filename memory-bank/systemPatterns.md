# System Patterns – Project QuickNav

## Architecture Overview

Project QuickNav has evolved from a three-tier navigation system into a comprehensive AV project intelligence platform with advanced analysis capabilities:

### Core Navigation System (Production Ready)
1. **Python Backend (`find_project_path.py`):**  
   Central logic for locating project directories based on a 5-digit code.
   **ENHANCED**: Now includes document discovery and training data generation capabilities.

2. **AutoHotkey (AHK) Frontend (`lld_navigator.ahk`):**  
   Provides the user interface and handles local user interaction.
   **ENHANCED**: Includes training data generation toggle for optional AI/ML workflow integration.

3. **MCP Server (`mcp_server/`):**  
   Exposes backend navigation functions to AI agents or external systems through the Model Context Protocol.

### NEW: Advanced Analysis & Intelligence Layer

4. **Hybrid Training Analyzer (`hybrid_training_analyzer.py`):**
   Processes training data using 90% rule-based + 10% AI approach for optimal speed and reliability.

5. **AI Agent Training Generator (`ai_agent_training_generator.py`):**
   Generates structured training examples and actionable insights for improving AI agents.

6. **Project Extractor (`project_extractor.py`):**
   Provides comprehensive project-specific analysis based on 5-digit project numbers.

## Key Technical Decisions

### Core System Architecture
- **Stdout-based IPC (Inter-Process Communication):**  
  The Python backend communicates results via standard output, enabling robust and platform-neutral IPC. This choice maximizes compatibility and simplifies integration with both the AHK frontend and MCP server.
  **EXTENDED**: Now includes training data generation status messages and analysis results.

- **Loose Coupling:**  
  Each component interacts only via clearly defined interfaces (CLI/stdout or MCP tools), minimizing dependencies and simplifying maintenance.
  **ENHANCED**: Analysis scripts maintain loose coupling while sharing common patterns and data structures.

- **Explicit Separation of Concerns:**  
  - The backend is focused purely on directory resolution and document discovery.
  - The frontend is concerned only with UI/UX and user actions.
  - The MCP server bridges AI and automation workflows to the backend logic.
  - **NEW**: Analysis scripts provide specialized intelligence capabilities without affecting core navigation.

### Advanced Analysis Architecture
- **Hybrid Processing Strategy:**
  - 90% rule-based processing for speed, reliability, and deterministic results
  - 10% strategic AI usage for insights that add genuine value
  - Eliminates JSON parsing errors and timeouts common in pure AI approaches

- **Modular Analysis Design:**
  - Each analysis script serves a specific purpose with clear input/output contracts
  - Common patterns and data structures shared across scripts for consistency
  - Independent execution capability while supporting pipeline workflows

- **Error Resilience:**
  - Robust fallback mechanisms across all analysis components
  - Graceful degradation when API calls fail or data is incomplete
  - Comprehensive logging and user feedback throughout processing

## Component Relationships

### Core Data Flow

1. **User-initiated Navigation Flow:**
   - User enters the project code into the AHK interface.
   - User optionally selects "Generate Training Data" checkbox.
   - AHK script invokes the Python backend on-demand, passing the job number and optional `--training-data` flag.
   - Backend returns resolved path via stdout.
   - **ENHANCED**: If training data flag is set, backend also discovers documents and generates training JSON.
   - AHK presents the result for user action and optionally displays training data confirmation.

2. **AI-initiated Navigation Flow:**
   - AI agent connects via MCP to the server.
   - MCP server invokes Python backend on-demand with the proper job number.
   - Path is returned to the AI agent, enabling automated operations.

### NEW: Analysis & Intelligence Flow

3. **Training Data Analysis Pipeline:**
   ```
   training_data/*.json → Hybrid Analyzer → hybrid_analysis_results.json
                                        ↓
   AI Training Generator → ai_agent_training_dataset.json + insights
   ```

4. **Project-Specific Analysis Flow:**
   ```
   Project ID → Project Extractor → project_profile_[id].json + summary
   ```

5. **Integrated Intelligence Workflow:**
   ```
   QuickNav → Training Data → Hybrid Analysis → AI Training → Project Profiles
   ```

### Analysis System Architecture

**Document Processing Flow:**
```
Raw Documents → Classification → Quality Assessment → Categorization → Analysis Output
```

**Intelligence Generation:**
```
Patterns → Rule-Based Processing → Strategic AI Enhancement → Actionable Insights
```

**Training Data Pipeline:**
```
Project Documents → Structured Examples → Training Dataset → Agent Improvement
```

### Enhanced Diagram (Textual)

```
Users ⇄ [AHK Frontend] ⇄ [Python Backend + Document Discovery]
                                    ↓ training_data/*.json
                          [Hybrid Training Analyzer]
                                    ↓ hybrid_analysis_results.json
                          [AI Agent Training Generator]
                                    ↓ training datasets + insights
AI/Automation ⇄ [MCP Server] ⇄ [Python Backend]
                                    ↓ project data
                          [Project Extractor] → project profiles

Project Managers ⇄ [Project Extractor] ⇄ training_data/training_data_[id].json
```

## Advanced Analysis Patterns

### Shared Classification Systems
- **AV Project Structure**: 6 standard categories (Sales, BOM, PMO, Designs, Handover, Support)
- **Document Types**: 15+ AV-specific classifications (system_design, as_built, floor_plan, etc.)
- **Quality Patterns**: Consistent high/medium/low value assessment across all scripts
- **Client Mappings**: Standard abbreviation expansions (DWNR → Downer, DOE → Department of Education)

### Processing Patterns
- **Rule-Based Foundation**: Deterministic regex and pattern matching for core processing
- **Strategic AI Enhancement**: AI usage only where it provides genuine added value
- **Revision Management**: Intelligent handling of document revisions (Rev 102 > Rev 101)
- **Quality Filtering**: Automated exclusion of low-value documents (templates, archives)

### Data Flow Patterns
- **Input Standardization**: All scripts work with consistent training data JSON format
- **Output Consistency**: Structured JSON outputs with comprehensive metadata
- **Error Handling**: Graceful degradation with informative user feedback
- **Performance Optimization**: Fast processing optimized for large document sets

## Intelligence Capabilities

### Project Health Assessment
- **Completeness Scoring**: Automated calculation of documentation completeness percentage
- **Risk Identification**: Pattern-based detection of project risks and gaps
- **Quality Metrics**: Multi-dimensional assessment of documentation quality
- **Timeline Analysis**: Document revision progression and phase identification

### Technology Analysis
- **AV System Detection**: Automated identification of audio, video, control, and conferencing systems
- **Brand Recognition**: Detection of major AV brands and technologies in documentation
- **Complexity Assessment**: Evaluation of technical complexity based on documentation patterns
- **Scope Analysis**: Comprehensive project scope assessment from document analysis

### AI Agent Development
- **Training Example Generation**: Structured examples for agent improvement across multiple categories
- **Pattern Recognition**: Identification of successful project patterns for agent training
- **Performance Metrics**: Specific KPIs for measuring and improving AI agent effectiveness
- **Automation Opportunities**: Data-driven identification of tasks ready for automation

## Summary

Project QuickNav's evolved architecture ensures:

### Operational Excellence
- **Modularity**: Each component serves a specific purpose with clear interfaces
- **Reliability**: Rule-based foundation provides consistent, predictable results
- **Extensibility**: New analysis capabilities added without affecting core navigation
- **Performance**: Optimized processing delivers results in under 2 minutes

### Intelligence Capabilities
- **Comprehensive Analysis**: Full project profiling from basic navigation data
- **AI Development Support**: Structured training data and improvement insights
- **Business Intelligence**: Project health monitoring and portfolio analysis
- **Quality Assurance**: Automated documentation assessment and risk identification

The platform now serves as a **comprehensive AV project intelligence system** that transforms simple navigation into actionable business and technical insights while maintaining the reliability and simplicity of the original core functionality.