# Project QuickNav – Project Brief

## Overview

Project QuickNav is a utility designed to streamline navigation, access, and context sharing for large-scale project directories, specifically targeting workflows that rely on a structured project numbering system. It provides a rapid, user-friendly means to locate, open, and interact with project folders based on a 5-digit project code, integrating with both human and AI users.

## Core Requirements & Goals

- **Rapid project folder location** via 5-digit project numbers.
- **Seamless user interface** for quick input and navigation.
- **AI interoperability** through an MCP (Model Context Protocol) server for toolchain integration.
- **Preservation of project context and quick recall** through a persistent memory/documentation system.
- **Cross-component, reliable communication** (backend, frontend, AI).

## Overall Objectives

- Improve developer and power-user efficiency for project-based workflows.
- Minimize search time and friction when switching between projects.
- Enable advanced automation and AI-assisted navigation scenarios.

## Target Users

- Developers, engineers, and project managers working with a large number of projects organized by standard codes.
- AI systems and tools seeking direct access to project directories for processing or analysis.

## Primary Functionality

- **Input**: User enters a 5-digit project number.
- **Processing**: Python backend locates the corresponding project folder.
- **Output**: GUI (via AutoHotkey) displays path and provides navigation actions.
- **Integration**: MCP server exposes backend logic for AI/automation workflows.

## High-Level Requirements by Component

### Python Backend (`find_project_path.py`)
- Traverse root directories to locate project folders.
- Expose a CLI interface for folder search.
- Provide results via stdout for IPC.

### AHK Frontend (`lld_navigator.ahk`)
- Present a GUI for user interaction.
- Accept project code input.
- Display resolved paths and navigation options.
- Launch folders in File Explorer or Terminal.

### MCP Server (`mcp_server/`)
- Wrap backend logic as MCP tools.
- Provide standardized protocol for AI access.
- Allow remote or local AI agents to trigger project navigation actions.

## Scope & Foundation

This document defines the core scope of Project QuickNav and serves as the authoritative reference for requirements, stakeholders, and initial technical approach.