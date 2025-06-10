#!/usr/bin/env python3
"""
AI-Powered Scope of Work Extractor for QuickNav
Finds and extracts Scope of Work content from project documents using AI analysis and OCR.
Only generates scope documents when actual scope content is found.

Usage:
    python scope_generator.py 12345
    python scope_generator.py 12345 --output custom_scope.md
    
Requirements:
    export OPENROUTER_API_KEY='your_api_key_here'
"""

import os
import re
import argparse
import sys
import math
import tempfile
import requests
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

class AIScopeExtractor:
    """AI-powered Scope of Work extractor with OCR and reasoning capabilities."""
    
    def __init__(self, openrouter_api_key: Optional[str] = None):
        self.api_key = openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        } if self.api_key else None
        
        # Priority order for searching documents
        self.priority_folders = [
            "1. Sales Handover",
            "Sales Handover", 
            "Handover",
            "1. sales handover",
            "sales handover"
        ]
        
        # Scope document patterns (more comprehensive)
        self.scope_patterns = [
            r'scope.*work',
            r'sow\b',
            r'statement.*work',
            r'project.*scope',
            r'requirements?',
            r'specification',
            r'tender.*doc',
            r'proposal',
            r'brief',
            r'charter',
            r'scope.*doc',
            r'deliverables?',
            r'objectives?',
            r'quotation',
            r'quote'
        ]

    def get_onedrive_folder(self):
        """Locate the user's OneDrive - Pro AV Solutions directory."""
        user_profile = os.environ.get("UserProfile")
        if not user_profile:
            return self.setup_test_environment()

        onedrive_path = os.path.join(user_profile, "OneDrive - Pro AV Solutions")
        if not os.path.isdir(onedrive_path):
            return self.setup_test_environment()

        project_files_path = os.path.join(onedrive_path, "Project Files")
        if os.path.isdir(project_files_path):
            return project_files_path

        return onedrive_path

    def setup_test_environment(self):
        """Create a simulated project folder structure in the temp directory."""
        base_dir = os.path.join(tempfile.gettempdir(), "Project Folders")
        os.makedirs(os.path.join(base_dir, "10000 - 10999"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "17000 - 17999"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "10000 - 10999", "10123 - Project A"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "10000 - 10999", "10456 - Project B"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "17000 - 17999", "17741 - Test Project"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "17000 - 17999", "17742 - Another Project"), exist_ok=True)

        # Create test documents in Sales Handover
        project_dir = os.path.join(base_dir, "17000 - 17999", "17741 - Test Project")
        sales_handover_dir = os.path.join(project_dir, "1. Sales Handover")
        os.makedirs(sales_handover_dir, exist_ok=True)
        
        # Create a sample scope document
        scope_file = os.path.join(sales_handover_dir, "Project Scope Document.txt")
        with open(scope_file, 'w') as f:
            f.write("""SCOPE OF WORK
Project Overview
This project involves the design, supply, installation, and commissioning of a complete audio visual system for the main conference room.

Project Requirements
The client requires a state-of-the-art presentation and video conferencing solution capable of supporting:
- Local presentations from laptops and mobile devices  
- Video conferencing with remote participants
- High-quality audio capture and reproduction
- Automated system control via touch panel interface

Deliverables
- System design drawings
- Equipment supply and installation
- Programming and commissioning
- User training and documentation
- 12-month warranty and support
""")

        return base_dir

    def get_project_folders(self, onedrive_path):
        """Get the Project Folders directory path."""
        if os.path.basename(onedrive_path) == "Project Folders":
            return onedrive_path
        pf_path = os.path.join(onedrive_path, "Project Folders")
        if not os.path.isdir(pf_path):
            return self.setup_test_environment()
        return pf_path

    def get_range_folder(self, proj_num: str, pf_path: str):
        """Get the directory path for the thousand-range containing the project."""
        num = int(proj_num)
        start = int(math.floor(num / 1000) * 1000)
        end = start + 999
        range_name = f"{start} - {end}"
        range_path = os.path.join(pf_path, range_name)

        if not os.path.isdir(range_path):
            if os.path.basename(os.path.dirname(pf_path)) == "Temp" or os.path.basename(pf_path) == "Project Folders":
                try:
                    os.makedirs(range_path, exist_ok=True)
                except Exception:
                    raise Exception(f"Failed to create range folder {range_name}")
            else:
                raise Exception(f"Range folder {range_name} not found")
        return range_path

    def search_project_dirs(self, proj_num: str, range_path: str):
        """Search for all directories matching '[ProjNum] - *' in the range folder."""
        pat = re.compile(rf"^{proj_num} - .+")
        try:
            entries = os.listdir(range_path)
        except Exception:
            raise Exception("Unable to list range folder contents")
        
        matches = []
        for entry in entries:
            full_path = os.path.join(range_path, entry)
            if os.path.isdir(full_path) and pat.match(entry):
                matches.append(os.path.abspath(full_path))
        return matches

    def find_project_folder(self, project_id: str) -> Optional[str]:
        """Locate the project folder using the same logic as find_project_path.py."""
        try:
            if not re.fullmatch(r"\d{5}", project_id):
                print(f"❌ Invalid project ID format: {project_id} (must be 5 digits)")
                return None

            onedrive_folder = self.get_onedrive_folder()
            pfolder = self.get_project_folders(onedrive_folder)
            range_folder = self.get_range_folder(project_id, pfolder)
            matches = self.search_project_dirs(project_id, range_folder)
            
            if not matches:
                print(f"❌ No project folder found for number {project_id}")
                return None
            elif len(matches) == 1:
                return matches[0]
            else:
                print(f"⚠️  Multiple matches found for project {project_id}, using first match")
                for i, match in enumerate(matches):
                    print(f"   {i+1}. {os.path.basename(match)}")
                return matches[0]
                
        except Exception as e:
            print(f"❌ Error locating project folder: {e}")
            return None

    def find_sales_handover_folder(self, project_folder: str) -> Optional[str]:
        """Find the Sales Handover folder within the project directory."""
        for folder_name in self.priority_folders:
            potential_path = os.path.join(project_folder, folder_name)
            if os.path.isdir(potential_path):
                print(f"📁 Found Sales Handover folder: {folder_name}")
                return potential_path
        
        # Search for any folder containing "sales" or "handover"
        try:
            for item in os.listdir(project_folder):
                item_path = os.path.join(project_folder, item)
                if os.path.isdir(item_path):
                    item_lower = item.lower()
                    if 'sales' in item_lower or 'handover' in item_lower:
                        print(f"📁 Found potential Sales Handover folder: {item}")
                        return item_path
        except Exception as e:
            print(f"⚠️  Error searching for Sales Handover folder: {e}")
        
        return None

    def find_all_documents(self, search_path: str, extensions: List[str] = None) -> List[str]:
        """Find all documents in the given path, prioritizing scope-related documents."""
        if extensions is None:
            extensions = ['.pdf', '.docx', '.doc', '.txt', '.rtf']
        
        all_docs = []
        scope_docs = []
        
        try:
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    file_lower = file.lower()
                    file_ext = os.path.splitext(file_lower)[1]
                    
                    if file_ext in extensions:
                        full_path = os.path.join(root, file)
                        all_docs.append(full_path)
                        
                        # Check if this matches scope patterns
                        for pattern in self.scope_patterns:
                            if re.search(pattern, file_lower):
                                scope_docs.append(full_path)
                                break
        except Exception as e:
            print(f"❌ Error searching for documents: {e}")
        
        # Return scope documents first, then other documents
        return scope_docs + [doc for doc in all_docs if doc not in scope_docs]

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file types with enhanced OCR capabilities."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_word(file_path)
        elif file_ext == '.rtf':
            return self.extract_text_from_rtf(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                return f"[Error reading text file: {e}]"
        else:
            return f"[Unsupported file type: {file_ext}]"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with multiple fallback methods."""
        text = ""
        
        # Try PyPDF2 first
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try pdfplumber as fallback
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text
        except ImportError:
            pass
        except Exception:
            pass
        
        # If all methods fail, return an indicator that OCR might be needed
        return f"[PDF text extraction failed for {os.path.basename(pdf_path)} - may require OCR]"

    def extract_text_from_word(self, doc_path: str) -> str:
        """Extract text from Word document."""
        try:
            import docx
            doc = docx.Document(doc_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text if text.strip() else f"[No text extracted from {os.path.basename(doc_path)}]"
        except ImportError:
            return f"[Word content from {os.path.basename(doc_path)} - python-docx library not available]"
        except Exception as e:
            return f"[Error extracting Word content: {e}]"

    def extract_text_from_rtf(self, rtf_path: str) -> str:
        """Extract text from RTF file with enhanced parsing."""
        try:
            # Try using striprtf library if available
            try:
                from striprtf.striprtf import rtf_to_text
                with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                    rtf_content = f.read()
                text = rtf_to_text(rtf_content)
                if text and text.strip() and not text.startswith('adeflang'):
                    return text
            except ImportError:
                print("     striprtf library not available, using fallback parser")
            except Exception as e:
                print(f"     striprtf failed: {e}, using fallback parser")
            
            # Enhanced fallback RTF parsing
            with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            import re
            
            # Try multiple encoding approaches
            encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(rtf_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    break
                except:
                    continue
            
            # Enhanced RTF to text conversion
            original_content = content
            
            # Remove font table and other header information
            content = re.sub(r'\\fonttbl.*?}', '', content, flags=re.DOTALL)
            content = re.sub(r'\\colortbl.*?}', '', content, flags=re.DOTALL)
            content = re.sub(r'\\stylesheet.*?}', '', content, flags=re.DOTALL)
            content = re.sub(r'\\info.*?}', '', content, flags=re.DOTALL)
            
            # Remove RTF version and other header info
            content = re.sub(r'\\rtf\d+', '', content)
            content = re.sub(r'\\ansi', '', content)
            content = re.sub(r'\\deff\d+', '', content)
            content = re.sub(r'\\deflang\d+', '', content)
            
            # Remove control words with parameters
            content = re.sub(r'\\[a-z]+\d*\s*', ' ', content, flags=re.IGNORECASE)
            
            # Remove control symbols
            content = re.sub(r'\\[^a-zA-Z\s]', '', content)
            
            # Handle special characters
            content = re.sub(r'\\\'([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), content)
            
            # Remove remaining braces and control sequences
            # Remove nested braces by repeatedly removing innermost braces
            prev_len = len(content)
            while True:
                content = re.sub(r'{[^{}]*}', ' ', content)
                if len(content) == prev_len:
                    break
                prev_len = len(content)
            
            # Remove any remaining braces
            content = re.sub(r'[{}]', ' ', content)
            
            # Clean up whitespace and special characters
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', content)
            
            # Remove common RTF artifacts
            content = re.sub(r'\\par\b', '\n', content)
            content = re.sub(r'\\line\b', '\n', content)
            content = re.sub(r'\\tab\b', '\t', content)
            
            # Final cleanup
            content = content.strip()
            
            # If we still have mostly control characters, try a different approach
            if not content or len([c for c in content if c.isalnum()]) < len(content) * 0.3:
                print("     Fallback RTF parsing produced poor results, trying simplified extraction")
                
                # Super basic extraction - just find text between common RTF patterns
                simple_text = []
                text_parts = re.findall(r'[a-zA-Z][a-zA-Z0-9\s.,;:!?\'"()-]{10,}', original_content)
                for part in text_parts:
                    cleaned = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'"()-]', ' ', part)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    if len(cleaned) > 20 and any(word in cleaned.lower() for word in ['project', 'scope', 'work', 'install', 'system']):
                        simple_text.append(cleaned)
                
                if simple_text:
                    content = '\n'.join(simple_text)
            
            return content if content.strip() else f"[No readable text extracted from RTF: {os.path.basename(rtf_path)}]"
            
        except Exception as e:
            return f"[Error extracting RTF content: {e}]"

    def call_openrouter_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """Make an API call to OpenRouter for AI analysis."""
        if not self.api_key or not self.headers:
            return "[AI Analysis Error: No API key provided]"
            
        payload = {
            "model": "anthropic/claude-3-haiku",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"[AI Analysis Error: {e}]"

    def analyze_document_for_scope(self, file_path: str, document_text: str) -> Tuple[bool, str, str]:
        """Use AI to analyze if document contains scope information and extract it."""
        if not document_text or "[Error" in document_text:
            return False, "", "Unable to extract text from document"
        
        # Check if we have an API key for AI analysis
        if not self.api_key:
            return self.analyze_document_pattern_based(file_path, document_text)
        
        prompt = f"""Analyze this document for Scope of Work content. This is from an AV (Audio Visual) project document.

Document: {os.path.basename(file_path)}

Content:
{document_text[:3000]}...

Tasks:
1. Determine if this document contains actual Scope of Work information (requirements, deliverables, project objectives, work to be performed)
2. If it contains scope information, extract and format it cleanly
3. Focus on: project objectives, deliverables, requirements, systems to be installed, work to be performed

Respond in this format:
CONTAINS_SCOPE: [YES/NO]
EXTRACTED_SCOPE: [If YES, provide the cleaned and formatted scope content. If NO, leave empty]
REASON: [Brief explanation of your determination]"""

        response = self.call_openrouter_api(prompt, max_tokens=1500)
        
        # Parse the AI response
        contains_scope = False
        extracted_scope = ""
        reason = ""
        
        contains_match = re.search(r"CONTAINS_SCOPE:\s*(YES|NO)", response, re.IGNORECASE)
        if contains_match and contains_match.group(1).upper() == "YES":
            contains_scope = True
        
        scope_match = re.search(r"EXTRACTED_SCOPE:\s*(.*?)(?=REASON:|$)", response, re.DOTALL | re.IGNORECASE)
        if scope_match:
            extracted_scope = scope_match.group(1).strip()
        
        reason_match = re.search(r"REASON:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        
        return contains_scope, extracted_scope, reason

    def analyze_document_pattern_based(self, file_path: str, document_text: str) -> Tuple[bool, str, str]:
        """Fallback pattern-based analysis when no AI is available."""
        if not document_text or "[Error" in document_text:
            return False, "", "Unable to extract text from document"
        
        # Debug: Show first 200 characters of extracted text
        print(f"     Debug: First 200 chars: {document_text[:200]}...")
        
        text_lower = document_text.lower()
        
        # Enhanced scope detection patterns
        scope_indicators = [
            r'scope\s+of\s+work',
            r'project\s+scope',
            r'work\s+scope',
            r'scope\s+statement',
            r'scope\s+description',
            r'project\s+requirements',
            r'deliverables?',
            r'objectives?',
            r'goals?',
            r'project\s+overview',
            r'work\s+to\s+be\s+performed',
            r'systems?\s+to\s+be\s+installed',
            r'equipment\s+to\s+be\s+supplied',
            r'installation\s+requirements',
            r'technical\s+requirements',
            # Additional patterns for AV project documents
            r'audio\s+visual',
            r'av\s+system',
            r'conference\s+room',
            r'presentation\s+system',
            r'video\s+conferencing',
            r'supply\s+and\s+install',
            r'commissioning',
            r'training',
            r'warranty'
        ]
        
        # Check for scope indicators with debug output
        scope_found = False
        found_patterns = []
        for pattern in scope_indicators:
            if re.search(pattern, text_lower):
                scope_found = True
                found_patterns.append(pattern)
        
        if found_patterns:
            print(f"     Debug: Found patterns: {found_patterns}")
        
        if not scope_found:
            return False, "", "No scope indicators found in document"
        
        # If the filename contains "scope", assume the entire content is scope-related
        filename_lower = os.path.basename(file_path).lower()
        if 'scope' in filename_lower:
            print(f"     Debug: Filename contains 'scope', treating entire content as scope")
            formatted_content = self.clean_scope_text(document_text)
            return True, formatted_content, f"Scope document identified by filename, found patterns: {', '.join(found_patterns)}"
        
        # Extract scope sections using pattern matching
        extracted_sections = []
        lines = document_text.split('\n')
        
        current_section = []
        in_scope_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line starts a scope section
            for pattern in scope_indicators:
                if re.search(pattern, line_lower):
                    # If we were already in a scope section, save it
                    if in_scope_section and current_section:
                        extracted_sections.append('\n'.join(current_section))
                    
                    # Start new scope section
                    current_section = [line.strip()]
                    in_scope_section = True
                    break
            else:
                # Check if this line ends a scope section
                if in_scope_section:
                    end_patterns = [
                        r'exclusions?',
                        r'assumptions?',
                        r'limitations?',
                        r'out\s+of\s+scope',
                        r'not\s+included',
                        r'schedule',
                        r'timeline',
                        r'budget',
                        r'pricing',
                        r'terms\s+and\s+conditions'
                    ]
                    
                    end_section = False
                    for pattern in end_patterns:
                        if re.search(pattern, line_lower):
                            end_section = True
                            break
                    
                    if end_section:
                        # End current scope section
                        if current_section:
                            extracted_sections.append('\n'.join(current_section))
                        current_section = []
                        in_scope_section = False
                    else:
                        # Continue current scope section
                        if line.strip():  # Only add non-empty lines
                            current_section.append(line.strip())
        
        # Add any remaining scope section
        if in_scope_section and current_section:
            extracted_sections.append('\n'.join(current_section))
        
        if extracted_sections:
            # Clean and format the extracted content
            formatted_content = self.clean_scope_text('\n\n'.join(extracted_sections))
            return True, formatted_content, f"Scope content found using pattern matching, patterns: {', '.join(found_patterns)}"
        else:
            return False, "", f"Scope indicators found ({', '.join(found_patterns)}) but no extractable content"

    def clean_scope_text(self, text: str) -> str:
        """Clean and format scope text into proper markdown with headers and lists."""
        lines = text.split('\n')
        formatted_lines = []
        
        # Common section headers to format as markdown headers
        section_patterns = [
            r'project\s+objective',
            r'project\s+overview',
            r'scope\s+of\s+work',
            r'scope\s+description',
            r'locations?\s+and\s+room\s+details?',
            r'equipment\s+and\s+materials?',
            r'project\s+activities',
            r'installation\s+process',
            r'pre\s+installation\s+preparation',
            r'assumptions?\s+and\s+dependencies',
            r'deliverables?',
            r'exclusions?',
            r'requirements?',
            r'technical\s+requirements?',
            r'work\s+to\s+be\s+performed',
            r'responsibilities',
            r'schedule',
            r'timeline',
            r'commissioning',
            r'testing',
            r'training',
            r'warranty',
            r'support'
        ]
        
        # Sub-section patterns (level 3 headers)
        subsection_patterns = [
            r'santos[\'\']?\s*responsibilities',
            r'client\s+responsibilities',
            r'new\s+displays?',
            r'brackets?',
            r'friday\s*[–-]\s*initial\s+setup',
            r'saturday\s*[–-]\s*preparation',
            r'sunday\s*[–-]\s*display\s+installation',
            r'brisbane\s*[–-]',
            r'curtis\s+island',
            r'installation\s+process'
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            line_lower = line.lower()
            
            # Skip empty lines but preserve them for spacing
            if not line:
                formatted_lines.append('')
                i += 1
                continue
            
            # Remove page numbers and common artifacts
            if re.match(r'^\s*page\s+\d+\s+of\s+\d+\s*$', line_lower) or \
               re.match(r'^\s*\d+\s*$', line) or \
               len(line) < 3:
                i += 1
                continue
            
            # Check if this is a main section header
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, line_lower):
                    formatted_lines.append(f"### {line}")
                    formatted_lines.append('')
                    is_section_header = True
                    break
            
            if is_section_header:
                i += 1
                continue
            
            # Check if this is a subsection header
            is_subsection_header = False
            for pattern in subsection_patterns:
                if re.match(pattern, line_lower):
                    formatted_lines.append(f"#### {line}")
                    formatted_lines.append('')
                    is_subsection_header = True
                    break
            
            if is_subsection_header:
                i += 1
                continue
            
            # Check if this looks like a list item that should be formatted
            if self._is_list_item(line):
                # Format as list item
                formatted_line = self._format_list_item(line)
                formatted_lines.append(formatted_line)
            else:
                # Check if this is likely a standalone line that should be its own paragraph
                if len(line) < 80 and not line.endswith('.') and not line.endswith(':'):
                    # Check if the next line might be continuation or a list
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and (self._is_list_item(next_line) or len(next_line) > 80):
                            # This might be a header or standalone item
                            if not any(formatted_lines[-3:]) == ['']:  # Don't add extra spacing
                                formatted_lines.append('')
                            formatted_lines.append(f"**{line}**")
                            formatted_lines.append('')
                        else:
                            formatted_lines.append(line)
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            
            i += 1
        
        # Post-processing cleanup
        result = '\n'.join(formatted_lines)
        
        # Clean up excessive whitespace
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        # Fix bullet points that weren't caught
        result = re.sub(r'^\s*[•▪▫·]\s*', '- ', result, flags=re.MULTILINE)
        
        # Fix numbered lists
        result = re.sub(r'^\s*(\d+\.)\s*([A-Z])', r'1. \2', result, flags=re.MULTILINE)
        
        # Remove excessive spaces
        result = re.sub(r' {2,}', ' ', result)
        
        return result.strip()
    
    def _is_list_item(self, line: str) -> bool:
        """Check if a line appears to be a list item."""
        line = line.strip()
        return bool(re.match(r'^[-•▪▫·]\s+', line) or 
                   re.match(r'^\d+\.\s+', line) or
                   re.match(r'^[a-z]\)\s+', line) or
                   re.match(r'^[IVX]+\.\s+', line, re.IGNORECASE) or
                   (line.lower().startswith(('arrange ', 'provide ', 'ensure ', 'manage ', 
                                           'install ', 'mount ', 'perform ', 'relocate ',
                                           'commission ', 'secure ', 'mark ', 'retrieve ',
                                           'decommission ', 'remove ', 'electronically ')) and
                    len(line) > 20))
    
    def _format_list_item(self, line: str) -> str:
        """Format a line as a proper markdown list item."""
        line = line.strip()
        
        # If it already has a bullet point, clean it up
        if re.match(r'^[-•▪▫·]\s+', line):
            return re.sub(r'^[-•▪▫·]\s+', '- ', line)
        
        # If it has a number, keep it
        if re.match(r'^\d+\.\s+', line):
            return line
        
        # If it starts with action words, make it a bullet point
        if line.lower().startswith(('arrange ', 'provide ', 'ensure ', 'manage ', 
                                  'install ', 'mount ', 'perform ', 'relocate ',
                                  'commission ', 'secure ', 'mark ', 'retrieve ',
                                  'decommission ', 'remove ', 'electronically ')):
            return f"- {line}"
        
        return line

    def generate_scope_markdown(self, project_id: str, project_folder: str, 
                               scope_content: List[Dict[str, Any]]) -> str:
        """Generate formatted markdown scope document."""
        md_content = []
        
        # Header
        project_name = os.path.basename(project_folder)
        md_content.append(f"# Scope of Work - {project_id}")
        md_content.append(f"**Project:** {project_name}")
        md_content.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md_content.append("")
        
        # Overview
        md_content.append("## Overview")
        md_content.append("")
        md_content.append("This document contains the Scope of Work extracted from project documentation using AI analysis.")
        md_content.append(f"**Source Documents:** {len(scope_content)} documents analyzed")
        md_content.append("")
        
        # Scope Content
        md_content.append("## Scope of Work")
        md_content.append("")
        
        for i, content in enumerate(scope_content, 1):
            source_doc = os.path.basename(content['source'])
            
            if len(scope_content) > 1:
                md_content.append(f"### From: {source_doc}")
                md_content.append("")
            
            # Add the extracted scope content
            md_content.append(content['scope'])
            md_content.append("")
        
        # Source Documents
        md_content.append("## Source Documents")
        md_content.append("")
        for content in scope_content:
            doc_name = os.path.basename(content['source'])
            rel_path = os.path.relpath(content['source'], project_folder)
            md_content.append(f"- **{doc_name}**")
            md_content.append(f"  - Path: `{rel_path}`")
            md_content.append(f"  - Analysis: {content['reason']}")
            md_content.append("")
        
        md_content.append("---")
        md_content.append("*This scope document was generated using AI analysis of project documentation.*")
        md_content.append("*Please verify accuracy and completeness by reviewing the source documents.*")
        
        return "\n".join(md_content)

    def extract_scope(self, project_id: str, output_file: Optional[str] = None) -> bool:
        """Main function to extract scope of work for a project."""
        
        mode = "AI-Powered" if self.api_key else "Pattern-Based"
        print(f"🔍 {mode} Scope Extraction for project {project_id}...")
        
        # Find project folder
        project_folder = self.find_project_folder(project_id)
        if not project_folder:
            return False
        
        print(f"📁 Found project folder: {os.path.basename(project_folder)}")
        
        # Look for Sales Handover folder first
        sales_handover_folder = self.find_sales_handover_folder(project_folder)
        
        # Determine search scope
        if sales_handover_folder:
            print(f"🔍 Searching in Sales Handover folder: {os.path.basename(sales_handover_folder)}")
            search_paths = [sales_handover_folder]
        else:
            print("⚠️  No Sales Handover folder found, searching entire project folder")
            search_paths = [project_folder]
        
        # Find all documents in priority order
        all_documents = []
        for search_path in search_paths:
            documents = self.find_all_documents(search_path)
            all_documents.extend(documents)
        
        if not all_documents:
            print("❌ No documents found in project folder")
            return False
        
        print(f"📄 Found {len(all_documents)} documents to analyze")
        
        # Analyze documents for scope content
        scope_content = []
        documents_processed = 0
        max_documents = 25 if sales_handover_folder else 10  # More docs in Sales Handover
        
        # Prioritize documents with "scope" in filename
        priority_docs = [doc for doc in all_documents if 'scope' in os.path.basename(doc).lower()]
        other_docs = [doc for doc in all_documents if 'scope' not in os.path.basename(doc).lower()]
        prioritized_documents = priority_docs + other_docs
        
        print(f"📊 Processing {len(priority_docs)} scope documents first, then {len(other_docs)} other documents")
        
        for doc_path in prioritized_documents:
            print(f"📖 Analyzing: {os.path.basename(doc_path)}")
            
            # Extract text from document
            text = self.extract_text_from_file(doc_path)
            
            # Use AI to analyze for scope content
            contains_scope, extracted_scope, reason = self.analyze_document_for_scope(doc_path, text)
            
            if contains_scope and extracted_scope:
                scope_content.append({
                    'source': doc_path,
                    'scope': extracted_scope,
                    'reason': reason
                })
                print(f"   ✅ Scope content found!")
            else:
                print(f"   ❌ No scope content: {reason}")
            
            documents_processed += 1
            
            # Limit to prevent excessive processing
            if documents_processed >= max_documents:
                print(f"   ⚠️  Analyzed maximum {max_documents} documents")
                break
        
        # Only generate scope document if we found actual scope content
        if not scope_content:
            print("❌ No scope content found in any documents")
            print("💡 Scope documents should be present in '1. Sales Handover' folder")
            print("💡 Ensure documents contain project requirements, deliverables, or objectives")
            return False
        
        # Generate markdown
        print("📝 Generating scope document...")
        markdown_content = self.generate_scope_markdown(
            project_id, project_folder, scope_content
        )
        
        # Save to file
        if not output_file:
            output_file = f"{project_id}_project_scope.md"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✅ Scope document saved to: {output_file}")
            print(f"📊 Successfully extracted scope from {len(scope_content)} documents")
            return True
        except Exception as e:
            print(f"❌ Error saving scope document: {e}")
            return False

def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Scope of Work Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scope_generator.py 17741
  python scope_generator.py 17741 --output custom_scope.md

Requirements:
  export OPENROUTER_API_KEY='your_api_key_here'
  
Optional Dependencies for enhanced PDF support:
  pip install PyPDF2 python-docx pdfplumber
        """
    )
    
    parser.add_argument(
        "project_id",
        help="5-digit project number"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Custom output filename (default: [project_id]_project_scope.md)"
    )
    
    args = parser.parse_args()
    
    # Validate project ID
    if not re.match(r'^\d{5}$', args.project_id):
        print("❌ Error: Project ID must be exactly 5 digits")
        sys.exit(1)
    
    # Get API key (optional for pattern-based fallback)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not set - using pattern-based analysis")
        print("💡 For enhanced AI analysis, set: export OPENROUTER_API_KEY='your_api_key_here'")
    
    # Extract scope
    extractor = AIScopeExtractor(api_key)
    success = extractor.extract_scope(args.project_id, args.output)
    
    if success:
        print("\n🎉 Scope extraction completed successfully!")
    else:
        print("\n❌ Scope extraction failed - no valid scope content found!")
        sys.exit(1)

if __name__ == "__main__":
    main() 