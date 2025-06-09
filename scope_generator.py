#!/usr/bin/env python3
"""
Scope of Work Extractor for QuickNav
Finds and extracts Scope of Work content from project documents and formats it into markdown.

Usage:
    python scope_generator.py 12345
    python scope_generator.py 12345 --output custom_scope.md
"""

import os
import re
import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

class ScopeExtractor:
    """Extracts Scope of Work content from project documents."""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Initialize patterns for scope document identification."""
        
        # Document filename patterns that likely contain scope information
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
            r'scope.*doc'
        ]
        
        # Content patterns to identify scope sections in documents
        self.scope_section_patterns = [
            r'scope\s+of\s+work',
            r'project\s+scope',
            r'work\s+scope',
            r'scope\s+statement',
            r'scope\s+description',
            r'project\s+requirements',
            r'deliverables?',
            r'objectives?',
            r'goals?'
        ]
        
        # Patterns to identify end of scope sections
        self.scope_end_patterns = [
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

    def find_project_folder(self, project_id: str) -> Optional[str]:
        """Use find_project_path.py to locate the project folder."""
        try:
            # Run the find_project_path.py script
            result = subprocess.run(
                [sys.executable, "find_project_path.py", project_id],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output.startswith("SUCCESS:"):
                    project_path = output[8:]  # Remove "SUCCESS:" prefix
                    if os.path.isdir(project_path):
                        return project_path
                elif output.startswith("SELECT:"):
                    # Multiple matches found, use the first one
                    paths = output[7:].split("|")  # Remove "SELECT:" prefix
                    if paths and os.path.isdir(paths[0]):
                        return paths[0]
            
            return None
        except Exception as e:
            print(f"❌ Error running find_project_path.py: {e}")
            return None

    def find_scope_documents(self, project_folder: str) -> List[str]:
        """Find documents that likely contain scope of work information."""
        scope_docs = []
        
        # Document extensions to search
        doc_extensions = ['.pdf', '.docx', '.doc', '.txt', '.rtf']
        
        try:
            for root, dirs, files in os.walk(project_folder):
                for file in files:
                    file_lower = file.lower()
                    file_ext = os.path.splitext(file_lower)[1]
                    
                    if file_ext in doc_extensions:
                        # Check if filename matches scope patterns
                        for pattern in self.scope_patterns:
                            if re.search(pattern, file_lower):
                                full_path = os.path.join(root, file)
                                scope_docs.append(full_path)
                                break
        except Exception as e:
            print(f"❌ Error searching for scope documents: {e}")
        
        return scope_docs

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            # Try using PyPDF2 first
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                pass
            
            # Try using pdfplumber as fallback
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
            except ImportError:
                pass
            
            # If no PDF libraries available, return placeholder
            return f"[PDF content from {os.path.basename(pdf_path)} - PDF extraction libraries not available]"
            
        except Exception as e:
            return f"[Error extracting PDF content: {e}]"

    def extract_text_from_word(self, doc_path: str) -> str:
        """Extract text from Word document."""
        try:
            import docx
            doc = docx.Document(doc_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            return f"[Word content from {os.path.basename(doc_path)} - python-docx library not available]"
        except Exception as e:
            return f"[Error extracting Word content: {e}]"

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file types."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_word(file_path)
        elif file_ext in ['.txt', '.rtf']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                return f"[Error reading text file: {e}]"
        else:
            return f"[Unsupported file type: {file_ext}]"

    def extract_scope_sections(self, text: str) -> List[str]:
        """Extract scope-related sections from document text."""
        scope_sections = []
        lines = text.split('\n')
        
        current_section = []
        in_scope_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line starts a scope section
            for pattern in self.scope_section_patterns:
                if re.search(pattern, line_lower):
                    # If we were already in a scope section, save it
                    if in_scope_section and current_section:
                        scope_sections.append('\n'.join(current_section))
                    
                    # Start new scope section
                    current_section = [line.strip()]
                    in_scope_section = True
                    break
            else:
                # Check if this line ends a scope section
                if in_scope_section:
                    end_section = False
                    for pattern in self.scope_end_patterns:
                        if re.search(pattern, line_lower):
                            end_section = True
                            break
                    
                    if end_section:
                        # End current scope section
                        if current_section:
                            scope_sections.append('\n'.join(current_section))
                        current_section = []
                        in_scope_section = False
                    else:
                        # Continue current scope section
                        if line.strip():  # Only add non-empty lines
                            current_section.append(line.strip())
        
        # Add any remaining scope section
        if in_scope_section and current_section:
            scope_sections.append('\n'.join(current_section))
        
        return scope_sections

    def clean_scope_text(self, text: str) -> str:
        """Clean and format scope text for better readability."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up bullet points and numbering
        text = re.sub(r'^\s*[•▪▫·]\s*', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', text, flags=re.MULTILINE)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def generate_scope_markdown(self, project_id: str, project_folder: str, 
                               scope_documents: List[str], scope_content: List[Dict[str, Any]]) -> str:
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
        md_content.append(f"This document contains the Scope of Work extracted from project documentation.")
        md_content.append(f"**Source Documents:** {len(scope_documents)} documents analyzed")
        md_content.append("")
        
        # Scope Content
        if scope_content:
            md_content.append("## Scope of Work")
            md_content.append("")
            
            for i, content in enumerate(scope_content, 1):
                source_doc = os.path.basename(content['source'])
                
                if len(scope_content) > 1:
                    md_content.append(f"### From: {source_doc}")
                    md_content.append("")
                
                # Add the scope sections
                for section in content['sections']:
                    # Format the section content
                    formatted_section = self.clean_scope_text(section)
                    md_content.append(formatted_section)
                    md_content.append("")
        else:
            md_content.append("## Scope of Work")
            md_content.append("")
            md_content.append("*No specific scope of work content was found in the analyzed documents.*")
            md_content.append("")
            
            if scope_documents:
                md_content.append("### Documents Analyzed")
                md_content.append("")
                for doc in scope_documents:
                    doc_name = os.path.basename(doc)
                    md_content.append(f"- {doc_name}")
                md_content.append("")
                
                md_content.append("**Note:** While scope-related documents were found, ")
                md_content.append("automatic text extraction may have been limited. ")
                md_content.append("Please review the source documents manually for complete scope information.")
            else:
                md_content.append("**Note:** No documents matching typical scope of work patterns were found in the project folder.")
            md_content.append("")
        
        # Source Documents
        if scope_documents:
            md_content.append("## Source Documents")
            md_content.append("")
            for doc in scope_documents:
                doc_name = os.path.basename(doc)
                rel_path = os.path.relpath(doc, project_folder)
                md_content.append(f"- **{doc_name}**")
                md_content.append(f"  - Path: `{rel_path}`")
                md_content.append("")
        
        md_content.append("---")
        md_content.append("*This scope document was generated automatically from project documentation.*")
        md_content.append("*Please verify accuracy and completeness by reviewing the source documents.*")
        
        return "\n".join(md_content)

    def extract_scope(self, project_id: str, output_file: Optional[str] = None) -> bool:
        """Main function to extract scope of work for a project."""
        
        print(f"🔍 Extracting Scope of Work for project {project_id}...")
        
        # Find project folder
        project_folder = self.find_project_folder(project_id)
        if not project_folder:
            print(f"❌ Could not locate project folder for {project_id}")
            return False
        
        print(f"📁 Found project folder: {os.path.basename(project_folder)}")
        
        # Find scope documents
        print("🔎 Searching for scope-related documents...")
        scope_documents = self.find_scope_documents(project_folder)
        
        if not scope_documents:
            print("⚠️  No scope-related documents found")
        else:
            print(f"📄 Found {len(scope_documents)} scope-related documents:")
            for doc in scope_documents:
                print(f"   - {os.path.basename(doc)}")
        
        # Extract scope content
        scope_content = []
        
        if scope_documents:
            print("📖 Extracting content from documents...")
            
            for doc_path in scope_documents:
                print(f"   Processing: {os.path.basename(doc_path)}")
                
                # Extract text from document
                text = self.extract_text_from_file(doc_path)
                
                # Extract scope sections
                sections = self.extract_scope_sections(text)
                
                if sections:
                    scope_content.append({
                        'source': doc_path,
                        'sections': sections
                    })
                    print(f"   ✅ Found {len(sections)} scope sections")
                else:
                    print(f"   ⚠️  No scope sections identified")
        
        # Generate markdown
        print("📝 Generating scope document...")
        markdown_content = self.generate_scope_markdown(
            project_id, project_folder, scope_documents, scope_content
        )
        
        # Save to file
        if not output_file:
            output_file = f"{project_id}_project_scope.md"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✅ Scope document saved to: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving scope document: {e}")
            return False

def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Extract Scope of Work from project documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scope_generator.py 17741
  python scope_generator.py 17741 --output custom_scope.md

Dependencies:
  For full functionality, install optional packages:
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
    
    # Extract scope
    extractor = ScopeExtractor()
    success = extractor.extract_scope(args.project_id, args.output)
    
    if success:
        print("\n🎉 Scope extraction completed successfully!")
    else:
        print("\n❌ Scope extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 