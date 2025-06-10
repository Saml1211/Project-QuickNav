import os
import json
import requests
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

class HybridTrainingAnalyzer:
    """
    Hybrid rule-based + AI training data analyzer for AV project documentation.
    
    Uses deterministic rules for 90% of processing (classification, filtering, metadata extraction)
    and AI strategically for 10% (project summaries, scope analysis).
    """
    
    def __init__(self, openrouter_api_key: str, training_data_dir: str = "training_data"):
        self.api_key = openrouter_api_key
        self.training_data_dir = training_data_dir
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rule-based classification patterns
        self.folder_categories = {
            "sales_handover": ["1. Sales Handover", "Sales Handover", "Handover"],
            "bom_orders": ["2. BOM & Orders", "BOM", "Orders", "Purchase", "Procurement"],
            "project_management": ["3. PMO", "PMO", "Project Management", "PM"],
            "system_designs": ["4. System Designs", "System Designs", "Designs", "Engineering"],
            "customer_handover": ["5. Customer Handover Documents", "Customer Handover", "Handover Documents"],
            "avcare_handover": ["6. AVCare Handover", "AVCare", "Support", "Maintenance"]
        }
        
        # Document type patterns
        self.document_types = {
            "floor_plan": r"(?i)(floor\s*plan|floorplan|plan|layout)",
            "system_design": r"(?i)(system\s*design|design|schematic|rack\s*layout)",
            "as_built": r"(?i)(as\s*built|asbuilt|final)",
            "handover": r"(?i)(handover|delivery|completion)",
            "purchase_order": r"(?i)(purchase\s*order|po\s*\d+|order)",
            "change_order": r"(?i)(change\s*order|co\s*\d+|variation)",
            "quote": r"(?i)(quote|quotation|proposal|estimate)",
            "invoice": r"(?i)(invoice|bill|payment)",
            "manual": r"(?i)(manual|guide|instruction)",
            "test_report": r"(?i)(test\s*report|testing|qa|quality)",
            "commissioning": r"(?i)(commission|testing|verification)"
        }
        
        # Quality filters (documents to exclude)
        self.exclude_patterns = [
            r"(?i)/old/",
            r"(?i)/archive/",
            r"(?i)template",
            r"(?i)master",
            r"(?i)\.tmp$",
            r"(?i)~\$",
            r"(?i)backup"
        ]
        
        # Low-value document patterns
        self.low_value_patterns = [
            r"(?i)invoice",
            r"(?i)receipt",
            r"(?i)email",
            r"(?i)correspondence"
        ]
        
        # Client name mappings for common abbreviations
        self.client_mappings = {
            "DWNR": "Downer",
            "DOE": "Department of Education",
            "QLDT": "Queensland Department of Transport",
            "QLD": "Queensland",
            "NSW": "New South Wales",
            "VIC": "Victoria",
            "WA": "Western Australia",
            "SA": "South Australia",
            "TAS": "Tasmania",
            "NT": "Northern Territory",
            "ACT": "Australian Capital Territory"
        }
    
    def discover_training_files(self) -> List[str]:
        """Discover all training_data_*.json files."""
        pattern = r"training_data_\d{5}\.json$"
        files = []
        
        if not os.path.exists(self.training_data_dir):
            print(f"❌ Training data directory '{self.training_data_dir}' not found!")
            return files
            
        for filename in os.listdir(self.training_data_dir):
            if re.match(pattern, filename):
                files.append(os.path.join(self.training_data_dir, filename))
        
        return sorted(files)
    
    def extract_project_metadata(self, project_folder: str) -> Dict[str, str]:
        """Extract project ID, client, and name from folder name using rules."""
        # Pattern: "20692 - DWNR - Project B1012"
        pattern = r"(\d{5})\s*-\s*([^-]+?)\s*-\s*(.+)"
        match = re.match(pattern, project_folder.strip())
        
        if match:
            project_id = match.group(1)
            client_abbrev = match.group(2).strip()
            project_name = match.group(3).strip()
            
            # Expand client abbreviation if known
            client_full = self.client_mappings.get(client_abbrev, client_abbrev)
            
            return {
                "project_id": project_id,
                "client_abbreviation": client_abbrev,
                "client_name": client_full,
                "project_name": project_name,
                "full_project_name": f"{client_full} - {project_name}"
            }
        
        # Fallback: try to extract just the project ID
        id_match = re.search(r"(\d{5})", project_folder)
        if id_match:
            return {
                "project_id": id_match.group(1),
                "client_abbreviation": "",
                "client_name": "",
                "project_name": project_folder,
                "full_project_name": project_folder
            }
        
        return {
            "project_id": "unknown",
            "client_abbreviation": "",
            "client_name": "",
            "project_name": project_folder,
            "full_project_name": project_folder
        }
    
    def classify_document_category(self, document_path: str) -> str:
        """Classify document into category based on folder structure."""
        path_lower = document_path.lower()
        
        for category, patterns in self.folder_categories.items():
            for pattern in patterns:
                if pattern.lower() in path_lower:
                    return category
        
        return "uncategorized"
    
    def classify_document_type(self, document_name: str, document_path: str) -> str:
        """Classify document type based on filename and path patterns."""
        combined_text = f"{document_name} {document_path}".lower()
        
        for doc_type, pattern in self.document_types.items():
            if re.search(pattern, combined_text):
                return doc_type
        
        return "other"
    
    def extract_revision_number(self, document_name: str) -> Optional[int]:
        """Extract revision number from document name."""
        patterns = [
            r"rev\s*(\d+)",
            r"revision\s*(\d+)",
            r"r(\d+)",
            r"v(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, document_name.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def is_excluded_document(self, document_path: str, document_name: str) -> bool:
        """Check if document should be excluded based on quality filters."""
        combined_text = f"{document_path} {document_name}"
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, combined_text):
                return True
        
        return False
    
    def is_low_value_document(self, document_name: str) -> bool:
        """Check if document is low-value (invoices, emails, etc.)."""
        for pattern in self.low_value_patterns:
            if re.search(pattern, document_name):
                return True
        
        return False
    
    def filter_documents_by_revision(self, documents: List[Dict]) -> List[Dict]:
        """Keep only the latest revision of documents with similar names."""
        grouped_docs = defaultdict(list)
        
        # Group documents by base name (without revision info)
        for doc in documents:
            # Remove revision patterns to group similar documents
            base_name = re.sub(r"(?i)\s*-?\s*(rev|revision|r|v)\s*\d+", "", doc["document_name"])
            base_name = re.sub(r"\s+", " ", base_name.strip())
            grouped_docs[base_name].append(doc)
        
        filtered_docs = []
        for base_name, doc_group in grouped_docs.items():
            if len(doc_group) == 1:
                filtered_docs.append(doc_group[0])
            else:
                # Sort by revision number, keeping highest
                doc_group.sort(key=lambda x: x.get("revision", 0) or 0, reverse=True)
                filtered_docs.append(doc_group[0])
        
        return filtered_docs
    
    def process_documents_with_rules(self, documents: List[Dict]) -> List[Dict]:
        """Apply rule-based processing to documents."""
        processed_docs = []
        
        for doc in documents:
            # Skip excluded documents
            if self.is_excluded_document(doc["document_path"], doc["document_name"]):
                continue
            
            # Skip low-value documents
            if self.is_low_value_document(doc["document_name"]):
                continue
            
            # Extract metadata
            project_metadata = self.extract_project_metadata(doc["project_folder"])
            
            # Classify document
            category = self.classify_document_category(doc["document_path"])
            doc_type = self.classify_document_type(doc["document_name"], doc["document_path"])
            revision = self.extract_revision_number(doc["document_name"])
            
            processed_doc = {
                **doc,
                **project_metadata,
                "category": category,
                "document_type": doc_type,
                "revision": revision,
                "filtered": True
            }
            
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def group_documents_by_project(self, documents: List[Dict]) -> Dict[str, Dict]:
        """Group processed documents by project."""
        projects = defaultdict(lambda: {
            "documents": [],
            "metadata": {},
            "categories": defaultdict(list)
        })
        
        for doc in documents:
            project_id = doc["project_id"]
            projects[project_id]["documents"].append(doc)
            
            # Store project metadata (use first document's metadata)
            if not projects[project_id]["metadata"]:
                projects[project_id]["metadata"] = {
                    "project_id": doc["project_id"],
                    "client_name": doc["client_name"],
                    "project_name": doc["project_name"],
                    "full_project_name": doc["full_project_name"]
                }
            
            # Categorize documents
            projects[project_id]["categories"][doc["category"]].append(doc)
        
        return dict(projects)
    
    def call_openrouter_api(self, prompt: str, max_tokens: int = 1000) -> str:
        """Make a focused API call to OpenRouter for project summaries."""
        payload = {
            "model": "anthropic/claude-3-haiku",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error: {e}")
            return f"Error generating summary: {e}"
    
    def generate_project_summary(self, project_data: Dict) -> Tuple[str, str]:
        """Generate AI-powered project summary and scope analysis."""
        metadata = project_data["metadata"]
        categories = project_data["categories"]
        
        # Create a concise summary of document types per category
        summary_info = []
        for category, docs in categories.items():
            if docs:
                doc_types = [doc["document_type"] for doc in docs]
                unique_types = list(set(doc_types))
                summary_info.append(f"{category}: {len(docs)} docs ({', '.join(unique_types)})")
        
        prompt = f"""Based on this AV project data, provide:
1. A 2-sentence project summary
2. A 1-sentence scope analysis

Project: {metadata['full_project_name']}
Document categories: {'; '.join(summary_info)}

Format your response as:
SUMMARY: [2 sentences about what this project is]
SCOPE: [1 sentence about the technical scope/systems involved]"""
        
        response = self.call_openrouter_api(prompt, max_tokens=200)
        
        # Parse response
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?=SCOPE:|$)", response, re.DOTALL)
        scope_match = re.search(r"SCOPE:\s*(.+)", response, re.DOTALL)
        
        summary = summary_match.group(1).strip() if summary_match else "AV system installation project."
        scope = scope_match.group(1).strip() if scope_match else "Audio-visual systems and integration."
        
        return summary, scope
    
    def analyze_training_data(self) -> Dict[str, Any]:
        """Run the complete hybrid analysis."""
        print("🔍 Starting Hybrid Training Data Analysis...")
        start_time = time.time()
        
        # Step 1: Discover files
        print("\n📁 Step 1: Discovering training files...")
        files = self.discover_training_files()
        
        if not files:
            return {"error": "No training data files found"}
        
        print(f"Found {len(files)} training files")
        
        # Step 2: Load and process all documents
        print("\n📊 Step 2: Loading documents...")
        all_documents = []
        files_processed = 0
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_documents.extend(data)
                    files_processed += 1
                    print(f"✓ Loaded {os.path.basename(file_path)}: {len(data)} documents")
            except Exception as e:
                print(f"✗ Error loading {os.path.basename(file_path)}: {e}")
        
        print(f"Total documents loaded: {len(all_documents)}")
        
        # Step 3: Rule-based processing
        print("\n⚙️  Step 3: Rule-based processing...")
        processed_docs = self.process_documents_with_rules(all_documents)
        print(f"Documents after filtering: {len(processed_docs)}")
        
        # Step 4: Filter by revision
        print("\n🔄 Step 4: Filtering revisions...")
        filtered_docs = self.filter_documents_by_revision(processed_docs)
        print(f"Documents after revision filtering: {len(filtered_docs)}")
        
        # Step 5: Group by project
        print("\n📁 Step 5: Grouping by project...")
        projects = self.group_documents_by_project(filtered_docs)
        print(f"Projects identified: {len(projects)}")
        
        # Step 6: AI enhancement (selective)
        print("\n🤖 Step 6: AI enhancement for project summaries...")
        ai_enhanced_projects = 0
        
        for project_id, project_data in projects.items():
            if len(project_data["documents"]) >= 3:  # Only enhance projects with significant documentation
                try:
                    summary, scope = self.generate_project_summary(project_data)
                    project_data["ai_generated_summary"] = summary
                    project_data["scope_analysis"] = scope
                    ai_enhanced_projects += 1
                    print(f"✓ Enhanced project {project_id}")
                except Exception as e:
                    print(f"✗ Failed to enhance project {project_id}: {e}")
                    project_data["ai_generated_summary"] = "AV installation project."
                    project_data["scope_analysis"] = "Audio-visual systems integration."
            else:
                project_data["ai_generated_summary"] = "AV installation project."
                project_data["scope_analysis"] = "Audio-visual systems integration."
        
        # Step 7: Generate final output
        print("\n📄 Step 7: Generating output...")
        
        output = {
            "processing_summary": {
                "total_files_processed": files_processed,
                "total_documents": len(all_documents),
                "documents_after_filtering": len(filtered_docs),
                "ai_enhanced_projects": ai_enhanced_projects,
                "processing_time_seconds": round(time.time() - start_time, 2)
            },
            "projects": []
        }
        
        for project_id, project_data in projects.items():
            project_output = {
                "project_id": project_id,
                "project_name": project_data["metadata"]["full_project_name"],
                "client_name": project_data["metadata"]["client_name"],
                "document_categories": {},
                "ai_generated_summary": project_data.get("ai_generated_summary", ""),
                "scope_analysis": project_data.get("scope_analysis", "")
            }
            
            # Organize documents by category
            for category, docs in project_data["categories"].items():
                if docs:
                    project_output["document_categories"][category] = [
                        {
                            "document_name": doc["document_name"],
                            "document_type": doc["document_type"],
                            "revision": doc.get("revision")
                        }
                        for doc in docs
                    ]
            
            output["projects"].append(project_output)
        
        # Sort projects by ID
        output["projects"].sort(key=lambda x: x["project_id"])
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Analysis complete in {elapsed_time:.1f} seconds!")
        print(f"📊 Processed {len(all_documents)} → {len(filtered_docs)} documents across {len(projects)} projects")
        print(f"🤖 AI enhanced {ai_enhanced_projects} projects")
        
        return output
    
    def save_results(self, results: Dict[str, Any], output_file: str = "hybrid_analysis_results.json"):
        """Save analysis results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {output_file}")


def main():
    """Main execution function."""
    import sys
    
    # Get OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY='your_api_key_here'")
        return
    
    # Get training data directory from command line argument
    training_dir = sys.argv[1] if len(sys.argv) > 1 else "training_data"
    
    # Initialize analyzer
    analyzer = HybridTrainingAnalyzer(api_key, training_dir)
    
    # Run analysis
    results = analyzer.analyze_training_data()
    
    if "error" not in results:
        # Save results
        analyzer.save_results(results)
        
        # Print summary
        summary = results["processing_summary"]
        print(f"\n📈 Final Summary:")
        print(f"   📁 Files processed: {summary['total_files_processed']}")
        print(f"   📄 Total documents: {summary['total_documents']}")
        print(f"   ✅ After filtering: {summary['documents_after_filtering']}")
        print(f"   🤖 AI enhanced projects: {summary['ai_enhanced_projects']}")
        print(f"   ⏱️  Processing time: {summary['processing_time_seconds']}s")
        print(f"\n🎯 Check 'hybrid_analysis_results.json' for detailed results")
    else:
        print(f"❌ Analysis failed: {results['error']}")


if __name__ == "__main__":
    main() 