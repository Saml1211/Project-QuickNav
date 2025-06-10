import os
import json
import argparse
import re
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

class ProjectExtractor:
    """
    Extracts and structures comprehensive project-specific information for AV projects.
    
    Takes a 5-digit project number and generates a complete project profile including
    documents, metadata, analysis, and actionable insights.
    """
    
    def __init__(self, project_id: str, training_data_dir: str = "training_data"):
        self.project_id = project_id
        self.training_data_dir = training_data_dir
        
        # Project structure patterns
        self.folder_categories = {
            "sales_handover": ["1. Sales Handover", "Sales Handover", "Handover"],
            "bom_orders": ["2. BOM & Orders", "BOM", "Orders", "Purchase", "Procurement"],
            "project_management": ["3. PMO", "PMO", "Project Management", "PM"],
            "system_designs": ["4. System Designs", "System Designs", "Designs", "Engineering"],
            "customer_handover": ["5. Customer Handover Documents", "Customer Handover", "Handover Documents"],
            "avcare_handover": ["6. AVCare Handover", "AVCare", "Support", "Maintenance"]
        }
        
        # Document type classifications
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
            "commissioning": r"(?i)(commission|testing|verification)",
            "meeting_minutes": r"(?i)(minutes|meeting|notes)",
            "specification": r"(?i)(spec|specification|requirement)",
            "drawing": r"(?i)(drawing|dwg|cad|autocad)",
            "cable_schedule": r"(?i)(cable|schedule|wiring)",
            "equipment_list": r"(?i)(equipment|list|schedule|bom)"
        }
        
        # Quality indicators
        self.quality_patterns = {
            "high_value": [
                r"(?i)(final|as\s*built|approved)",
                r"(?i)(rev\s*\d+|revision\s*\d+)",
                r"(?i)(system\s*design|schematic)",
                r"(?i)(handover|delivery)"
            ],
            "medium_value": [
                r"(?i)(draft|preliminary)",
                r"(?i)(meeting|minutes)",
                r"(?i)(specification|requirement)"
            ],
            "low_value": [
                r"(?i)(email|correspondence)",
                r"(?i)(invoice|bill)",
                r"(?i)(template|example)"
            ]
        }
        
        # Client mappings
        self.client_mappings = {
            "DWNR": "Downer",
            "DOE": "Department of Education",
            "QLDT": "Queensland Department of Transport",
            "QLD": "Queensland Government",
            "NSW": "New South Wales Government",
            "VIC": "Victoria Government",
            "WA": "Western Australia Government",
            "SA": "South Australia Government",
            "TAS": "Tasmania Government",
            "NT": "Northern Territory Government",
            "ACT": "Australian Capital Territory Government"
        }
        
        # AV technology keywords for scope analysis
        self.av_keywords = {
            "audio": ["speakers", "microphones", "amplifier", "mixer", "audio", "sound", "acoustic"],
            "video": ["display", "projector", "screen", "monitor", "video", "camera", "recording"],
            "control": ["crestron", "amx", "extron", "control", "automation", "processor"],
            "connectivity": ["hdmi", "hdbaset", "fiber", "network", "ip", "dante", "aes67"],
            "conferencing": ["conference", "meeting", "zoom", "teams", "video conference", "vc"],
            "digital_signage": ["signage", "digital display", "wayfinding", "information display"]
        }

    def find_training_data_file(self) -> Optional[str]:
        """Find the training data file for this project."""
        filename = f"training_data_{self.project_id}.json"
        filepath = os.path.join(self.training_data_dir, filename)
        
        if os.path.exists(filepath):
            return filepath
        
        print(f"⚠️  Training data file not found: {filepath}")
        return None

    def load_project_data(self) -> List[Dict]:
        """Load project data from training data file."""
        filepath = self.find_training_data_file()
        if not filepath:
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data)} documents for project {self.project_id}")
                return data
        except Exception as e:
            print(f"❌ Error loading project data: {e}")
            return []

    def extract_project_metadata(self, project_folder: str) -> Dict[str, str]:
        """Extract comprehensive project metadata from folder name."""
        # Pattern: "20692 - DWNR - Project B1012"
        pattern = r"(\d{5})\s*-\s*([^-]+?)\s*-\s*(.+)"
        match = re.match(pattern, project_folder.strip())
        
        if match:
            project_id = match.group(1)
            client_abbrev = match.group(2).strip()
            project_name = match.group(3).strip()
            
            # Expand client abbreviation
            client_full = self.client_mappings.get(client_abbrev, client_abbrev)
            
            return {
                "project_id": project_id,
                "client_abbreviation": client_abbrev,
                "client_name": client_full,
                "project_name": project_name,
                "full_project_name": f"{client_full} - {project_name}",
                "folder_structure": project_folder
            }
        
        # Fallback
        return {
            "project_id": self.project_id,
            "client_abbreviation": "Unknown",
            "client_name": "Unknown Client",
            "project_name": project_folder,
            "full_project_name": project_folder,
            "folder_structure": project_folder
        }

    def classify_document(self, doc_path: str, doc_name: str) -> Dict[str, Any]:
        """Classify a document and extract metadata."""
        path_lower = doc_path.lower()
        name_lower = doc_name.lower()
        combined = f"{doc_path} {doc_name}".lower()
        
        # Determine category based on folder structure
        category = "uncategorized"
        for cat, patterns in self.folder_categories.items():
            for pattern in patterns:
                if pattern.lower() in path_lower:
                    category = cat
                    break
            if category != "uncategorized":
                break
        
        # Determine document type
        doc_type = "other"
        for dtype, pattern in self.document_types.items():
            if re.search(pattern, combined):
                doc_type = dtype
                break
        
        # Extract revision number
        revision = None
        rev_patterns = [r"rev\s*(\d+)", r"revision\s*(\d+)", r"r(\d+)", r"v(\d+)"]
        for pattern in rev_patterns:
            match = re.search(pattern, name_lower)
            if match:
                revision = int(match.group(1))
                break
        
        # Assess quality/importance
        quality_score = self.assess_document_quality(doc_path, doc_name)
        
        return {
            "document_name": doc_name,
            "document_path": doc_path,
            "category": category,
            "document_type": doc_type,
            "revision": revision,
            "quality_score": quality_score,
            "file_extension": Path(doc_name).suffix.lower()
        }

    def assess_document_quality(self, doc_path: str, doc_name: str) -> Dict[str, Any]:
        """Assess the quality and importance of a document."""
        combined = f"{doc_path} {doc_name}".lower()
        
        # Check quality patterns
        quality_level = "medium"
        indicators = []
        
        for level, patterns in self.quality_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined):
                    quality_level = level
                    indicators.append(pattern)
        
        # Check for exclusion patterns (low quality indicators)
        exclude_patterns = [
            r"(?i)/old/", r"(?i)/archive/", r"(?i)template", 
            r"(?i)\.tmp$", r"(?i)~\$", r"(?i)backup"
        ]
        
        is_excluded = any(re.search(pattern, combined) for pattern in exclude_patterns)
        
        return {
            "quality_level": quality_level,
            "indicators": indicators,
            "is_excluded": is_excluded,
            "importance_score": 0.9 if quality_level == "high" else 0.6 if quality_level == "medium" else 0.3
        }

    def analyze_project_scope(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze project scope based on documents and content."""
        scope_analysis = {
            "technology_areas": defaultdict(int),
            "project_phases": [],
            "complexity_indicators": {},
            "estimated_scope": "Unknown"
        }
        
        # Analyze technology areas
        all_text = " ".join([doc.get("document_name", "") + " " + doc.get("document_path", "") for doc in documents]).lower()
        
        for tech_area, keywords in self.av_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in all_text)
            if matches > 0:
                scope_analysis["technology_areas"][tech_area] = matches
        
        # Determine project phases
        categories_present = set(doc.get("category", "") for doc in documents)
        
        if "sales_handover" in categories_present:
            scope_analysis["project_phases"].append("Sales/Planning")
        if "system_designs" in categories_present:
            scope_analysis["project_phases"].append("Design/Engineering")
        if "bom_orders" in categories_present:
            scope_analysis["project_phases"].append("Procurement")
        if "customer_handover" in categories_present:
            scope_analysis["project_phases"].append("Delivery/Completion")
        if "avcare_handover" in categories_present:
            scope_analysis["project_phases"].append("Support/Maintenance")
        
        # Complexity indicators
        total_docs = len(documents)
        unique_categories = len(categories_present)
        unique_doc_types = len(set(doc.get("document_type", "") for doc in documents))
        
        scope_analysis["complexity_indicators"] = {
            "total_documents": total_docs,
            "document_categories": unique_categories,
            "document_types": unique_doc_types,
            "revision_control": len([doc for doc in documents if doc.get("revision")]),
            "technical_documents": len([doc for doc in documents if doc.get("document_type") in ["system_design", "as_built", "specification"]])
        }
        
        # Estimate overall scope
        if total_docs > 100 and unique_categories >= 5:
            scope_analysis["estimated_scope"] = "Large/Complex"
        elif total_docs > 30 and unique_categories >= 3:
            scope_analysis["estimated_scope"] = "Medium"
        elif total_docs > 10:
            scope_analysis["estimated_scope"] = "Small/Standard"
        else:
            scope_analysis["estimated_scope"] = "Minimal/Planning"
        
        return scope_analysis

    def generate_document_timeline(self, documents: List[Dict]) -> List[Dict]:
        """Generate a timeline of document creation/revision."""
        timeline = []
        
        # Group documents by revision and category
        revision_docs = [doc for doc in documents if doc.get("revision")]
        revision_docs.sort(key=lambda x: x.get("revision", 0))
        
        for doc in revision_docs:
            timeline.append({
                "document": doc["document_name"],
                "revision": doc["revision"],
                "category": doc["category"],
                "document_type": doc["document_type"],
                "estimated_phase": self.estimate_document_phase(doc)
            })
        
        return timeline

    def estimate_document_phase(self, document: Dict) -> str:
        """Estimate project phase based on document characteristics."""
        doc_type = document.get("document_type", "")
        category = document.get("category", "")
        revision = document.get("revision", 0)
        
        if category == "sales_handover":
            return "Project Initiation"
        elif doc_type in ["quote", "proposal"] and revision == 0:
            return "Pre-Sales"
        elif doc_type in ["specification", "floor_plan"] and revision <= 101:
            return "Design Phase"
        elif doc_type in ["system_design", "drawing"] and revision >= 100:
            return "Engineering Phase"
        elif doc_type == "as_built" or revision >= 102:
            return "Implementation/Completion"
        elif category == "customer_handover":
            return "Project Handover"
        else:
            return "Unknown Phase"

    def identify_key_deliverables(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Identify key project deliverables and their status."""
        deliverables = {
            "design_deliverables": [],
            "technical_deliverables": [],
            "handover_deliverables": [],
            "support_deliverables": []
        }
        
        for doc in documents:
            doc_type = doc.get("document_type", "")
            quality = doc.get("quality_score", {}).get("quality_level", "medium")
            
            if doc_type in ["system_design", "floor_plan", "specification"] and quality == "high":
                deliverables["design_deliverables"].append({
                    "name": doc["document_name"],
                    "type": doc_type,
                    "revision": doc.get("revision"),
                    "status": "Complete" if doc.get("revision", 0) >= 100 else "In Progress"
                })
            
            elif doc_type in ["as_built", "drawing", "cable_schedule"] and quality == "high":
                deliverables["technical_deliverables"].append({
                    "name": doc["document_name"],
                    "type": doc_type,
                    "revision": doc.get("revision"),
                    "status": "Complete" if doc_type == "as_built" else "In Progress"
                })
            
            elif doc.get("category") == "customer_handover" and quality == "high":
                deliverables["handover_deliverables"].append({
                    "name": doc["document_name"],
                    "type": doc_type,
                    "status": "Complete"
                })
            
            elif doc.get("category") == "avcare_handover":
                deliverables["support_deliverables"].append({
                    "name": doc["document_name"],
                    "type": doc_type,
                    "status": "Complete"
                })
        
        return deliverables

    def assess_project_health(self, documents: List[Dict], scope_analysis: Dict) -> Dict[str, Any]:
        """Assess overall project health and completeness."""
        health_metrics = {
            "completeness_score": 0,
            "documentation_quality": "Unknown",
            "risk_indicators": [],
            "recommendations": []
        }
        
        # Calculate completeness score
        categories_present = set(doc.get("category") for doc in documents if doc.get("category") != "uncategorized")
        expected_categories = set(self.folder_categories.keys())
        completeness = len(categories_present) / len(expected_categories)
        health_metrics["completeness_score"] = round(completeness * 100)
        
        # Assess documentation quality
        high_quality_docs = len([doc for doc in documents if doc.get("quality_score", {}).get("quality_level") == "high"])
        total_docs = len(documents)
        
        if high_quality_docs / total_docs > 0.5:
            health_metrics["documentation_quality"] = "High"
        elif high_quality_docs / total_docs > 0.3:
            health_metrics["documentation_quality"] = "Medium"
        else:
            health_metrics["documentation_quality"] = "Low"
        
        # Identify risk indicators
        if completeness < 0.5:
            health_metrics["risk_indicators"].append("Incomplete documentation categories")
        
        if not any(doc.get("document_type") == "as_built" for doc in documents):
            health_metrics["risk_indicators"].append("Missing as-built documentation")
        
        if "customer_handover" not in categories_present:
            health_metrics["risk_indicators"].append("No customer handover documentation")
        
        # Generate recommendations
        if completeness < 0.8:
            health_metrics["recommendations"].append("Complete missing documentation categories")
        
        if health_metrics["documentation_quality"] == "Low":
            health_metrics["recommendations"].append("Improve document quality and organization")
        
        if scope_analysis.get("estimated_scope") == "Large/Complex" and completeness < 0.7:
            health_metrics["recommendations"].append("Large project requires comprehensive documentation")
        
        return health_metrics

    def extract_project_profile(self) -> Dict[str, Any]:
        """Extract comprehensive project profile."""
        print(f"🔍 Extracting project profile for {self.project_id}...")
        start_time = time.time()
        
        # Load project data
        raw_documents = self.load_project_data()
        if not raw_documents:
            return {"error": f"No data found for project {self.project_id}"}
        
        # Get project metadata from first document
        project_folder = raw_documents[0].get("project_folder", "")
        project_metadata = self.extract_project_metadata(project_folder)
        
        # Process all documents
        print("📄 Processing documents...")
        processed_documents = []
        for doc in raw_documents:
            classified_doc = self.classify_document(doc.get("document_path", ""), doc.get("document_name", ""))
            processed_documents.append(classified_doc)
        
        # Filter out excluded documents
        filtered_documents = [doc for doc in processed_documents if not doc.get("quality_score", {}).get("is_excluded", False)]
        
        print(f"✅ Processed {len(filtered_documents)} documents (filtered from {len(processed_documents)})")
        
        # Analyze project scope
        print("📊 Analyzing project scope...")
        scope_analysis = self.analyze_project_scope(filtered_documents)
        
        # Generate timeline
        print("📅 Generating document timeline...")
        timeline = self.generate_document_timeline(filtered_documents)
        
        # Identify deliverables
        print("📋 Identifying key deliverables...")
        deliverables = self.identify_key_deliverables(filtered_documents)
        
        # Assess project health
        print("🏥 Assessing project health...")
        health_assessment = self.assess_project_health(filtered_documents, scope_analysis)
        
        # Compile comprehensive profile
        project_profile = {
            "extraction_metadata": {
                "project_id": self.project_id,
                "extraction_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "total_documents_found": len(raw_documents),
                "documents_processed": len(filtered_documents)
            },
            "project_metadata": project_metadata,
            "scope_analysis": scope_analysis,
            "document_summary": {
                "categories": dict(Counter(doc.get("category") for doc in filtered_documents)),
                "document_types": dict(Counter(doc.get("document_type") for doc in filtered_documents)),
                "quality_distribution": dict(Counter(doc.get("quality_score", {}).get("quality_level") for doc in filtered_documents)),
                "file_types": dict(Counter(doc.get("file_extension") for doc in filtered_documents))
            },
            "project_timeline": timeline,
            "key_deliverables": deliverables,
            "health_assessment": health_assessment,
            "detailed_documents": filtered_documents
        }
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Project profile extraction complete in {elapsed_time:.1f} seconds!")
        
        return project_profile

    def save_project_profile(self, profile: Dict[str, Any], output_file: Optional[str] = None):
        """Save project profile to JSON file."""
        if not output_file:
            output_file = f"project_profile_{self.project_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Project profile saved to {output_file}")

    def generate_summary_report(self, profile: Dict[str, Any]) -> str:
        """Generate human-readable summary report."""
        metadata = profile.get("project_metadata", {})
        scope = profile.get("scope_analysis", {})
        health = profile.get("health_assessment", {})
        doc_summary = profile.get("document_summary", {})
        
        report = f"""
📁 PROJECT PROFILE REPORT
Project: {metadata.get('full_project_name', 'Unknown')}
Project ID: {metadata.get('project_id', 'Unknown')}
Client: {metadata.get('client_name', 'Unknown')}

📊 PROJECT SCOPE:
• Estimated Scope: {scope.get('estimated_scope', 'Unknown')}
• Project Phases: {', '.join(scope.get('project_phases', []))}
• Technology Areas: {', '.join(scope.get('technology_areas', {}).keys())}

📄 DOCUMENTATION SUMMARY:
• Total Documents: {profile.get('extraction_metadata', {}).get('documents_processed', 0)}
• Categories: {len(doc_summary.get('categories', {}))}
• Document Types: {len(doc_summary.get('document_types', {}))}

🏥 PROJECT HEALTH:
• Completeness Score: {health.get('completeness_score', 0)}%
• Documentation Quality: {health.get('documentation_quality', 'Unknown')}
• Risk Indicators: {len(health.get('risk_indicators', []))}

🎯 KEY DELIVERABLES:
• Design Deliverables: {len(profile.get('key_deliverables', {}).get('design_deliverables', []))}
• Technical Deliverables: {len(profile.get('key_deliverables', {}).get('technical_deliverables', []))}
• Handover Deliverables: {len(profile.get('key_deliverables', {}).get('handover_deliverables', []))}

⚠️  RECOMMENDATIONS:
"""
        
        for rec in health.get('recommendations', []):
            report += f"• {rec}\n"
        
        return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract comprehensive project profile for AV projects")
    parser.add_argument("project_id", help="5-digit project number")
    parser.add_argument("--training-dir", default="training_data", help="Training data directory")
    parser.add_argument("--output", help="Output file for project profile")
    parser.add_argument("--summary", action="store_true", help="Generate summary report")
    
    args = parser.parse_args()
    
    # Validate project ID
    if not re.match(r"^\d{5}$", args.project_id):
        print("❌ Project ID must be a 5-digit number")
        return
    
    # Initialize extractor
    extractor = ProjectExtractor(args.project_id, args.training_dir)
    
    # Extract project profile
    profile = extractor.extract_project_profile()
    
    if "error" in profile:
        print(f"❌ {profile['error']}")
        return
    
    # Save profile
    extractor.save_project_profile(profile, args.output)
    
    # Generate summary report if requested
    if args.summary:
        report = extractor.generate_summary_report(profile)
        report_file = f"project_summary_{args.project_id}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 Summary report saved to {report_file}")
        print(report)
    
    # Print key metrics
    metadata = profile.get("extraction_metadata", {})
    health = profile.get("health_assessment", {})
    scope = profile.get("scope_analysis", {})
    
    print(f"\n📈 Project {args.project_id} Summary:")
    print(f"   📄 Documents processed: {metadata.get('documents_processed', 0)}")
    print(f"   🏥 Health score: {health.get('completeness_score', 0)}%")
    print(f"   📊 Estimated scope: {scope.get('estimated_scope', 'Unknown')}")
    print(f"   ⏱️  Processing time: {metadata.get('processing_time_seconds', 0)}s")


if __name__ == "__main__":
    main() 