import os
import json
import requests
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass

@dataclass
class TrainingExample:
    """Structured training example for AI agents."""
    input_context: str
    expected_output: str
    example_type: str
    confidence_score: float
    metadata: Dict[str, Any]

class AIAgentTrainingGenerator:
    """
    Generates training data and actionable insights for AV project AI agents.
    
    Focuses on creating structured training examples, identifying patterns,
    and providing insights to improve AI agent performance on AV projects.
    """
    
    def __init__(self, openrouter_api_key: str, training_data_dir: str = "training_data"):
        self.api_key = openrouter_api_key
        self.training_data_dir = training_data_dir
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Training example categories for AI agents
        self.training_categories = {
            "project_identification": "Given project folder names, extract project metadata",
            "document_classification": "Given document paths/names, classify document types",
            "scope_assessment": "Given project documents, assess project scope and complexity",
            "client_communication": "Given project context, generate appropriate client communications",
            "phase_identification": "Given document sets, identify current project phase",
            "quality_assessment": "Given document quality indicators, assess completeness",
            "technical_extraction": "Given technical documents, extract key specifications",
            "change_detection": "Given document revisions, identify scope changes"
        }
        
        # Patterns for generating insights
        self.insight_patterns = {
            "common_failures": "What patterns indicate project issues?",
            "success_indicators": "What patterns indicate successful projects?",
            "documentation_gaps": "What types of documentation are commonly missing?",
            "client_patterns": "What are common client-specific patterns?",
            "technical_complexity": "How to assess technical complexity from documents?",
            "timeline_indicators": "What document patterns indicate project timeline status?",
            "quality_metrics": "What metrics indicate documentation quality?",
            "automation_opportunities": "What tasks could be automated based on patterns?"
        }
        
        # Common AV project terminology and patterns
        self.av_terminology = {
            "systems": ["conference room", "digital signage", "audio system", "video wall", 
                       "projection", "displays", "speakers", "microphones", "cameras"],
            "brands": ["Crestron", "AMX", "Extron", "Biamp", "QSC", "Sony", "Panasonic", 
                      "Samsung", "LG", "Christie", "Barco", "Shure", "Sennheiser"],
            "technologies": ["4K", "HDBaseT", "Dante", "AES67", "HDMI", "VGA", "USB-C",
                           "PoE", "IP", "streaming", "recording", "control system"],
            "project_types": ["corporate", "education", "healthcare", "retail", "hospitality",
                            "government", "house of worship", "entertainment"]
        }

    def load_hybrid_analysis_results(self, results_file: str = "hybrid_analysis_results.json") -> Dict[str, Any]:
        """Load results from the hybrid analyzer."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Hybrid analysis results not found at {results_file}")
            print("🔄 Please run hybrid_training_analyzer.py first")
            return {}
        except Exception as e:
            print(f"❌ Error loading hybrid results: {e}")
            return {}

    def call_openrouter_api(self, prompt: str, max_tokens: int = 2000, model: str = "anthropic/claude-3-haiku") -> str:
        """Make API call to OpenRouter for insight generation."""
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error: {e}")
            return f"Error generating insight: {e}"

    def generate_project_identification_examples(self, projects: List[Dict]) -> List[TrainingExample]:
        """Generate training examples for project metadata extraction."""
        examples = []
        
        for project in projects[:10]:  # Limit to prevent too many API calls
            folder_name = f"{project['project_id']} - {project['client_name']} - {project['project_name']}"
            
            example = TrainingExample(
                input_context=f"Extract project metadata from folder name: '{folder_name}'",
                expected_output=json.dumps({
                    "project_id": project['project_id'],
                    "client_name": project['client_name'],
                    "project_name": project['project_name']
                }, indent=2),
                example_type="project_identification",
                confidence_score=0.95,
                metadata={"source": "rule_based_extraction"}
            )
            examples.append(example)
        
        return examples

    def generate_document_classification_examples(self, projects: List[Dict]) -> List[TrainingExample]:
        """Generate training examples for document classification."""
        examples = []
        
        for project in projects[:5]:
            for category, docs in project.get('document_categories', {}).items():
                for doc in docs[:3]:  # Limit examples per category
                    example = TrainingExample(
                        input_context=f"Classify this document: '{doc['document_name']}'",
                        expected_output=json.dumps({
                            "category": category,
                            "document_type": doc['document_type'],
                            "confidence": "high"
                        }, indent=2),
                        example_type="document_classification",
                        confidence_score=0.9,
                        metadata={"project_id": project['project_id'], "category": category}
                    )
                    examples.append(example)
        
        return examples

    def analyze_documentation_patterns(self, projects: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across all projects for insights."""
        patterns = {
            "category_distribution": defaultdict(int),
            "document_type_frequency": defaultdict(int),
            "project_completeness": defaultdict(list),
            "common_gaps": [],
            "quality_indicators": {}
        }
        
        for project in projects:
            categories = project.get('document_categories', {})
            
            # Count category distribution
            for category in categories.keys():
                patterns["category_distribution"][category] += 1
            
            # Count document types
            for category, docs in categories.items():
                for doc in docs:
                    patterns["document_type_frequency"][doc['document_type']] += 1
            
            # Assess completeness
            completeness_score = len(categories) / 6  # Out of 6 standard categories
            patterns["project_completeness"][project['project_id']] = {
                "score": completeness_score,
                "categories_present": list(categories.keys()),
                "total_documents": sum(len(docs) for docs in categories.values())
            }
        
        return patterns

    def generate_ai_training_insights(self, projects: List[Dict], patterns: Dict[str, Any]) -> str:
        """Generate AI-powered insights for agent improvement."""
        
        # Prepare analysis data
        project_summary = {
            "total_projects": len(projects),
            "avg_documents_per_project": sum(
                sum(len(docs) for docs in p.get('document_categories', {}).values()) 
                for p in projects
            ) / len(projects) if projects else 0,
            "most_common_categories": dict(Counter(patterns["category_distribution"]).most_common(5)),
            "most_common_doc_types": dict(Counter(patterns["document_type_frequency"]).most_common(10))
        }
        
        prompt = f"""As an AI training expert for AV project management systems, analyze this data and provide actionable insights for improving AI agents:

PROJECT DATA SUMMARY:
{json.dumps(project_summary, indent=2)}

PATTERN ANALYSIS:
- Category Distribution: {dict(patterns["category_distribution"])}
- Document Type Frequency: {dict(patterns["document_type_frequency"])}

Please provide insights in this format:

TRAINING DATA GAPS:
- [List 3-5 specific areas where AI agents need more training examples]

AGENT IMPROVEMENT OPPORTUNITIES:
- [List 3-5 specific ways to improve AI agent performance]

AUTOMATION POTENTIAL:
- [List 3-5 tasks that could be automated based on patterns]

QUALITY METRICS:
- [Suggest 3-5 metrics to measure AI agent performance]

TRAINING PRIORITIES:
- [Rank top 5 training priorities for AV project AI agents]

Keep responses practical and focused on actionable improvements."""

        return self.call_openrouter_api(prompt, max_tokens=2500, model="anthropic/claude-3-sonnet")

    def generate_scope_assessment_examples(self, projects: List[Dict]) -> List[TrainingExample]:
        """Generate examples for project scope assessment."""
        examples = []
        
        for project in projects[:8]:
            categories = project.get('document_categories', {})
            doc_count = sum(len(docs) for docs in categories.values())
            
            # Determine complexity based on document patterns
            complexity = "low"
            if doc_count > 50:
                complexity = "high"
            elif doc_count > 20:
                complexity = "medium"
            
            scope_indicators = []
            if 'system_designs' in categories:
                scope_indicators.append("technical design phase")
            if 'customer_handover' in categories:
                scope_indicators.append("delivery phase")
            if len(categories) >= 4:
                scope_indicators.append("comprehensive project")
            
            example = TrainingExample(
                input_context=f"Assess project scope for: {project['project_name']}\nDocument categories: {list(categories.keys())}\nTotal documents: {doc_count}",
                expected_output=json.dumps({
                    "complexity": complexity,
                    "scope_indicators": scope_indicators,
                    "estimated_size": f"{doc_count} documents across {len(categories)} categories",
                    "phase": "implementation" if 'system_designs' in categories else "planning"
                }, indent=2),
                example_type="scope_assessment",
                confidence_score=0.85,
                metadata={"project_id": project['project_id'], "doc_count": doc_count}
            )
            examples.append(example)
        
        return examples

    def generate_quality_assessment_examples(self, projects: List[Dict]) -> List[TrainingExample]:
        """Generate examples for documentation quality assessment."""
        examples = []
        
        for project in projects[:6]:
            categories = project.get('document_categories', {})
            
            # Calculate quality indicators
            completeness = len(categories) / 6  # Out of 6 standard categories
            has_designs = 'system_designs' in categories
            has_handover = 'customer_handover' in categories
            doc_diversity = len(set(
                doc['document_type'] 
                for docs in categories.values() 
                for doc in docs
            ))
            
            quality_score = "high" if completeness > 0.8 and has_designs and has_handover else \
                           "medium" if completeness > 0.5 and has_designs else "low"
            
            quality_factors = []
            if completeness > 0.8:
                quality_factors.append("comprehensive documentation")
            if has_designs:
                quality_factors.append("technical documentation present")
            if has_handover:
                quality_factors.append("delivery documentation complete")
            if doc_diversity > 5:
                quality_factors.append("diverse document types")
            
            example = TrainingExample(
                input_context=f"Assess documentation quality for project {project['project_id']}\nCategories: {list(categories.keys())}\nCompleteness: {completeness:.1%}",
                expected_output=json.dumps({
                    "quality_score": quality_score,
                    "completeness_percentage": round(completeness * 100),
                    "quality_factors": quality_factors,
                    "recommendations": ["Add missing categories"] if completeness < 0.8 else ["Maintain current standards"]
                }, indent=2),
                example_type="quality_assessment",
                confidence_score=0.8,
                metadata={"project_id": project['project_id'], "completeness": completeness}
            )
            examples.append(example)
        
        return examples

    def generate_comprehensive_training_dataset(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training dataset for AI agents."""
        if not results or 'projects' not in results:
            return {"error": "Invalid hybrid analysis results"}
        
        projects = results['projects']
        print(f"🎯 Generating AI agent training data from {len(projects)} projects...")
        
        # Generate different types of training examples
        training_examples = []
        
        print("📋 Generating project identification examples...")
        training_examples.extend(self.generate_project_identification_examples(projects))
        
        print("📄 Generating document classification examples...")
        training_examples.extend(self.generate_document_classification_examples(projects))
        
        print("📊 Generating scope assessment examples...")
        training_examples.extend(self.generate_scope_assessment_examples(projects))
        
        print("✅ Generating quality assessment examples...")
        training_examples.extend(self.generate_quality_assessment_examples(projects))
        
        # Analyze patterns
        print("🔍 Analyzing documentation patterns...")
        patterns = self.analyze_documentation_patterns(projects)
        
        # Generate AI insights
        print("🧠 Generating AI training insights...")
        ai_insights = self.generate_ai_training_insights(projects, patterns)
        
        # Compile training dataset
        training_dataset = {
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_projects": len(projects),
                "total_training_examples": len(training_examples),
                "example_types": list(set(ex.example_type for ex in training_examples))
            },
            "training_examples": [
                {
                    "input": ex.input_context,
                    "output": ex.expected_output,
                    "type": ex.example_type,
                    "confidence": ex.confidence_score,
                    "metadata": ex.metadata
                }
                for ex in training_examples
            ],
            "pattern_analysis": patterns,
            "ai_improvement_insights": ai_insights,
            "agent_training_recommendations": self.generate_training_recommendations(patterns, training_examples)
        }
        
        return training_dataset

    def generate_training_recommendations(self, patterns: Dict[str, Any], examples: List[TrainingExample]) -> Dict[str, Any]:
        """Generate specific recommendations for AI agent training."""
        recommendations = {
            "priority_training_areas": [],
            "example_distribution": {},
            "suggested_improvements": [],
            "performance_metrics": []
        }
        
        # Analyze example distribution
        example_types = Counter(ex.example_type for ex in examples)
        recommendations["example_distribution"] = dict(example_types)
        
        # Identify priority areas
        if example_types.get("document_classification", 0) > 20:
            recommendations["priority_training_areas"].append("Document classification has good coverage")
        else:
            recommendations["priority_training_areas"].append("Need more document classification examples")
        
        if example_types.get("scope_assessment", 0) < 10:
            recommendations["priority_training_areas"].append("Scope assessment needs more examples")
        
        # Suggest improvements
        total_categories = len(patterns["category_distribution"])
        if total_categories < 5:
            recommendations["suggested_improvements"].append("Increase document category diversity")
        
        recommendations["suggested_improvements"].extend([
            "Add more complex multi-project scenarios",
            "Include edge cases and error handling examples",
            "Develop client-specific pattern training",
            "Create time-based project progression examples"
        ])
        
        # Performance metrics
        recommendations["performance_metrics"] = [
            "Classification accuracy per document type",
            "Project metadata extraction accuracy",
            "Scope assessment correlation with actual project outcomes",
            "Quality score prediction accuracy",
            "Processing time per document"
        ]
        
        return recommendations

    def save_training_dataset(self, dataset: Dict[str, Any], output_file: str = "ai_agent_training_dataset.json"):
        """Save the training dataset to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"💾 AI agent training dataset saved to {output_file}")

    def generate_training_summary_report(self, dataset: Dict[str, Any]) -> str:
        """Generate a human-readable summary report."""
        metadata = dataset.get("generation_metadata", {})
        examples = dataset.get("training_examples", [])
        recommendations = dataset.get("agent_training_recommendations", {})
        
        report = f"""
🤖 AI AGENT TRAINING DATASET REPORT
Generated: {metadata.get('timestamp', 'Unknown')}

📊 DATASET OVERVIEW:
• Source Projects: {metadata.get('source_projects', 0)}
• Training Examples: {metadata.get('total_training_examples', 0)}
• Example Types: {', '.join(metadata.get('example_types', []))}

📈 EXAMPLE DISTRIBUTION:
"""
        
        for ex_type, count in recommendations.get("example_distribution", {}).items():
            report += f"• {ex_type.replace('_', ' ').title()}: {count} examples\n"
        
        report += f"""
🎯 PRIORITY TRAINING AREAS:
"""
        for priority in recommendations.get("priority_training_areas", []):
            report += f"• {priority}\n"
        
        report += f"""
🔧 SUGGESTED IMPROVEMENTS:
"""
        for improvement in recommendations.get("suggested_improvements", []):
            report += f"• {improvement}\n"
        
        report += f"""
📏 PERFORMANCE METRICS TO TRACK:
"""
        for metric in recommendations.get("performance_metrics", []):
            report += f"• {metric}\n"
        
        return report

    def run_training_generation(self) -> Dict[str, Any]:
        """Run the complete AI agent training data generation process."""
        print("🚀 Starting AI Agent Training Data Generation...")
        start_time = time.time()
        
        # Load hybrid analysis results
        print("\n📁 Loading hybrid analysis results...")
        results = self.load_hybrid_analysis_results()
        
        if not results:
            return {"error": "Could not load hybrid analysis results"}
        
        # Generate training dataset
        print("\n🎓 Generating comprehensive training dataset...")
        dataset = self.generate_comprehensive_training_dataset(results)
        
        if "error" in dataset:
            return dataset
        
        # Save dataset
        print("\n💾 Saving training dataset...")
        self.save_training_dataset(dataset)
        
        # Generate and save summary report
        print("\n📋 Generating summary report...")
        report = self.generate_training_summary_report(dataset)
        
        with open("ai_agent_training_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training data generation complete in {elapsed_time:.1f} seconds!")
        print(f"📊 Generated {len(dataset['training_examples'])} training examples")
        print(f"🎯 Check 'ai_agent_training_dataset.json' and 'ai_agent_training_report.txt'")
        
        return dataset


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
    
    # Initialize generator
    generator = AIAgentTrainingGenerator(api_key, training_dir)
    
    # Run training data generation
    results = generator.run_training_generation()
    
    if "error" not in results:
        print("\n🎉 AI Agent Training Data Generation Successful!")
        print("\nNext steps:")
        print("1. Review 'ai_agent_training_dataset.json' for training examples")
        print("2. Read 'ai_agent_training_report.txt' for actionable insights")
        print("3. Use the training examples to improve your AI agents")
        print("4. Track the suggested performance metrics")
    else:
        print(f"❌ Training generation failed: {results['error']}")


if __name__ == "__main__":
    main() 