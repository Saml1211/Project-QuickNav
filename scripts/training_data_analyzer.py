import os
import json
import requests
from typing import Dict, List, Any
import re

class TrainingDataAnalyzer:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def discover_training_files(self, directory: str = ".") -> List[str]:
        """Discover all training_data_*.json files with 5-digit project numbers."""
        pattern = r"training_data_\d{5}\.json$"
        files = []
        for filename in os.listdir(directory):
            if re.match(pattern, filename):
                files.append(os.path.join(directory, filename))
        return sorted(files)
    
    def load_file_details(self, files: List[str]) -> Dict[str, Any]:
        """Load and summarize details of each training data file."""
        files_details = {}
        
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                files_details[filename] = {
                    'file_path': file_path,
                    'num_records': len(data),
                    'full_data': data,
                    'project_number': filename.split('_')[2].split('.')[0]
                }
                print(f"✓ Loaded {filename}: {len(data)} records")
                
            except Exception as e:
                files_details[filename] = {
                    'file_path': file_path,
                    'error': str(e),
                    'project_number': filename.split('_')[2].split('.')[0]
                }
                print(f"✗ Error loading {filename}: {e}")
                
        return files_details
    
    def call_openrouter_ai(self, system_prompt: str, user_prompt: str, model: str, max_tokens: int = 4000) -> str:
        """Make API call to OpenRouter."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error: {e}")
            return f"Error: {str(e)}"
    
    def first_ai_analysis(self, files_details: Dict[str, Any]) -> Dict[str, Any]:
        """First AI: Large context analysis to filter unnecessary data."""
        
        system_prompt = """You are an expert data analyst specializing in AV (Audio-Visual) project documentation. 
        Your task is to analyze training data from multiple AV system design projects and identify what information is essential vs. unnecessary.
        
        Focus on:
        1. Documents that contain actual scope of work and system requirements
        2. Records with meaningful, actionable information
        3. Filtering out empty, duplicate, or low-quality entries
        4. Identifying patterns in document structure and content quality
        
        Return your analysis as JSON with:
        - filtered_data: Only essential records that should be passed to the next stage
        - removal_summary: Brief explanation of what was filtered out and why
        - quality_insights: Observations about data quality and patterns"""
        
        user_prompt = f"""Analyze this AV project training data and filter out unnecessary information:

{json.dumps(files_details, indent=2)}

Return only the essential data that would be valuable for training an AI agent to extract scope of work and system requirements from AV project documents."""
        
        print("🤖 Stage 1: Analyzing data with large context model...")
        
        # Using a large context model for comprehensive analysis
        response = self.call_openrouter_ai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="anthropic/claude-3.5-sonnet",  # Large context, good for analysis
            max_tokens=8000
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("⚠️  First AI response was not valid JSON, returning raw response")
            return {"raw_response": response}
    
    def second_ai_optimization(self, filtered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Second AI: Reasoning model for optimal output generation."""
        
        system_prompt = """You are an expert AI architect designing training strategies for document extraction systems.
        
        Your task is to take filtered AV project data and produce the OPTIMAL training dataset structure that will:
        1. Maximize extraction accuracy for scope of work and system requirements
        2. Provide clear examples for few-shot learning
        3. Create the most effective training patterns
        4. Generate actionable insights for improving the AI agent
        
        Output a structured response with:
        - optimized_training_examples: Best examples formatted for training
        - extraction_patterns: Common patterns found in successful extractions  
        - recommendations: Specific suggestions for improving the AI agent
        - implementation_strategy: How to use this data effectively"""
        
        user_prompt = f"""Based on this filtered AV project data, create the optimal training dataset and strategy:

{json.dumps(filtered_data, indent=2)}

Focus on creating the most effective training approach for an AI agent that needs to extract scope of work and system requirements from AV project documents."""
        
        print("🧠 Stage 2: Optimizing with reasoning model...")
        
        # Using a reasoning model for optimization
        response = self.call_openrouter_ai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="deepseek/deepseek-r1",  # Reasoning model for optimization
            max_tokens=6000
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("⚠️  Second AI response was not valid JSON, returning raw response")
            return {"raw_response": response}
    
    def run_analysis(self, directory: str = ".") -> Dict[str, Any]:
        """Run the complete two-stage analysis."""
        print("🔍 Starting AV Project Training Data Analysis...")
        
        # Step 1: Discover and load files
        print("\n📁 Step 1: Discovering training data files...")
        files = self.discover_training_files(directory)
        
        if not files:
            print("❌ No training data files found!")
            return {"error": "No training_data_*.json files found"}
        
        print(f"Found {len(files)} training data files")
        
        # Step 2: Load file details
        print("\n📊 Step 2: Loading file details...")
        files_details = self.load_file_details(files)
        
        # Step 3: First AI analysis
        print("\n🤖 Step 3: First AI Analysis (Filtering)...")
        filtered_results = self.first_ai_analysis(files_details)
        
        # Step 4: Second AI optimization
        print("\n🧠 Step 4: Second AI Analysis (Optimization)...")
        optimal_results = self.second_ai_optimization(filtered_results)
        
        # Combine results
        final_output = {
            "analysis_summary": {
                "total_files": len(files),
                "files_processed": len([f for f in files_details.values() if 'error' not in f]),
                "files_with_errors": len([f for f in files_details.values() if 'error' in f])
            },
            "stage_1_filtering": filtered_results,
            "stage_2_optimization": optimal_results,
            "timestamp": json.dumps(files_details, default=str)
        }
        
        # Save results
        output_file = "ai_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
        return final_output

def main():
    """Main execution function."""
    # You need to set your OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY='your_api_key_here'")
        return
    
    # Initialize analyzer
    analyzer = TrainingDataAnalyzer(api_key)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Print summary
    if "error" not in results:
        print("\n📈 Analysis Summary:")
        summary = results["analysis_summary"]
        print(f"   - Files processed: {summary['files_processed']}/{summary['total_files']}")
        if summary['files_with_errors'] > 0:
            print(f"   - Files with errors: {summary['files_with_errors']}")
        print("   - Stage 1: Data filtering completed")
        print("   - Stage 2: Optimization completed")
        print("\n🎯 Check 'ai_analysis_results.json' for detailed results")

if __name__ == "__main__":
    main()
