import os
import json
import requests
from typing import Dict, List, Any
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        """Make API call to OpenRouter with retries."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        retries = 3
        delay = 5  # seconds
        for attempt in range(retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except requests.exceptions.Timeout:
                print(f"API Error (attempt {attempt + 1}/{retries}): Request timed out. Retrying in {delay}s...")
            except Exception as e:
                print(f"API Error (attempt {attempt + 1}/{retries}): {e}. Retrying in {delay}s...")
            
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        
        print(f"API Error: Failed after {retries} attempts.")
        return f"Error: Failed after {retries} attempts."
    
    def extract_json_from_response(self, response_text: str) -> str:
        """Extracts JSON content from a string, handling markdown code blocks."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if match:
            return match.group(1).strip()
        
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return response_text[start:end]
        except ValueError:
            pass
            
        return response_text
    
    def first_ai_analysis(self, files_details: Dict[str, Any]) -> Dict[str, Any]:
        """First AI: Large context analysis to filter unnecessary data."""
        
        system_prompt = """You are an expert data analyst specializing in AV (Audio-Visual) project documentation. 
        Your task is to analyze training data from multiple AV system design projects and identify what information is essential vs. unnecessary.
        
        Focus on:
        1. Documents that contain actual scope of work and system requirements.
        2. Records with meaningful, actionable information.
        3. Filtering out empty, duplicate, or low-quality entries.
        4. Identifying patterns in document structure and content quality.

        Return your analysis as a single JSON object with the following keys:
        - "filtered_data": A JSON array of record objects. Each object should be a complete record from the input data that you deem essential. Do not return just filenames.
        - "removal_summary": A brief string explaining what was filtered out and why.
        - "quality_insights": A string containing observations about data quality and patterns."""
        
        user_prompt = f"""Analyze this AV project training data and filter out unnecessary information:

{json.dumps(files_details, indent=2)}

Return ONLY a single, valid JSON object with the keys "filtered_data", "removal_summary", and "quality_insights". Do not include any other text or explanations."""
        
        response = self.call_openrouter_ai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="google/gemini-flash-1.5",
            max_tokens=4096
        )
        
        try:
            json_str = self.extract_json_from_response(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("⚠️  First AI response was not valid JSON, returning raw response")
            with open("error_log.txt", "a", encoding='utf-8') as f:
                f.write("--- STAGE 1 ERROR ---\n")
                f.write(response + "\n\n")
            return {"raw_response": response}
    
    def second_ai_optimization(self, filtered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Second AI: Reasoning model for optimal output generation."""
        
        system_prompt = """You are a data transformation API. You accept a JSON array of AV project document records and you MUST return a single, valid JSON object.
Your response MUST conform to the schema provided and contain NO other text, explanations, or code.

**Input Format:** An array of objects, each with 'project_folder', 'document_path', 'document_name'.
**Output Format:** A single JSON object with the key 'optimized_training_data', which contains an array of objects matching the specified schema.

**Example:**
**Input Record:**
```json
{
  "project_folder": "20692 - DWNR  - Project B1012",
  "document_path": "C:\\\\Users\\\\SamLyndon\\\\Project Files\\\\20692 - DWNR  - Project B1012\\\\4. System Designs\\\\20692 - DOWNER PROJECT B1012 - Rev 102 - AS BUILT.pdf",
  "document_name": "20692 - DOWNER PROJECT B1012 - Rev 102 - AS BUILT.pdf"
}
```

**Required Output Object for this Record:**
```json
{
  "project_id": "20692",
  "project_name": "DWNR - Project B1012",
  "document_type": "As-Built System Design",
  "document_path": "C:\\\\Users\\\\SamLyndon\\\\Project Files\\\\20692 - DWNR  - Project B1012\\\\4. System Designs\\\\20692 - DOWNER PROJECT B1012 - Rev 102 - AS BUILT.pdf",
  "document_name": "20692 - DOWNER PROJECT B1012 - Rev 102 - AS BUILT.pdf",
  "scope_summary": "Final as-built system design (Rev 102) for the Downer Project B1012."
}
```
"""
        
        user_prompt = f"""Transform the following records into the specified JSON format.

{json.dumps(filtered_data, indent=2)}

Return ONLY the final JSON object."""
        
        print("🧠 Stage 2: Optimizing with reasoning model...")
        
        response = self.call_openrouter_ai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="anthropic/claude-3-haiku",
            max_tokens=4096
        )
        
        try:
            json_str = self.extract_json_from_response(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("⚠️  Second AI response was not valid JSON, returning raw response")
            with open("error_log.txt", "a", encoding='utf-8') as f:
                f.write("--- STAGE 2 ERROR ---\n")
                f.write(response + "\n\n")
            return {"raw_response": response}
    
    def run_analysis(self, directory: str = ".") -> Dict[str, Any]:
        """Run the complete two-stage analysis."""
        print("🔍 Starting AV Project Training Data Analysis...")
        
        print("\n📁 Step 1: Discovering training data files...")
        files = self.discover_training_files(directory)
        
        if not files:
            print("❌ No training data files found!")
            return {"error": "No training_data_*.json files found"}
        
        print(f"Found {len(files)} training data files")
        
        print("\n📊 Step 2: Loading file details...")
        files_details = self.load_file_details(files)
        
        print("\n🤖 Step 3: First AI Analysis (Filtering)...")
        all_filtered_results = {"filtered_data": [], "removal_summary": [], "quality_insights": []}
        
        stage1_chunks_to_process = []
        for filename, details in files_details.items():
            records = details.get('full_data', [])
            if not records:
                print(f"   - Skipping {filename}: No data.")
                continue

            chunk_size = 20
            if details.get('num_records', 0) > chunk_size:
                for i in range(0, details['num_records'], chunk_size):
                    chunk = records[i:i+chunk_size]
                    chunk_details = {
                        filename: {**details, 'full_data': chunk, 'num_records': len(chunk)}
                    }
                    stage1_chunks_to_process.append(chunk_details)
            else:
                stage1_chunks_to_process.append({filename: details})

        with ThreadPoolExecutor(max_workers=5) as executor:
            print(f"   - Processing {len(stage1_chunks_to_process)} chunks in parallel...")
            future_to_chunk = {executor.submit(self.first_ai_analysis, chunk): chunk for chunk in stage1_chunks_to_process}
            for future in as_completed(future_to_chunk):
                filtered_result = future.result()
                if "raw_response" in filtered_result:
                    print(f"   ⚠️  Skipping a chunk due to analysis error.")
                    all_filtered_results["removal_summary"].append(f"Analysis failed for a chunk: {filtered_result['raw_response']}")
                    continue

                if filtered_result.get("filtered_data"):
                    all_filtered_results["filtered_data"].extend(filtered_result["filtered_data"])
                if filtered_result.get("removal_summary"):
                    all_filtered_results["removal_summary"].append(filtered_result["removal_summary"])
                if filtered_result.get("quality_insights"):
                    all_filtered_results["quality_insights"].append(filtered_result["quality_insights"])

        print("\n🧠 Step 4: Second AI Analysis (Optimization)...")
        
        final_optimized_results = {"optimized_training_data": []}
        filtered_records = all_filtered_results.get("filtered_data", [])
        
        stage2_chunks_to_process = []
        if filtered_records:
            optimization_chunk_size = 25
            for i in range(0, len(filtered_records), optimization_chunk_size):
                stage2_chunks_to_process.append(filtered_records[i:i+optimization_chunk_size])
        
        if stage2_chunks_to_process:
            with ThreadPoolExecutor(max_workers=5) as executor:
                print(f"   - Optimizing {len(filtered_records)} filtered records in {len(stage2_chunks_to_process)} parallel chunks...")
                future_to_chunk = {executor.submit(self.second_ai_optimization, chunk): chunk for chunk in stage2_chunks_to_process}
                for future in as_completed(future_to_chunk):
                    optimal_result_chunk = future.result()
                    if "raw_response" in optimal_result_chunk:
                        print(f"   ⚠️  Skipping optimization chunk due to analysis error.")
                        continue
                    
                    if optimal_result_chunk.get("optimized_training_data"):
                        final_optimized_results["optimized_training_data"].extend(optimal_result_chunk["optimized_training_data"])
        else:
            print("   - No data to optimize.")

        final_output = {
            "analysis_summary": {
                "total_files": len(files),
                "files_processed": len([f for f in files_details.values() if 'error' not in f]),
                "files_with_errors": len([f for f in files_details.values() if 'error' in f])
            },
            "stage_1_filtering": all_filtered_results,
            "stage_2_optimization": final_optimized_results,
        }
        
        output_file = "ai_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
        return final_output

def main():
    """Main execution function."""
    import sys
    
    # You need to set your OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY='your_api_key_here'")
        return
    
    # Get directory from command line argument or use current directory
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Initialize analyzer
    analyzer = TrainingDataAnalyzer(api_key)
    
    # Run analysis
    results = analyzer.run_analysis(directory)
    
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
