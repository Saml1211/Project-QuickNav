# Project QuickNav – Idea Logs Note-Taking System

This subfolder provides an AI-powered, structured note-taking system to efficiently capture, organize, and review spontaneous insights and ideas related to Project QuickNav.

## Features

- **CLI Utility:** Quickly append timestamped notes from the command line, with optional tags.
- **Dual Format Storage:** Notes are saved in both JSON (for structured processing) and Markdown (for human-readable review).
- **Automatic Contextual Organization:** Notes are grouped by date and can be filtered by tags. Keyword extraction (simple or AI-powered) tags each note for better searchability.
- **No Dependency (Basic Mode):** The default extractor requires no additional Python packages.
- **AI-Powered Extraction (Optional):** Enable true AI keyword extraction with OpenAI and a single extra dependency (`requests`).

## Usage

From this directory (or with full path):

```bash
python note_append.py "Your insightful idea goes here." --tags quicknav,ai,workflow
```

- `note` _(required)_: The note or idea text to log.
- `--tags` _(optional)_: Comma-separated tags for topic/context classification.
- `--ai` _(optional)_: Use OpenAI-powered keyword extraction (see below).

**Example:**
```bash
python note_append.py "Implemented a new navigation shortcuts feature for the CLI." --tags feature,cli
```

After running, your note will be:
- Appended to `idea_log.json` (grouped by date for programmatic access).
- Appended to `idea_log.md` (grouped by date for easy human review).

## AI-powered Keyword Extraction (Optional)

To use advanced AI-powered keyword extraction via OpenAI (recommended for maximum relevance):

1. **Install Dependencies**
   - Required: [`requests`](https://pypi.org/project/requests/)
   - Optional (for `.env` support): [`python-dotenv`](https://pypi.org/project/python-dotenv/)
<br/>

## Environment Variables & .env Configuration

You can control all API keys, LLM endpoints, and model names via environment variables or a `.env` file in this directory.
- **.env file:** Simply copy `.env.example` to `.env`, edit the values, and they will be loaded automatically at script start (if you have `python-dotenv` installed).
- **Precedence order:**  
  1. CLI arguments  
  2. Environment variables (including those loaded from `.env`)  
  3. Defaults (used only if not set elsewhere)  
- **Best practices:**  
  - Do **not** commit your real `.env` to version control.
  - Use `.env.example` as a template for sharing configuration requirements.

### Supported Variables

- `OPENAI_API_KEY`  (required for `--ai` mode): Your OpenAI API key.
- `LLM_ENDPOINT`    (for `--llm` mode): REST API endpoint for your local LLM (e.g., Ollama, LM Studio).  
  Default: `http://localhost:11434/api/generate`
- `LLM_MODEL`       (for `--llm` mode): Model name for your local LLM.  
  Default: `llama3`

**Example `.env`:**
```
OPENAI_API_KEY=sk-your-openai-key
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL=llama3
```

If a required value (such as `OPENAI_API_KEY` for `--ai` or `LLM_ENDPOINT`/`LLM_MODEL` for `--llm`) is missing, a clear error message will be printed with instructions.

See `.env.example` in this folder for a full template.

   ```bash
   pip install requests python-dotenv
   ```

2. **Configure API Key**
   - Set the environment variable `OPENAI_API_KEY` before running the script, _or_
   - Create a `.env` file in this folder containing:
     ```
     OPENAI_API_KEY=sk-...
     ```

3. **Run with `--ai` flag**

   ```bash
   python note_append.py "A new generative AI note for QuickNav." --tags ai,log --ai
   ```

   - If the key is missing or there is an API error, the script will fall back to the built-in keyword extractor and notify you.
   - CLI output will indicate when AI-powered extraction is used or if a fallback occurs.

## Local LLM Keyword Extraction (Optional)

You can extract keywords using local Large Language Models (LLMs) such as [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/) running on your machine, without using OpenAI or internet connectivity.

**Requirements:**  
- `requests` Python package (see install above)
- A supported local LLM server (Ollama or LM Studio)

**Usage Examples:**

```bash
# Use Ollama at default endpoint (http://localhost:11434/api/generate) with 'llama3'
python note_append.py "A locally processed note for QuickNav." --tags local,llm --llm

# Specify a custom local endpoint and model (e.g., LM Studio or custom Ollama model)
python note_append.py "Try another local model." --llm --llm-endpoint http://localhost:1234/api/generate --llm-model mistral
```

**Options:**
- `--llm`: Enables local LLM keyword extraction (takes priority if set).
- `--llm-endpoint`: Specify the base URL for your running local LLM API (default: Ollama's `http://localhost:11434/api/generate`).
- `--llm-model`: Model name to use (default: `llama3` for Ollama).

**Fallback order:**  
If local LLM extraction fails, the script will automatically:
1. Fall back to OpenAI-powered extraction if `--ai` and an API key are present.
2. Otherwise, revert to the classic built-in extractor.

**CLI output** will always indicate which extractor was used and if a fallback occurred.
## File Structure

- `idea_log.json`: Structured daily log of notes with timestamps, tags, and AI-extracted keywords.
- `idea_log.md`: Markdown file grouping all notes by date, readable and searchable.
- `note_append.py`: Command-line script for quick note-logging (see code docstring for details).

## Advanced Use

- The script includes a simple keyword extraction algorithm for basic tagging. With the `--ai` flag, it integrates OpenAI's API for true AI-driven keywords (see above), falling back to the basic algorithm on error or missing key.

## Integration

To use this system as part of broader memory-bank/QuickNav workflows, simply call the script and read from the log files as needed.

---
Designed for fast, organized, and intelligent capture of fleeting inspiration.