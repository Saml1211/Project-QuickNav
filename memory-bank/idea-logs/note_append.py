#!/usr/bin/env python3
"""
Note Append Utility for Project QuickNav - Idea Logs

Usage:
  python note_append.py "Your insightful idea goes here." --tags tag1,tag2

Description:
  Appends a timestamped note (with optional tags) to both a JSON log and a Markdown log.
  Designed for fast, structured, and human-readable knowledge capture in Project QuickNav.

Dependencies:
  - Python 3.7+
  - AI mode: Requires `requests` (for API call) and `python-dotenv` (optional, for local .env dev)
    - Set your OpenAI API key in the environment variable `OPENAI_API_KEY`
    - Or create a `.env` file in this directory containing: OPENAI_API_KEY=sk-...
    - The --ai flag enables AI-powered keyword extraction via OpenAI
    - Falls back to built-in extractor on error or missing key, with CLI notification

Features:
  - Stores log in both structured JSON and human-readable Markdown.
  - Organizes notes with date, tags, and AI-assisted keyword tagging.
  - Each note entry includes timestamp, text, tags, and extracted keywords.
  - Contextual organization by date; notes grouped per day in JSON, sequential in Markdown.
  - Easy CLI usage for fast note capture.
"""

import argparse
import json
from datetime import datetime
import os
import re
import sys
import tempfile
import shutil

# Optional dependencies for AI-powered extraction
import importlib
from typing import Any, Optional, Dict, List, Union

# Handle optional imports
requests: Optional[Any] = None
load_dotenv: Optional[Any] = None

try:
    import requests
except ImportError:
    requests = None  # Will error at runtime if --ai is used without requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Always load .env at script start if present and python-dotenv is available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
if load_dotenv and os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH, override=False)

try:
    import fcntl  # Unix-based locking
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_LOG = os.path.join(BASE_DIR, 'idea_log.json')
MD_LOG = os.path.join(BASE_DIR, 'idea_log.md')

def extract_keywords(note, extra_stopwords=None):
    """
    Extracts unique, lowercase, alphanumeric keywords not in a (customizable) stoplist.
    Optionally accepts extra stopwords.
    For advanced extraction, replace body with a call to AI/LLM service.

    Args:
        note (str): The note text.
        extra_stopwords (set, optional): Additional stopwords to exclude.

    Returns:
        List[str]: Sorted list of extracted keywords.
    """
    # Extended stoplist
    stopwords = set("""
        the a an and or but if in on for with as by to from is are of this that it at we he she you your our their my
        be was were been have has had about after before again against all am any because been being between both
        can did do does doing don down during each few further here how into itself just more most other over own same
        should so some such than too very
    """.split())
    if extra_stopwords:
        stopwords |= set(extra_stopwords)
    tokens = re.findall(r'\b\w+\b', note.lower())
    keywords = sorted(set(t for t in tokens if t not in stopwords and len(t) > 2))
    return keywords

def extract_keywords_local_llm(note, endpoint=None, model=None):
    """
    Keyword extraction using a local LLM (Ollama, LM Studio, etc.) via REST API.

    Args:
        note (str): The note to extract keywords from.
        endpoint (str): The API endpoint for the local LLM.
        model (str): Name of the LLM model.

    Returns:
        List[str]: List of extracted keywords (from local LLM or fallback).
    """
    if not requests:
        print("Local LLM mode: 'requests' library is required for LLM API. Falling back to basic extractor.", file=sys.stderr)
        return extract_keywords(note)

    endpoint = endpoint or "http://localhost:11434/api/generate"
    model = model or "llama3"
    system_prompt = (
        "Extract 5-10 concise, relevant keywords for this note, comma-separated: "
    )
    payload = {
        "model": model,
        "prompt": f"{system_prompt}{note.strip()}",
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            print(f"Local LLM mode: LLM endpoint error ({resp.status_code} {resp.reason}). Falling back.", file=sys.stderr)
            return extract_keywords(note)
        data = resp.json()
        # Ollama: {'model': ..., 'created_at': ..., 'response': 'keyword1, keyword2, ...'}
        # LM Studio: {'choices':[{'text':'keyword1, keyword2, ...'}]}
        if "response" in data:
            raw_keywords = data["response"].strip()
        elif "choices" in data and data["choices"]:
            raw_keywords = data["choices"][0].get("text", "").strip()
        else:
            raw_keywords = ""
        # Parse keywords (robust: comma/semicolon/line split)
        keywords = [
            x.strip().lower() for x in re.split(r"[,;\n]", raw_keywords)
            if x.strip() and len(x.strip()) > 1 and x.strip().isalpha()
        ]
        keywords = sorted(set([k for k in keywords if k not in {
            "the","and","for","with","you","are","but","can","was","has","that","your","our","their","from"
        }]))
        if 2 <= len(keywords) <= 15:
            print(f"Local LLM-powered keyword extraction used via {endpoint} (model: {model}).")
            return keywords
        else:
            print("Local LLM mode: Unexpected response, falling back to next extractor.", file=sys.stderr)
            return extract_keywords(note)
    except Exception as e:
        print(f"Local LLM mode: API error ({e}). Falling back to next extractor.", file=sys.stderr)
        return extract_keywords(note)


def extract_keywords_llm(note):
    """
    AI-powered keyword extraction using OpenAI API.

    - Uses OpenAI Chat API (gpt-3.5/4) to extract 5–10 concise, relevant keywords.
    - Requires OPENAI_API_KEY in environment or .env file.
    - Falls back to built-in extractor on API error or missing key, and notifies user.
    - Only dependencies required: requests, (optional) python-dotenv for .env support.

    Returns:
        List[str]: List of extracted keywords (from AI or fallback).
    Raises:
        RuntimeError if requests is not installed and --ai is used.
    """
    # Try loading dotenv for local dev
    if load_dotenv:
        load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("AI mode: No OPENAI_API_KEY found in environment. Falling back to basic extractor.", file=sys.stderr)
        return extract_keywords(note)
    if not requests:
        print("AI mode: 'requests' library is required for OpenAI API. Falling back to basic extractor.", file=sys.stderr)
        return extract_keywords(note)

    openai_url = "https://api.openai.com/v1/chat/completions"
    system_prompt = (
        "You are an expert assistant for quickly tagging notes. "
        "Given a user note, extract 5-10 concise, relevant, comma-separated keywords (no phrases, no explanations). "
        "Do not include stopwords or duplicates. Return only the keyword list."
    )
    user_prompt = f"Note: {note.strip()}\nExtract 5-10 keywords:"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 60,
        "temperature": 0.2,
        "n": 1
    }
    try:
        resp = requests.post(openai_url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 401:
            print("AI mode: Invalid or missing OpenAI API key. Falling back to basic extractor.", file=sys.stderr)
            return extract_keywords(note)
        resp.raise_for_status()
        raw = resp.json()
        ai_text = raw["choices"][0]["message"]["content"].strip()
        # Accept keywords as comma/semicolon/line separated
        keywords = [
            x.strip().lower() for x in re.split(r"[,;\n]", ai_text)
            if x.strip() and len(x.strip()) > 1 and x.strip().isalpha()
        ]
        # Remove stopwords and deduplicate
        keywords = sorted(set([k for k in keywords if k not in {
            "the","and","for","with","you","are","but","can","was","has","that","your","our","their","from"
        }]))
        if 2 <= len(keywords) <= 15:
            print("AI-powered keyword extraction used.")
            return keywords
        else:
            print("AI mode: Unexpected response from OpenAI, falling back to basic extractor.", file=sys.stderr)
            return extract_keywords(note)
    except Exception as e:
        print(f"AI mode: OpenAI API error ({e}). Falling back to basic extractor.", file=sys.stderr)
        return extract_keywords(note)

def atomic_write(filepath, data, mode="w", encoding="utf-8"):
    """
    Writes data atomically to the given filepath.

    Platform compatibility:
    - Uses a temporary file and moves it into place.
    - On Unix, applies file locking via fcntl if available.
    - On Windows (or if fcntl is unavailable), moves file without locking.
      This is as atomic as possible for normal file use, but if the file is in use (e.g., opened in a text editor with exclusive lock),
      behavior gracefully falls back to shutil.move, which will overwrite the original.
    - All other logic (tempfile, os.path, encoding) is cross-platform and works on macOS and Windows.

    Accepts either a string or a callable for data.
    """
    dir_path = os.path.dirname(filepath)
    fd, temp_path = tempfile.mkstemp(dir=dir_path)
    try:
        with os.fdopen(fd, mode, encoding=encoding) as tmpf:
            if callable(data):
                data(tmpf)
            else:
                tmpf.write(data)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        if HAS_FCNTL:
            try:
                with open(filepath, "a+", encoding=encoding) as lockf:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
                    shutil.move(temp_path, filepath)
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            except Exception:
                print("Warning: Unix file locking failed. Proceeding without lock.", file=sys.stderr)
                shutil.move(temp_path, filepath)
        else:
            # Windows/common fallback: no file locking, move temp file.
            shutil.move(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def append_to_json(note_entry, log_path=JSON_LOG):
    """
    Appends the note_entry (dict) to the appropriate day in the JSON log.
    Creates or updates the JSON log file (atomically).
    """
    today = note_entry['date']
    data = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    if today not in data:
        data[today] = []
    data[today].append(note_entry)
    def dump_json(f):
        json.dump(data, f, indent=2, ensure_ascii=False)
    atomic_write(log_path, dump_json)

def append_to_markdown(note_entry, log_path=MD_LOG):
    """
    Appends the note_entry as a Markdown-formatted section to the log file.
    Organizes notes under a daily header.
    Atomic write for integrity.
    """
    md_date_header = f"## {note_entry['date']}\n"
    md_entry = (
        f"- **Time:** {note_entry['time']}\n"
        f"  - **Tags:** {', '.join(note_entry['tags']) if note_entry['tags'] else 'None'}\n"
        f"  - **AI Keywords:** {', '.join(note_entry['ai_keywords']) if note_entry['ai_keywords'] else 'None'}\n"
        f"  - **Note:** {note_entry['note']}\n\n"
    )
    # Read current content
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""
    # Insert under the correct date header
    if md_date_header not in content:
        content += f"{md_date_header}\n"
    # Place entry after the last occurrence of the date header
    parts = content.split(md_date_header)
    if len(parts) == 2:
        pre, post = parts
        post = md_entry + post
        new_content = pre + md_date_header + post
    else:
        new_content = content + md_entry

    atomic_write(log_path, new_content)

def main():
    parser = argparse.ArgumentParser(
        description="Quickly append a timestamped idea/note to Project QuickNav's idea logs (JSON & Markdown)."
    )
    parser.add_argument("note", type=str, help="The idea or note to log (required, max 2000 chars).")
    parser.add_argument("--tags", type=str, default="", help="Comma-separated list of topic tags (optional, each max 32 chars).")
    parser.add_argument("--ai", action="store_true",
                        help="Use OpenAI-powered keyword extraction if available (default: basic extractor).")
    parser.add_argument("--llm", action="store_true",
                        help="Use a local LLM (e.g. Ollama, LM Studio) for keyword extraction via REST API. If set, tries local LLM first, then OpenAI (--ai), then classic extractor.")
    parser.add_argument("--llm-endpoint", type=str, default=None,
                        help="Local LLM API endpoint URL. Can be set as LLM_ENDPOINT in .env. Default: http://localhost:11434/api/generate (Ollama).")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Local LLM model name. Can be set as LLM_MODEL in .env. Default: llama3 (Ollama).")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key for --ai mode. Can be set as OPENAI_API_KEY in .env.")
    args = parser.parse_args()

    # Unified config resolution: CLI > environment > .env > default
    def get_config(cli_val, env_key, default_val=None):
        if cli_val is not None:
            return cli_val
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val
        return default_val

    note_text = args.note.strip()
    if not note_text:
        print("Error: Note text cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(note_text) > 2000:
        print("Error: Note text too long (max 2000 characters).", file=sys.stderr)
        sys.exit(1)
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    for tag in tags:
        if len(tag) > 32:
            print(f"Error: Tag '{tag}' is too long (max 32 chars).", file=sys.stderr)
            sys.exit(1)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Keyword extraction config
    llm_endpoint = get_config(args.llm_endpoint, "LLM_ENDPOINT", "http://localhost:11434/api/generate")
    llm_model = get_config(args.llm_model, "LLM_MODEL", "llama3")
    openai_api_key = get_config(args.openai_api_key, "OPENAI_API_KEY", None)

    # Patch environment for extractor functions (for compatibility)
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if llm_endpoint:
        os.environ["LLM_ENDPOINT"] = llm_endpoint
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model

    # Keyword extraction strategy
    ai_keywords = []
    extractor_used = "classic"
    if args.llm:
        # If LLM config missing, print message and abort
        if not llm_endpoint or not llm_model:
            print(
                "Error: Local LLM mode requires LLM_ENDPOINT and LLM_MODEL to be set via command-line or .env file.",
                file=sys.stderr,
            )
            print("Tip: See .env.example for configuration help.", file=sys.stderr)
            sys.exit(1)
        ai_keywords = extract_keywords_local_llm(note_text, llm_endpoint, llm_model)
        if ai_keywords and len(ai_keywords) >= 2:
            extractor_used = f"local LLM ({llm_endpoint}, model: {llm_model})"
        elif args.ai:
            if not openai_api_key:
                print(
                    "Error: --ai mode requires OPENAI_API_KEY. Set it via --openai-api-key or in .env.",
                    file=sys.stderr,
                )
                print("Tip: See .env.example for configuration help.", file=sys.stderr)
                sys.exit(1)
            ai_keywords = extract_keywords_llm(note_text)
            if ai_keywords and len(ai_keywords) >= 2:
                extractor_used = "OpenAI API"
            else:
                ai_keywords = extract_keywords(note_text)
                extractor_used = "classic"
        else:
            ai_keywords = extract_keywords(note_text)
            extractor_used = "classic"
    elif args.ai:
        if not openai_api_key:
            print(
                "Error: --ai mode requires OPENAI_API_KEY. Set it via --openai-api-key or in .env.",
                file=sys.stderr,
            )
            print("Tip: See .env.example for configuration help.", file=sys.stderr)
            sys.exit(1)
        ai_keywords = extract_keywords_llm(note_text)
        if ai_keywords and len(ai_keywords) >= 2:
            extractor_used = "OpenAI API"
        else:
            ai_keywords = extract_keywords(note_text)
            extractor_used = "classic"
    else:
        ai_keywords = extract_keywords(note_text)
        extractor_used = "classic"

    note_entry = {
        "date": date_str,
        "time": time_str,
        "note": note_text,
        "tags": tags,
        "ai_keywords": ai_keywords
    }

    # Try writing both logs atomically; if either fails, do not commit partial logs.
    try:
        append_to_json(note_entry)
        append_to_markdown(note_entry)
        print(f"Note logged successfully for {date_str} at {time_str}.\nTags: {tags}\nAI Keywords: {ai_keywords}")
        print(f"(Extractor used: {extractor_used}. See above for any fallback messages or errors.)")
    except Exception as e:
        print("ERROR: Failed to log note due to file I/O or system error.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(2)