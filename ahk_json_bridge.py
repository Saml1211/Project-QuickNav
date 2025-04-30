import sys
import json
import os

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # For AHK, output as a compact JSON string
        print(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        print("{}", file=sys.stderr)
        sys.exit(1)

def dump_json(path, data_str):
    try:
        data = json.loads(data_str)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Failed to write JSON", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print("Usage: ahk_json_bridge.py <load|dump> <path> [content_for_dump]", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]
    path = sys.argv[2]

    if mode == "load":
        load_json(path)
    elif mode == "dump":
        if len(sys.argv) < 4:
            print("Missing JSON string for dump", file=sys.stderr)
            sys.exit(1)
        data_str = sys.argv[3]
        dump_json(path, data_str)
    else:
        print("Unknown mode", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()