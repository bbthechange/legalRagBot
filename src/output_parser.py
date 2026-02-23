"""
Output Parser — Extract structured JSON from LLM responses.

Handles common LLM output patterns: pure JSON, code-fenced JSON,
JSON with preamble/trailing text, and malformed responses.
"""

import json
import re


def parse_json_response(text: str) -> dict | None:
    """
    Attempt to parse JSON from an LLM response.

    Tries (in order):
    1. Direct JSON parse
    2. Code-fence extraction (```json ... ``` or ``` ... ```)
    3. Brace-matching extraction

    Returns parsed dict or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Code-fence extraction
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: Brace-matching
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, dict):
                            return result
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    return None


def parse_json_response_or_raw(text: str) -> dict:
    """
    Parse JSON from LLM response, falling back to raw text wrapper.

    Returns parsed dict, or {"raw_response": text, "parse_error": True}
    if no valid JSON could be extracted.
    """
    result = parse_json_response(text)
    if result is not None:
        return result
    return {"raw_response": text, "parse_error": True}
