"""Tests for src/output_parser.py"""

import pytest

from src.output_parser import parse_json_response, parse_json_response_or_raw


class TestParseJsonResponse:
    def test_pure_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_in_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_preamble_and_json(self):
        text = 'Here is the analysis:\n{"key": "value"}'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_trailing(self):
        text = '{"key": "value"}\nEnd of response.'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_preamble_fenced_trailing(self):
        text = 'Here it is:\n```json\n{"key": "value"}\n```\nDone.'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_no_json_returns_none(self):
        result = parse_json_response("This has no JSON at all.")
        assert result is None

    def test_malformed_json_returns_none(self):
        result = parse_json_response('{"key": value}')
        assert result is None

    def test_nested_objects(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_response(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_empty_string_returns_none(self):
        result = parse_json_response("")
        assert result is None

    def test_escaped_quotes(self):
        text = '{"key": "value with \\"quotes\\""}'
        result = parse_json_response(text)
        assert result is not None
        assert "quotes" in result["key"]

    def test_none_input_returns_none(self):
        assert parse_json_response(None) is None

    def test_json_array_returns_none(self):
        assert parse_json_response("[1, 2, 3]") is None


class TestParseJsonResponseOrRaw:
    def test_valid_json_returns_parsed(self):
        result = parse_json_response_or_raw('{"key": "value"}')
        assert result == {"key": "value"}
        assert "parse_error" not in result

    def test_invalid_json_returns_raw(self):
        text = "Not valid JSON"
        result = parse_json_response_or_raw(text)
        assert result["raw_response"] == text
        assert result["parse_error"] is True
