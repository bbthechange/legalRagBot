"""Tests for the breach analysis pipeline."""

import json

import pytest

from src.breach_analysis import (
    validate_breach_params,
    retrieve_applicable_statutes,
    analyze_breach_for_state,
    generate_breach_report,
    _build_summary,
)


# --- validate_breach_params ---

class TestValidateBreachParams:
    def test_valid_params_no_errors(self):
        """Valid breach parameters produce no errors."""
        params = {
            "data_types_compromised": ["ssn", "email"],
            "affected_states": ["CA", "NY"],
        }
        errors = validate_breach_params(params)
        assert errors == []

    def test_missing_data_types(self):
        """Missing data_types_compromised produces error."""
        params = {"affected_states": ["CA"]}
        errors = validate_breach_params(params)
        assert any("data_types_compromised" in e for e in errors)

    def test_missing_affected_states(self):
        """Missing affected_states produces error."""
        params = {"data_types_compromised": ["ssn"]}
        errors = validate_breach_params(params)
        assert any("affected_states" in e for e in errors)

    def test_empty_data_types(self):
        """Empty data_types_compromised list produces error."""
        params = {
            "data_types_compromised": [],
            "affected_states": ["CA"],
        }
        errors = validate_breach_params(params)
        assert any("data_types_compromised" in e for e in errors)

    def test_empty_affected_states(self):
        """Empty affected_states list produces error."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": [],
        }
        errors = validate_breach_params(params)
        assert any("affected_states" in e for e in errors)

    def test_missing_both_fields(self):
        """Missing both required fields produces two errors."""
        errors = validate_breach_params({})
        assert len(errors) == 2


# --- retrieve_applicable_statutes ---

class TestRetrieveApplicableStatutes:
    def test_returns_dict_keyed_by_state(self, loaded_multi_source_db):
        """retrieve_applicable_statutes returns dict with state keys."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["UK"],
            "encryption_status": "unknown",
        }
        results = retrieve_applicable_statutes(params, loaded_multi_source_db)
        assert isinstance(results, dict)
        assert "UK" in results

    def test_returns_list_for_each_state(self, loaded_multi_source_db):
        """Each state key maps to a list of retrieval results."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["UK"],
            "encryption_status": "unknown",
        }
        results = retrieve_applicable_statutes(params, loaded_multi_source_db)
        assert isinstance(results["UK"], list)


# --- analyze_breach_for_state ---

class TestAnalyzeBreachForState:
    def test_returns_parsed_dict(self, mock_provider):
        """analyze_breach_for_state returns a parsed dict."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA"],
        }
        # Mock statute results in the format that format_retrieval_results expects
        statute_results = [
            {
                "clause": {
                    "id": "statute-ca-summary",
                    "title": "California Data Breach Notification Law",
                    "type": "",
                    "category": "",
                    "text": "California breach notification law requires...",
                    "risk_level": "",
                    "notes": "",
                    "source": "statutes",
                    "doc_type": "statute",
                    "jurisdiction": "CA",
                    "citation": "Cal. Civ. Code 1798.29",
                    "position": "",
                },
                "score": 0.95,
            },
        ]
        result = analyze_breach_for_state(params, "CA", statute_results, mock_provider)
        assert isinstance(result, dict)


# --- generate_breach_report ---

class TestGenerateBreachReport:
    def test_invalid_params_returns_error(self):
        """Invalid params returns error dict without crashing."""
        report = generate_breach_report({}, {})
        assert "error" in report
        assert "details" in report

    def test_invalid_params_error_details(self):
        """Invalid params error includes specific field errors."""
        report = generate_breach_report({"data_types_compromised": []}, {})
        assert "error" in report
        details = report["details"]
        assert any("data_types_compromised" in d for d in details)

    def test_full_pipeline_returns_report(self, loaded_multi_source_db):
        """Full pipeline with mock data returns report structure."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["UK"],
            "encryption_status": "unencrypted",
            "number_of_affected_individuals": 1000,
        }
        report = generate_breach_report(params, loaded_multi_source_db)
        assert "summary" in report
        assert "state_analyses" in report
        assert "review_status" in report
        assert "disclaimer" in report

    def test_report_has_pending_review_status(self, loaded_multi_source_db):
        """Report review_status is 'pending_review'."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["UK"],
        }
        report = generate_breach_report(params, loaded_multi_source_db)
        assert report["review_status"] == "pending_review"

    def test_report_has_draft_disclaimer(self, loaded_multi_source_db):
        """Report disclaimer contains 'DRAFT'."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["UK"],
        }
        report = generate_breach_report(params, loaded_multi_source_db)
        assert "DRAFT" in report["disclaimer"]

    def test_report_summary_has_required_fields(self, loaded_multi_source_db):
        """Summary includes total_jurisdictions and notifications_required."""
        params = {
            "data_types_compromised": ["ssn"],
            "affected_states": ["UK"],
        }
        report = generate_breach_report(params, loaded_multi_source_db)
        summary = report["summary"]
        assert "total_jurisdictions" in summary
        assert "notifications_required" in summary
        assert "earliest_deadline" in summary


# --- _build_summary ---

class TestBuildSummary:
    def test_counts_notifications_required(self):
        """_build_summary correctly counts notifications required."""
        analyses = [
            {"jurisdiction": "CA", "notification_required": True, "notify_ag": True,
             "ag_notification_details": "500+ affected"},
            {"jurisdiction": "NY", "notification_required": True, "notify_ag": False},
            {"jurisdiction": "TX", "notification_required": False},
        ]
        params = {"data_types_compromised": ["ssn"], "encryption_status": "unknown"}
        summary = _build_summary(params, analyses)
        assert summary["notifications_required"] == 2

    def test_collects_ag_notifications(self):
        """_build_summary collects AG notification details."""
        analyses = [
            {"jurisdiction": "CA", "notification_required": True, "notify_ag": True,
             "ag_notification_details": "500+ affected"},
            {"jurisdiction": "NY", "notification_required": True, "notify_ag": True,
             "ag_notification_details": "All breaches"},
        ]
        params = {"data_types_compromised": ["ssn"]}
        summary = _build_summary(params, analyses)
        assert len(summary["ag_notifications_required"]) == 2

    def test_total_jurisdictions(self):
        """_build_summary reports correct total jurisdictions."""
        analyses = [
            {"jurisdiction": "CA"},
            {"jurisdiction": "NY"},
        ]
        params = {"data_types_compromised": ["ssn"]}
        summary = _build_summary(params, analyses)
        assert summary["total_jurisdictions"] == 2

    def test_earliest_deadline_detected(self):
        """_build_summary detects the earliest deadline."""
        analyses = [
            {"jurisdiction": "FL", "deadline": "30 days"},
            {"jurisdiction": "CA", "deadline": "Without unreasonable delay"},
        ]
        params = {"data_types_compromised": ["ssn"]}
        summary = _build_summary(params, analyses)
        assert "30 days" in summary["earliest_deadline"]

    def test_no_deadline_fallback(self):
        """_build_summary falls back when no deadline has day/hour."""
        analyses = [
            {"jurisdiction": "CA", "deadline": "Without unreasonable delay"},
        ]
        params = {"data_types_compromised": ["ssn"]}
        summary = _build_summary(params, analyses)
        assert summary["earliest_deadline"] == "See individual state analyses"
