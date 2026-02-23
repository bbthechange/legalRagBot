"""
Ingestor for state breach notification statute JSON files.

Loads structured statute data and creates two document types per state:
1. A structured summary document (for deterministic lookups)
2. Individual provision documents (for semantic search within a state)
"""

import json
import glob
import logging

from src.ingest.base import BaseIngestor

logger = logging.getLogger(__name__)


class StatuteIngestor(BaseIngestor):
    source_name = "statutes"

    def __init__(self, data_dir: str = "data/statutes"):
        self.data_dir = data_dir

    def load_raw(self) -> list[dict]:
        """Load all statute JSON files from the data directory."""
        files = sorted(glob.glob(f"{self.data_dir}/*.json"))
        statutes = []
        for path in files:
            with open(path) as f:
                statutes.append(json.load(f))
        return statutes

    def transform(self, raw_data: list[dict]) -> list[dict]:
        """
        Create multiple documents per statute for rich retrieval.

        For each state, creates:
        - One summary document (full statute text for broad queries)
        - One document per key provision (for targeted queries)
        """
        docs = []
        for statute in raw_data:
            jurisdiction = statute["jurisdiction"]
            abbr = statute["jurisdiction_abbr"]
            citation = statute["statute_citation"]

            # 1. Full summary document
            summary_text = self._build_summary_text(statute)
            docs.append({
                "doc_id": f"statute-{abbr.lower()}-summary",
                "source": "statutes",
                "doc_type": "statute",
                "title": f"{jurisdiction} Data Breach Notification Law",
                "text": summary_text,
                "metadata": {
                    "jurisdiction": abbr,
                    "citation": citation,
                    "practice_area": "privacy",
                    "category": "breach_notification",
                },
            })

            # 2. PI definition document
            pi_def = "; ".join(statute.get("personal_information_definition", []))
            if pi_def:
                docs.append({
                    "doc_id": f"statute-{abbr.lower()}-pi-definition",
                    "source": "statutes",
                    "doc_type": "statute",
                    "title": f"{jurisdiction} — Personal Information Definition",
                    "text": f"Under {citation}, personal information is defined as: {pi_def}",
                    "metadata": {
                        "jurisdiction": abbr,
                        "citation": citation,
                        "practice_area": "privacy",
                        "category": "pi_definition",
                    },
                })

            # 3. Notification timeline document
            timeline = statute.get("notification_timeline", "")
            days = statute.get("notification_timeline_days")
            timeline_text = f"Notification timeline: {timeline}"
            if days:
                timeline_text += f" ({days} days)"
            docs.append({
                "doc_id": f"statute-{abbr.lower()}-timeline",
                "source": "statutes",
                "doc_type": "statute",
                "title": f"{jurisdiction} — Notification Timeline",
                "text": timeline_text,
                "metadata": {
                    "jurisdiction": abbr,
                    "citation": citation,
                    "practice_area": "privacy",
                    "category": "notification_timeline",
                },
            })

            # 4. Safe harbor document
            if statute.get("encryption_safe_harbor"):
                docs.append({
                    "doc_id": f"statute-{abbr.lower()}-safe-harbor",
                    "source": "statutes",
                    "doc_type": "statute",
                    "title": f"{jurisdiction} — Encryption Safe Harbor",
                    "text": f"Encryption safe harbor: {statute.get('encryption_safe_harbor_details', 'Yes')}",
                    "metadata": {
                        "jurisdiction": abbr,
                        "citation": citation,
                        "practice_area": "privacy",
                        "category": "safe_harbor",
                    },
                })

        return docs

    def _build_summary_text(self, statute: dict) -> str:
        """Build a comprehensive text summary of a statute for embedding."""
        parts = [
            f"{statute['jurisdiction']} Data Breach Notification Law ({statute['statute_citation']})",
            f"Breach definition: {statute.get('breach_definition', 'N/A')}",
            f"Notification timeline: {statute.get('notification_timeline', 'N/A')}",
        ]
        if statute.get("notification_timeline_days"):
            parts.append(f"Deadline: {statute['notification_timeline_days']} days")
        if statute.get("encryption_safe_harbor"):
            parts.append(f"Encryption safe harbor: {statute.get('encryption_safe_harbor_details', 'Yes')}")
        pi_defs = statute.get("personal_information_definition", [])
        if pi_defs:
            parts.append(f"Personal information includes: {'; '.join(pi_defs)}")
        penalties = statute.get("penalties", "")
        if penalties:
            parts.append(f"Penalties: {penalties}")
        return "\n".join(parts)
