"""Ingestor for firm playbook JSON files."""

import json
from glob import glob

from src.ingest.base import BaseIngestor


class PlaybookIngestor(BaseIngestor):
    source_name = "common_paper"

    def __init__(self, data_dir: str = "data/playbooks"):
        self.data_dir = data_dir

    def load_raw(self) -> list[dict]:
        raw = []
        for path in sorted(glob(f"{self.data_dir}/*.json")):
            with open(path) as f:
                playbook = json.load(f)
            for clause in playbook.get("clauses", []):
                clause["_playbook_id"] = playbook["playbook_id"]
                clause["_playbook_name"] = playbook["name"]
                raw.append(clause)
        return raw

    def transform(self, raw_data: list[dict]) -> list[dict]:
        docs = []
        for clause in raw_data:
            playbook_id = clause["_playbook_id"]
            clause_type = clause["clause_type"]
            position_text = (
                f"Preferred: {clause['preferred_position']}\n"
                f"Fallback: {clause['fallback_position']}\n"
                f"Walk-away: {clause['walk_away']}"
            )
            docs.append({
                "doc_id": f"playbook-{playbook_id}-{clause_type}",
                "source": "common_paper",
                "doc_type": "playbook",
                "title": f"{clause['_playbook_name']} - {clause_type.replace('_', ' ').title()}",
                "text": position_text,
                "metadata": {
                    "clause_type": clause_type,
                    "position": position_text,
                    "risk_factors": ", ".join(clause.get("risk_factors", [])),
                    "notes": clause.get("notes", ""),
                },
            })
        return docs
