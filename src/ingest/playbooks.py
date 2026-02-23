"""Ingestor for firm playbook JSON files."""

import json
import glob
import logging

from src.ingest.base import BaseIngestor

logger = logging.getLogger(__name__)


class PlaybookIngestor(BaseIngestor):
    source_name = "common_paper"

    def __init__(self, data_dir: str = "data/playbooks"):
        self.data_dir = data_dir

    def load_raw(self) -> list[dict]:
        files = sorted(glob.glob(f"{self.data_dir}/*.json"))
        playbooks = []
        for path in files:
            with open(path) as f:
                playbooks.append(json.load(f))
        return playbooks

    def transform(self, raw_data: list[dict]) -> list[dict]:
        docs = []
        for playbook in raw_data:
            playbook_id = playbook["playbook_id"]

            for clause in playbook.get("clauses", []):
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
                    "title": f"{playbook['name']} \u2014 {clause_type.replace('_', ' ').title()}",
                    "text": position_text,
                    "metadata": {
                        "clause_type": clause_type,
                        "practice_area": "commercial_contracts",
                        "risk_level": None,
                        "notes": clause.get("notes", ""),
                        "category": clause_type,
                    },
                })
        return docs
