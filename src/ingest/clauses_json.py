"""Ingestor for the original hand-authored clauses.json file."""

import json
from src.ingest.base import BaseIngestor


class ClausesJsonIngestor(BaseIngestor):
    source_name = "clauses_json"

    def __init__(self, data_path: str = "data/clauses.json"):
        self.data_path = data_path

    def load_raw(self) -> list[dict]:
        with open(self.data_path) as f:
            return json.load(f)

    def transform(self, raw_data: list[dict]) -> list[dict]:
        practice_area_map = {
            "NDA": "intellectual_property",
            "Employment": "employment_labor",
            "Service Agreement": "commercial_contracts",
        }

        docs = []
        for clause in raw_data:
            docs.append({
                "doc_id": clause["id"],
                "source": "clauses_json",
                "doc_type": "clause",
                "title": clause["title"],
                "text": clause["text"],
                "metadata": {
                    "clause_type": clause["type"],
                    "category": clause["category"],
                    "risk_level": clause["risk_level"],
                    "notes": clause["notes"],
                    "practice_area": practice_area_map.get(clause["type"], "general"),
                },
            })
        return docs
