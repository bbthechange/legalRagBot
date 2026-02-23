"""
CUAD (Contract Understanding Atticus Dataset) ingestor.

Downloads CUAD from HuggingFace, extracts clause annotations,
and converts them to the unified document schema.

Expected output: ~4,000-5,000 documents (non-empty clause annotations).
"""

import hashlib
import logging
import re

from src.ingest.base import BaseIngestor

logger = logging.getLogger(__name__)

# Map CUAD question text to clean clause type labels
QUESTION_TO_CLAUSE_TYPE = {
    "Highlight the parts (if any) of this contract related to \"Parties\" that should be reviewed by a lawyer.": "parties",
    "Highlight the parts (if any) of this contract related to \"Agreement Date\"": "agreement_date",
    "Highlight the parts (if any) of this contract related to \"Effective Date\"": "effective_date",
    "Highlight the parts (if any) of this contract related to \"Expiration Date\"": "expiration_date",
    "Highlight the parts (if any) of this contract related to \"Renewal Term\"": "renewal_term",
    "Highlight the parts (if any) of this contract related to \"Notice Period To Terminate Renewal\"": "notice_period",
    "Highlight the parts (if any) of this contract related to \"Governing Law\"": "governing_law",
    "Highlight the parts (if any) of this contract related to \"Most Favored Nation\"": "most_favored_nation",
    "Highlight the parts (if any) of this contract related to \"Non-Compete\"": "non_compete",
    "Highlight the parts (if any) of this contract related to \"Exclusivity\"": "exclusivity",
    "Highlight the parts (if any) of this contract related to \"No-Solicit Of Customers\"": "no_solicit_customers",
    "Highlight the parts (if any) of this contract related to \"No-Solicit Of Employees\"": "no_solicit_employees",
    "Highlight the parts (if any) of this contract related to \"Non-Disparagement\"": "non_disparagement",
    "Highlight the parts (if any) of this contract related to \"Termination For Convenience\"": "termination_for_convenience",
    "Highlight the parts (if any) of this contract related to \"Change Of Control\"": "change_of_control",
    "Highlight the parts (if any) of this contract related to \"Anti-Assignment\"": "anti_assignment",
    "Highlight the parts (if any) of this contract related to \"Revenue/Profit Sharing\"": "revenue_profit_sharing",
    "Highlight the parts (if any) of this contract related to \"Price Restrictions\"": "price_restrictions",
    "Highlight the parts (if any) of this contract related to \"Minimum Commitment\"": "minimum_commitment",
    "Highlight the parts (if any) of this contract related to \"Volume Restriction\"": "volume_restriction",
    "Highlight the parts (if any) of this contract related to \"Ip Ownership Assignment\"": "ip_ownership_assignment",
    "Highlight the parts (if any) of this contract related to \"Joint Ip Ownership\"": "joint_ip_ownership",
    "Highlight the parts (if any) of this contract related to \"License Grant\"": "license_grant",
    "Highlight the parts (if any) of this contract related to \"Non-Transferable License\"": "non_transferable_license",
    "Highlight the parts (if any) of this contract related to \"Affiliate License-Loss Of Ip\"": "affiliate_license_ip_loss",
    "Highlight the parts (if any) of this contract related to \"Cap On Liability\"": "cap_on_liability",
    "Highlight the parts (if any) of this contract related to \"Liquidated Damages\"": "liquidated_damages",
    "Highlight the parts (if any) of this contract related to \"Warranty Duration\"": "warranty_duration",
    "Highlight the parts (if any) of this contract related to \"Insurance\"": "insurance",
    "Highlight the parts (if any) of this contract related to \"Covenant Not To Sue\"": "covenant_not_to_sue",
    "Highlight the parts (if any) of this contract related to \"Third Party Beneficiary\"": "third_party_beneficiary",
    "Highlight the parts (if any) of this contract related to \"Indemnification\"": "indemnification",
    "Highlight the parts (if any) of this contract related to \"Uncapped Liability\"": "uncapped_liability",
    "Highlight the parts (if any) of this contract related to \"Audit Rights\"": "audit_rights",
    "Highlight the parts (if any) of this contract related to \"Post-Termination Services\"": "post_termination_services",
    "Highlight the parts (if any) of this contract related to \"Competitive Restriction Exception\"": "competitive_restriction_exception",
    "Highlight the parts (if any) of this contract related to \"Unlimited/All-You-Can-Eat-License\"": "unlimited_license",
    "Highlight the parts (if any) of this contract related to \"Irrevocable Or Perpetual License\"": "irrevocable_perpetual_license",
    "Highlight the parts (if any) of this contract related to \"Source Code Escrow\"": "source_code_escrow",
    "Highlight the parts (if any) of this contract related to \"Rofr/Rofo/Rofn\"": "right_of_first_refusal",
}


class CuadIngestor(BaseIngestor):
    source_name = "cuad"

    def __init__(self, split: str = "test", max_docs: int | None = None):
        """
        Args:
            split: HuggingFace dataset split to use (CUAD only has "test")
            max_docs: Maximum number of output documents (for testing/cost control).
                      None = no limit.
        """
        self.split = split
        self.max_docs = max_docs

    def load_raw(self) -> list[dict]:
        """Download CUAD from HuggingFace."""
        from datasets import load_dataset
        dataset = load_dataset("theatticusproject/cuad-qa", split=self.split)
        return [dict(row) for row in dataset]

    def _extract_clause_type(self, question: str) -> str:
        """Extract a clean clause type label from a CUAD question string."""
        # Try exact match first
        if question in QUESTION_TO_CLAUSE_TYPE:
            return QUESTION_TO_CLAUSE_TYPE[question]

        # Fuzzy: extract the quoted part
        match = re.search(r'"([^"]+)"', question)
        if match:
            return match.group(1).lower().replace(" ", "_").replace("-", "_")

        return "unknown"

    def _make_doc_id(self, contract_title: str, clause_type: str, text: str) -> str:
        """Generate a deterministic, unique document ID."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', contract_title)[:30]
        return f"cuad-{safe_title}-{clause_type}-{content_hash}"

    def _infer_practice_area(self, clause_type: str) -> str:
        """Map CUAD clause types to practice areas."""
        ip_types = {"ip_ownership_assignment", "joint_ip_ownership", "license_grant",
                     "non_transferable_license", "affiliate_license_ip_loss",
                     "source_code_escrow", "irrevocable_perpetual_license", "unlimited_license"}
        employment_types = {"non_compete", "no_solicit_employees", "non_disparagement",
                           "competitive_restriction_exception"}
        commercial_types = {"cap_on_liability", "indemnification", "uncapped_liability",
                           "insurance", "liquidated_damages", "warranty_duration",
                           "minimum_commitment", "volume_restriction", "price_restrictions",
                           "revenue_profit_sharing"}

        if clause_type in ip_types:
            return "intellectual_property"
        elif clause_type in employment_types:
            return "employment_labor"
        elif clause_type in commercial_types:
            return "commercial_contracts"
        return "general"

    def transform(self, raw_data: list[dict]) -> list[dict]:
        """
        Extract clause annotations from CUAD rows.

        Each row with non-empty answers produces one document per
        unique answer text. Deduplicates by text hash to avoid
        indexing the same clause span multiple times.
        """
        docs = []
        seen_hashes = set()

        for row in raw_data:
            answers = row.get("answers", {})
            answer_texts = answers.get("text", [])
            if not answer_texts:
                continue  # Skip rows with no annotations

            question = row.get("question", "")
            clause_type = self._extract_clause_type(question)
            contract_title = row.get("title", "unknown_contract")

            for text in answer_texts:
                text = text.strip()
                if not text or len(text) < 10:
                    continue  # Skip very short fragments

                # Deduplicate
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)

                doc_id = self._make_doc_id(contract_title, clause_type, text)
                clause_type_display = clause_type.replace("_", " ").title()

                docs.append({
                    "doc_id": doc_id,
                    "source": "cuad",
                    "doc_type": "clause",
                    "title": f"{clause_type_display} — {contract_title}",
                    "text": text,
                    "metadata": {
                        "clause_type": clause_type,
                        "source_contract": contract_title,
                        "practice_area": self._infer_practice_area(clause_type),
                        "category": clause_type,
                    },
                })

                if self.max_docs and len(docs) >= self.max_docs:
                    return docs

        return docs
