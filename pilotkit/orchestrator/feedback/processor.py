from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import textdistance

from ..models.schemas import FeedbackItem

SEVERITY_MAP = {"P0": 4, "P1": 3, "P2": 2, "P3": 1}


class FeedbackProcessor:
    def __init__(self, pii_allowed: bool = False):
        self.pii_allowed = pii_allowed
        self._seen_hashes: set[str] = set()

    def _hash_text(self, text: str) -> str:
        redacted = text if self.pii_allowed else self._redact(text)
        return hashlib.sha256(redacted.encode("utf-8")).hexdigest()

    @staticmethod
    def _redact(text: str) -> str:
        return text.replace("@", "[at]").replace("+", "[plus]")

    def auto_tag(self, item: FeedbackItem) -> List[str]:
        tags = set(item.tags)
        text_lower = item.text.lower()
        if "slow" in text_lower or "waiting" in text_lower:
            tags.add("latency")
        if "error" in text_lower or "fail" in text_lower:
            tags.add("defect")
        if "confusing" in text_lower or "unclear" in text_lower:
            tags.add("usability")
        if item.step:
            tags.add(item.step)
        return sorted(tags)

    def deduplicate(self, items: Iterable[FeedbackItem]) -> List[FeedbackItem]:
        unique_items: List[FeedbackItem] = []
        for item in items:
            digest = self._hash_text(item.text)
            if digest in self._seen_hashes:
                continue
            self._seen_hashes.add(digest)
            unique_items.append(item)
        return unique_items

    def cluster(self, items: List[FeedbackItem]) -> Dict[str, List[FeedbackItem]]:
        clusters: Dict[str, List[FeedbackItem]] = {}
        for item in items:
            assigned = False
            for theme, theme_items in clusters.items():
                similarity = textdistance.jaccard.normalized_similarity(
                    item.text.lower(), theme_items[0].text.lower()
                )
                if similarity > 0.5:
                    theme_items.append(item)
                    assigned = True
                    break
            if not assigned:
                clusters[item.text[:40]] = [item]
        return clusters

    def triage(self, clusters: Dict[str, List[FeedbackItem]]) -> List[Tuple[str, List[FeedbackItem], str]]:
        triaged = []
        for theme, items in clusters.items():
            max_sev = max(SEVERITY_MAP[item.severity] for item in items)
            triage_level = next(k for k, v in SEVERITY_MAP.items() if v == max_sev)
            triaged.append((theme, items, triage_level))
        triaged.sort(key=lambda t: SEVERITY_MAP[t[2]], reverse=True)
        return triaged

    def aggregate_insights(self, items: Iterable[FeedbackItem]):
        deduped = self.deduplicate([item.copy(update={"tags": self.auto_tag(item)}) for item in items])
        clusters = self.cluster(deduped)
        triaged = self.triage(clusters)
        themes = []
        for theme, cluster_items, triage_level in triaged:
            summary = self._summarize_cluster(cluster_items)
            themes.append(
                {
                    "theme": theme,
                    "triage": triage_level,
                    "count": len(cluster_items),
                    "summary": summary,
                    "proposed_fix": self._proposed_fix(theme, cluster_items),
                }
            )
        return themes

    def generate_interview_guide(self, themes: List[str]) -> str:
        agenda = ["# 30-Minute Pilot Interview", "## Agenda", "1. Warm-up (5 min)", "2. Deep dive on pilot workflow (15 min)", "3. Wrap + next steps (10 min)"]
        prompts = ["## Thematic Prompts"]
        for theme in themes:
            prompts.append(f"- Describe a recent moment related to '{theme}'. What worked? What broke?")
            prompts.append("  - What evidence would demonstrate improvement?")
        prompts.extend(
            [
                "## Closing",
                "- If we solved one thing next sprint, what should it be?",
                "- What guardrails should we respect while iterating?",
            ]
        )
        return "\n".join(agenda + [""] + prompts)

    def rag_stub(self, theme: str) -> str:
        return (
            "RAG placeholder: connect to knowledge base to retrieve change logs and prior mitigations for theme '"
            + theme
            + "'."
        )

    @staticmethod
    def _summarize_cluster(items: List[FeedbackItem]) -> str:
        frustrations = [item.frustration or 3 for item in items]
        avg = sum(frustrations) / len(frustrations)
        return f"Avg frustration {avg:.1f} across {len(items)} reports."

    @staticmethod
    def _proposed_fix(theme: str, items: List[FeedbackItem]) -> str:
        if any("error" in item.text.lower() for item in items):
            return "Add guardrails + QA checks for error condition."
        if any("slow" in item.text.lower() for item in items):
            return "Streamline queue transitions + surface blockers in dashboard."
        return "Schedule usability review and co-design session."
