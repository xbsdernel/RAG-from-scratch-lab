"""
Reciprocal Rank Fusion implementation.

RRF merges multiple ranked document lists into a single ranked list.
Formula: RRF_score(d) = sum_r( 1 / (c + rank_r(d)) )
where rank_r(d) is the 1-based rank of document d in list r,
and c is a constant (default 60, as recommended in the original paper).

Documents not appearing in a list contribute 0 from that list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class RRFResult:
    """A document with its RRF aggregated score and per-list rank info."""

    document: Document
    rrf_score: float
    chunk_id: str
    ranks: dict[int, int] = field(default_factory=dict)  # {query_index: rank}
    final_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_rank": self.final_rank,
            "rrf_score": round(self.rrf_score, 6),
            "chunk_id": self.chunk_id,
            "source": self.document.metadata.get("source", "unknown"),
            "page": self.document.metadata.get("page", "?"),
            "ranks_per_query": self.ranks,
            "preview": self.document.page_content[:120] + "…",
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    c: int = 60,
) -> list[RRFResult]:
    """
    Merge multiple ranked document lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a ranked sequence of Documents
                      (index 0 = rank 1, i.e. most relevant first).
        c: RRF constant (default 60 as per original paper).
           Higher c gives a more equal weighting across ranks.

    Returns:
        List of RRFResult, sorted descending by rrf_score (highest first).
        Ties are broken deterministically by chunk_id (alphabetical).
    """
    scores: dict[str, float] = {}
    id_to_doc: dict[str, Document] = {}
    id_to_ranks: dict[str, dict[int, int]] = {}

    for query_idx, ranked_docs in enumerate(ranked_lists):
        for rank_0based, doc in enumerate(ranked_docs):
            cid = doc.metadata.get("chunk_id", "")
            if not cid:
                # Fallback: use a hash of page_content for deduplication
                import hashlib
                cid = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]

            rank_1based = rank_0based + 1
            contribution = 1.0 / (c + rank_1based)

            scores[cid] = scores.get(cid, 0.0) + contribution
            id_to_doc[cid] = doc  # last occurrence wins for doc reference
            id_to_ranks.setdefault(cid, {})[query_idx] = rank_1based

    # Sort: descending by score, then ascending by chunk_id for deterministic ties
    sorted_items = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    results: list[RRFResult] = []
    for final_rank, (cid, score) in enumerate(sorted_items, 1):
        results.append(
            RRFResult(
                document=id_to_doc[cid],
                rrf_score=score,
                chunk_id=cid,
                ranks=id_to_ranks[cid],
                final_rank=final_rank,
            )
        )

    return results
