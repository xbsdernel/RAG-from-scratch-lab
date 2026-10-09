"""
Unit tests for the RRF algorithm.
No LLM, no vector store, no network.
"""
import pytest

from langchain_core.documents import Document

from rag_app.pipelines.rrf import RRFResult, reciprocal_rank_fusion


def make_doc(chunk_id: str, content: str = "test") -> Document:
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id, "source": "test.txt", "page": 1},
    )


class TestRRF:
    def test_single_list(self):
        docs = [make_doc("a"), make_doc("b"), make_doc("c")]
        results = reciprocal_rank_fusion([docs], c=60)
        assert len(results) == 3
        # Rank 1 in single list has highest score
        assert results[0].chunk_id == "a"

    def test_rrf_formula(self):
        """Verify the formula: 1/(c+rank) for c=60."""
        docs = [make_doc("x")]
        results = reciprocal_rank_fusion([docs], c=60)
        expected = 1.0 / (60 + 1)
        assert abs(results[0].rrf_score - expected) < 1e-9

    def test_document_in_multiple_lists_scores_higher(self):
        """doc 'a' appears in both lists at rank 1 → should outscore 'b' which appears only once."""
        list1 = [make_doc("a"), make_doc("b")]
        list2 = [make_doc("a"), make_doc("c")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a_result = next(r for r in results if r.chunk_id == "a")
        b_result = next(r for r in results if r.chunk_id == "b")
        c_result = next(r for r in results if r.chunk_id == "c")
        assert a_result.rrf_score > b_result.rrf_score
        assert a_result.rrf_score > c_result.rrf_score

    def test_aggregated_score_sum(self):
        """Manually verify the aggregated sum for a doc in two lists."""
        # Doc 'a' is rank 1 in list1 and rank 2 in list2
        list1 = [make_doc("a"), make_doc("b")]
        list2 = [make_doc("b"), make_doc("a")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a_result = next(r for r in results if r.chunk_id == "a")
        expected = 1 / (60 + 1) + 1 / (60 + 2)
        assert abs(a_result.rrf_score - expected) < 1e-9

    def test_rank_tracking(self):
        """Each result should track which rank it held in each query list."""
        list1 = [make_doc("a"), make_doc("b")]
        list2 = [make_doc("b"), make_doc("a")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a = next(r for r in results if r.chunk_id == "a")
        assert a.ranks[0] == 1  # rank 1 in list1
        assert a.ranks[1] == 2  # rank 2 in list2

    def test_tie_breaking_deterministic(self):
        """Documents with equal RRF scores should be ordered deterministically by chunk_id."""
        # 'x' and 'y' each appear only in one list at rank 1 → equal scores
        list1 = [make_doc("x")]
        list2 = [make_doc("y")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        chunk_ids = [r.chunk_id for r in results]
        # Running again should give same order
        results2 = reciprocal_rank_fusion([list1, list2], c=60)
        chunk_ids2 = [r.chunk_id for r in results2]
        assert chunk_ids == chunk_ids2

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], c=60)
        assert results == []

    def test_empty_inner_list(self):
        list1 = [make_doc("a")]
        results = reciprocal_rank_fusion([list1, []], c=60)
        assert len(results) == 1

    def test_final_rank_assigned(self):
        docs = [make_doc("a"), make_doc("b"), make_doc("c")]
        results = reciprocal_rank_fusion([docs], c=60)
        ranks = [r.final_rank for r in results]
        assert ranks == [1, 2, 3]

    def test_three_lists_convergence(self):
        """Document ranked #1 in all 3 lists should be #1 overall."""
        star = make_doc("star")
        list1 = [star, make_doc("b"), make_doc("c")]
        list2 = [star, make_doc("d"), make_doc("e")]
        list3 = [star, make_doc("f"), make_doc("g")]
        results = reciprocal_rank_fusion([list1, list2, list3], c=60)
        assert results[0].chunk_id == "star"

    def test_c_parameter_effect(self):
        """Higher c → smaller differences between ranks. Lower c → bigger differences."""
        doc_top = make_doc("top")
        doc_low = make_doc("low")
        results_low_c = reciprocal_rank_fusion([[doc_top, doc_low]], c=1)
        results_high_c = reciprocal_rank_fusion([[doc_top, doc_low]], c=1000)

        # With low c: difference between rank 1 and rank 2 is larger
        diff_low = results_low_c[0].rrf_score - results_low_c[1].rrf_score
        diff_high = results_high_c[0].rrf_score - results_high_c[1].rrf_score
        assert diff_low > diff_high

    def test_to_dict_structure(self):
        docs = [make_doc("myid")]
        results = reciprocal_rank_fusion([docs])
        d = results[0].to_dict()
        assert "final_rank" in d
        assert "rrf_score" in d
        assert "chunk_id" in d
        assert "source" in d
        assert "ranks_per_query" in d
