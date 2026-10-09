"""
Standalone test runner for the RRF algorithm.
Tests only rag_app.pipelines.rrf — which has NO external dependencies.
Run with:
    python tests/test_rrf_standalone.py
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa — will fail if not installed

# ── Minimal no-dep test harness ──────────────────────────────────────────────
_passed = 0
_failed = 0


def test(name: str, fn):
    global _passed, _failed
    try:
        fn()
        print(f"  ✅ {name}")
        _passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        _failed += 1


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: expected {b!r}, got {a!r}"


def assert_approx(a, b, tol=1e-9, msg=""):
    assert abs(a - b) < tol, f"{msg}: expected ~{b}, got {a}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def doc(chunk_id: str, content: str = "test") -> Document:
    return Document(page_content=content, metadata={"chunk_id": chunk_id, "source": "t.txt", "page": 1})


# ── Tests ────────────────────────────────────────────────────────────────────

def run_tests():
    from rag_app.pipelines.rrf import reciprocal_rank_fusion

    print("\n─── RRF Tests ───────────────────────────────────────────────────────")

    def t_single_list():
        docs = [doc("a"), doc("b"), doc("c")]
        results = reciprocal_rank_fusion([docs], c=60)
        assert len(results) == 3
        assert results[0].chunk_id == "a"

    test("Single list — rank 1 wins", t_single_list)

    def t_formula():
        results = reciprocal_rank_fusion([[doc("x")]], c=60)
        expected = 1.0 / (60 + 1)
        assert_approx(results[0].rrf_score, expected, msg="RRF formula")

    test("Formula: 1/(c+rank)", t_formula)

    def t_multi_list_boosts():
        list1 = [doc("a"), doc("b")]
        list2 = [doc("a"), doc("c")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a = next(r for r in results if r.chunk_id == "a")
        b = next(r for r in results if r.chunk_id == "b")
        assert a.rrf_score > b.rrf_score

    test("Document in 2 lists scores higher than in 1 list", t_multi_list_boosts)

    def t_aggregated_sum():
        list1 = [doc("a"), doc("b")]
        list2 = [doc("b"), doc("a")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a = next(r for r in results if r.chunk_id == "a")
        expected = 1/(60+1) + 1/(60+2)
        assert_approx(a.rrf_score, expected, msg="Aggregated sum")

    test("Aggregated score sum is correct", t_aggregated_sum)

    def t_rank_tracking():
        list1 = [doc("a"), doc("b")]
        list2 = [doc("b"), doc("a")]
        results = reciprocal_rank_fusion([list1, list2], c=60)
        a = next(r for r in results if r.chunk_id == "a")
        assert a.ranks[0] == 1
        assert a.ranks[1] == 2

    test("Per-query rank tracking", t_rank_tracking)

    def t_deterministic_tie():
        r1 = reciprocal_rank_fusion([[doc("x")], [doc("y")]], c=60)
        r2 = reciprocal_rank_fusion([[doc("x")], [doc("y")]], c=60)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    test("Tie-breaking is deterministic", t_deterministic_tie)

    def t_empty_lists():
        results = reciprocal_rank_fusion([], c=60)
        assert results == []

    test("Empty input returns empty list", t_empty_lists)

    def t_empty_inner():
        results = reciprocal_rank_fusion([[doc("a")], []], c=60)
        assert len(results) == 1

    test("Empty inner list handled", t_empty_inner)

    def t_final_rank():
        docs = [doc("a"), doc("b"), doc("c")]
        results = reciprocal_rank_fusion([docs], c=60)
        assert [r.final_rank for r in results] == [1, 2, 3]

    test("Final ranks assigned 1-based", t_final_rank)

    def t_star_wins_three_lists():
        star = doc("star")
        results = reciprocal_rank_fusion([
            [star, doc("b")],
            [star, doc("c")],
            [star, doc("d")],
        ], c=60)
        assert results[0].chunk_id == "star"

    test("Doc ranked #1 in 3 lists wins overall", t_star_wins_three_lists)

    def t_c_effect():
        d1, d2 = doc("top"), doc("low")
        low_c = reciprocal_rank_fusion([[d1, d2]], c=1)
        high_c = reciprocal_rank_fusion([[d1, d2]], c=1000)
        diff_low = low_c[0].rrf_score - low_c[1].rrf_score
        diff_high = high_c[0].rrf_score - high_c[1].rrf_score
        assert diff_low > diff_high

    test("Higher c → smaller rank differences", t_c_effect)

    def t_to_dict():
        results = reciprocal_rank_fusion([[doc("myid")]])
        d = results[0].to_dict()
        for key in ["final_rank", "rrf_score", "chunk_id", "source", "ranks_per_query"]:
            assert key in d, f"Missing key: {key}"

    test("to_dict() has required fields", t_to_dict)

    print(f"\n─── Results: {_passed} passed, {_failed} failed ─────────────────────")
    return _failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
