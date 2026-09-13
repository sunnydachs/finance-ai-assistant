"""Tests for the char n-gram BM25 retrieval (offline, no API)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import Doc, load_corpus  # noqa: E402
from src.retrieval import BM25Index, tokenize  # noqa: E402
from src import config  # noqa: E402


def make_index(chunked=False):
    corpus = load_corpus(config.FAQ_PATH, config.NOTES_DIR, chunked=chunked)
    return BM25Index(corpus.docs)


def test_tokenize_japanese_and_ascii():
    tokens = tokenize("住宅ローンのATM手数料 NISA")
    assert "住宅ロ" in tokens and "ンnisa" not in tokens
    assert "atm" in tokens  # ASCII words indexed whole
    assert "nisa" in tokens


def test_retrieval_finds_relevant_faq():
    index = make_index()
    hits = index.search("繰上返済を店頭で申し込むと手数料はいくら？", k=4)
    ids = [doc.display_id() for doc, _ in hits]
    assert "FAQ-013" in ids
    assert ids[0] == "FAQ-013"  # the most relevant doc ranks first


def test_retrieval_finds_note_sections_when_chunked():
    index = make_index(chunked=True)
    hits = index.search("投資助言業を行うには登録が必要ですか", k=5)
    ids = [doc.display_id() for doc, _ in hits]
    assert "NOTE-002-A" in ids[:3]


def test_retrieval_top1_accuracy_over_faq_questions():
    """Sanity: paraphrased product questions should retrieve the right FAQ."""
    index = make_index()
    expectations = {
        "コンビニATMは月に何回まで無料ですか？": "FAQ-002",
        "変動金利の基準金利は？": "FAQ-012",
        "カードローンの金利の幅を教えて": "FAQ-021",
        "米ドルの為替手数料は？": "FAQ-034",
        "つみたて投資枠対応商品は何本？": "FAQ-031",
    }
    misses = []
    for q, expected in expectations.items():
        hits = index.search(q, k=4)
        ids = [doc.display_id() for doc, _ in hits]
        if expected not in ids:
            misses.append((q, expected, ids))
    assert not misses, f"retrieval misses: {misses}"


def test_search_filters_zero_score_hits():
    """Docs with no token overlap (score 0) must not be returned."""
    index = BM25Index([Doc(id="FAQ-999", title="t", text="全く関係ない内容です", kind="faq")])
    assert index.search("xyzzy plugh", k=1) == []  # ASCII tokens absent from doc
    hits = index.search("宇宙船の燃料は何ですか", k=1)  # only incidental ngram overlap
    assert hits and hits[0][1] < 1.0  # incidental overlap scores near zero
