import pytest
import rag


def test_faq_dataset_loaded():
    assert not rag.faq_df.empty


def test_required_faq_columns_exist():
    assert {"question", "answer", "class"}.issubset(
        rag.faq_df.columns
    )


def test_faq_questions_are_not_empty():
    assert rag.faq_df["question"].notna().all()
    assert rag.faq_df["question"].str.strip().ne("").all()


def test_tfidf_matrix_matches_faq_rows():
    assert rag.faq_matrix.shape[0] == len(rag.faq_df)
    assert rag.faq_matrix.shape[1] > 0


def test_empty_query_returns_no_results():
    assert rag.retrieve_faqs("") == []
    assert rag.retrieve_faqs("   ") == []


def test_non_string_query_returns_no_results():
    assert rag.retrieve_faqs(None) == []
    assert rag.retrieve_faqs(123) == []


def test_exact_faq_question_is_retrieved_first():
    question = str(rag.faq_df.iloc[0]["question"])

    results = rag.retrieve_faqs(question)

    assert results
    assert results[0]["question"] == question
    assert results[0]["similarity"] >= rag.SIMILARITY_THRESHOLD


def test_retrieval_result_has_expected_fields():
    question = str(rag.faq_df.iloc[0]["question"])
    results = rag.retrieve_faqs(question)

    assert results

    expected_fields = {
        "question",
        "answer",
        "class",
        "similarity",
    }
    assert expected_fields.issubset(results[0].keys())


def test_retrieval_respects_top_k():
    question = str(rag.faq_df.iloc[0]["question"])

    results = rag.retrieve_faqs(question, top_k=2)

    assert len(results) <= 2


def test_results_are_sorted_by_similarity():
    question = str(rag.faq_df.iloc[0]["question"])
    results = rag.retrieve_faqs(question)

    scores = [item["similarity"] for item in results]

    assert scores == sorted(scores, reverse=True)


def test_all_retrieved_results_meet_similarity_threshold():
    question = str(rag.faq_df.iloc[0]["question"])
    results = rag.retrieve_faqs(question)

    assert all(
        item["similarity"] >= rag.SIMILARITY_THRESHOLD
        for item in results
    )


def test_blank_question_returns_safe_response():
    result = rag.ask_question("   ")

    assert isinstance(result, dict)
    assert result["answer"] == "Please enter a valid, non-empty question."
    assert result["sources"] == []


def test_ask_question_returns_expected_structure(monkeypatch):
    fake_faqs = [
        {
            "question": "How can I check my account?",
            "answer": "Use the bank's approved account enquiry process.",
            "class": "accounts",
            "similarity": 0.95,
        }
    ]

    monkeypatch.setattr(
        rag, "retrieve_faqs", lambda question: fake_faqs
    )
    monkeypatch.setattr(
        rag, "generate_answer", lambda question, faqs: "Test answer"
    )

    result = rag.ask_question("How can I check my account?")

    assert result["question"] == "How can I check my account?"
    assert result["answer"] == "Test answer"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["question"] == fake_faqs[0]["question"]


def test_no_retrieved_faqs_returns_fallback():
    answer = rag.generate_answer("unrelated query", [])

    assert isinstance(answer, str)
    assert "could not find" in answer.lower()