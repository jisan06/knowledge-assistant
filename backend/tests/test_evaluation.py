import json
import os
import pytest

from app.schemas.search import SearchResultItem
from app.services.search_service import SearchService
from scripts.evaluate_rag import (
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    extract_cited_pages,
    check_citation_correctness,
    check_refusal_accuracy,
    RAGEvaluator,
)
from scripts.generate_sample_pdf import HANDBOOK_PAGES, create_pdf_bytes


class TestMetricCalculations:
    """Verifies retrieval and answer quality metric calculations."""

    def test_recall_at_k(self):
        # Expected on page 1; retrieved [1, 2, 3] -> Hit
        assert calculate_recall_at_k(retrieved_pages=[1, 2, 3], expected_pages=[1]) is True
        # Expected on page 4; retrieved [1, 2, 3] -> Miss
        assert calculate_recall_at_k(retrieved_pages=[1, 2, 3], expected_pages=[4]) is False
        # Cross-page: expected [1, 4]; retrieved [2, 4] -> Hit
        assert calculate_recall_at_k(retrieved_pages=[2, 4], expected_pages=[1, 4]) is True
        # Empty expected pages (unanswerable)
        assert calculate_recall_at_k(retrieved_pages=[1, 2], expected_pages=[]) is True

    def test_reciprocal_rank(self):
        # First rank hit: 1 / 1 = 1.0
        assert calculate_reciprocal_rank(retrieved_pages=[2, 3, 4], expected_pages=[2]) == 1.0
        # Second rank hit: 1 / 2 = 0.5
        assert calculate_reciprocal_rank(retrieved_pages=[1, 2, 3], expected_pages=[2]) == 0.5
        # Third rank hit: 1 / 3 = ~0.3333
        assert abs(calculate_reciprocal_rank(retrieved_pages=[1, 3, 2], expected_pages=[2]) - (1.0 / 3.0)) < 1e-4
        # Complete miss in top-K: 0.0
        assert calculate_reciprocal_rank(retrieved_pages=[1, 3, 4], expected_pages=[2]) == 0.0

    def test_extract_cited_pages(self):
        text = "Acme offers 20 vacation days [Source: company_handbook.pdf, Page 3]."
        assert extract_cited_pages(text) == [3]

        text_multi = "Headquarters are in Seattle [Source: doc.pdf, Page 1] and issued laptop is a MacBook [Source: doc.pdf, Page 4]."
        assert extract_cited_pages(text_multi) == [1, 4]

        text_none = "I could not find any relevant information about this topic."
        assert extract_cited_pages(text_none) == []

    def test_citation_correctness(self):
        valid = "Employees must work on-site on Tuesdays [Source: handbook.pdf, Page 1]."
        assert check_citation_correctness(valid, expected_pages=[1]) is True

        wrong_page = "Employees must work on-site on Tuesdays [Source: handbook.pdf, Page 4]."
        assert check_citation_correctness(wrong_page, expected_pages=[1]) is False

        no_citation = "Employees must work on-site on Tuesdays and Thursdays."
        assert check_citation_correctness(no_citation, expected_pages=[1]) is False

    def test_refusal_accuracy(self):
        # Correct refusal: clear absence statement and no citations
        refusal = "The uploaded documents do not provide enough information to answer this question."
        assert check_refusal_accuracy(refusal) is True

        # Invalid refusal: states absence but fabricates a citation
        refusal_with_citation = (
            "The uploaded documents do not provide enough information [Source: doc.pdf, Page 2]."
        )
        assert check_refusal_accuracy(refusal_with_citation) is False

        # Invalid refusal: hallucinated answer without disclaimer
        hallucination = "Acme offers pet insurance through HealthyPaws with a 15% corporate discount."
        assert check_refusal_accuracy(hallucination) is False


class TestEvaluationDatasetIntegrity:
    """Validates structure and content of eval_dataset.json."""

    @pytest.fixture
    def dataset(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "data", "eval_dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_dataset_size_and_categories(self, dataset):
        assert len(dataset) == 20, "Evaluation dataset must contain exactly 20 questions"

        categories = [item["category"] for item in dataset]
        direct_count = categories.count("direct_single_page")
        multi_count = categories.count("multi_section")
        unanswerable_count = categories.count("unanswerable")

        assert direct_count == 10, "Should have 10 direct single-page questions"
        assert multi_count == 5, "Should have 5 multi-section questions"
        assert unanswerable_count == 5, "Should have 5 unanswerable questions"

    def test_dataset_item_schema(self, dataset):
        for item in dataset:
            assert "id" in item
            assert "category" in item
            assert "question" in item
            assert "is_answerable" in item
            assert "expected_pages" in item

            if item["is_answerable"]:
                assert len(item["expected_pages"]) >= 1, f"Answerable question {item['id']} must specify expected_pages"
                assert len(item["expected_facts"]) >= 1, f"Answerable question {item['id']} must specify expected_facts"
            else:
                assert len(item["expected_pages"]) == 0, f"Unanswerable question {item['id']} should have empty expected_pages"


class TestEvaluationPipelineExecution:
    """Verifies that the RAGEvaluator runs end-to-end deterministically."""

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_run(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "data", "eval_dataset.json")

        evaluator = RAGEvaluator(
            top_k=4,
            use_mock_llm=True,
            use_mock_embeddings=True,
            store_type="memory",
        )

        results = await evaluator.evaluate_dataset(dataset_path)

        summary = results["summary"]
        assert summary["total_questions"] == 20
        assert summary["answerable_questions"] == 15
        assert summary["unanswerable_questions"] == 5

        metrics = summary["metrics"]
        assert "recall_at_k" in metrics
        assert "mrr" in metrics
        assert "citation_correctness" in metrics
        assert "refusal_accuracy" in metrics

        # With mock embeddings and deterministic mock LLM, all 20 questions are evaluated
        assert len(results["detailed_results"]) == 20
        assert metrics["recall_at_k"] >= 70.0
        assert metrics["mrr"] > 0.5
        assert metrics["refusal_accuracy"] == 100.0

        # Verify structured retrieval diagnostics are present on every question
        for item in results["detailed_results"]:
            assert "retrieval_diagnostics" in item
            diags = item["retrieval_diagnostics"]
            assert len(diags) > 0
            for d in diags:
                assert "rank" in d
                assert "chunk_id" in d
                assert "filename" in d
                assert "page_number" in d
                assert "chunk_index" in d
                assert "similarity_score" in d
                assert "distance" in d
                assert "text" in d
                assert "text_snippet" in d


class TestRetrievalQualityRegression:
    """
    Regression tests validating retrieval precision, irrelevant query refusal,
    cross-page retrieval coverage, citation grounding, and hybrid scoring.
    """

    @pytest.fixture
    async def configured_evaluator(self):
        evaluator = RAGEvaluator(
            top_k=4,
            use_mock_llm=True,
            use_mock_embeddings=True,
            store_type="memory",
        )
        await evaluator.index_document_from_pages("company_handbook.pdf", HANDBOOK_PAGES)
        return evaluator

    @pytest.mark.asyncio
    async def test_relevant_retrieval_and_diagnostics_structure(self, configured_evaluator):
        evaluator = configured_evaluator
        # Direct query about password rotation (page 4)
        query = "What are the minimum password length requirements and rotation interval?"
        search_resp = await evaluator.search_service.search(query=query, top_k=4)

        assert len(search_resp.results) > 0
        top_chunk = search_resp.results[0]
        # Target chunk must be from expected page 4
        assert top_chunk.page_number == 4
        assert "password" in top_chunk.text.lower()
        assert top_chunk.chunk_id.startswith("company_handbook_pdf_p4")
        assert top_chunk.similarity_score > 0.0
        assert top_chunk.distance is not None

    @pytest.mark.asyncio
    async def test_multi_section_cross_page_retrieval(self, configured_evaluator):
        evaluator = configured_evaluator
        # Multi-section query requiring information from page 1 (office location) and page 4 (hardware laptop)
        query = "What hardware laptop is issued to engineering staff and where is Acme's regional engineering center located?"
        search_resp = await evaluator.search_service.search(query=query, top_k=4)

        retrieved_pages = [c.page_number for c in search_resp.results]
        # Both page 1 and page 4 must be captured in top-k
        assert 1 in retrieved_pages
        assert 4 in retrieved_pages
        # Recall@4 should be True for expected [1, 4]
        assert calculate_recall_at_k(retrieved_pages, [1, 4]) is True

    @pytest.mark.asyncio
    async def test_irrelevant_query_retrieval_and_refusal(self, configured_evaluator):
        evaluator = configured_evaluator
        irrelevant_questions = [
            "Does Acme Cloud Technologies provide pet insurance discounts or pet bereavement leave?",
            "What is the current stock strike price for employee incentive stock options (ISOs)?",
            "Does the Seattle office cafeteria provide catered hot lunches and barista coffee?",
        ]

        for q in irrelevant_questions:
            # Check retrieval behavior
            search_resp = await evaluator.search_service.search(query=q, top_k=4)
            assert len(search_resp.results) <= 4

            # Check RAG refusal behavior
            answer = await evaluator.rag_service.chat(q, top_k=4)
            assert check_refusal_accuracy(answer) is True, f"Failed refusal for irrelevant query: {q}"
            assert len(extract_cited_pages(answer)) == 0, f"Unanswerable question must not cite pages: {answer}"

    @pytest.mark.asyncio
    async def test_document_citation_grounding_regression(self, configured_evaluator):
        evaluator = configured_evaluator
        # Query: paid parental leave (page 3)
        query = "How many weeks of paid parental leave does Acme provide and what is the tenure eligibility requirement?"
        answer = await evaluator.rag_service.chat(query, top_k=4)

        cited_pages = extract_cited_pages(answer)
        assert len(cited_pages) > 0, "Grounded answer must include citations"
        assert 3 in cited_pages, "Citations must include expected page 3"
        assert check_citation_correctness(answer, expected_pages=[3]) is True

    def test_hybrid_reranking_preserves_vector_similarity_when_no_lexical_match(self):
        candidates = [
            SearchResultItem(
                chunk_id="doc_p1_c1",
                text="Acme Cloud Technologies overview and hybrid workplace schedule.",
                filename="doc.pdf",
                page_number=1,
                chunk_index=1,
                similarity_score=0.75,
                distance=0.25,
            ),
            SearchResultItem(
                chunk_id="doc_p2_c1",
                text="Compensation, payroll cycles, and 401k benefits.",
                filename="doc.pdf",
                page_number=2,
                chunk_index=1,
                similarity_score=0.60,
                distance=0.40,
            ),
        ]

        # Query with content terms that do not appear lexically in any candidate
        query = "zenith quantum robotics astrophysics"
        reranked = SearchService._apply_hybrid_reranking(query=query, candidates=candidates, top_k=2)

        assert len(reranked) == 2
        # Original vector similarity should be preserved (NOT multiplied by 0.65)
        assert reranked[0].chunk_id == "doc_p1_c1"
        assert reranked[0].similarity_score == 0.75
        assert reranked[1].chunk_id == "doc_p2_c1"
        assert reranked[1].similarity_score == 0.60

