import json
import os
import pytest

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
