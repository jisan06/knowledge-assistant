import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Add backend directory to sys.path to allow clean app imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.config import settings
from app.providers.base import AIProvider
from app.providers.embeddings.mock_provider import MockEmbeddingProvider
from app.schemas.search import SearchResultItem
from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
from app.services.search_service import SearchService
from app.vector_store.factory import get_vector_store
from app.vector_store.memory_store import InMemoryVectorStore
from scripts.generate_sample_pdf import HANDBOOK_PAGES, generate_handbook_pdf

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_rag")


class MockDeterministicRAGProvider(AIProvider):
    """
    Deterministic AI Provider simulating Gemini's RAG behavior offline.
    Uses exact ground-truth matching and citation generation to evaluate
    the pipeline deterministically without requiring API keys or external network calls.
    """

    async def generate(self, message: str) -> str:
        # Check if prompt contains the RAG context header
        if "=== RETRIEVED DOCUMENT CONTEXT ===" not in message:
            return "General response without document context."

        # Extract the user question from the prompt
        q_match = re.search(r"User Question:\s*(.*?)\s*\n\s*Grounded Answer:", message, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""

        # Extract context block
        ctx_match = re.search(r"=== RETRIEVED DOCUMENT CONTEXT ===\s*(.*?)\s*=== END OF CONTEXT ===", message, re.DOTALL)
        context_text = ctx_match.group(1).strip() if ctx_match else ""

        # Check for unanswerable topics
        unanswerable_triggers = [
            "pet insurance", "pet bereavement", "stock strike price",
            "incentive stock options", "cafeteria", "hot lunches",
            "tuition reimbursement", "chief executive officer", "ceo", "cto"
        ]
        if any(trigger in question.lower() for trigger in unanswerable_triggers):
            return "The uploaded documents do not provide enough information to answer this question."

        # Parse available pages in the retrieved context
        pages_in_context = set(re.findall(r"Page (\d+)", context_text))

        # Synthesize answers and cite the actual retrieved pages
        citations = [f"[Source: company_handbook.pdf, Page {p}]" for p in sorted(pages_in_context, key=int)]
        citation_str = " ".join(citations) if citations else ""

        if "year was acme cloud technologies founded" in question.lower():
            return f"Acme Cloud Technologies was founded in 2018 in Seattle, Washington {citation_str}."
        elif "on-site in the office" in question.lower():
            return f"Employees are required to work on-site on Tuesdays and Thursdays {citation_str}."
        elif "core collaboration hours" in question.lower():
            return f"Core collaboration hours are 10:00 AM to 3:00 PM PST {citation_str}."
        elif "payroll frequency" in question.lower():
            return f"Employees are paid bi-weekly on alternating Fridays via direct deposit {citation_str}."
        elif "health insurance premiums" in question.lower():
            return f"Acme covers 90 percent of premiums for employees and 70 percent for dependents through BlueCross BlueShield {citation_str}."
        elif "401(k)" in question.lower() or "401k" in question.lower():
            return f"Acme matches 100 percent of contributions up to 4 percent of base salary with immediate vesting {citation_str}."
        elif "technology stipend" in question.lower() or "home office" in question.lower():
            return f"The annual technology stipend is 1,200 dollars reimbursable via Expensify within 60 days {citation_str}."
        elif "vacation days" in question.lower():
            return f"Full-time employees accrue 20 vacation days (160 hours) per calendar year {citation_str}."
        elif "parental leave" in question.lower():
            return f"Acme provides 16 weeks of 100 percent paid parental leave after 6 months of service {citation_str}."
        elif "password" in question.lower():
            return f"Passwords must be at least 16 characters and rotated every 90 days {citation_str}."
        elif "laptop is issued" in question.lower() or "regional engineering center" in question.lower():
            return f"Engineering staff receive an Apple MacBook Pro 16-inch or Dell XPS 15, and the regional center is in Austin, TX {citation_str}."
        elif "sick leave" in question.lower():
            return f"Employees receive 10 paid sick days annually which do not roll over, compared to 20 vacation days with up to 5 unused PTO days rollover {citation_str}."
        elif "lost company laptop" in question.lower() or "reporting deadline" in question.lower():
            return f"Incident reporting deadline is 1 hour to security@acmecloud.example.com, and home office items are expensed through Expensify {citation_str}."
        elif "flexible days" in question.lower() or "remote" in question.lower():
            return f"Employees working remote on Mondays, Wednesdays, and Fridays receive a 1,200 dollars annual stipend {citation_str}."
        elif "rollover" in question.lower() or "pii" in question.lower():
            return f"Up to 5 unused PTO days roll over on December 31st, and customer PII is encrypted with AES-256 as Restricted Confidential {citation_str}."

        return f"Based on the provided excerpts, the answer is grounded in context {citation_str}."

    async def generate_stream(self, message: str):
        full = await self.generate(message)
        words = full.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")


# ---------------------------------------------------------------------------
# Metric Calculation & Answer Quality Verification Utilities
# ---------------------------------------------------------------------------

def calculate_recall_at_k(retrieved_pages: List[int], expected_pages: List[int]) -> bool:
    """
    Returns True if at least one expected page is present in the top-K retrieved chunks.
    For questions expecting multiple pages, checks if any expected page was retrieved.
    """
    if not expected_pages:
        return True
    return any(p in retrieved_pages for p in expected_pages)


def calculate_reciprocal_rank(retrieved_pages: List[int], expected_pages: List[int]) -> float:
    """
    Calculates Reciprocal Rank (1 / rank of first relevant chunk).
    Returns 0.0 if no expected page appears in the retrieved results.
    """
    if not expected_pages:
        return 0.0
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return 1.0 / rank
    return 0.0


def extract_cited_pages(answer_text: str) -> List[int]:
    """
    Extracts page numbers cited in an answer string.
    Supports formats like:
      - [Source: file.pdf, Page 2]
      - [Source: Page 3]
      - (Page 1)
      - p. 2
    """
    patterns = [
        r"Page\s+(\d+)",
        r"p\.\s*(\d+)",
        r"Page:\s*(\d+)",
    ]
    found = set()
    for pat in patterns:
        for match in re.finditer(pat, answer_text, re.IGNORECASE):
            found.add(int(match.group(1)))
    return sorted(list(found))


def check_refusal_accuracy(answer_text: str) -> bool:
    """
    Checks if the model correctly refused an unanswerable question
    without fabricating facts or citations.
    """
    refusal_patterns = [
        r"do not provide enough information",
        r"does not provide enough information",
        r"not mentioned in the (documents|handbook|provided)",
        r"no information (is )?provided",
        r"not contain (enough|any) information",
        r"unable to answer .* from the provided",
        r"cannot answer .* based on the provided",
    ]
    is_refused = any(re.search(pat, answer_text, re.IGNORECASE) for pat in refusal_patterns)
    cited_pages = extract_cited_pages(answer_text)
    # Refusal is valid only if it clearly states absence and does NOT cite pages
    return is_refused and len(cited_pages) == 0


def check_citation_correctness(answer_text: str, expected_pages: List[int]) -> bool:
    """
    Verifies that the answer includes citations and that all cited pages belong to the expected pages.
    """
    cited = extract_cited_pages(answer_text)
    if not cited:
        return False
    # At least one cited page must be in expected_pages, and no cited page should be completely hallucinated
    return any(p in expected_pages for p in cited)


def normalize_fact_text(text: str) -> str:
    """
    Normalizes text for robust lexical fact comparison:
    - Lowercase
    - Replaces symbols like '%' with 'percent'
    - Replaces '$1,200' with '1200 dollars'
    - Normalizes common variants (e.g. 'biweekly' -> 'bi weekly')
    - Strips commas within numbers (e.g. '1,200' -> '1200')
    - Normalizes hyphens and punctuation
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"(\d+)\s*%", r"\1 percent", t)
    t = re.sub(r"\$\s*([0-9,]+)", r"\1 dollars", t)
    t = re.sub(r"(\d+),(\d+)", r"\1\2", t)
    t = re.sub(r"\bbiweekly\b", "bi weekly", t)
    t = re.sub(r"\b401\s*k\b", "401k", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def evaluate_answer_facts(
    answer_text: str, expected_facts: List[str]
) -> Tuple[bool, float, List[str], List[str]]:
    """
    Evaluates answer quality against expected facts using grounded lexical fact verification.
    
    Practical Method:
    1. Normalizes answer and fact strings (case, percentages, currencies, number punctuation).
    2. Checks whether each expected fact is represented either as an exact phrase match or
       through its essential constituent tokens.
    3. Computes fact coverage = matched_facts / total_expected_facts.
    4. Passes verification if coverage meets threshold (>= 75% for 3+ facts, or 100% for <= 2 facts).
    
    Returns:
        (facts_verified: bool, fact_coverage: float, matched_facts: List[str], missing_facts: List[str])
    """
    if not expected_facts:
        return True, 1.0, [], []

    norm_answer = normalize_fact_text(answer_text)
    matched: List[str] = []
    missing: List[str] = []

    for fact in expected_facts:
        norm_fact = normalize_fact_text(fact)
        if not norm_fact:
            matched.append(fact)
            continue

        # 1. Exact normalized phrase match
        if norm_fact in norm_answer:
            matched.append(fact)
            continue

        # 2. Token-level match for multi-word phrases
        tokens = [tk for tk in norm_fact.split() if len(tk) > 1]
        if tokens and all(tk in norm_answer for tk in tokens):
            matched.append(fact)
        else:
            missing.append(fact)

    coverage = len(matched) / len(expected_facts)
    is_verified = (coverage >= 0.75) if len(expected_facts) > 2 else (len(missing) == 0)
    return is_verified, coverage, matched, missing


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

class RAGEvaluator:
    def __init__(
        self,
        top_k: int = 4,
        use_mock_llm: bool = True,
        use_mock_embeddings: bool = False,
        store_type: str = "memory",
        collection_name: str = "eval_rag_collection",
    ):
        self.top_k = top_k
        self.use_mock_llm = use_mock_llm
        self.use_mock_embeddings = use_mock_embeddings
        self.store_type = store_type
        self.collection_name = collection_name

        # Initialize vector store & embeddings
        if use_mock_embeddings:
            self.embedding_service = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        else:
            self.embedding_service = EmbeddingService()

        if store_type == "memory":
            self.vector_store = InMemoryVectorStore(collection_name=self.collection_name)
        elif store_type == "chroma":
            from app.vector_store.chroma_store import ChromaVectorStore
            try:
                self.vector_store = ChromaVectorStore(
                    persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                    collection_name=self.collection_name,
                )
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed ({e}). Falling back to factory vector store.")
                self.vector_store = get_vector_store()
        else:
            self.vector_store = get_vector_store(store_type=store_type)

        self.search_service = SearchService(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

        self.ai_service = AIService()
        if self.use_mock_llm:
            self.ai_service.provider = MockDeterministicRAGProvider()

        self.rag_service = RAGService(
            search_service=self.search_service,
            ai_service=self.ai_service,
        )

    async def index_document_from_pages(
        self, filename: str, pages_text: List[List[str]], reindex: bool = False
    ):
        """Builds chunks and indexes the sample document into the vector store."""
        if reindex and self.vector_store.count() > 0:
            try:
                self.vector_store.delete_by_filename(filename)
            except Exception as e:
                logger.warning(f"Failed to delete existing document before reindexing: {e}")

        from app.schemas.document import DocumentPage
        doc_pages = []
        for i, lines in enumerate(pages_text, start=1):
            text = "\n".join(lines)
            doc_pages.append(
                DocumentPage(
                    page_number=i,
                    text=text,
                    char_count=len(text),
                )
            )
        await self.search_service.index_document_pages(pages=doc_pages, filename=filename)

    async def _generate_rag_answer(
        self,
        question: str,
        top_k: int,
        max_retries: int = 3,
        initial_backoff: float = 2.0,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Executes RAG generation with exponential backoff for rate limits and transient errors.
        Returns: (answer_text, error_message)
        """
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                answer = await self.rag_service.chat(question, top_k=top_k)
                if answer and answer.startswith("An error occurred while retrieving document context"):
                    return None, answer
                return answer, None
            except Exception as exc:
                last_exception = exc
                err_text = str(exc)
                is_rate_limit = any(
                    token in err_text.lower()
                    for token in ["429", "resource_exhausted", "quota", "rate limit", "rate_limit", "too many requests"]
                )
                is_transient = is_rate_limit or any(
                    token in err_text.lower()
                    for token in ["503", "502", "504", "timeout", "timed out", "connection error", "temporary"]
                )

                if is_transient and attempt < max_retries:
                    wait_time = initial_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"Transient error querying LLM for '{question[:30]}...' (attempt {attempt}/{max_retries}): {exc}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    break

        error_msg = f"{type(last_exception).__name__}: {str(last_exception)}" if last_exception else "Unknown error"
        return None, error_msg

    async def evaluate_dataset(
        self,
        dataset_path: str,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        question_ids: Optional[List[str]] = None,
        reindex: bool = False,
    ) -> Dict[str, Any]:
        """Runs the evaluation across questions in the dataset, supporting subset filtering."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            all_items = json.load(f)

        total_in_file = len(all_items)
        items = list(all_items)

        # Apply category filter
        if category:
            cat_clean = category.strip().lower()
            items = [item for item in items if item.get("category", "").lower() == cat_clean]

        # Apply question IDs filter
        if question_ids:
            target_ids = {qid.strip().lower() for qid in question_ids}
            items = [item for item in items if item.get("id", "").lower() in target_ids]

        # Apply subset limit
        if limit is not None and limit > 0:
            items = items[:limit]

        if not items:
            raise ValueError(
                f"No questions matched filter (limit={limit}, category={category}, question_ids={question_ids})"
            )

        # Ensure document is indexed
        if self.vector_store.count() == 0 or reindex:
            await self.index_document_from_pages("company_handbook.pdf", HANDBOOK_PAGES, reindex=reindex)

        results = []
        answerable_count = 0
        unanswerable_count = 0
        recall_hits = 0
        reciprocal_ranks = []
        citation_correct_count = 0
        refusal_correct_count = 0
        fact_coverage_total = 0.0
        facts_verified_count = 0
        api_errors_count = 0

        for item in items:
            qid = item["id"]
            question = item["question"]
            item_cat = item["category"]
            expected_pages = item.get("expected_pages", [])
            expected_facts = item.get("expected_facts", [])
            is_answerable = item.get("is_answerable", True)

            # 1. Retrieval Evaluation
            search_resp = await self.search_service.search(query=question, top_k=self.top_k)
            retrieved_chunks = search_resp.results
            retrieved_pages = [c.page_number for c in retrieved_chunks]

            retrieval_diagnostics = [
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "similarity_score": c.similarity_score,
                    "distance": c.distance,
                    "text_snippet": (c.text[:140] + "...") if len(c.text) > 140 else c.text,
                    "text": c.text,
                }
                for rank, c in enumerate(retrieved_chunks, start=1)
            ]

            # 2. Answer Generation Evaluation
            answer_text, error_msg = await self._generate_rag_answer(question, top_k=self.top_k)

            # 3. Compute Metrics
            recall_pass = False
            rr = 0.0
            citation_pass = None
            refusal_pass = None
            facts_pass = False
            fact_cov = 0.0
            matched_facts = []
            missing_facts = []
            api_error = False

            if error_msg is not None:
                api_error = True
                api_errors_count += 1
                answer_text = f"[API ERROR: {error_msg}]"

                if is_answerable:
                    answerable_count += 1
                    recall_pass = calculate_recall_at_k(retrieved_pages, expected_pages)
                    rr = calculate_reciprocal_rank(retrieved_pages, expected_pages)
                    if recall_pass:
                        recall_hits += 1
                    reciprocal_ranks.append(rr)
                    citation_pass = False
                    facts_pass = False
                    fact_cov = 0.0
                else:
                    unanswerable_count += 1
                    refusal_pass = False
                    facts_pass = False
                    fact_cov = 0.0
            else:
                if is_answerable:
                    answerable_count += 1
                    recall_pass = calculate_recall_at_k(retrieved_pages, expected_pages)
                    rr = calculate_reciprocal_rank(retrieved_pages, expected_pages)
                    if recall_pass:
                        recall_hits += 1
                    reciprocal_ranks.append(rr)

                    citation_pass = check_citation_correctness(answer_text, expected_pages)
                    if citation_pass:
                        citation_correct_count += 1

                    facts_pass, fact_cov, matched_facts, missing_facts = evaluate_answer_facts(
                        answer_text, expected_facts
                    )
                    fact_coverage_total += fact_cov
                    if facts_pass:
                        facts_verified_count += 1
                else:
                    unanswerable_count += 1
                    refusal_pass = check_refusal_accuracy(answer_text)
                    if refusal_pass:
                        refusal_correct_count += 1
                    facts_pass = refusal_pass
                    fact_cov = 1.0 if refusal_pass else 0.0

            results.append({
                "id": qid,
                "category": item_cat,
                "question": question,
                "is_answerable": is_answerable,
                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages,
                "recall_at_k": recall_pass if is_answerable else None,
                "reciprocal_rank": rr if is_answerable else None,
                "retrieval_diagnostics": retrieval_diagnostics,
                "answer": answer_text,
                "cited_pages": extract_cited_pages(answer_text) if not api_error else [],
                "citation_correct": citation_pass,
                "refusal_correct": refusal_pass,
                "facts_verified": facts_pass,
                "fact_coverage": round(fact_cov * 100, 1),
                "matched_facts": matched_facts,
                "missing_facts": missing_facts,
                "api_error": api_error,
                "error_message": error_msg,
            })

        recall_at_k_score = (recall_hits / answerable_count) if answerable_count > 0 else 0.0
        mrr_score = (sum(reciprocal_ranks) / answerable_count) if answerable_count > 0 else 0.0
        citation_acc = (citation_correct_count / answerable_count) if answerable_count > 0 else 0.0
        refusal_acc = (refusal_correct_count / unanswerable_count) if unanswerable_count > 0 else 0.0
        avg_fact_coverage = (fact_coverage_total / answerable_count) if answerable_count > 0 else 0.0
        facts_verified_acc = (facts_verified_count / answerable_count) if answerable_count > 0 else 0.0

        is_real = (not self.use_mock_llm) and (not self.use_mock_embeddings)
        pipeline_mode = "real_pipeline" if is_real else ("live_llm_mock_embeddings" if not self.use_mock_llm else "mock_pipeline")

        return {
            "summary": {
                "pipeline_mode": pipeline_mode,
                "is_real_pipeline": is_real,
                "total_questions": len(items),
                "dataset_total": total_in_file,
                "subset_evaluated": len(items) < total_in_file,
                "answerable_questions": answerable_count,
                "unanswerable_questions": unanswerable_count,
                "top_k": self.top_k,
                "store_type": self.store_type,
                "vector_store_class": self.vector_store.__class__.__name__,
                "llm_mode": "live_gemini" if not self.use_mock_llm else "mock_deterministic",
                "use_mock_llm": self.use_mock_llm,
                "use_mock_embeddings": self.use_mock_embeddings,
                "embedding_model": self.embedding_service.model_name,
                "llm_model": settings.GEMINI_MODEL if not self.use_mock_llm else "MockDeterministicRAGProvider",
                "metrics": {
                    "recall_at_k": round(recall_at_k_score * 100, 2),
                    "mrr": round(mrr_score, 4),
                    "citation_correctness": round(citation_acc * 100, 2),
                    "refusal_accuracy": round(refusal_acc * 100, 2),
                    "fact_coverage": round(avg_fact_coverage * 100, 2),
                    "facts_verified_rate": round(facts_verified_acc * 100, 2),
                    "api_errors": api_errors_count,
                },
            },
            "detailed_results": results,
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint & Formatted Output
# ---------------------------------------------------------------------------

def print_evaluation_report(results: Dict[str, Any], show_diagnostics: bool = False):
    summary = results["summary"]
    metrics = summary["metrics"]
    details = results["detailed_results"]

    mode_title = "REAL PIPELINE (LIVE GEMINI)" if summary.get("is_real_pipeline") else (
        "LIVE LLM PIPELINE" if not summary.get("use_mock_llm") else "MOCK DETERMINISTIC PIPELINE (OFFLINE)"
    )

    print("\n" + "=" * 88)
    print(f"       RAG EVALUATION REPORT — {mode_title}")
    print("=" * 88)
    print(f"Pipeline Mode       : {summary.get('pipeline_mode')}")
    print(f"Evaluated Questions : {summary['total_questions']}" + (f" (Subset of {summary['dataset_total']})" if summary.get("subset_evaluated") else ""))
    print(f"  - Answerable      : {summary['answerable_questions']}")
    print(f"  - Unanswerable    : {summary['unanswerable_questions']}")
    print(f"Retrieval Top-K     : {summary['top_k']}")
    print(f"Vector Store Type   : {summary['store_type']} ({summary.get('vector_store_class', '')})")
    print(f"LLM Generation Mode : {summary['llm_mode']} ({summary.get('llm_model', '')})")
    print(f"Embedding Model     : {summary['embedding_model']}")
    print("-" * 88)
    print("OVERALL METRICS:")
    print(f"  * Recall@{summary['top_k']}             : {metrics['recall_at_k']}%")
    print(f"  * Mean Reciprocal Rank (MRR) : {metrics['mrr']}")
    print(f"  * Citation Correctness      : {metrics['citation_correctness']}%")
    print(f"  * Refusal / Absence Handling: {metrics['refusal_accuracy']}%")
    print(f"  * Fact Coverage (Answer Qlt): {metrics.get('fact_coverage', 0.0)}%")
    print(f"  * Facts Fully Verified Rate : {metrics.get('facts_verified_rate', 0.0)}%")
    if metrics.get("api_errors", 0) > 0:
        print(f"  * API Errors / Rate Limits  : {metrics['api_errors']}")
    print("-" * 88)

    print("\nQUESTION-BY-QUESTION BREAKDOWN:")
    print(f"{'ID':<5} {'Category':<18} {'Exp. Pgs':<10} {'Ret. Pgs':<12} {'Recall':<8} {'MRR':<6} {'Citations':<10} {'Fact Cov':<9} {'Status'}")
    print("-" * 88)

    failed_questions = []
    for item in details:
        qid = item["id"]
        cat = item["category"][:16]
        exp = str(item["expected_pages"]) if item["expected_pages"] else "None"
        ret = str(item["retrieved_pages"][:3])
        rec = "PASS" if item["recall_at_k"] else ("N/A" if item["recall_at_k"] is None else "FAIL")
        rr = f"{item['reciprocal_rank']:.2f}" if item["reciprocal_rank"] is not None else "N/A"
        cit = "PASS" if item["citation_correct"] else ("N/A" if item["citation_correct"] is None else "FAIL")
        cov = f"{item.get('fact_coverage', 0.0):.0f}%" if item["is_answerable"] else "N/A"

        status = "OK"
        if item.get("api_error"):
            status = "API_ERROR"
            failed_questions.append((qid, item["question"], f"API Error: {item.get('error_message')}"))
        elif item["is_answerable"]:
            reasons = []
            if not item["recall_at_k"]:
                reasons.append("Retrieval miss")
            if not item["citation_correct"]:
                reasons.append("Citation mismatch")
            if not item.get("facts_verified", True):
                reasons.append("Fact coverage low")
            if reasons:
                status = "FAILED"
                failed_questions.append((qid, item["question"], ", ".join(reasons)))
        else:
            if not item["refusal_correct"]:
                status = "FAILED"
                failed_questions.append((qid, item["question"], "Failed to refuse unanswerable question"))

        print(f"{qid:<5} {cat:<18} {exp:<10} {ret:<12} {rec:<8} {rr:<6} {cit:<10} {cov:<9} {status}")

    print("-" * 88)
    if failed_questions:
        print(f"\nFAILED / ERRORED QUESTIONS ({len(failed_questions)}):")
        for qid, qtext, reason in failed_questions:
            print(f"  - [{qid}] '{qtext}' -> {reason}")
    else:
        print("\nAll evaluated test questions passed their respective evaluation criteria!")

    if any(item.get("api_error") for item in details):
        print("\n[WARNING] One or more questions failed due to Gemini API rate limits or missing credentials.")
        print("          Ensure GEMINI_API_KEY is configured in backend/.env, or run with --mock-llm for offline test.")

    if show_diagnostics:
        print("\n" + "=" * 88)
        print("               DETAILED RETRIEVAL DIAGNOSTICS PER QUERY                 ")
        print("=" * 88)
        for item in details:
            qid = item["id"]
            question = item["question"]
            exp = item["expected_pages"]
            rec_status = "PASS" if item["recall_at_k"] else ("N/A" if item["recall_at_k"] is None else "FAIL")
            print(f"\n[{qid}] {question}")
            print(f"  Category: {item['category']} | Expected Pages: {exp or 'None (Unanswerable)'} | Retrieval: {rec_status}")
            print("  Retrieved Chunks:")
            for diag in item.get("retrieval_diagnostics", []):
                rank = diag["rank"]
                cid = diag["chunk_id"]
                fn = diag["filename"]
                pg = diag["page_number"]
                sim = diag["similarity_score"]
                dist = diag["distance"]
                snip = diag["text_snippet"].replace("\n", " ")
                print(f"    #{rank} | Page {pg} | Sim: {sim:.4f} | Dist: {dist:.4f} | File: {fn} | ID: {cid}")
                print(f"         Excerpt: \"{snip}\"")
            if item.get("answer"):
                first_line = item["answer"].split("\n")[0]
                if len(first_line) > 120:
                    first_line = first_line[:120] + "..."
                print(f"  Answer: \"{first_line}\"")
                print(f"  Citations: {item.get('cited_pages', [])}")
        print("=" * 88)
    else:
        print("\n[Tip] Run with --diagnostics (or -d) to view chunk text, chunk IDs, and similarity scores per query.")

    print("=" * 88 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval & Answer Quality")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--real-pipeline", action="store_true", help="Run full real pipeline: live Gemini LLM, persistent Chroma store, and local embedding model")
    parser.add_argument("--live-llm", action="store_true", help="Call live Gemini API (default: uses deterministic mock LLM)")
    parser.add_argument("--mock-llm", action="store_true", default=False, help="Use deterministic mock LLM for offline evaluation (active by default unless --live-llm/--real-pipeline is passed)")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use MockEmbeddingProvider for offline test environments")
    parser.add_argument("--diagnostics", "-d", action="store_true", help="Display detailed retrieval diagnostics (chunk ID, text snippet, similarity & distance scores)")
    parser.add_argument("--store-type", type=str, default="memory", choices=["memory", "chroma"], help="Vector store implementation (default: memory)")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing handbook into vector store")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Evaluate a small subset of questions (e.g. --limit 3)")
    parser.add_argument("--category", type=str, default=None, choices=["direct_single_page", "multi_section", "unanswerable"], help="Filter evaluation by question category")
    parser.add_argument("--question-ids", type=str, default=None, help="Comma-separated question IDs to evaluate (e.g. q01,q05,q16)")
    parser.add_argument("--dataset", type=str, default=os.path.join(BACKEND_DIR, "data", "eval_dataset.json"), help="Path to evaluation dataset JSON")
    parser.add_argument("--output", type=str, default=os.path.join(BACKEND_DIR, "data", "eval_results.json"), help="Path to save evaluation output JSON")

    args = parser.parse_args()

    # Generate test PDF if it doesn't already exist
    pdf_path = os.path.join(BACKEND_DIR, "data", "sample_documents", "company_handbook.pdf")
    if not os.path.exists(pdf_path):
        generate_handbook_pdf(pdf_path)

    # Resolve pipeline flags
    if args.real_pipeline:
        use_mock_llm = False
        use_mock_embeddings = False
        store_type = "chroma" if args.store_type == "memory" else args.store_type
    else:
        use_mock_llm = not args.live_llm
        use_mock_embeddings = args.mock_embeddings
        store_type = args.store_type

    q_ids = [q.strip() for q in args.question_ids.split(",")] if args.question_ids else None

    evaluator = RAGEvaluator(
        top_k=args.top_k,
        use_mock_llm=use_mock_llm,
        use_mock_embeddings=use_mock_embeddings,
        store_type=store_type,
    )

    results = await evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        limit=args.limit,
        category=args.category,
        question_ids=q_ids,
        reindex=args.reindex,
    )
    print_evaluation_report(results, show_diagnostics=args.diagnostics)

    # Save results to JSON file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed evaluation metrics saved to: {args.output}\n")


if __name__ == "__main__":
    asyncio.run(main())
