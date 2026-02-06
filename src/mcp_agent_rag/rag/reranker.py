"""Reranker for improving retrieval precision.

Provides reranking capabilities to improve the precision of retrieval results
by re-scoring based on query-document relevance.
"""

from typing import Dict, List, Tuple

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class SimpleReranker:
    """Simple lexical reranker based on term overlap and position.
    
    This is a lightweight reranker that doesn't require additional models.
    For production use, consider using a cross-encoder model.
    """

    def __init__(self):
        """Initialize simple reranker."""
        pass

    def rerank(
        self,
        query: str,
        results: List[Tuple[float, Dict]],
        top_k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """Rerank results based on query-document relevance.

        Args:
            query: Search query
            results: List of (score, document) tuples
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Extract query terms (simple tokenization)
        query_terms = self._tokenize(query.lower())
        query_set = set(query_terms)

        reranked = []
        for score, doc in results:
            text = doc.get("text", "").lower()
            doc_terms = self._tokenize(text)

            # Compute relevance factors
            relevance_score = self._compute_relevance(
                query_terms, query_set, doc_terms, text
            )

            # Combine with original score
            # Original score is typically from vector similarity or BM25
            # Weight: 70% original, 30% relevance
            combined_score = 0.7 * score + 0.3 * relevance_score

            reranked.append((combined_score, doc))

        # Sort by combined score and return top k
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple split on whitespace and punctuation
        import re

        tokens = re.findall(r"\w+", text)
        return [t for t in tokens if len(t) > 2]  # Filter short tokens

    def _compute_relevance(
        self,
        query_terms: List[str],
        query_set: set,
        doc_terms: List[str],
        doc_text: str,
    ) -> float:
        """Compute relevance score.

        Args:
            query_terms: Query terms (list)
            query_set: Query terms (set)
            doc_terms: Document terms
            doc_text: Full document text

        Returns:
            Relevance score [0, 1]
        """
        if not query_terms or not doc_terms:
            return 0.0

        # Factor 1: Term overlap (Jaccard similarity)
        doc_set = set(doc_terms)
        intersection = len(query_set & doc_set)
        union = len(query_set | doc_set)
        jaccard = intersection / union if union > 0 else 0.0

        # Factor 2: Query term coverage
        coverage = intersection / len(query_set) if query_set else 0.0

        # Factor 3: Term position bonus (early matches are better)
        position_score = 0.0
        for term in query_set:
            if term in doc_text:
                # Find first occurrence position
                pos = doc_text.find(term)
                # Normalize by doc length, invert so early = high score
                if len(doc_text) > 0:
                    position_score += 1.0 - (pos / len(doc_text))

        position_score /= len(query_set) if query_set else 1.0

        # Factor 4: Exact phrase match bonus
        phrase_bonus = 0.0
        query_phrase = " ".join(query_terms)
        if query_phrase in doc_text:
            phrase_bonus = 0.5

        # Combine factors
        relevance = (
            0.3 * jaccard + 0.3 * coverage + 0.2 * position_score + 0.2 * phrase_bonus
        )

        return relevance


class MMRReranker:
    """Maximal Marginal Relevance (MMR) reranker for diversity.
    
    Reranks results to maximize both relevance and diversity,
    reducing redundancy in the result set.
    """

    def __init__(self, lambda_param: float = 0.5):
        """Initialize MMR reranker.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
                         Default 0.5 is equal weight
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        query: str,
        results: List[Tuple[float, Dict]],
        top_k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """Rerank using MMR for diversity.

        Args:
            query: Search query (not used in simple version)
            results: List of (score, document) tuples
            top_k: Number of diverse results to select

        Returns:
            Reranked diverse results
        """
        if not results or len(results) <= top_k:
            return results

        # Start with the highest scoring document
        selected = [results[0]]
        remaining = list(results[1:])

        # Iteratively select documents that are relevant but diverse
        while len(selected) < top_k and remaining:
            best_mmr_score = -float("inf")
            best_idx = 0

            for idx, (score, doc) in enumerate(remaining):
                # Compute max similarity to already selected documents
                max_sim = self._max_similarity(doc, selected)

                # MMR score = λ * relevance - (1-λ) * similarity
                mmr_score = self.lambda_param * score - (1 - self.lambda_param) * max_sim

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            # Add best MMR document to selected
            selected.append(remaining.pop(best_idx))

        return selected

    def _max_similarity(
        self, doc: Dict, selected: List[Tuple[float, Dict]]
    ) -> float:
        """Compute maximum similarity to selected documents.

        Args:
            doc: Document to compare
            selected: List of selected (score, doc) tuples

        Returns:
            Maximum similarity score
        """
        if not selected:
            return 0.0

        doc_text = doc.get("text", "")
        doc_tokens = set(self._tokenize(doc_text))

        max_sim = 0.0
        for _, sel_doc in selected:
            sel_text = sel_doc.get("text", "")
            sel_tokens = set(self._tokenize(sel_text))

            # Jaccard similarity
            if doc_tokens or sel_tokens:
                intersection = len(doc_tokens & sel_tokens)
                union = len(doc_tokens | sel_tokens)
                similarity = intersection / union if union > 0 else 0.0
                max_sim = max(max_sim, similarity)

        return max_sim

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        import re

        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if len(t) > 2]


class ChainReranker:
    """Chain multiple rerankers together.
    
    Applies rerankers in sequence, passing output of one
    as input to the next.
    """

    def __init__(self, rerankers: List):
        """Initialize chain reranker.

        Args:
            rerankers: List of reranker instances
        """
        self.rerankers = rerankers

    def rerank(
        self,
        query: str,
        results: List[Tuple[float, Dict]],
        top_k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """Apply rerankers in sequence.

        Args:
            query: Search query
            results: Initial results
            top_k: Final number of results

        Returns:
            Reranked results
        """
        current_results = results

        for reranker in self.rerankers:
            # Each reranker processes all results, final top_k at end
            k = top_k if reranker == self.rerankers[-1] else len(current_results)
            current_results = reranker.rerank(query, current_results, top_k=k)

        return current_results[:top_k]
