"""BM25-style tool search engine for progressive tool discovery."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchDocument:
    """Indexed tool document."""

    name: str
    domain: str
    terms: dict[str, int]  # term -> raw count
    field_length: int  # total token count


@dataclass(frozen=True)
class SearchHit:
    """Search result with BM25 relevance score."""

    tool_name: str
    domain: str
    score: float
    description: str


_SPLIT_RE = re.compile(r"[_\s/\-]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, splitting on underscores/spaces/hyphens."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


class ToolSearchEngine:
    """BM25-style tool search engine.

    Improves on simple word-count matching with:
    - Term frequency (TF) with saturation
    - Inverse document frequency (IDF)
    - Document length normalization
    - Field weighting (name matches boost, tag matches boost)

    Usage: call reindex_all(registry) after all tools are registered,
    then use search() for queries.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._documents: dict[str, SearchDocument] = {}
        self._descriptions: dict[str, str] = {}
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0

    def reindex_all(self, registry) -> int:
        """Index all tools from a ToolRegistry. Returns count indexed.

        Computes IDF across entire corpus.
        """
        self._documents.clear()
        self._descriptions.clear()

        for name, info in registry._tools.items():
            func = info["func"]
            docstring = func.__doc__ or ""
            first_line = docstring.strip().split("\n")[0].strip() if docstring else ""
            tags = info.get("tags", [])
            domain = info["domain"]

            # Tokenize each field separately for weighting
            name_tokens = _tokenize(name)
            desc_tokens = _tokenize(first_line)
            tag_tokens = []
            for tag in tags:
                tag_tokens.extend(_tokenize(tag))

            # Build weighted term counts:
            # name tokens get 3x, tag tokens get 2x, description tokens get 1x
            terms: dict[str, int] = {}
            for t in name_tokens:
                terms[t] = terms.get(t, 0) + 3
            for t in tag_tokens:
                terms[t] = terms.get(t, 0) + 2
            for t in desc_tokens:
                terms[t] = terms.get(t, 0) + 1

            field_length = len(name_tokens) * 3 + len(tag_tokens) * 2 + len(desc_tokens)

            doc = SearchDocument(
                name=name,
                domain=domain,
                terms=terms,
                field_length=field_length,
            )
            self._documents[name] = doc
            self._descriptions[name] = first_line

        # Compute IDF for all terms
        self._compute_idf()
        return len(self._documents)

    def _compute_idf(self) -> None:
        """Compute IDF values and average document length."""
        n = len(self._documents)
        if n == 0:
            self._idf = {}
            self._avg_dl = 0.0
            return

        # Collect all terms and their document frequencies
        df: dict[str, int] = {}
        total_dl = 0
        for doc in self._documents.values():
            total_dl += doc.field_length
            for term in doc.terms:
                df[term] = df.get(term, 0) + 1

        self._avg_dl = total_dl / n

        # IDF = log((N - n + 0.5) / (n + 0.5) + 1)
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)

    def search(
        self, query: str, domain: str | None = None, limit: int = 10
    ) -> list[SearchHit]:
        """Search indexed tools using BM25 scoring.

        Optional domain filter. Returns up to limit results sorted by score.
        """
        query_terms = _tokenize(query)

        results: list[SearchHit] = []

        for name, doc in self._documents.items():
            # Domain filter
            if domain is not None and doc.domain != domain:
                continue

            # If no query terms, return all matching docs (domain-only search)
            if not query_terms:
                results.append(
                    SearchHit(
                        tool_name=name,
                        domain=doc.domain,
                        score=0.0,
                        description=self._descriptions[name],
                    )
                )
                continue

            # BM25 scoring
            score = 0.0
            dl = doc.field_length
            avgdl = self._avg_dl if self._avg_dl > 0 else 1.0

            for term in query_terms:
                if term not in doc.terms:
                    continue
                tf = doc.terms[term]
                idf = self._idf.get(term, 0.0)
                # BM25 formula
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / avgdl)
                score += idf * numerator / denominator

            if score > 0:
                results.append(
                    SearchHit(
                        tool_name=name,
                        domain=doc.domain,
                        score=score,
                        description=self._descriptions[name],
                    )
                )

        # Sort by score descending
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:limit]
