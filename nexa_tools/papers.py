"""Lightweight integrations for scholarly search and metadata fetch."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, List


LOGGER = logging.getLogger(__name__)


class PaperSearcher:
    """Search scholarly corpora using simple HTTP APIs."""

    _CROSSREF_ENDPOINT = "https://api.crossref.org/works"

    def __init__(self, *, user_agent: str = "NexaTools/1.0") -> None:
        self._user_agent = user_agent

    def search(self, query: str, *, top_k: int, corpus: str) -> List[Dict[str, Any]]:
        """Query the requested corpus for papers relevant to the prompt."""

        if not query.strip():
            raise ValueError("Query string must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        corpus = corpus.lower()
        if corpus != "crossref":
            raise NotImplementedError(f"Corpus '{corpus}' is not supported yet.")

        params = {"query": query, "rows": str(top_k)}
        url = f"{self._CROSSREF_ENDPOINT}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        items = payload.get("message", {}).get("items", [])
        return [self._normalise_crossref_item(item) for item in items]

    @staticmethod
    def _normalise_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce Crossref API responses into a consistent schema."""

        authors = []
        for author in item.get("author", []):
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            if given or family:
                authors.append(" ".join(part for part in [given, family] if part))
        doi = item.get("DOI")
        return {
            "title": item.get("title", [""])[0] if item.get("title") else "",
            "authors": authors,
            "year": _first_int(item.get("issued", {}).get("date-parts", [[None]])[0]),
            "doi": doi,
            "url": item.get("URL"),
            "abstract": item.get("abstract"),
            "source": "crossref",
        }


class PaperFetcher:
    """Fetch canonical metadata for a paper identified by DOI."""

    _CROSSREF_WORKS_ENDPOINT = "https://api.crossref.org/works"

    def __init__(self, *, user_agent: str = "NexaTools/1.0") -> None:
        self._user_agent = user_agent

    def fetch(self, doi: str) -> Dict[str, Any]:
        """Retrieve metadata for the requested DOI."""

        cleaned = doi.strip()
        if not cleaned:
            raise ValueError("DOI must not be empty.")

        encoded = urllib.parse.quote(cleaned)
        url = f"{self._CROSSREF_WORKS_ENDPOINT}/{encoded}"
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        message = payload.get("message", {})
        authors = []
        for author in message.get("author", []):
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            if given or family:
                authors.append(" ".join(part for part in [given, family] if part))

        return {
            "title": message.get("title", [""])[0] if message.get("title") else "",
            "abstract": message.get("abstract"),
            "year": _first_int(message.get("issued", {}).get("date-parts", [[None]])[0]),
            "bibtex": message.get("citation", ""),
            "authors": authors,
            "doi": message.get("DOI"),
            "url": message.get("URL"),
            "source": "crossref",
        }


def _first_int(values: List[Any]) -> int | None:
    """Extract the first integer-like value from a list."""

    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            LOGGER.debug("Unable to coerce '%s' to int when parsing year.", value)
            continue
    return None


__all__ = ["PaperSearcher", "PaperFetcher"]

