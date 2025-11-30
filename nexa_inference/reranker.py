"""Reranking utilities for search results."""

from __future__ import annotations

from typing import Dict, List, Optional


class Reranker:
    """Apply metadata filters and learned reranking."""

    def __init__(self, rt_tolerance: float = 0.5) -> None:
        self.rt_tolerance = rt_tolerance

    def rerank(
        self,
        results: List[Dict],
        query_metadata: Dict,
        *,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        filtered = self._filter_by_rt_delta(results, query_metadata)
        filtered = self._filter_by_metadata(filtered, metadata_filters)
        return self._sort_by_score(filtered)

    def _filter_by_rt_delta(self, results: List[Dict], query_metadata: Dict) -> List[Dict]:
        if "retention_time" not in query_metadata:
            return results
        query_rt = float(query_metadata["retention_time"])
        filtered = []
        for result in results:
            rt = result.get("metadata", {}).get("retention_time")
            if rt is None or abs(float(rt) - query_rt) <= self.rt_tolerance:
                filtered.append(result)
        return filtered

    def _filter_by_metadata(self, results: List[Dict], metadata_filters: Optional[Dict[str, str]]) -> List[Dict]:
        if not metadata_filters:
            return results
        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            if all(str(metadata.get(key)) == str(value) for key, value in metadata_filters.items()):
                filtered.append(result)
        return filtered

    def _sort_by_score(self, results: List[Dict]) -> List[Dict]:
        return sorted(results, key=lambda r: r.get("score", 0), reverse=True)

