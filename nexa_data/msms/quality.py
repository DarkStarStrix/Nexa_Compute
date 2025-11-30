"""Quality ranking integration wrapper."""

from typing import Dict, List, Optional

try:
    from scripts.rank_data_quality import rank_data, RankerSpec
    from nexa_eval.clients import OpenRouterConfig
    RANKING_AVAILABLE = True
except ImportError:
    RANKING_AVAILABLE = False


class QualityRanker:
    """Wrapper for quality ranking functionality."""

    def __init__(self, model_id: str = "openai/gpt-4o-mini", dry_run: bool = False):
        """Initialize quality ranker.

        Args:
            model_id: Model ID for ranking
            dry_run: If True, skip API calls
        """
        if not RANKING_AVAILABLE:
            raise ImportError("Quality ranking dependencies not available")

        self.model_id = model_id
        self.dry_run = dry_run
        self.ranker = RankerSpec(model_id=model_id)
        self.client_config = None if dry_run else OpenRouterConfig(model=model_id)
        self.calls_made = 0

    def rank_spectrum(self, record: Dict) -> Optional[Dict]:
        """Rank a single spectrum.

        Args:
            record: Spectrum record

        Returns:
            Record with quality scores added, or None if failed
        """
        import pandas as pd

        df = pd.DataFrame([record])
        ranked_df = rank_data(
            df,
            ranker=self.ranker,
            client_config=self.client_config,
            dry_run=self.dry_run,
            max_workers=1,
        )

        if len(ranked_df) == 0:
            return None

        ranked_record = ranked_df.iloc[0].to_dict()
        self.calls_made += 1
        return ranked_record

    def rank_batch(self, records: List[Dict]) -> List[Dict]:
        """Rank a batch of spectra.

        Args:
            records: List of spectrum records

        Returns:
            List of records with quality scores
        """
        import pandas as pd

        df = pd.DataFrame(records)
        ranked_df = rank_data(
            df,
            ranker=self.ranker,
            client_config=self.client_config,
            dry_run=self.dry_run,
            max_workers=4,
        )

        self.calls_made += len(records)
        return ranked_df.to_dict("records")

