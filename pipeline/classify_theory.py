"""Классификация теоретических рамок (промпт theory_v3).

python -m pipeline.classify_theory --config config.yaml [--model slug] [--repeats 2] [--limit N]
Результат: runs/<slug>_<date>/theory_r<k>.json + meta.json + cache/ (сырые ответы).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .aggregate import aggregate_theory, validate_theory_response
from .runner import build_argparser, run_kind


def process_theory(data: Optional[Dict[str, Any]], row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Валидация evidence + агрегация для одного текста. data=None при ошибке разбора → UND."""
    records, stats = validate_theory_response(data, row["text"], cfg.get("verification", {}))
    agg = aggregate_theory(records, int(row["n_words"]), cfg.get("aggregation", {}))
    out: Dict[str, Any] = {"evidence": records, "filter_stats": stats,
                           "notes": (data or {}).get("notes", "") if isinstance(data, dict) else ""}
    out.update(agg)
    return out


def main(argv=None) -> None:
    args = build_argparser("Классификация теоретических рамок (v3)").parse_args(argv)
    run_kind("theory", process_theory, args)


if __name__ == "__main__":
    main()
