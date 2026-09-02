"""Слой трудной проблемы (промпт hardproblem_v1), независимый проход по каждому тексту.

python -m pipeline.classify_hardproblem --config config.yaml [--model slug] [--repeats 2]
Результат: runs/<slug>_<date>/hardproblem_r<k>.json
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .aggregate import aggregate_hp, validate_hp_response
from .runner import build_argparser, run_kind


def process_hardproblem(data: Optional[Dict[str, Any]], row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    records, stats = validate_hp_response(data, row["text"], cfg.get("verification", {}))
    agg = aggregate_hp(data, records)
    out: Dict[str, Any] = {"hp_spans": records, "filter_stats": stats,
                           "notes": (data or {}).get("notes", "") if isinstance(data, dict) else ""}
    out.update(agg)
    return out


def main(argv=None) -> None:
    args = build_argparser("Слой трудной проблемы (v1)").parse_args(argv)
    run_kind("hardproblem", process_hardproblem, args)


if __name__ == "__main__":
    main()
