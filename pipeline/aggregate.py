"""Валидация ответов модели и агрегация: n_spans, density, primary/secondary, is_hybrid, confidence.

Правила (ТЗ v3, раздел 4):
  - score класса = число независимых верифицированных фрагментов с mapping (n_spans);
  - density = n_spans / words * 100;
  - primary = класс с максимальным n_spans; при равенстве — по покрытию текста
    фрагментами (слов в фрагментах), затем по порядку классов; факт равенства
    фиксируется флагом primary_tie;
  - secondary = все классы с n_spans >= secondary_min_spans и density >= порога;
    secondary_raw — то же без порога плотности (для отчёта о влиянии длины);
  - confidence вычисляется из n_spans primary-класса, у модели не запрашивается;
  - UND, если ни у одного класса нет ни одного фрагмента.

CLI: python -m pipeline.aggregate --config config.yaml
  Собирает все v3-прогоны в reports/results_theory_long.csv и results_hardproblem_long.csv,
  пишет отчёт о фильтрации свидетельств и отчёт «длина × число меток» до и после нормировки.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import (ALL_CLASSES, CLASSES, HP_STANCES, HP_SUBJECTS, LEVELS, count_words,
                     list_runs, load_config, load_json, reports_dir)
from .verify_quotes import select_independent, verify_span


# --------------------------------------------------------------------------- #
# Теоретические рамки
# --------------------------------------------------------------------------- #

def _is_true(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def validate_theory_response(data: Optional[Dict[str, Any]], text: str, vcfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Нормализует evidence из ответа модели, помечает отброшенные с причиной.

    Возвращает (records, stats). records — все фрагменты с полем kept и drop_reason.
    Причины отбрасывания: malformed, invalid_class, span_too_short, no_mapping,
    exclusion_not_checked, unverified, duplicate.
    """
    threshold = float(vcfg.get("token_overlap_threshold", 0.7))
    require_exact = bool(vcfg.get("require_exact", False))
    min_words = int(vcfg.get("min_span_words", 2))
    require_checked = bool(vcfg.get("require_exclusion_checked", True))
    items = data.get("evidence") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = []
    records: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            records.append({"kept": False, "drop_reason": "malformed", "raw": item})
            continue
        cls = str(item.get("class", "")).strip().upper()
        span = str(item.get("span", "") or "").strip()
        mapping = item.get("mapping")
        level = str(item.get("level", "") or "").strip().lower()
        rec: Dict[str, Any] = {
            "class": cls,
            "span": span,
            "mapping": mapping if isinstance(mapping, dict) else {"source": "", "target": ""},
            "level": level if level in LEVELS else "unspecified",
            "exclusion_checked": _is_true(item.get("exclusion_checked")),
            "reasoning": str(item.get("reasoning", "") or ""),
            "kept": False,
            "drop_reason": None,
            "verification": None,
        }
        if cls not in CLASSES:
            rec["drop_reason"] = "invalid_class"
        elif count_words(span) < min_words:
            rec["drop_reason"] = "span_too_short"
        elif not (isinstance(mapping, dict)
                  and str(mapping.get("source", "") or "").strip()
                  and str(mapping.get("target", "") or "").strip()):
            rec["drop_reason"] = "no_mapping"
        elif require_checked and not rec["exclusion_checked"]:
            rec["drop_reason"] = "exclusion_not_checked"
        else:
            v = verify_span(span, text, threshold=threshold, require_exact=require_exact)
            rec["verification"] = v
            rec["loc"] = v.get("loc")
            if not v["verified"]:
                rec["drop_reason"] = "unverified"
        records.append(rec)

    # независимость фрагментов внутри класса
    for cls in CLASSES:
        cand = [r for r in records if r.get("class") == cls and r["drop_reason"] is None]
        kept, dup = select_independent(cand)
        for r in kept:
            r["kept"] = True
        for r in dup:
            r["drop_reason"] = "duplicate"
    for r in records:
        r.pop("loc", None)

    dropped = Counter(r["drop_reason"] for r in records if not r["kept"])
    stats = {
        "n_raw": len(records),
        "n_kept": sum(1 for r in records if r["kept"]),
        "dropped": dict(dropped),
    }
    return records, stats


def aggregate_theory(records: List[Dict[str, Any]], n_words: int, acfg: Dict[str, Any]) -> Dict[str, Any]:
    kept = [r for r in records if r.get("kept")]
    min_spans = int(acfg.get("secondary_min_spans", 2))
    thr = float(acfg.get("secondary_density_threshold", 1.0))
    cthr = acfg.get("confidence_thresholds", {}) or {}
    med, high = int(cthr.get("medium", 2)), int(cthr.get("high", 3))

    n_spans = {c: 0 for c in CLASSES}
    coverage = {c: 0 for c in CLASSES}
    for r in kept:
        n_spans[r["class"]] += 1
        coverage[r["class"]] += count_words(r["span"])
    words = max(int(n_words or 0), 1)
    density = {c: round(n_spans[c] / words * 100, 4) for c in CLASSES}

    present = [c for c in CLASSES if n_spans[c] > 0]
    if present:
        ranked = sorted(present, key=lambda c: (-n_spans[c], -coverage[c], CLASSES.index(c)))
        primary = ranked[0]
        primary_tie = any(
            n_spans[c] == n_spans[primary] and coverage[c] == coverage[primary]
            for c in ranked[1:]
        )
    else:
        primary, primary_tie = "UND", False

    secondary_raw = [c for c in CLASSES if c != primary and n_spans[c] >= min_spans]
    secondary = [c for c in secondary_raw if density[c] >= thr]
    n_primary = n_spans.get(primary, 0) if primary != "UND" else 0
    if primary == "UND":
        confidence = "none"
    elif n_primary >= high:
        confidence = "high"
    elif n_primary >= med:
        confidence = "medium"
    else:
        confidence = "low"
    labels = [primary] + secondary if primary != "UND" else ["UND"]
    return {
        "n_spans": n_spans,
        "density": density,
        "coverage_words": coverage,
        "n_spans_total": sum(n_spans.values()),
        "primary": primary,
        "primary_tie": primary_tie,
        "secondary": secondary,
        "secondary_raw": secondary_raw,
        "is_hybrid": bool(secondary),
        "is_hybrid_raw": bool(secondary_raw),
        "confidence": confidence,
        "labels": labels,
    }


# --------------------------------------------------------------------------- #
# Трудная проблема
# --------------------------------------------------------------------------- #

def validate_hp_response(data: Optional[Dict[str, Any]], text: str, vcfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    threshold = float(vcfg.get("token_overlap_threshold", 0.7))
    require_exact = bool(vcfg.get("require_exact", False))
    items = data.get("hp_spans") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = []
    records: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            item = {"span": item, "subject": "неясно"}
        if not isinstance(item, dict):
            records.append({"kept": False, "drop_reason": "malformed", "raw": item})
            continue
        span = str(item.get("span", "") or "").strip()
        subject = str(item.get("subject", "") or "").strip().lower()
        rec = {
            "span": span,
            "subject": subject if subject in HP_SUBJECTS else "неясно",
            "reasoning": str(item.get("reasoning", "") or ""),
            "kept": False, "drop_reason": None, "verification": None,
        }
        if not span:
            rec["drop_reason"] = "empty_span"
        else:
            v = verify_span(span, text, threshold=threshold, require_exact=require_exact)
            rec["verification"] = v
            rec["loc"] = v.get("loc")
            if not v["verified"]:
                rec["drop_reason"] = "unverified"
            elif rec["subject"] == "человек":
                rec["drop_reason"] = "human_subject"
        records.append(rec)
    cand = [r for r in records if r["drop_reason"] is None]
    kept, dup = select_independent(cand)
    for r in kept:
        r["kept"] = True
    for r in dup:
        r["drop_reason"] = "duplicate"
    for r in records:
        r.pop("loc", None)
    dropped = Counter(r["drop_reason"] for r in records if not r["kept"])
    return records, {"n_raw": len(records), "n_kept": len(kept), "dropped": dict(dropped)}


def aggregate_hp(data: Optional[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """hp_level модели принимается только при наличии верифицированных фрагментов не о человеке."""
    kept = [r for r in records if r.get("kept")]
    flags: List[str] = []
    try:
        level_model = int(data.get("hp_level", 0)) if isinstance(data, dict) else 0
    except (TypeError, ValueError):
        level_model, flags = 0, ["level_not_int"]
    level_model = max(0, min(2, level_model))
    stance_model = data.get("hp_stance") if isinstance(data, dict) else None
    stance_model = str(stance_model).strip().lower() if stance_model else None
    if stance_model not in HP_STANCES:
        stance_model = None

    if not kept:
        level = 0
        if level_model > 0:
            flags.append("level_reset_no_verified_spans")
    else:
        level = level_model
        if level == 0:
            flags.append("spans_without_level")
    stance = stance_model if level >= 1 else None
    if level >= 1 and stance is None:
        flags.append("stance_missing")
    subjects = Counter(r["subject"] for r in kept)
    subject = subjects.most_common(1)[0][0] if subjects else None
    return {
        "hp_level": level,
        "hp_stance": stance,
        "hp_subject": subject,
        "hp_n_spans": len(kept),
        "hp_level_model": level_model,
        "hp_stance_model": stance_model,
        "hp_flags": flags,
    }


# --------------------------------------------------------------------------- #
# Длинные таблицы по всем прогонам
# --------------------------------------------------------------------------- #

def theory_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: rec.get(k) for k in ("id", "title", "source", "n_words", "model", "slug", "repeat",
                                   "prompt_hash", "status", "primary", "primary_tie", "is_hybrid",
                                   "is_hybrid_raw", "confidence", "n_spans_total")}
    row["secondary"] = "|".join(rec.get("secondary") or [])
    row["secondary_raw"] = "|".join(rec.get("secondary_raw") or [])
    row["labels"] = "|".join(rec.get("labels") or ["UND"])
    row["n_labels"] = len(rec.get("labels") or []) if rec.get("primary") != "UND" else 0
    row["n_labels_raw"] = (1 + len(rec.get("secondary_raw") or [])) if rec.get("primary") != "UND" else 0
    for c in CLASSES:
        row[f"n_{c}"] = (rec.get("n_spans") or {}).get(c, 0)
        row[f"d_{c}"] = (rec.get("density") or {}).get(c, 0.0)
    fs = rec.get("filter_stats") or {}
    row["ev_raw"] = fs.get("n_raw", 0)
    row["ev_kept"] = fs.get("n_kept", 0)
    for reason in ("no_mapping", "unverified", "exclusion_not_checked", "duplicate", "span_too_short", "invalid_class"):
        row[f"drop_{reason}"] = (fs.get("dropped") or {}).get(reason, 0)
    return row


def hp_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: rec.get(k) for k in ("id", "title", "source", "n_words", "model", "slug", "repeat",
                                   "prompt_hash", "status", "hp_level", "hp_stance", "hp_subject",
                                   "hp_n_spans", "hp_level_model", "hp_stance_model")}
    row["hp_flags"] = "|".join(rec.get("hp_flags") or [])
    fs = rec.get("filter_stats") or {}
    row["hp_ev_raw"] = fs.get("n_raw", 0)
    row["hp_ev_kept"] = fs.get("n_kept", 0)
    row["hp_drop_unverified"] = (fs.get("dropped") or {}).get("unverified", 0)
    row["hp_drop_human"] = (fs.get("dropped") or {}).get("human_subject", 0)
    return row


def load_long(cfg: Dict[str, Any], kind: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    conv = theory_row if kind == "theory" else hp_row
    for run in list_runs(cfg, kind):
        for rep, path in run["files"].items():
            for rec in load_json(path):
                r = conv(rec)
                r["run"] = run["run_dir"].name
                r.setdefault("slug", run["meta"].get("slug"))
                r.setdefault("model", run["meta"].get("model"))
                r["repeat"] = rep
                rows.append(r)
    return pd.DataFrame(rows)


def filter_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "Нет прогонов.\n"
    cols = [c for c in df.columns if c.startswith("drop_")]
    g = df.groupby(["run", "repeat"]).agg(texts=("id", "count"), ev_raw=("ev_raw", "sum"), ev_kept=("ev_kept", "sum"),
                                          **{c: (c, "sum") for c in cols}).reset_index()
    g["share_dropped"] = ((g["ev_raw"] - g["ev_kept"]) / g["ev_raw"].replace(0, np.nan)).round(3)
    lines = ["# Фильтрация свидетельств (theory v3)", "",
             "Каждая строка — прогон (модель × повтор). Доля отброшенных = (ev_raw − ev_kept) / ev_raw.", "",
             g.to_markdown(index=False), ""]
    return "\n".join(lines)


def length_report(df: pd.DataFrame) -> str:
    """Корреляция длины с числом меток до (secondary_raw) и после (secondary с порогом плотности) нормировки."""
    if df.empty:
        return "Нет прогонов.\n"
    from scipy import stats as st
    rows = []
    for (run, rep), g in df.groupby(["run", "repeat"]):
        g = g[g["status"] == "ok"]
        if len(g) < 5:
            continue
        r_raw, p_raw = st.spearmanr(g["n_words"], g["n_labels_raw"])
        r_new, p_new = st.spearmanr(g["n_words"], g["n_labels"])
        rows.append({
            "run": run, "repeat": rep, "n": len(g),
            "spearman_before": round(float(r_raw), 3), "p_before": round(float(p_raw), 4),
            "spearman_after": round(float(r_new), 3), "p_after": round(float(p_new), 4),
            "median_words_secondary_before": g.loc[g["is_hybrid_raw"] == True, "n_words"].median(),
            "median_words_no_secondary_before": g.loc[g["is_hybrid_raw"] != True, "n_words"].median(),
            "median_words_secondary_after": g.loc[g["is_hybrid"] == True, "n_words"].median(),
            "median_words_no_secondary_after": g.loc[g["is_hybrid"] != True, "n_words"].median(),
            "n_hybrid_before": int((g["is_hybrid_raw"] == True).sum()),
            "n_hybrid_after": int((g["is_hybrid"] == True).sum()),
        })
    out = pd.DataFrame(rows)
    lines = ["# Длина текста и число меток: до и после нормировки", "",
             "«До» — вторичная метка по правилу n_spans ≥ 2 без порога плотности; "
             "«после» — с порогом density ≥ порога из config.yaml. Spearman ρ между числом слов и числом меток.", "",
             out.to_markdown(index=False) if not out.empty else "недостаточно данных", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Агрегация v3-прогонов в длинные таблицы и отчёты")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rd = reports_dir(cfg)

    th = load_long(cfg, "theory")
    th.to_csv(rd / "results_theory_long.csv", index=False)
    (rd / "filter_report.md").write_text(filter_report(th), encoding="utf-8")
    (rd / "length_report.md").write_text(length_report(th), encoding="utf-8")
    hp = load_long(cfg, "hardproblem")
    hp.to_csv(rd / "results_hardproblem_long.csv", index=False)
    print(f"theory: {len(th)} строк из {th['run'].nunique() if not th.empty else 0} прогонов; "
          f"hardproblem: {len(hp)} строк. Отчёты в {rd}")


if __name__ == "__main__":
    main()
