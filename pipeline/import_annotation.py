"""Импорт заполненной таблицы ручной разметки: κ человек/человек, человек/модель, человек/consensus,
таблица разногласий для согласительной сессии.

python -m pipeline.import_annotation --config config.yaml [--input reports/annotation_sheet_filled.xlsx]
Выход: reports/human_agreement.md, reports/human_labels.csv, reports/disagreements.csv, reports/author_type.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .agreement import cohen_kappa, jaccard, krippendorff_alpha, confusion
from .common import ALL_CLASSES, CLASSES, load_config, reports_dir, resolve

CODERS = ("A", "B")


def _norm_class(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    return s if s in ALL_CLASSES else (None if not s else s)


def _norm_labels(primary: Optional[str], secondary: Any) -> Optional[List[str]]:
    if primary is None:
        return None
    if primary == "UND":
        return ["UND"]
    sec = [] if secondary is None or (isinstance(secondary, float) and np.isnan(secondary)) else \
        [s.strip().upper() for s in str(secondary).replace(",", "|").split("|") if s.strip()]
    return [primary] + [s for s in sec if s in CLASSES and s != primary]


def _norm_level(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def read_filled(path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    ann = pd.read_excel(xl, "annotation")
    author = pd.read_excel(xl, "author_type") if "author_type" in xl.sheet_names else None
    out = pd.DataFrame({"id": ann["id"].astype(int)})
    for c in CODERS:
        out[f"theory_primary_{c}"] = ann.get(f"theory_primary_{c}", pd.Series([None] * len(ann))).map(_norm_class)
        out[f"theory_labels_{c}"] = [
            _norm_labels(p, s) for p, s in zip(out[f"theory_primary_{c}"], ann.get(f"theory_secondary_{c}", pd.Series([None] * len(ann))))
        ]
        out[f"hp_level_{c}"] = ann.get(f"hp_level_{c}", pd.Series([None] * len(ann))).map(_norm_level)
        st = ann.get(f"hp_stance_{c}", pd.Series([None] * len(ann)))
        out[f"hp_stance_{c}"] = st.map(lambda v: str(v).strip().lower() if isinstance(v, str) and v.strip() else None)
        out[f"note_{c}"] = ann.get(f"note_{c}", pd.Series([""] * len(ann)))
    for col in ("source", "title", "text", "annotate_theory", "annotate_hp"):
        if col in ann.columns:
            out[col] = ann[col]
    return {"annotation": out, "author_type": author}


def _kappa_row(name_a: str, a: pd.Series, name_b: str, b: pd.Series, weights: Optional[str] = None) -> Dict[str, Any]:
    both = pd.DataFrame({"a": a, "b": b}).dropna()
    if both.empty:
        return {"A": name_a, "B": name_b, "kappa": float("nan"), "exact": float("nan"), "n": 0}
    row = {"A": name_a, "B": name_b, "kappa": round(cohen_kappa(both["a"], both["b"]), 3),
           "exact": round(float((both["a"] == both["b"]).mean()), 3), "n": len(both)}
    if weights:
        row["kappa_weighted"] = round(cohen_kappa(both["a"], both["b"], weights=weights), 3)
    return row


def build(cfg: Dict[str, Any], input_path, out_dir) -> None:
    data = read_filled(input_path)
    ann = data["annotation"]
    cons_path = out_dir / "consensus.csv"
    cons = pd.read_csv(cons_path) if cons_path.exists() else None
    lines = ["# Согласие людей между собой и с моделями", ""]

    th = ann[ann.get("annotate_theory", 1) == 1] if "annotate_theory" in ann.columns else ann
    a, b = th["theory_primary_A"], th["theory_primary_B"]
    rows = [_kappa_row("human_A", a, "human_B", b)]
    both = th[["theory_primary_A", "theory_primary_B"]].dropna()
    lines += ["## Теоретическая рамка (primary)", "",
              f"Размечено обоими: {len(both)} из {len(th)} текстов подвыборки.", ""]
    if len(both):
        lines += ["Матрица согласия A (строки) × B (столбцы):", "",
                  confusion(both["theory_primary_A"], both["theory_primary_B"], ALL_CLASSES).to_markdown(), ""]
        jl = [jaccard(x, y) for x, y in zip(th["theory_labels_A"], th["theory_labels_B"]) if x is not None and y is not None]
        lines += [f"Multi-label Jaccard A/B (primary + secondary): {np.mean(jl):.3f}" if jl else "", ""]
    model_cols = []
    if cons is not None:
        m = th.merge(cons, on="id", how="left")
        model_cols = [c for c in cons.columns if c.startswith("primary_")]
        for coder in CODERS:
            rows.append(_kappa_row(f"human_{coder}", m[f"theory_primary_{coder}"], "consensus", m["consensus_primary"]))
            for mc in model_cols:
                rows.append(_kappa_row(f"human_{coder}", m[f"theory_primary_{coder}"], mc.replace("primary_", ""), m[mc]))
        units = m[["theory_primary_A", "theory_primary_B"] + model_cols].values.tolist()
        lines += [f"Krippendorff α (люди + модели): {krippendorff_alpha(units):.3f}", ""]
    lines += [pd.DataFrame(rows).to_markdown(index=False), ""]

    # hp
    hpa = ann[ann.get("annotate_hp", 1) == 1] if "annotate_hp" in ann.columns else ann
    hrows = [_kappa_row("human_A", hpa["hp_level_A"], "human_B", hpa["hp_level_B"], weights="quadratic")]
    if cons is not None and "hp_level" in cons.columns:
        m = hpa.merge(cons, on="id", how="left")
        hp_model_cols = [c for c in cons.columns if c.startswith("hp_level_")]
        for coder in CODERS:
            hrows.append(_kappa_row(f"human_{coder}", m[f"hp_level_{coder}"], "consensus", m["hp_level"], weights="quadratic"))
            for mc in hp_model_cols:
                hrows.append(_kappa_row(f"human_{coder}", m[f"hp_level_{coder}"], mc.replace("hp_level_", ""), m[mc], weights="quadratic"))
        stance = m[["hp_stance_A", "hp_stance_B"]].dropna()
        if len(stance):
            hrows.append(_kappa_row("human_A (stance)", stance["hp_stance_A"], "human_B (stance)", stance["hp_stance_B"]))
    lines += ["## Трудная проблема (hp_level; weighted = quadratic)", "", pd.DataFrame(hrows).to_markdown(index=False), ""]

    # human labels для stats.py
    hl = ann[["id"] + [f"theory_primary_{c}" for c in CODERS] + [f"hp_level_{c}" for c in CODERS] + [f"hp_stance_{c}" for c in CODERS]].copy()
    hl["human_primary"] = np.where(hl["theory_primary_A"] == hl["theory_primary_B"], hl["theory_primary_A"], None)
    hl["human_labels"] = ["|".join(x) if (x is not None and y is not None and x == y) else "" for x, y in zip(ann["theory_labels_A"], ann["theory_labels_B"])]
    hl["human_hp_level"] = np.where(hl["hp_level_A"] == hl["hp_level_B"], hl["hp_level_A"], np.nan)
    hl["human_hp_stance"] = np.where(hl["hp_stance_A"] == hl["hp_stance_B"], hl["hp_stance_A"], None)
    hl.to_csv(out_dir / "human_labels.csv", index=False)

    # разногласия
    dis = ann[(ann["theory_primary_A"].notna()) & (ann["theory_primary_B"].notna()) & (ann["theory_primary_A"] != ann["theory_primary_B"])
              | (ann["hp_level_A"].notna()) & (ann["hp_level_B"].notna()) & (ann["hp_level_A"] != ann["hp_level_B"])].copy()
    keep = ["id", "source", "title", "theory_primary_A", "theory_primary_B", "hp_level_A", "hp_level_B",
            "hp_stance_A", "hp_stance_B", "note_A", "note_B", "text"]
    dis = dis[[c for c in keep if c in dis.columns]]
    if cons is not None:
        dis = dis.merge(cons[["id", "consensus_primary", "disputed"] + model_cols + [c for c in cons.columns if c.startswith("hp_level_")]], on="id", how="left")
    dis.to_csv(out_dir / "disagreements.csv", index=False)
    lines += [f"## Разногласия для согласительной сессии: {len(dis)} текстов → disagreements.csv", ""]

    if data["author_type"] is not None:
        at = data["author_type"][["id", "author_type"]].copy()
        at["author_type"] = at["author_type"].map(lambda v: str(v).strip().lower() if isinstance(v, str) and v.strip() else None)
        at.to_csv(out_dir / "author_type.csv", index=False)
        lines += [f"author_type: заполнено {int(at['author_type'].notna().sum())} из {len(at)} → author_type.csv", ""]

    (out_dir / "human_agreement.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Импорт ручной разметки и расчёт κ")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default=None, help="заполненная таблица (по умолчанию paths.annotation_filled)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    inp = resolve(cfg, args.input or cfg["paths"]["annotation_filled"])
    if not inp.exists():
        raise SystemExit(f"Файл не найден: {inp}")
    rd = reports_dir(cfg)
    build(cfg, inp, rd)
    print(f"Готово: {rd / 'human_agreement.md'}, human_labels.csv, disagreements.csv")


if __name__ == "__main__":
    main()
