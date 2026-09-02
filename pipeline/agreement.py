"""Согласие между прогонами: Cohen κ, Krippendorff α (nominal), Jaccard по multi-label,
матрицы согласия, consensus-разметка. Legacy-прогоны (декабрь, май) включаются для документирования.

python -m pipeline.agreement --config config.yaml
Выход: reports/consensus.csv, reports/agreement_summary.md, reports/agreement_pairwise_kappa.csv,
       reports/confusion_<A>__<B>.md
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import ALL_CLASSES, CLASSES, list_runs, load_config, load_corpus, load_json, reports_dir, resolve


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #

def cohen_kappa(a: Sequence[Any], b: Sequence[Any], weights: Optional[str] = None) -> float:
    """κ Коэна. weights=None — номинальная; 'linear'/'quadratic' — взвешенная (для порядковых шкал)."""
    a, b = list(a), list(b)
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b), key=lambda x: (str(type(x)), x))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k == 1:
        return 1.0
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
    o /= n
    ra, cb = o.sum(axis=1), o.sum(axis=0)
    e = np.outer(ra, cb)
    if weights is None:
        w = 1 - np.eye(k)
    else:
        pos = np.array([float(c) for c in cats])
        d = np.abs(pos[:, None] - pos[None, :])
        w = d if weights == "linear" else d ** 2
        w = w / w.max() if w.max() > 0 else w
    de = (w * e).sum()
    if de == 0:
        return 1.0
    return float(1 - (w * o).sum() / de)


def krippendorff_alpha(units: Iterable[Sequence[Optional[Any]]]) -> float:
    """α Криппендорфа, номинальная шкала. units — по одному списку значений (по кодировщикам) на единицу; None = пропуск."""
    units = [[v for v in u if v is not None and v == v] for u in units]
    values = sorted({v for u in units for v in u}, key=str)
    if not values:
        return float("nan")
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[idx[u[i]], idx[u[j]]] += 1.0 / (m - 1)
    n = o.sum()
    if n <= 1:
        return float("nan")
    nc = o.sum(axis=1)
    do = n - np.trace(o)
    denom = n * n - (nc ** 2).sum()
    if denom == 0:
        return 1.0
    return float(1 - (n - 1) * do / denom)


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def confusion(a: Sequence[Any], b: Sequence[Any], labels: Sequence[Any]) -> pd.DataFrame:
    m = pd.DataFrame(0, index=list(labels), columns=list(labels))
    for x, y in zip(a, b):
        if x in m.index and y in m.columns:
            m.loc[x, y] += 1
    return m


def bootstrap_ci(fn, n: int, rng: np.random.Generator, reps: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    vals = []
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        v = fn(ix)
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------- #
# Загрузка прогонов
# --------------------------------------------------------------------------- #

def parse_labels(s: Any) -> List[str]:
    if isinstance(s, list):
        return [x for x in s if x]
    if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
        return []
    return [x for x in str(s).split("|") if x]


def load_v3_theory(cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in list_runs(cfg, "theory"):
        for rep, path in run["files"].items():
            for rec in load_json(path):
                rows.append({
                    "rater": f"{run['meta'].get('slug')}_r{rep}", "slug": run["meta"].get("slug"),
                    "model": run["meta"].get("model"), "run": run["run_dir"].name, "repeat": rep,
                    "id": rec["id"], "status": rec.get("status"),
                    "primary": rec.get("primary", "UND"),
                    "labels": rec.get("labels") or ["UND"],
                    "secondary_raw": rec.get("secondary_raw") or [],
                    "is_hybrid": bool(rec.get("is_hybrid")), "is_hybrid_raw": bool(rec.get("is_hybrid_raw")),
                })
    return pd.DataFrame(rows)


def load_v3_hp(cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in list_runs(cfg, "hardproblem"):
        for rep, path in run["files"].items():
            for rec in load_json(path):
                rows.append({
                    "rater": f"{run['meta'].get('slug')}_r{rep}", "slug": run["meta"].get("slug"),
                    "repeat": rep, "id": rec["id"], "status": rec.get("status"),
                    "hp_level": int(rec.get("hp_level", 0) or 0), "hp_stance": rec.get("hp_stance"),
                    "hp_subject": rec.get("hp_subject"),
                })
    return pd.DataFrame(rows)


def load_legacy_run(entry: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Форматы v1 (analysis_results) и v2 (enhanced_*): список dict с index, primary_class, secondary_class, is_hybrid."""
    path = resolve(cfg, entry["path"])
    if not path.exists():
        print(f"  legacy '{entry['name']}': файл не найден, пропускаю ({path})")
        return None
    rows = []
    for rec in load_json(path):
        primary = rec.get("primary_class") or "UND"
        if primary not in ALL_CLASSES:
            primary = "UND"
        labels = [primary] if primary != "UND" else ["UND"]
        sec = rec.get("secondary_class")
        if primary != "UND" and rec.get("is_hybrid") and sec in CLASSES and sec != primary:
            labels.append(sec)
        rows.append({"rater": entry["name"], "slug": entry["name"], "id": int(rec.get("index", len(rows))),
                     "primary": primary, "labels": labels, "is_hybrid": len(labels) > 1, "legacy": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-model метки и consensus
# --------------------------------------------------------------------------- #

def majority(values: Sequence[Any], default: Any = None) -> Tuple[Any, bool]:
    """Большинство; при равенстве — первое значение (повтор 1). Возвращает (значение, было_ли_равенство)."""
    vals = [v for v in values if v is not None and v == v]
    if not vals:
        return default, False
    c = Counter(vals)
    best = max(c.values())
    winners = [v for v in vals if c[v] == best]
    return winners[0], len(set(winners)) > 1


def per_model_labels(long: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на (slug, id): primary по большинству повторов, labels — классы, встречающиеся
    в большинстве повторов (при равенстве — из повтора 1), флаг unstable при расхождении primary."""
    rows = []
    for (slug, tid), g in long.sort_values("repeat").groupby(["slug", "id"]):
        prim, tie = majority(g["primary"].tolist(), "UND")
        reps = g["repeat"].nunique()
        counts = Counter(c for labs in g["labels"] for c in set(labs))
        labels = [c for c in ALL_CLASSES if counts.get(c, 0) * 2 > reps]
        if not labels:
            labels = list(g.iloc[0]["labels"])
        if prim != "UND":
            labels = [prim] + [c for c in labels if c not in ("UND", prim)]
        else:
            labels = ["UND"]
        raw_counts = Counter(c for labs in g["secondary_raw"] for c in set(labs))
        hybrid_raw = prim != "UND" and any(raw_counts.get(c, 0) * 2 > reps for c in CLASSES if c != prim)
        if reps == 1:
            hybrid_raw = bool(g.iloc[0]["is_hybrid_raw"])
        rows.append({"slug": slug, "id": tid, "primary": prim, "labels": labels,
                     "is_hybrid": len(labels) > 1, "is_hybrid_raw": hybrid_raw,
                     "unstable": g["primary"].nunique() > 1, "n_repeats": reps})
    return pd.DataFrame(rows)


def consensus_theory(pm: pd.DataFrame, min_agree: int) -> pd.DataFrame:
    rows = []
    for tid, g in pm.groupby("id"):
        prim, _ = majority(g["primary"].tolist(), "UND")
        n_agree = int((g["primary"] == prim).sum())
        disputed = n_agree < min_agree
        cons_primary = prim if not disputed else "UND"
        counts = Counter(c for labs in g["labels"] for c in set(labs))
        labels = [c for c in CLASSES if counts.get(c, 0) >= min_agree]
        if cons_primary != "UND":
            labels = [cons_primary] + [c for c in labels if c != cons_primary]
        else:
            labels = ["UND"]
        hybrid_raw_votes = int(g["is_hybrid_raw"].sum())
        row = {"id": tid, "consensus_primary": cons_primary, "disputed": disputed, "n_models_agree": n_agree,
               "n_models": len(g), "consensus_labels": labels, "is_hybrid": len(labels) > 1,
               "is_hybrid_raw": hybrid_raw_votes >= min_agree}
        for _, r in g.iterrows():
            row[f"primary_{r['slug']}"] = r["primary"]
            row[f"labels_{r['slug']}"] = "|".join(r["labels"])
        rows.append(row)
    return pd.DataFrame(rows)


def consensus_hp(hp_long: pd.DataFrame, min_agree: int) -> pd.DataFrame:
    if hp_long.empty:
        return pd.DataFrame(columns=["id"])
    rows = []
    for (slug, tid), g in hp_long.sort_values("repeat").groupby(["slug", "id"]):
        lvl, _ = majority(g["hp_level"].tolist(), 0)
        st, _ = majority([s for s in g["hp_stance"] if s], None)
        rows.append({"slug": slug, "id": tid, "hp_level": lvl, "hp_stance": st})
    pm = pd.DataFrame(rows)
    out = []
    for tid, g in pm.groupby("id"):
        lvl, _ = majority(g["hp_level"].tolist(), 0)
        n_agree = int((g["hp_level"] == lvl).sum())
        disputed = n_agree < min_agree
        level = lvl if not disputed else int(np.median(g["hp_level"]))
        stances = [s for s in g["hp_stance"] if s]
        st, _ = majority(stances, None)
        st_ok = st is not None and stances.count(st) >= min_agree
        row = {"id": tid, "hp_level": level, "hp_disputed": disputed, "hp_stance": st if (st_ok and level >= 1) else None,
               "hp_stance_disputed": bool(stances) and not st_ok}
        for _, r in g.iterrows():
            row[f"hp_level_{r['slug']}"] = r["hp_level"]
            row[f"hp_stance_{r['slug']}"] = r["hp_stance"]
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #

def _wide(long: pd.DataFrame, col: str) -> pd.DataFrame:
    return long.pivot_table(index="id", columns="rater", values=col, aggfunc="first")


def _pairwise_kappa(wide: pd.DataFrame, weights: Optional[str] = None) -> pd.DataFrame:
    raters = list(wide.columns)
    m = pd.DataFrame(np.nan, index=raters, columns=raters)
    for a in raters:
        for b in raters:
            both = wide[[a, b]].dropna()
            if len(both) == 0:
                continue
            m.loc[a, b] = round(cohen_kappa(both[a], both[b], weights=weights), 3)
    return m


def _md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(**kw)


def build_report(cfg: Dict[str, Any], out_dir) -> Dict[str, Any]:
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    min_agree = int(cfg.get("consensus", {}).get("min_models_agree", 2))
    corpus = load_corpus(cfg)[["id", "title", "source", "n_words"]]
    th = load_v3_theory(cfg)
    hp = load_v3_hp(cfg)
    legacy_frames = [f for f in (load_legacy_run(e, cfg) for e in cfg.get("legacy_runs", [])) if f is not None]
    lines: List[str] = ["# Согласие между прогонами", ""]
    summary: Dict[str, Any] = {}

    if th.empty:
        lines.append("v3-прогонов нет (runs/ пуст). Ниже только legacy.")
        pm = pd.DataFrame(columns=["slug", "id", "primary", "labels", "is_hybrid", "is_hybrid_raw", "unstable"])
        cons = corpus[["id"]].assign(consensus_primary="UND", disputed=True, consensus_labels=[["UND"]] * len(corpus), is_hybrid=False, is_hybrid_raw=False, n_models_agree=0, n_models=0)
    else:
        th = th[th["status"] == "ok"]
        pm = per_model_labels(th)
        cons = consensus_theory(pm, min_agree)

        # --- 1. все прогоны (модель × повтор) ---
        wide = _wide(th, "primary")
        pk = _pairwise_kappa(wide)
        pk.to_csv(out_dir / "agreement_pairwise_kappa_runs.csv")
        units = wide.values.tolist()
        alpha_runs = krippendorff_alpha(units)
        alpha_ci = bootstrap_ci(lambda ix: krippendorff_alpha([units[i] for i in ix]), len(units), rng)
        summary["alpha_runs"] = alpha_runs
        lines += ["## 1. Все v3-прогоны (модель × повтор), primary", "",
                  f"Krippendorff α (nominal) = **{alpha_runs:.3f}** (bootstrap 95% CI {alpha_ci[0]:.3f}–{alpha_ci[1]:.3f}), "
                  f"n = {len(units)} текстов, {wide.shape[1]} прогонов.", "",
                  "Попарные Cohen κ:", "", _md(pk), ""]

        # test-retest
        lines += ["### Test–retest (повтор 1 vs повтор 2 одной модели)", ""]
        tr_rows = []
        for slug, g in th.groupby("slug"):
            w = _wide(g, "primary")
            if w.shape[1] >= 2:
                cols = list(w.columns)[:2]
                both = w[cols].dropna()
                k = cohen_kappa(both[cols[0]], both[cols[1]])
                exact = float((both[cols[0]] == both[cols[1]]).mean())
                tr_rows.append({"model": slug, "kappa": round(k, 3), "exact_agreement": round(exact, 3), "n": len(both)})
        if tr_rows:
            lines += [_md(pd.DataFrame(tr_rows), index=False), ""]
        summary["test_retest"] = tr_rows

        # --- 2. per-model ---
        pmw = pm.pivot_table(index="id", columns="slug", values="primary", aggfunc="first")
        pk_m = _pairwise_kappa(pmw)
        pk_m.to_csv(out_dir / "agreement_pairwise_kappa_models.csv")
        units_m = pmw.values.tolist()
        alpha_models = krippendorff_alpha(units_m)
        alpha_m_ci = bootstrap_ci(lambda ix: krippendorff_alpha([units_m[i] for i in ix]), len(units_m), rng)
        summary["alpha_models"] = alpha_models
        lines += ["## 2. Между моделями (метка модели = большинство её повторов), primary", "",
                  f"Krippendorff α (nominal) = **{alpha_models:.3f}** (95% CI {alpha_m_ci[0]:.3f}–{alpha_m_ci[1]:.3f}), "
                  f"{pmw.shape[1]} моделей.", "", "Попарные Cohen κ (95% bootstrap CI в отдельной таблице):", "", _md(pk_m), ""]
        ci_rows = []
        for a, b in itertools.combinations(list(pmw.columns), 2):
            both = pmw[[a, b]].dropna()
            va, vb = both[a].tolist(), both[b].tolist()
            k = cohen_kappa(va, vb)
            lo, hi = bootstrap_ci(lambda ix: cohen_kappa([va[i] for i in ix], [vb[i] for i in ix]), len(va), rng)
            ci_rows.append({"A": a, "B": b, "kappa": round(k, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                            "exact": round(float((both[a] == both[b]).mean()), 3), "n": len(both)})
            confusion(va, vb, ALL_CLASSES).to_csv(out_dir / f"confusion_{a}__{b}.csv")
            (out_dir / f"confusion_{a}__{b}.md").write_text(
                f"# Матрица согласия: строки {a}, столбцы {b}\n\n" + _md(confusion(va, vb, ALL_CLASSES)) + "\n", encoding="utf-8")
        if ci_rows:
            lines += [_md(pd.DataFrame(ci_rows), index=False), ""]
        summary["pairwise_models"] = ci_rows

        # multi-label
        lab = pm.pivot_table(index="id", columns="slug", values="labels", aggfunc="first")
        jr = []
        for a, b in itertools.combinations(list(lab.columns), 2):
            both = lab[[a, b]].dropna()
            js = [jaccard(x, y) for x, y in zip(both[a], both[b])]
            jr.append({"A": a, "B": b, "mean_jaccard": round(float(np.mean(js)), 3), "n": len(js)})
        lines += ["### Multi-label (primary + все secondary): средний Jaccard по текстам", ""]
        if jr:
            lines += [_md(pd.DataFrame(jr), index=False), ""]
        # per-class κ on membership
        pc_rows = []
        for c in CLASSES:
            row = {"class": c}
            for a, b in itertools.combinations(list(lab.columns), 2):
                both = lab[[a, b]].dropna()
                xa = [c in x for x in both[a]]
                xb = [c in x for x in both[b]]
                row[f"{a}~{b}"] = round(cohen_kappa(xa, xb), 3) if (any(xa) or any(xb)) else float("nan")
            row["n_texts_any_model"] = int(sum(any(c in x for x in r if isinstance(x, list)) for r in lab.values.tolist()))
            pc_rows.append(row)
        lines += ["### κ по принадлежности к классу (бинарно, multi-label)", "", _md(pd.DataFrame(pc_rows), index=False), ""]

        # consensus stats
        dist = Counter(cons["consensus_primary"])
        lines += ["## 3. Consensus-разметка", "",
                  f"Класс присваивается при совпадении у ≥{min_agree} моделей, иначе UND с флагом disputed. "
                  f"Disputed: {int(cons['disputed'].sum())} из {len(cons)}.", "",
                  _md(pd.DataFrame([{"class": k, "n": v} for k, v in sorted(dist.items(), key=lambda kv: -kv[1])]), index=False), ""]

    # --- hp ---
    if not hp.empty:
        hp = hp[hp["status"] == "ok"]
        hw = _wide(hp, "hp_level")
        alpha_hp = krippendorff_alpha(hw.values.tolist())
        pk_hp = _pairwise_kappa(hw)
        pk_hp_w = _pairwise_kappa(hw, weights="quadratic")
        chp = consensus_hp(hp, min_agree)
        lvl_dist = Counter(chp["hp_level"])
        st_dist = Counter(s for s in chp["hp_stance"] if s)
        lines += ["## 4. Слой трудной проблемы (hp_level)", "",
                  f"Krippendorff α (nominal) по всем прогонам = **{alpha_hp:.3f}**.", "",
                  "Попарные κ (nominal):", "", _md(pk_hp), "", "Попарные κ (quadratic weighted, шкала 0/1/2):", "", _md(pk_hp_w), "",
                  f"Consensus hp_level: {dict(sorted(lvl_dist.items()))}; disputed: {int(chp['hp_disputed'].sum())}. "
                  f"Consensus hp_stance: {dict(st_dist)}.", ""]
        summary["alpha_hp"] = alpha_hp
    else:
        chp = pd.DataFrame(columns=["id"])
        lines += ["## 4. Слой трудной проблемы", "", "прогонов hardproblem нет.", ""]

    # --- legacy ---
    if legacy_frames:
        leg = pd.concat(legacy_frames, ignore_index=True)
        lines += ["## 5. Legacy-прогоны (для документирования)", "",
                  "Старые прогоны сделаны другими промптами (v1/v2) и другой схемой агрегации; "
                  "сравнение показывает, насколько разметка изменилась, а не надёжность v3.", ""]
        lw = _wide(leg, "primary")
        if lw.shape[1] >= 2:
            cols = list(lw.columns)
            both = lw.dropna()
            lines += [f"κ между legacy-прогонами ({' vs '.join(cols)}): **{cohen_kappa(both[cols[0]], both[cols[1]]):.3f}**, "
                      f"exact {float((both[cols[0]] == both[cols[1]]).mean()):.1%}, n={len(both)}.", ""]
        if not th.empty:
            merged = lw.join(cons.set_index("id")["consensus_primary"], how="inner")
            rows = []
            for c in lw.columns:
                both = merged[[c, "consensus_primary"]].dropna()
                rows.append({"legacy_run": c, "kappa_vs_consensus": round(cohen_kappa(both[c], both["consensus_primary"]), 3),
                             "exact": round(float((both[c] == both["consensus_primary"]).mean()), 3), "n": len(both)})
                for slug in pmw.columns:
                    b2 = lw[[c]].join(pmw[[slug]], how="inner").dropna()
                    rows[-1][f"kappa_vs_{slug}"] = round(cohen_kappa(b2[c], b2[slug]), 3)
            lines += [_md(pd.DataFrame(rows), index=False), ""]
            all_units = lw.join(pmw, how="outer").values.tolist()
            lines += [f"Krippendorff α по всем моделям v3 + legacy: **{krippendorff_alpha(all_units):.3f}**.", ""]
        legacy_wide = lw.add_prefix("primary_")
    else:
        legacy_wide = None

    # --- consensus.csv ---
    out = corpus.merge(cons, on="id", how="left")
    out["consensus_primary"] = out["consensus_primary"].fillna("UND")
    out["consensus_labels"] = out["consensus_labels"].apply(lambda x: "|".join(x) if isinstance(x, list) else "UND")
    for col, val in (("disputed", True), ("is_hybrid", False), ("is_hybrid_raw", False)):
        out[col] = out[col].fillna(val).astype(bool)
    if not chp.empty:
        out = out.merge(chp, on="id", how="left")
    if legacy_wide is not None:
        out = out.merge(legacy_wide, left_on="id", right_index=True, how="left")
    out.to_csv(out_dir / "consensus.csv", index=False)
    (out_dir / "agreement_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="κ/α между прогонами и consensus-разметка")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rd = reports_dir(cfg)
    s = build_report(cfg, rd)
    print(f"Готово: {rd / 'agreement_summary.md'}, {rd / 'consensus.csv'}")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
