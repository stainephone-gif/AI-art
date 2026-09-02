"""Статистика с базовой линией, раздельно по источнику и типу автора.

python -m pipeline.stats --config config.yaml [--labels consensus|human]
Выход: reports/stats_<labels>/ — csv + md для каждой таблицы, png для графиков:
  distribution_<subset>.csv/md          распределение primary (с UND и без)
  chi2_source_x_primary.md              χ² на различие распределений между источниками (asymptotic + Monte-Carlo p)
  chi2_author_type_x_primary.md         то же по типу автора (если reports/author_type.csv есть)
  cooccurrence_<subset>.csv/md          пары классов: наблюдаемое, ожидаемое, χ², станд. остатки, Fisher, PMI, permutation p
  theory_x_hp_level_<subset>.md         кросс-таблица + Fisher exact (2×2 точный, r×c — Monte-Carlo)
  theory_x_hp_stance_<subset>.md
  length_logit.md                       логистическая регрессия «есть вторичная метка ~ длина» до и после нормировки
  *.png
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from .agreement import parse_labels  # noqa: E402
from .common import ALL_CLASSES, CLASSES, load_config, reports_dir  # noqa: E402


# --------------------------------------------------------------------------- #
# Базовые тесты
# --------------------------------------------------------------------------- #

def chi2_report(ct: pd.DataFrame, n_mc: int, rng: np.random.Generator) -> Dict[str, Any]:
    """χ² независимости с ожидаемыми, скорректированными стандартизованными остатками и Monte-Carlo p (фиксированные маргиналы)."""
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    obs = ct.values.astype(float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return {"chi2": float("nan"), "df": 0, "p_asymptotic": float("nan"), "p_mc": float("nan"),
                "expected": ct * np.nan, "std_resid": ct * np.nan, "min_expected": float("nan"), "n": int(obs.sum())}
    chi2, p, dof, exp = st.chi2_contingency(obs, correction=False)
    n = obs.sum()
    r, c = obs.sum(axis=1), obs.sum(axis=0)
    adj = (obs - exp) / np.sqrt(exp * (1 - r[:, None] / n) * (1 - c[None, :] / n))
    dist = st.random_table(r.astype(int), c.astype(int), seed=rng)
    sims = dist.rvs(n_mc)
    chi2_sim = ((sims - exp) ** 2 / exp).sum(axis=(1, 2))
    p_mc = (float((chi2_sim >= chi2 - 1e-9).sum()) + 1) / (n_mc + 1)
    return {"chi2": float(chi2), "df": int(dof), "p_asymptotic": float(p), "p_mc": p_mc,
            "expected": pd.DataFrame(exp, index=ct.index, columns=ct.columns),
            "std_resid": pd.DataFrame(adj, index=ct.index, columns=ct.columns),
            "min_expected": float(exp.min()), "n": int(n), "cramers_v": float(math.sqrt(chi2 / (n * (min(obs.shape) - 1)))) if n else float("nan")}


def fisher_p(ct: pd.DataFrame, n_mc: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Точный Фишер для 2×2; для r×c — Monte-Carlo оценка (вероятность таблицы при фиксированных маргиналах)."""
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    obs = ct.values.astype(int)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return {"p": float("nan"), "method": "degenerate"}
    if obs.shape == (2, 2):
        return {"p": float(st.fisher_exact(obs)[1]), "method": "exact_2x2"}
    dist = st.random_table(obs.sum(axis=1), obs.sum(axis=0), seed=rng)
    lp_obs = dist.logpmf(obs)
    sims = dist.rvs(n_mc)
    lp_sim = dist.logpmf(sims)
    p = (float((lp_sim <= lp_obs + 1e-9).sum()) + 1) / (n_mc + 1)
    return {"p": p, "method": f"monte_carlo_{n_mc}"}


def permutation_pair(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> Dict[str, float]:
    """Двусторонний permutation-тест на со-встречаемость двух бинарных меток: перестановка одной метки
    между текстами сохраняет маргиналы обеих меток."""
    a, b = a.astype(bool), b.astype(bool)
    n = len(a)
    obs = int((a & b).sum())
    exp = a.sum() * b.sum() / n if n else float("nan")
    if a.sum() == 0 or b.sum() == 0:
        return {"observed": obs, "expected": exp, "p_perm": float("nan")}
    bb = b.copy()
    sims = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(bb)
        sims[i] = (a & bb).sum()
    dev = abs(obs - exp)
    p = (float((np.abs(sims - exp) >= dev - 1e-9).sum()) + 1) / (n_perm + 1)
    return {"observed": obs, "expected": float(exp), "p_perm": p}


def pmi(n_ab: float, n_a: float, n_b: float, n: float) -> float:
    if n_ab == 0 or n_a == 0 or n_b == 0 or n == 0:
        return float("-inf") if n_ab == 0 and n_a and n_b else float("nan")
    return math.log2((n_ab / n) / ((n_a / n) * (n_b / n)))


def logit_fit(x: Sequence[float], y: Sequence[int]) -> Dict[str, float]:
    """Логистическая регрессия y ~ x (IRLS). Возвращает коэффициент, SE, z, p, OR."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(y) < 5 or y.min() == y.max():
        return {"n": int(len(y)), "n_positive": int(y.sum()) if len(y) else 0, "coef": float("nan"), "se": float("nan"),
                "z": float("nan"), "p": float("nan"), "odds_ratio": float("nan"), "intercept": float("nan")}
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    H = None
    for _ in range(100):
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None])
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    try:
        cov = np.linalg.inv(H)
        se = float(np.sqrt(cov[1, 1]))
    except Exception:  # noqa: BLE001
        se = float("nan")
    z = beta[1] / se if se and se == se and se > 0 else float("nan")
    pval = 2 * (1 - st.norm.cdf(abs(z))) if z == z else float("nan")
    return {"n": int(len(y)), "n_positive": int(y.sum()), "coef": float(beta[1]), "se": se, "z": float(z), "p": float(pval),
            "odds_ratio": float(np.exp(beta[1])), "intercept": float(beta[0])}


# --------------------------------------------------------------------------- #
# Загрузка меток
# --------------------------------------------------------------------------- #

def load_labels(cfg: Dict[str, Any], which: str) -> pd.DataFrame:
    rd = reports_dir(cfg)
    cons = pd.read_csv(rd / "consensus.csv")
    if which == "consensus":
        df = cons.rename(columns={"consensus_primary": "primary", "consensus_labels": "labels"})
    elif which == "human":
        hp = rd / "human_labels.csv"
        if not hp.exists():
            raise SystemExit("reports/human_labels.csv нет — сначала python -m pipeline.import_annotation")
        h = pd.read_csv(hp)
        df = cons[["id", "title", "source", "n_words"]].merge(h, on="id", how="inner")
        df = df.rename(columns={"human_primary": "primary", "human_labels": "labels", "human_hp_level": "hp_level", "human_hp_stance": "hp_stance"})
        df = df[df["primary"].notna()]
        df["is_hybrid"] = df["labels"].map(lambda s: len(parse_labels(s)) > 1)
        df["is_hybrid_raw"] = df["is_hybrid"]
    else:
        raise ValueError(which)
    df["labels_list"] = df["labels"].map(parse_labels)
    df["primary"] = df["primary"].fillna("UND")
    for col in ("hp_level", "hp_stance", "is_hybrid", "is_hybrid_raw"):
        if col not in df.columns:
            df[col] = np.nan
    at = rd / "author_type.csv"
    if at.exists():
        a = pd.read_csv(at)
        df = df.merge(a[["id", "author_type"]], on="id", how="left")
    else:
        df["author_type"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Анализ одного подмножества
# --------------------------------------------------------------------------- #

def _write(df: pd.DataFrame, path: Path, title: str, index: bool = True, extra: str = "") -> None:
    df.to_csv(path.with_suffix(".csv"), index=index)
    path.with_suffix(".md").write_text(f"# {title}\n\n{extra}\n\n{df.to_markdown(index=index)}\n", encoding="utf-8")


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    cnt = df["primary"].value_counts().reindex(ALL_CLASSES, fill_value=0)
    n = len(df)
    non = cnt.drop("UND")
    out = pd.DataFrame({"n": cnt, "share_all": (cnt / n).round(3) if n else 0,
                        "share_non_und": (non / non.sum()).reindex(ALL_CLASSES).round(3) if non.sum() else np.nan})
    mem = {c: int(df["labels_list"].map(lambda l: c in l).sum()) for c in CLASSES}
    out["n_any_label"] = pd.Series(mem).reindex(ALL_CLASSES)
    out.loc["UND", "n_any_label"] = cnt["UND"]
    return out


def cooccurrence(df: pd.DataFrame, n_perm: int, rng: np.random.Generator, universe: str) -> pd.DataFrame:
    sub = df if universe == "all" else df[df["primary"] != "UND"]
    n = len(sub)
    mem = {c: sub["labels_list"].map(lambda l: c in l).values.astype(bool) for c in CLASSES}
    rows = []
    for a, b in itertools.combinations(CLASSES, 2):
        na, nb = int(mem[a].sum()), int(mem[b].sum())
        if na == 0 or nb == 0:
            continue
        nab = int((mem[a] & mem[b]).sum())
        ct = pd.DataFrame([[nab, na - nab], [nb - nab, n - na - nb + nab]], index=[f"{a}=1", f"{a}=0"], columns=[f"{b}=1", f"{b}=0"])
        exp = na * nb / n
        chi2_stat, p_chi, _, _ = st.chi2_contingency(ct.values, correction=False) if ct.values.min() >= 0 and (ct.values.sum(axis=0) > 0).all() and (ct.values.sum(axis=1) > 0).all() else (float("nan"),) * 4
        resid = (nab - exp) / math.sqrt(exp * (1 - na / n) * (1 - nb / n)) if exp > 0 and (1 - na / n) > 0 and (1 - nb / n) > 0 else float("nan")
        fisher = float(st.fisher_exact(ct.values)[1])
        perm = permutation_pair(mem[a], mem[b], n_perm, rng)
        rows.append({"universe": universe, "A": a, "B": b, "n": n, "n_A": na, "n_B": nb, "observed_AB": nab,
                     "expected_AB": round(exp, 2), "ratio_obs_exp": round(nab / exp, 2) if exp else float("nan"),
                     "chi2": round(float(chi2_stat), 3), "p_chi2": round(float(p_chi), 4), "std_resid": round(resid, 2),
                     "p_fisher": round(fisher, 4), "PMI": round(pmi(nab, na, nb, n), 3), "p_perm": round(perm["p_perm"], 4)})
    return pd.DataFrame(rows)


def analyze_subset(df: pd.DataFrame, name: str, out: Path, cfg: Dict[str, Any], rng: np.random.Generator) -> None:
    n_perm = int(cfg.get("stats", {}).get("n_permutations", 10000))
    n_mc = int(cfg.get("stats", {}).get("n_mc_tables", 10000))
    _write(distribution(df), out / f"distribution_{name}", f"Распределение primary — {name} (n={len(df)})",
           extra="share_non_und — доля среди не-UND; n_any_label — число текстов, где класс присутствует как primary или secondary.")
    co = pd.concat([cooccurrence(df, n_perm, rng, "all"), cooccurrence(df, n_perm, rng, "non_und")], ignore_index=True)
    if not co.empty:
        _write(co, out / f"cooccurrence_{name}", f"Со-встречаемость классов — {name}", index=False,
               extra=("universe=all: все тексты; non_und: только тексты с хотя бы одной меткой. expected_AB = n_A·n_B/n при независимости; "
                      "std_resid — скорректированный стандартизованный остаток ячейки (A=1,B=1); p_perm — двусторонний permutation-тест "
                      f"({n_perm} перестановок метки B между текстами, маргиналы сохраняются); p_fisher — точный тест Фишера 2×2."))
    if df["hp_level"].notna().any():
        sub = df[df["hp_level"].notna()].copy()
        sub["hp_level"] = sub["hp_level"].astype(int)
        ct = pd.crosstab(sub["primary"], sub["hp_level"]).reindex(ALL_CLASSES, fill_value=0)
        ct = ct.loc[ct.sum(axis=1) > 0]
        f = fisher_p(ct, n_mc, rng)
        _write(ct, out / f"theory_x_hp_level_{name}", f"theory × hp_level — {name}", extra=f"Fisher exact p = {f['p']:.4f} ({f['method']}).")
        sub2 = sub[sub["hp_stance"].notna() & (sub["hp_level"] >= 1)]
        if len(sub2):
            ct2 = pd.crosstab(sub2["primary"], sub2["hp_stance"]).reindex(ALL_CLASSES, fill_value=0)
            ct2 = ct2.loc[ct2.sum(axis=1) > 0]
            f2 = fisher_p(ct2, n_mc, rng)
            _write(ct2, out / f"theory_x_hp_stance_{name}", f"theory × hp_stance — {name} (только hp_level ≥ 1)", extra=f"Fisher exact p = {f2['p']:.4f} ({f2['method']}).")


def chi2_between_groups(df: pd.DataFrame, group: str, out: Path, cfg: Dict[str, Any], rng: np.random.Generator) -> None:
    n_mc = int(cfg.get("stats", {}).get("n_mc_tables", 10000))
    sub = df[df[group].notna()]
    if sub[group].nunique() < 2:
        return
    lines = [f"# χ²: {group} × primary", ""]
    for label, d in (("с UND", sub), ("без UND", sub[sub["primary"] != "UND"])):
        ct = pd.crosstab(d[group], d["primary"]).reindex(columns=[c for c in ALL_CLASSES if c in d["primary"].unique()], fill_value=0)
        r = chi2_report(ct, n_mc, rng)
        lines += [f"## {label} (n={r['n']})", "", ct.to_markdown(), "",
                  f"χ² = {r['chi2']:.3f}, df = {r['df']}, p (asymptotic) = {r['p_asymptotic']:.4f}, p (Monte-Carlo, {n_mc} таблиц с фиксированными маргиналами) = {r['p_mc']:.4f}, "
                  f"Cramér's V = {r.get('cramers_v', float('nan')):.3f}, min expected = {r['min_expected']:.2f}"
                  + (" (⚠ есть ожидаемые < 5, ориентируйтесь на Monte-Carlo p)" if r["min_expected"] == r["min_expected"] and r["min_expected"] < 5 else ""), "",
                  "Ожидаемые:", "", r["expected"].round(2).to_markdown() if isinstance(r["expected"], pd.DataFrame) else "", "",
                  "Скорректированные стандартизованные остатки (|z| > 1.96 — значимое отклонение ячейки):", "",
                  r["std_resid"].round(2).to_markdown() if isinstance(r["std_resid"], pd.DataFrame) else "", ""]
    (out / f"chi2_{group}_x_primary.md").write_text("\n".join(lines), encoding="utf-8")


def length_analysis(df: pd.DataFrame, out: Path, cfg: Dict[str, Any]) -> None:
    unit = float(cfg.get("stats", {}).get("length_unit_words", 100))
    sub = df[df["primary"] != "UND"].copy() if (df["primary"] != "UND").sum() >= 5 else df.copy()
    x = sub["n_words"].astype(float) / unit
    rows = []
    for label, col in (("до нормировки (secondary_raw: n_spans ≥ 2)", "is_hybrid_raw"), ("после нормировки (+ порог density)", "is_hybrid")):
        y = sub[col].fillna(False).astype(bool).astype(int).values
        r = logit_fit(x.values, y)
        r.update({"model": label, "median_words_with_secondary": float(sub.loc[y == 1, "n_words"].median()) if y.sum() else float("nan"),
                  "median_words_without": float(sub.loc[y == 0, "n_words"].median()) if (y == 0).sum() else float("nan")})
        rows.append(r)
    res = pd.DataFrame(rows)[["model", "n", "n_positive", "coef", "se", "z", "p", "odds_ratio", "median_words_with_secondary", "median_words_without"]]
    _write(res.round(4), out / "length_logit", "Логистическая регрессия: есть вторичная метка ~ длина", index=False,
           extra=f"Универсум: тексты с primary ≠ UND (n={len(sub)}). Коэффициент и OR — на +{int(unit)} слов.")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for a, (label, col) in zip(ax, (("до", "is_hybrid_raw"), ("после", "is_hybrid"))):
        y = sub[col].fillna(False).astype(bool)
        data = [sub.loc[~y, "n_words"], sub.loc[y, "n_words"]]
        a.boxplot(data, tick_labels=["нет secondary", "есть secondary"])
        a.set_title(f"{label} нормировки")
        a.set_yscale("log")
    ax[0].set_ylabel("слов в тексте (log)")
    fig.tight_layout()
    fig.savefig(out / "length_boxplot.png", dpi=150)
    plt.close(fig)


def figures(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["source"], df["primary"]).reindex(columns=ALL_CLASSES, fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    share.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("доля текстов источника")
    ax.set_title("Распределение primary по источникам")
    fig.tight_layout()
    fig.savefig(out / "distribution_by_source.png", dpi=150)
    plt.close(fig)

    mem = pd.DataFrame({c: df["labels_list"].map(lambda l: c in l) for c in CLASSES}).astype(int)
    n = len(mem)
    m = np.full((len(CLASSES), len(CLASSES)), np.nan)
    for i, a in enumerate(CLASSES):
        for j, b in enumerate(CLASSES):
            if i == j:
                continue
            na, nb, nab = mem[a].sum(), mem[b].sum(), int((mem[a] & mem[b]).sum())
            exp = na * nb / n if n else 0
            if exp > 0 and na < n and nb < n:
                m[i, j] = (nab - exp) / math.sqrt(exp * (1 - na / n) * (1 - nb / n))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if m[i, j] == m[i, j]:
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="станд. остаток (набл. − ожид.)")
    ax.set_title("Со-встречаемость классов: отклонение от независимости")
    fig.tight_layout()
    fig.savefig(out / "cooccurrence_residuals.png", dpi=150)
    plt.close(fig)


def run(cfg: Dict[str, Any], which: str) -> Path:
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    df = load_labels(cfg, which)
    out = reports_dir(cfg) / f"stats_{which}"
    out.mkdir(parents=True, exist_ok=True)
    analyze_subset(df, "all", out, cfg, rng)
    for src, d in df.groupby("source"):
        analyze_subset(d, f"source_{Path(str(src)).stem}", out, cfg, rng)
    chi2_between_groups(df, "source", out, cfg, rng)
    if df["author_type"].notna().any():
        for at, d in df[df["author_type"].notna()].groupby("author_type"):
            analyze_subset(d, f"author_{at}", out, cfg, rng)
        chi2_between_groups(df, "author_type", out, cfg, rng)
    length_analysis(df, out, cfg)
    figures(df, out)
    n_dis = int(df["disputed"].sum()) if "disputed" in df.columns else 0
    (out / "README.md").write_text(
        f"# Статистика ({which})\n\nn = {len(df)} текстов; disputed (consensus UND из-за расхождения моделей): {n_dis}.\n"
        "Файлы: distribution_*, cooccurrence_*, theory_x_hp_*, chi2_*, length_logit, *.png. "
        "Подмножества: all, source_<источник>, author_<тип автора>.\n", encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Статистика с базовой линией")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", choices=["consensus", "human"], default="consensus")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = run(cfg, args.labels)
    print(f"Готово: {out}")


if __name__ == "__main__":
    main()
