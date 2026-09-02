"""Регрессионные тесты пайплайна v3.

Оффлайн (без API): промпты и кодбук, верификация цитат, валидация evidence (mapping, исключения,
дубликаты), агрегация (UND по умолчанию, secondary по плотности, confidence), слой трудной проблемы
(человек → level 0), метрики согласия на известных примерах, consensus, слепой экспорт, импорт, статистика.

Live (нужен OPENROUTER_API_KEY; иначе пропускаются): контрпримеры из ТЗ п. 8.3–8.4 на реальной модели —
промпт v3 не должен давать ENACT за «работу с эмульсией в темноте», PAN за «древесина хранит аромат»,
COMP за «переводили стихотворение вместе с ИИ», и ещё 10 контрпримеров; «Демон» → hp_level 2,
словарь опыта у человека → hp_level 0.

    pytest test_classifier.py -q                 # оффлайн
    OPENROUTER_API_KEY=... pytest -m live -q     # живые тесты (модель: TEST_MODEL или первая из config.yaml)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline import agreement, aggregate, common, export_annotation, import_annotation, stats
from pipeline.classify_hardproblem import process_hardproblem
from pipeline.classify_theory import process_theory
from pipeline.verify_quotes import select_independent, verify_span

ROOT = Path(__file__).resolve().parent
CFG = common.load_config(ROOT / "config.yaml")
LIVE = pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="нужен OPENROUTER_API_KEY")


def _row(text: str, tid: int = 0, source: str = "test") -> pd.Series:
    return pd.Series({"id": tid, "title": f"t{tid}", "source": source, "text": text, "n_words": common.count_words(text)})


def _ev(cls, span, level="meta_metaphor", mapping=None, checked=True):
    return {"class": cls, "span": span, "level": level, "exclusion_checked": checked,
            "mapping": {"source": "машина", "target": "ум"} if mapping is None else mapping, "reasoning": "r"}


# --------------------------------------------------------------------------- #
# Контрпримеры (ТЗ 8.3): текст → класс, который НЕ должен быть присвоен
# --------------------------------------------------------------------------- #

COUNTEREXAMPLES = [
    ("ENACT", "Работа с цветной эмульсией происходит почти в полной темноте, поэтому художница выступает медиатором "
              "во взаимодействии между природным и цифровым. Отпечатки проявляются на ощупь, руками."),
    ("PAN", "Техника 3D-печати раскрывает фактурность; используемый древесный материал хранит аромат и тактильность "
            "живой материи. Arbor — киборганический живой объект, выращенный биоискусственным путём."),
    ("COMP", "Я решила обратиться за помощью к ИИ. Мы вместе слово за словом переводили это стихотворение, а затем "
             "я создала художественный перевод, основываясь на тех иероглифах, что предложил мне ИИ."),
    ("ENACT", "Зритель активирует инсталляцию прикосновением к сенсорной панели; свет и звук меняются в ответ "
              "на движения тела в пространстве зала."),
    ("PAN", "Из тьмы притворяющегося зеркалом экрана выходят те, кто стоял перед ним прежде. Зритель испытывает "
            "суеверный ужас — у зеркал и без того неоднозначная репутация."),
    ("EMERG", "Работа создана коллективом из двадцати художников, каждый принёс свой фрагмент; результат — "
              "мозаика голосов, случайных находок и импровизаций."),
    ("PRED", "Нейросеть галлюцинирует лишние пальцы и несуществующие здания; эти технические артефакты генерации "
             "художник оставляет в финальных изображениях как след машины."),
    ("GWT", "Инсталляция привлекает внимание зрителя к проблеме таяния ледников; на центральной сцене павильона "
            "установлен экран, где сменяются кадры Арктики."),
    ("IIT", "Композиция сохраняет целостность: отдельные кадры объединены в единый образ, где целое важнее частей, "
            "а монтаж создаёт синтез звука и изображения."),
    ("PAN", "По сюжету видео домовой оживляет старую мебель: стулья ходят по квартире, а шкаф рассказывает "
            "истории прежних хозяев. Фольклорные духи здесь — персонажи нарратива."),
    ("COMP", "Нейросеть, обученная на десяти тысячах картин русского авангарда, генерирует новые пейзажи; "
             "художник отбирает лучшие результаты и печатает их на холсте."),
    ("ENACT", "Перформанс: танцовщица в костюме с датчиками захвата движения, её пластика в реальном времени "
              "переводится в анимацию на экране позади сцены."),
    ("EMERG", "Генератор шума задаёт случайные параметры, поэтому каждый запуск даёт непредсказуемую картинку; "
              "число вариантов бесконечно, и ни один не повторяется."),
]

POSITIVE_THEORY = [
    ("COMP", "Подобно Демону, машина находится в промежуточном состоянии: она выполняет точные вычисления, "
             "но не имеет субъективного опыта, а смысл возникает только в сознании человека."),
    ("PAN", "Художник исходит из того, что у каждого пикселя, как и у каждого камня, есть зачаток опыта: "
            "материя чувствует, и экран — не исключение."),
]

HP_CASES = [
    (2, "Подобно Демону, ИИ находится в промежуточном состоянии: он выполняет точные вычисления, но не имеет "
        "субъективного опыта. Результаты его работы мы интерпретируем через призму человеческого восприятия."),
    (0, "Зритель, стоя перед экраном, переживает субъективный опыт узнавания: его личные воспоминания и квалиа "
        "детства всплывают при виде старых фотографий, которые нейросеть раскрашивает."),
    (0, "Нейросеть распознаёт эмоции посетителей по выражению лица и подбирает музыку; датасет эмоций собран "
        "на выставках прошлого года."),
]


# --------------------------------------------------------------------------- #
# Промпты и кодбук
# --------------------------------------------------------------------------- #

def test_prompts_render_codebook_and_hash_stable():
    p = common.load_prompt(CFG, "theory")
    h = common.load_prompt(CFG, "hardproblem")
    assert "{{codebook" not in p.system and "{{codebook" not in h.system
    assert p.hash == common.load_prompt(CFG, "theory").hash
    assert p.hash != h.hash
    assert "{text}" in p.user_template and "{text}" in h.user_template
    assert "XYZ" in p.user("XYZ")


def test_theory_prompt_contains_exclusions_from_spec():
    p = common.load_prompt(CFG, "theory").system
    for phrase in ("в темноте", "хранит аромат", "вместе с ИИ", "фольклорных духов", "бытовом смысле",
                   "множественность, коллективность, случайность", "только явные термины", "любую «генерацию»"):
        assert phrase in p, phrase
    for bad in ("×1.2", "КРИТИЧЕСКИ ВАЖНО", "вес 0.8", "confidence"):
        assert bad not in p, bad
    assert '"mapping"' in p and "exclusion_checked" in p


def test_hp_prompt_contains_levels_and_exclusions():
    h = common.load_prompt(CFG, "hardproblem").system
    for phrase in ("каково это быть", "философского зомби", "есть ли там кто-то", "denial", "reframing",
                   "эмоции как данные", "машина видит"):
        assert phrase in h, phrase


def test_codebook_sections_are_the_same_text_in_prompts():
    sections = common.codebook_sections(common.load_codebook(CFG))
    assert {"evidence", "theory", "levels", "hardproblem"} <= set(sections)
    assert sections["theory"] in common.load_prompt(CFG, "theory").system
    assert sections["hardproblem"] in common.load_prompt(CFG, "hardproblem").system


# --------------------------------------------------------------------------- #
# Верификация цитат
# --------------------------------------------------------------------------- #

TEXT = ("Подобно Демону, ИИ находится в промежуточном состоянии: он выполняет точные вычисления, "
        "но не имеет субъективного опыта. Инсталляция предлагает интерактивный опыт.")


def test_verify_exact_and_case_punct_insensitive():
    v = verify_span("он выполняет точные вычисления, но не имеет субъективного опыта", TEXT)
    assert v["verified"] and v["method"] == "exact" and v["loc"] is not None
    v2 = verify_span("ОН ВЫПОЛНЯЕТ ТОЧНЫЕ ВЫЧИСЛЕНИЯ", TEXT)
    assert v2["verified"] and v2["method"] == "exact"


def test_verify_token_overlap_and_reject():
    v = verify_span("выполняет точное вычисление, но не имеет субъективного опыта", TEXT, threshold=0.7)
    assert v["verified"] and v["method"] == "token_overlap_with_prefix"
    assert not verify_span("машина мечтает о свободе и любви", TEXT)["verified"]
    assert not verify_span("выполняет точное вычисление, но не имеет субъективного опыта", TEXT, require_exact=True)["verified"]


def test_select_independent_drops_overlaps():
    a = {"span": "выполняет точные вычисления", "loc": [7, 10]}
    b = {"span": "точные вычисления, но не имеет субъективного опыта", "loc": [8, 15]}
    c = {"span": "Инсталляция предлагает интерактивный опыт", "loc": [16, 20]}
    kept, dropped = select_independent([a, b, c])
    assert kept == [a, c] and dropped == [b]


# --------------------------------------------------------------------------- #
# Валидация и агрегация
# --------------------------------------------------------------------------- #

def test_validation_filters_by_reason():
    data = {"evidence": [
        _ev("COMP", "он выполняет точные вычисления, но не имеет субъективного опыта"),
        _ev("COMP", "выполняет точные вычисления", mapping={"source": "", "target": ""}),          # no_mapping
        _ev("PAN", "Инсталляция предлагает интерактивный опыт", checked=False),                     # exclusion_not_checked
        _ev("ENACT", "художница работает руками в темноте"),                                        # unverified
        _ev("COMP", "точные вычисления, но не имеет субъективного опыта"),                          # duplicate
        _ev("COMP", "вычисления"),                                                                  # too short
        _ev("XXX", "Инсталляция предлагает интерактивный опыт"),                                     # invalid_class
        "garbage",
    ]}
    recs, st = aggregate.validate_theory_response(data, TEXT, CFG["verification"])
    assert st["n_raw"] == 8 and st["n_kept"] == 1
    assert st["dropped"] == {"no_mapping": 1, "exclusion_not_checked": 1, "unverified": 1, "duplicate": 1,
                             "span_too_short": 1, "invalid_class": 1, "malformed": 1}


def test_und_default_and_confidence_from_counts():
    agg = aggregate.aggregate_theory([], 100, CFG["aggregation"])
    assert agg["primary"] == "UND" and agg["labels"] == ["UND"] and agg["confidence"] == "none" and not agg["is_hybrid"]
    one = [{"class": "COMP", "span": "a b c", "kept": True}]
    assert aggregate.aggregate_theory(one, 100, CFG["aggregation"])["confidence"] == "low"
    three = one * 3
    assert aggregate.aggregate_theory(three, 100, CFG["aggregation"])["confidence"] == "high"


def test_secondary_requires_two_spans_and_density():
    kept = [{"class": "COMP", "span": "a b c", "kept": True}] * 3 + [{"class": "PAN", "span": "x y", "kept": True}] * 2
    agg = aggregate.aggregate_theory(kept, 100, CFG["aggregation"])
    assert agg["primary"] == "COMP" and agg["secondary"] == ["PAN"] and agg["is_hybrid"]
    assert agg["density"]["PAN"] == 2.0 and agg["n_spans"]["COMP"] == 3
    long = aggregate.aggregate_theory(kept, 400, CFG["aggregation"])       # PAN density 0.5 < 1.0
    assert long["secondary_raw"] == ["PAN"] and long["secondary"] == [] and not long["is_hybrid"] and long["is_hybrid_raw"]
    single = [{"class": "COMP", "span": "a b c", "kept": True}] * 2 + [{"class": "PAN", "span": "x y", "kept": True}]
    assert aggregate.aggregate_theory(single, 50, CFG["aggregation"])["secondary_raw"] == []


def test_primary_tie_break_by_coverage_and_flag():
    kept = [{"class": "PAN", "span": "один два три четыре", "kept": True}, {"class": "COMP", "span": "один два", "kept": True}]
    agg = aggregate.aggregate_theory(kept, 100, CFG["aggregation"])
    assert agg["primary"] == "PAN" and not agg["primary_tie"]
    kept2 = [{"class": "PAN", "span": "один два", "kept": True}, {"class": "COMP", "span": "три четыре", "kept": True}]
    agg2 = aggregate.aggregate_theory(kept2, 100, CFG["aggregation"])
    assert agg2["primary"] == "COMP" and agg2["primary_tie"]


def test_process_theory_parse_error_gives_und():
    out = process_theory(None, _row(TEXT), CFG)
    assert out["primary"] == "UND" and out["filter_stats"]["n_raw"] == 0


def test_hp_human_only_span_resets_level():
    text = "Зритель переживает субъективный опыт узнавания. Нейросеть раскрашивает фотографии."
    data = {"hp_level": 2, "hp_stance": "attribution",
            "hp_spans": [{"span": "Зритель переживает субъективный опыт узнавания", "subject": "человек"}]}
    out = process_hardproblem(data, _row(text), CFG)
    assert out["hp_level"] == 0 and out["hp_stance"] is None and "level_reset_no_verified_spans" in out["hp_flags"]
    assert out["filter_stats"]["dropped"] == {"human_subject": 1}


def test_hp_machine_span_keeps_level_and_stance():
    data = {"hp_level": 2, "hp_stance": "denial",
            "hp_spans": [{"span": "выполняет точные вычисления, но не имеет субъективного опыта", "subject": "машина"}]}
    out = process_hardproblem(data, _row(TEXT), CFG)
    assert out["hp_level"] == 2 and out["hp_stance"] == "denial" and out["hp_subject"] == "машина"
    bad = {"hp_level": 2, "hp_stance": "denial", "hp_spans": [{"span": "выдуманная цитата про квалиа", "subject": "машина"}]}
    assert process_hardproblem(bad, _row(TEXT), CFG)["hp_level"] == 0


# --------------------------------------------------------------------------- #
# Метрики согласия
# --------------------------------------------------------------------------- #

def test_cohen_kappa_known_value():
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert abs(agreement.cohen_kappa(a, b) - 0.4) < 1e-9
    assert agreement.cohen_kappa(a, a) == 1.0
    assert agreement.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2], weights="quadratic") == 1.0


def test_krippendorff_alpha_canonical_example():
    # Krippendorff (2011), nominal, 3 coders, 15 units, пропуски: α = 0.691
    A = [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3]
    B = [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None]
    C = [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4]
    units = list(zip(A, B, C))
    assert abs(agreement.krippendorff_alpha(units) - 0.691) < 0.002


def test_consensus_rules():
    pm = pd.DataFrame([
        {"slug": "m1", "id": 0, "primary": "COMP", "labels": ["COMP", "PAN"], "is_hybrid": True, "is_hybrid_raw": True},
        {"slug": "m2", "id": 0, "primary": "COMP", "labels": ["COMP"], "is_hybrid": False, "is_hybrid_raw": True},
        {"slug": "m3", "id": 0, "primary": "PAN", "labels": ["PAN", "COMP"], "is_hybrid": True, "is_hybrid_raw": False},
        {"slug": "m1", "id": 1, "primary": "COMP", "labels": ["COMP"], "is_hybrid": False, "is_hybrid_raw": False},
        {"slug": "m2", "id": 1, "primary": "PAN", "labels": ["PAN"], "is_hybrid": False, "is_hybrid_raw": False},
        {"slug": "m3", "id": 1, "primary": "EMERG", "labels": ["EMERG"], "is_hybrid": False, "is_hybrid_raw": False},
    ])
    c = agreement.consensus_theory(pm, 2).set_index("id")
    assert c.loc[0, "consensus_primary"] == "COMP" and not c.loc[0, "disputed"]
    assert c.loc[0, "consensus_labels"] == ["COMP", "PAN"] and c.loc[0, "is_hybrid"] and c.loc[0, "is_hybrid_raw"]
    assert c.loc[1, "consensus_primary"] == "UND" and c.loc[1, "disputed"] and c.loc[1, "consensus_labels"] == ["UND"]


def test_per_model_majority_over_repeats():
    long = pd.DataFrame([
        {"slug": "m", "id": 0, "repeat": 1, "primary": "COMP", "labels": ["COMP"], "secondary_raw": [], "is_hybrid_raw": False},
        {"slug": "m", "id": 0, "repeat": 2, "primary": "PAN", "labels": ["PAN"], "secondary_raw": [], "is_hybrid_raw": False},
        {"slug": "m", "id": 0, "repeat": 3, "primary": "PAN", "labels": ["PAN"], "secondary_raw": [], "is_hybrid_raw": False},
    ])
    pm = agreement.per_model_labels(long).iloc[0]
    assert pm["primary"] == "PAN" and pm["unstable"] and pm["labels"] == ["PAN"]


def test_legacy_loader_reads_december_run():
    entry = CFG["legacy_runs"][0]
    df = agreement.load_legacy_run(entry, CFG)
    assert df is not None and len(df) == 211 and set(df["primary"]) <= set(common.ALL_CLASSES)


# --------------------------------------------------------------------------- #
# Сквозной прогон на mock-провайдере: экспорт, импорт, статистика
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def mock_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v3")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["llm"]["provider"] = "mock"
    cfg["corpus"]["path"] = str(ROOT / "combined_ai_preprocessed.xlsx")
    cfg["paths"]["prompts_dir"] = str(ROOT / "prompts")
    cfg["paths"]["runs_dir"] = str(tmp / "runs")
    cfg["paths"]["reports_dir"] = str(tmp / "reports")
    cfg["paths"]["annotation_sheet"] = str(tmp / "reports" / "annotation_sheet.xlsx")
    cfg["legacy_runs"][0]["path"] = str(ROOT / cfg["legacy_runs"][0]["path"])
    cfg["stats"].update({"n_permutations": 300, "n_mc_tables": 300})
    cpath = tmp / "config.yaml"
    cpath.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    from pipeline import classify_hardproblem, classify_theory
    classify_theory.main(["--config", str(cpath), "--limit", "60", "--repeats", "2"])
    classify_hardproblem.main(["--config", str(cpath), "--limit", "60", "--repeats", "1"])
    cfg = common.load_config(cpath)
    aggregate.main(["--config", str(cpath)])
    agreement.build_report(cfg, common.reports_dir(cfg))
    export_annotation.build_sheet(cfg)
    return cfg


@pytest.fixture(scope="module")
def filled_env(mock_env):
    """Имитация заполненной кодировщиками таблицы: A = consensus, B = consensus с 10% случайных расхождений."""
    rd = Path(mock_env["paths"]["reports_dir"])
    path = rd / "annotation_sheet.xlsx"
    ann = pd.read_excel(path, "annotation")
    cons = pd.read_csv(rd / "consensus.csv").set_index("id")
    rng = np.random.default_rng(0)
    for col in ("theory_primary_A", "theory_primary_B", "hp_level_A", "hp_level_B"):
        ann[col] = ann[col].astype(object)
    for i in range(len(ann)):
        tid = int(ann.loc[i, "id"])
        lab = cons.loc[tid, "consensus_primary"]
        ann.loc[i, "theory_primary_A"] = lab
        ann.loc[i, "theory_primary_B"] = lab if rng.random() < 0.9 else "PAN"
        lvl = cons.loc[tid, "hp_level"]
        lvl = 0 if pd.isna(lvl) else int(lvl)
        ann.loc[i, "hp_level_A"] = lvl
        ann.loc[i, "hp_level_B"] = lvl
    author = pd.read_excel(path, "author_type")
    author["author_type"] = ["artist" if i % 2 else "curator" for i in range(len(author))]
    filled = rd / "filled.xlsx"
    with pd.ExcelWriter(filled) as xw:
        ann.to_excel(xw, sheet_name="annotation", index=False)
        author.to_excel(xw, sheet_name="author_type", index=False)
    import_annotation.build(mock_env, filled, rd)
    return mock_env


def test_runs_have_metadata(mock_env):
    runs = common.list_runs(mock_env, "theory")
    assert len(runs) == 3
    meta = runs[0]["meta"]
    assert meta["prompts"]["theory"]["hash"] == common.load_prompt(mock_env, "theory").hash
    assert meta["code_version"] and meta["temperature"] == 0.0 and meta["seed"] == 42
    rec = common.load_json(runs[0]["files"][1])[52]
    assert rec["title"] == "Демон" and rec["prompt_hash"] and rec["n_words"] > 0
    assert all("mapping" in e for e in rec["evidence"] if e.get("kept"))


def test_aggregate_reports(mock_env):
    rd = Path(mock_env["paths"]["reports_dir"])
    assert (rd / "filter_report.md").exists() and "share_dropped" in (rd / "filter_report.md").read_text(encoding="utf-8")
    assert "spearman_before" in (rd / "length_report.md").read_text(encoding="utf-8")


def test_agreement_and_consensus(mock_env):
    rd = Path(mock_env["paths"]["reports_dir"])
    cons = pd.read_csv(rd / "consensus.csv")
    assert len(cons) == 211 and {"consensus_primary", "disputed", "consensus_labels", "hp_level", "primary_legacy_deepseek-v3.2_2025-12"} <= set(cons.columns)
    md = (rd / "agreement_summary.md").read_text(encoding="utf-8")
    assert "Krippendorff α" in md and "Legacy" in md and "Test–retest" in md


def test_export_annotation_is_blind_and_deterministic(mock_env):
    rd = Path(mock_env["paths"]["reports_dir"])
    s1 = export_annotation.build_sheet(mock_env)
    s2 = export_annotation.build_sheet(mock_env)
    assert s1["id"].tolist() == s2["id"].tolist() and s1["id"].tolist() != sorted(s1["id"].tolist())
    xl = pd.ExcelFile(rd / "annotation_sheet.xlsx")
    assert {"annotation", "codebook", "author_type", "instructions"} <= set(xl.sheet_names)
    ann = pd.read_excel(xl, "annotation")
    assert len(ann) == 211 and ann["annotate_hp"].eq(1).all()
    for c in ("A", "B"):
        for col in (f"theory_primary_{c}", f"theory_secondary_{c}", f"hp_level_{c}", f"hp_stance_{c}", f"note_{c}"):
            assert col in ann.columns and ann[col].isna().all()
    assert not any(col.startswith(("primary_", "labels_", "consensus")) for col in ann.columns)
    assert ann["annotate_theory"].sum() >= 1
    cb = pd.read_excel(xl, "codebook")
    assert cb["codebook"].astype(str).str.contains("хранит аромат").any()
    assert "author_type" in pd.read_excel(xl, "author_type").columns


def test_import_annotation_computes_kappa(filled_env):
    rd = Path(filled_env["paths"]["reports_dir"])
    md = (rd / "human_agreement.md").read_text(encoding="utf-8")
    assert "human_A" in md and "consensus" in md and "Krippendorff" in md
    hl = pd.read_csv(rd / "human_labels.csv")
    assert "human_primary" in hl.columns and hl["human_primary"].notna().sum() > 150
    dis = pd.read_csv(rd / "disagreements.csv")
    assert len(dis) > 0 and "consensus_primary" in dis.columns
    assert (rd / "author_type.csv").exists()


def test_stats_outputs(filled_env):
    mock_env = filled_env
    out = stats.run(mock_env, "consensus")
    co = pd.read_csv(out / "cooccurrence_all.csv") if (out / "cooccurrence_all.csv").exists() else pd.DataFrame()
    if not co.empty:
        assert {"observed_AB", "expected_AB", "p_perm", "PMI", "std_resid", "p_fisher"} <= set(co.columns)
    assert (out / "chi2_source_x_primary.md").exists() and (out / "length_logit.md").exists()
    assert (out / "distribution_by_source.png").exists() and (out / "cooccurrence_residuals.png").exists()
    assert (out / "chi2_author_type_x_primary.md").exists()
    out_h = stats.run(mock_env, "human")
    assert (out_h / "distribution_all.md").exists()


def test_stat_helpers():
    rng = np.random.default_rng(1)
    ct = pd.DataFrame([[10, 2], [3, 12]])
    r = stats.chi2_report(ct, 500, rng)
    assert r["p_asymptotic"] < 0.01 and r["p_mc"] < 0.05 and r["std_resid"].shape == (2, 2)
    assert stats.fisher_p(ct, 500, rng)["method"] == "exact_2x2"
    big = pd.DataFrame([[8, 1, 1], [1, 8, 1], [1, 1, 8]])
    assert stats.fisher_p(big, 500, rng)["p"] < 0.05
    a = np.array([1] * 10 + [0] * 10); b = a.copy()
    assert stats.permutation_pair(a, b, 500, rng)["p_perm"] < 0.05
    x = np.arange(40) / 10.0; y = (x > 2).astype(int)
    y[5] = 1; y[30] = 0
    assert stats.logit_fit(x, y)["coef"] > 0


# --------------------------------------------------------------------------- #
# Live: реальная модель и промпт v3 (пропускаются без ключа)
# --------------------------------------------------------------------------- #

def _live_client():
    models = common.model_entries(CFG)
    model = os.getenv("TEST_MODEL") or models[0]["id"]
    return common.LLMClient(CFG, model, cache_dir=ROOT / "runs" / "_tests_cache")


def _live_theory(text: str):
    p = common.load_prompt(CFG, "theory")
    resp = _live_client().complete(p.system, p.user(text), seed=CFG["seed"])
    assert not resp.error, resp.error
    return process_theory(common.extract_json(resp.content), _row(text), CFG)


def _live_hp(text: str):
    p = common.load_prompt(CFG, "hardproblem")
    resp = _live_client().complete(p.system, p.user(text), seed=CFG["seed"])
    assert not resp.error, resp.error
    return process_hardproblem(common.extract_json(resp.content), _row(text), CFG)


@LIVE
@pytest.mark.live
@pytest.mark.parametrize("forbidden,text", COUNTEREXAMPLES, ids=[f"not_{c}_{i}" for i, (c, _) in enumerate(COUNTEREXAMPLES)])
def test_live_counterexamples(forbidden, text):
    out = _live_theory(text)
    assert forbidden not in out["labels"], json.dumps(out["evidence"], ensure_ascii=False)


@LIVE
@pytest.mark.live
@pytest.mark.parametrize("expected,text", POSITIVE_THEORY, ids=["comp_denial", "pan_attribution"])
def test_live_positive_examples(expected, text):
    out = _live_theory(text)
    assert out["primary"] == expected, json.dumps(out["evidence"], ensure_ascii=False)


@LIVE
@pytest.mark.live
@pytest.mark.parametrize("expected,text", HP_CASES, ids=["demon_level2", "human_experience_level0", "emotion_data_level0"])
def test_live_hardproblem(expected, text):
    out = _live_hp(text)
    assert out["hp_level"] == expected, json.dumps(out, ensure_ascii=False)


@LIVE
@pytest.mark.live
def test_live_demon_from_corpus():
    corpus = common.load_corpus(CFG)
    row = corpus[corpus["title"] == "Демон"].iloc[0]
    out = _live_hp(row["text"])
    assert out["hp_level"] == 2 and out["hp_stance"] == "denial", json.dumps(out, ensure_ascii=False)
