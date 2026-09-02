"""Детерминированная заглушка LLM для тестов и сквозной проверки пайплайна без API.

НЕ является классификатором: отвечает по нескольким регулярным выражениям,
чтобы прогнать валидацию, агрегацию, согласие и статистику на реальном корпусе.
Включается через llm.provider: mock в config.yaml.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

_THEORY_RULES = [
    ("COMP", re.compile(r"(нейросет|алгоритм|ии|машин|модел)\w*[^.!?]{0,60}(мысл|дума|понима|соавтор|автор|созна)", re.I),
     {"source": "вычислительная система", "target": "мыслящий субъект"}),
    ("COMP", re.compile(r"(вычислен)\w*[^.!?]{0,80}(не имеет|нет|лишен)[^.!?]{0,30}(субъективн|опыт|пережива)", re.I),
     {"source": "вычислительная система", "target": "субъект опыта (отрицание)"}),
    ("PRED", re.compile(r"(предсказ|предикт|ожидан|галлюцинац)\w*[^.!?]{0,60}(восприят|познан|мозг)", re.I),
     {"source": "предсказание", "target": "восприятие"}),
    ("PAN", re.compile(r"(матери|объект|вещ|камн|пиксел)\w*[^.!?]{0,40}(чувств|пережива|протоопыт|внутренн\w+ сторон)", re.I),
     {"source": "материя", "target": "носитель опыта"}),
    ("EMERG", re.compile(r"(эмерджент|самоорганиз|роев\w+ разум|коллективн\w+ интеллект)", re.I),
     {"source": "множество элементов", "target": "новое качество ума"}),
    ("ENACT", re.compile(r"(энактив|воплощ\w+ познан|познание\W+(есть|как)\W+действ|тело мысл|сенсомотор)", re.I),
     {"source": "тело/действие", "target": "познание"}),
    ("GWT", re.compile(r"(глобальн\w+ рабоч\w+ пространств|театр разума|прожектор внимания)", re.I),
     {"source": "сцена/доступ", "target": "сознание"}),
    ("IIT", re.compile(r"(интегрированн\w+ информац|\bphi\b|тонони)", re.I),
     {"source": "интеграция информации", "target": "сознание"}),
]

_HP2 = re.compile(r"(субъективн\w+ опыт|квалиа|каково это|переживает ли|чувствует ли|философск\w+ зомби|объяснительн\w+ разрыв|феноменальн)", re.I)
_HP1 = re.compile(r"(есть ли там кто|понимает ли|только имитир|за поверхностью|пустая оболочка|внутренн\w+ мир\w* (машин|систем|нейросет))", re.I)
_HUMAN = re.compile(r"(зрител|посетител|художни|автор|человек)", re.I)
_DENIAL = re.compile(r"(не имеет|нет|лишен|но не|только имитир)", re.I)
_ATTR = re.compile(r"(обладает|чувствует|переживает|есть внутренн)", re.I)
_QUESTION = re.compile(r"\?|вопрос|можем ли|неизвестно", re.I)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class MockLLM:
    def complete(self, system: str, user: str) -> str:
        m = re.search(r'"""\n(.*?)\n"""', user, re.S)
        text = m.group(1) if m else user
        if "hp_level" in system:
            return json.dumps(self._hardproblem(text), ensure_ascii=False)
        return json.dumps(self._theory(text), ensure_ascii=False)

    def _theory(self, text: str) -> Dict[str, Any]:
        ev: List[Dict[str, Any]] = []
        for sent in _sentences(text):
            for cls, rx, mapping in _THEORY_RULES:
                if rx.search(sent):
                    ev.append({
                        "class": cls, "span": sent, "mapping": dict(mapping),
                        "level": "meta_metaphor", "exclusion_checked": True,
                        "reasoning": "mock: совпадение по регулярному выражению",
                    })
                    break
        return {"evidence": ev, "notes": "mock"}

    def _hardproblem(self, text: str) -> Dict[str, Any]:
        spans, level = [], 0
        for sent in _sentences(text):
            lvl = 2 if _HP2.search(sent) else (1 if _HP1.search(sent) else 0)
            if not lvl:
                continue
            subject = "человек" if (_HUMAN.search(sent) and not re.search(r"(ии|нейросет|машин|алгоритм|робот|систем)", sent, re.I)) else "машина"
            spans.append({"span": sent, "subject": subject, "reasoning": "mock"})
            if subject != "человек":
                level = max(level, lvl)
        stance = None
        if level:
            joined = " ".join(s["span"] for s in spans)
            stance = "denial" if _DENIAL.search(joined) else ("attribution" if _ATTR.search(joined) else ("open_question" if _QUESTION.search(joined) else "open_question"))
        return {"hp_level": level, "hp_stance": stance, "hp_spans": spans, "notes": "mock"}
