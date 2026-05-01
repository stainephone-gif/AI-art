"""
Enhanced Consciousness Theory Classifier with Advanced Metaphor Analysis
Integrates metaphor_analyzer for multi-level metaphor detection
"""

import os
import json
import re
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import time
import warnings
from typing import Dict, List, Optional

from metaphor_analyzer import (
    MetaphorAnalyzer,
    create_enhanced_system_prompt,
    MetaphorLevel,
    MetaphorType
)

warnings.filterwarnings('ignore')
load_dotenv()


# Enhanced system prompt with metaphor analysis
ENHANCED_SYSTEM_PROMPT = """Вы — эксперт-социолог, специализирующийся на философии сознания и медиаискусстве.

Ваша задача: классифицировать описания художественных работ по теории сознания, которая неявно или явно лежит в их основе.

КЛАССЫ (7 основных + 1 неопределённый):
1. COMP — Computational Functionalism (вычислительный функционализм)
2. IIT — Integrated Information Theory (интегрированная информация)
3. PRED — Predictive Processing / Free Energy (предиктивное кодирование)
4. GWT — Global Workspace Theory (глобальное рабочее пространство)
5. ENACT — Enactivism (энактивизм/воплощённое познание)
6. PAN — Panpsychism (панпсихизм)
7. EMERG — Emergentism (эмерджентизм)
8. UND — Undetermined (неопределённо)

УРОВНИ ИНДИКАТОРОВ - РАСШИРЕННАЯ СИСТЕМА:

1. EXPLICIT_TERM (вес 1.0): Прямое упоминание термина, теории или автора
   - Примеры: "интегрированная информация Tononi", "предиктивное кодирование Friston"

2. SCIENTIFIC_METAPHOR (вес 0.7): Метафора в научном дискурсе о сознании
   - Примеры: "мозг как компьютер", "сознание как театр разума", "разум как предиктор"

3. META_METAPHOR (вес 0.8): Художественная интерпретация научной метафоры
   - КРИТИЧЕСКИ ВАЖНО для AI-арта!
   - Примеры:
     * "алгоритм как художник" (COMP)
     * "генеративная модель как воображение" (PRED)
     * "нейросеть создаёт сознание" (COMP/EMERG)
     * "тело как медиум познания" (ENACT)
     * "роевой разум" (EMERG)

4. NESTED_METAPHOR (вес 0.5): Многослойная метафорическая трансформация

ТИП МЕТАФОРЫ (множитель к весу):
- ONTOLOGICAL (×1.2): О природе сознания/разума - ПРИОРИТЕТ!
- STRUCTURAL (×1.0): О структуре/архитектуре системы
- ORIENTATIONAL (×0.8): О пространственно-временных отношениях
- DECORATIVE (×0.3): Декоративная, не теоретическая - ИГНОРИРОВАТЬ

КРИТЕРИИ ОЦЕНКИ С ФОКУСОМ НА МЕТАФОРЫ:

COMP - Computational Functionalism:
- EXPLICIT: Тьюринг, функционализм, символьная обработка, алгоритм (в теоретическом контексте)
- SCIENTIFIC_METAPHOR: "мозг как компьютер", "мышление как вычисление"
- META_METAPHOR: "алгоритм как автор/художник", "данные как материал искусства", "код генерирует сознание"

IIT - Integrated Information Theory:
- EXPLICIT: phi, Tononi, квалиа, интегрированная информация, несводимость
- SCIENTIFIC_METAPHOR: "опыт как неделимое целое", "phi как мера сознания"
- META_METAPHOR: "единство противоположностей", "интегрированное восприятие", "неделимый образ"

PRED - Predictive Processing:
- EXPLICIT: Friston, free energy, байесовский вывод, предиктивное кодирование, prior
- SCIENTIFIC_METAPHOR: "мозг как байесовский предсказатель", "восприятие как контролируемая галлюцинация"
- META_METAPHOR: "генеративная модель как воображение", "ошибка предсказания как креативность", "синтез восприятия"

GWT - Global Workspace Theory:
- EXPLICIT: Baars, глобальное рабочее пространство, broadcast, доступ к сознанию
- SCIENTIFIC_METAPHOR: "сознание как театр разума", "внимание как прожектор"
- META_METAPHOR: "сцена восприятия", "фокус внимания как выбор образа", "освещение смысла"

ENACT - Enactivism:
- EXPLICIT: Varela, embodied cognition, сенсомоторные петли, enacted mind
- SCIENTIFIC_METAPHOR: "познание как действие", "разум как воплощённый процесс"
- META_METAPHOR: "тело как медиум", "жест как мысль", "движение как познание", "материальность разума"

PAN - Panpsychism:
- EXPLICIT: Strawson, Goff, протоквалиа, панпсихизм, протоопыт
- SCIENTIFIC_METAPHOR: "материя как чувствующая", "сознание как фундаментальное свойство"
- META_METAPHOR: "живая материя", "одушевлённые объекты", "чувствующие системы", "агентность материала"

EMERG - Emergentism:
- EXPLICIT: эмерджентность, самоорганизация, фазовый переход, коллективный интеллект
- SCIENTIFIC_METAPHOR: "сознание как эмерджентное свойство", "разум как самоорганизующаяся система"
- META_METAPHOR: "роевой разум", "коллективное творчество", "спонтанный порядок", "множественность в единстве"

ПАТТЕРН "МЕТАФОРА МЕТАФОРЫ" для AI-арта:

Если работа использует AI/ML и описание содержит:
1. Технический термин (нейросеть, алгоритм, генеративная модель)
2. + Художественную интерпретацию (создаёт, воображает, творит, познаёт)
3. + Связь с сознанием/разумом/опытом

→ Это META_METAPHOR! Вес 0.8, умноженный на тип метафоры.

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
1. Для каждого класса оцените вероятность от 0.0 до 1.0
2. ДЛЯ ЛЮБОЙ ОЦЕНКИ > 0.3 обязательно приведите ТОЧНЫЕ ЦИТАТЫ
3. Укажите уровень метафоры: explicit_term / scientific_metaphor / meta_metaphor / nested_metaphor
4. Укажите тип метафоры: ontological / structural / orientational / decorative
5. Если уверенности нет (все оценки < 0.5) → основной класс UND
6. Отвечайте ТОЛЬКО в формате JSON (без преамбулы, без markdown-блоков)

ФОРМАТ ОТВЕТА (строго JSON):
{
  "primary_class": "PRED",
  "confidence": "high",
  "scores": {
    "COMP": 0.3,
    "IIT": 0.2,
    "PRED": 0.9,
    "GWT": 0.0,
    "ENACT": 0.4,
    "PAN": 0.0,
    "EMERG": 0.2,
    "UND": 0.0
  },
  "evidence": [
    {
      "class": "PRED",
      "metaphor_level": "meta_metaphor",
      "metaphor_type": "ontological",
      "span": "генеративная модель создаёт образы, подобно воображению",
      "weight": 0.96,
      "reasoning": "Художественная интерпретация предиктивного кодирования: генеративная модель (научный термин) трансформирована в метафору воображения/творчества"
    },
    {
      "class": "COMP",
      "metaphor_level": "meta_metaphor",
      "metaphor_type": "ontological",
      "span": "алгоритм как художник",
      "weight": 0.80,
      "reasoning": "META_METAPHOR: технический термин 'алгоритм' используется метафорически для описания творческого агента"
    },
    {
      "class": "ENACT",
      "metaphor_level": "scientific_metaphor",
      "metaphor_type": "structural",
      "span": "сенсомоторное взаимодействие со средой",
      "weight": 0.70,
      "reasoning": "Научная метафора энактивизма в контексте воплощённого познания"
    }
  ],
  "secondary_class": "COMP",
  "is_hybrid": true,
  "metaphor_analysis": {
    "meta_metaphor_detected": true,
    "meta_metaphor": {
      "present": true,
      "source_phrase": "генеративная модель создаёт образы",
      "transformation_phrase": "подобно воображению",
      "theory": "PRED",
      "confidence": "high"
    },
    "dominant_semantic_field": "predictive_generative",
    "artistic_transformation": "scientific concept → artistic practice",
    "theoretical_grounding": "PRED + COMP hybrid"
  },
  "notes": "Работа демонстрирует паттерн 'метафора метафоры': научные концепции предиктивного кодирования и вычислительности художественно переосмыслены через практику генеративного искусства"
}

ВАЖНО:
- Не придумывайте цитаты — копируйте ТОЧНО из описания
- META_METAPHOR — ключевой индикатор для AI-арта!
- Если термин технический, но используется метафорически → meta_metaphor, не explicit_term
- Метафоры учитываются только если ONTOLOGICAL или STRUCTURAL
- При обнаружении meta_metaphor укажите в reasoning трансформацию: "X (научное) → Y (художественное)"
- Декоративные метафоры (metaphor_type: decorative) получают низкий вес ×0.3
- Поле meta_metaphor ОБЯЗАТЕЛЬНО заполнить если meta_metaphor_detected=true:
    source_phrase — цитата из текста с исходной научной концепцией
    transformation_phrase — цитата из текста с художественным переосмыслением
    theory — теория, к которой относится (COMP/IIT/PRED/GWT/ENACT/PAN/EMERG)
    confidence — уверенность в детекции (high/medium/low)
- Если meta_metaphor_detected=false → meta_metaphor.present=false, остальные поля пустые строки
"""


USER_PROMPT_TEMPLATE = """Проанализируй следующее описание художественной работы и классифицируй по теории сознания.

КОНТЕКСТ: Это работа AI-арта. Художники используют алгоритмы ML как инструмент и часто метафорически переосмысливают научные концепции сознания.

ОПИСАНИЕ:
\"\"\"
{description}
\"\"\"

Проведи анализ поэтапно:
1. Найди все явные термины (explicit_term)
2. Выяви научные метафоры (scientific_metaphor)
3. КРИТИЧЕСКИ: Определи художественные интерпретации научных метафор (meta_metaphor)
4. Оцени тип каждой метафоры (ontological/structural/orientational/decorative)
5. Рассчитай веса с учётом уровня и типа метафоры
6. Определи паттерн "метафора метафоры" если присутствует
7. Выведи результат в JSON-формате

Ответь ТОЛЬКО JSON без дополнительного текста."""


class EnhancedMetaphorClassifier:
    """
    Enhanced classifier with integrated metaphor analysis
    """

    def __init__(self):
        """Initialize with OpenRouter API and metaphor analyzer"""
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = os.getenv(
            'API_BASE_URL', 'https://openrouter.ai/api/v1/chat/completions'
        )
        self.model = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        self.request_delay = float(os.getenv('RATE_LIMIT_DELAY', '2.0'))

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv('HTTP_REFERER', 'http://localhost:3000'),
            "X-Title": os.getenv('X_TITLE', 'Enhanced Consciousness Classifier')
        }

        # Initialize metaphor analyzer
        self.metaphor_analyzer = MetaphorAnalyzer()

    def pre_analyze_metaphors(self, description: str) -> Dict:
        """
        Pre-analyze metaphors before LLM classification.
        Provides keyword-based hints to the LLM.

        DEPRECATED (meta_metaphor_present field only):
            The `meta_metaphor_present` field in the returned dict is always False
            because MetaphorAnalyzer.analyze_metaphor_network() cannot detect
            meta-metaphors via keyword co-occurrence in real curatorial texts.
            Retained as a scaffold; all other fields (detected_metaphors, semantic_fields,
            evidence_preview) remain valid and are used as pre-classification hints.
        """
        evidence = self.metaphor_analyzer.extract_metaphors(description)
        network = self.metaphor_analyzer.analyze_metaphor_network(evidence)

        return {
            'detected_metaphors': len(evidence),
            'meta_metaphor_present': network.get('meta_metaphor_detected', False),
            'dominant_theory': network.get('dominant_pattern'),
            'semantic_fields': [e.semantic_field for e in evidence[:5]],
            'evidence_preview': [
                {
                    'theory': e.theory_class,
                    'level': e.level.value,
                    'type': e.type.value,
                    'span': e.span[:100]
                }
                for e in evidence[:3]
            ]
        }

    # --- JSON Validation & Fallback ---

    CLASSIFIER_VERSION = "v2.0-structured-meta-metaphor-2026-05"

    VALID_CLASSES = {'COMP', 'IIT', 'PRED', 'GWT', 'ENACT', 'PAN', 'EMERG', 'UND'}
    VALID_CONFIDENCE = {'high', 'medium', 'low'}
    VALID_METAPHOR_LEVELS = {'explicit_term', 'scientific_metaphor', 'meta_metaphor', 'nested_metaphor'}
    VALID_METAPHOR_TYPES = {'ontological', 'structural', 'orientational', 'decorative'}

    # --- Verbatim quote verification ---

    @staticmethod
    def _normalize_for_match(s: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _verify_phrase_in_source(self, phrase: str, source_text: str,
                                   token_overlap_threshold: float = 0.7) -> Dict:
        """
        Verify that a quoted phrase actually exists in the source text.
        Returns metadata dict so downstream code can audit / filter.

        Methods (in order of strictness):
          - exact:           normalized substring match
          - token_overlap:   fraction of phrase tokens (>2 chars) present in source
        """
        if not phrase or not phrase.strip():
            return {'verified': True, 'token_overlap': 1.0, 'method': 'empty'}

        norm_phrase = self._normalize_for_match(phrase)
        norm_source = self._normalize_for_match(source_text)

        if not norm_phrase:
            return {'verified': True, 'token_overlap': 1.0, 'method': 'empty_after_norm'}

        if norm_phrase in norm_source:
            return {'verified': True, 'token_overlap': 1.0, 'method': 'exact'}

        phrase_tokens = [t for t in norm_phrase.split() if len(t) > 2]
        if not phrase_tokens:
            return {'verified': True, 'token_overlap': 1.0, 'method': 'too_short'}

        source_tokens = set(norm_source.split())

        def token_matches(t: str) -> bool:
            # Exact match
            if t in source_tokens:
                return True
            # Prefix match for words >5 chars (handles Russian inflection:
            # "генеративная" / "генеративную" share 9-char prefix)
            if len(t) > 5:
                prefix = t[:max(5, len(t) - 3)]
                return any(s.startswith(prefix) for s in source_tokens)
            return False

        overlap = sum(1 for t in phrase_tokens if token_matches(t)) / len(phrase_tokens)

        return {
            'verified': overlap >= token_overlap_threshold,
            'token_overlap': round(overlap, 2),
            'method': 'token_overlap_with_prefix'
        }

    def _annotate_quote_verification(self, classification: Dict, description: str) -> Dict:
        """
        Add quote_verification metadata to every cited span:
          - evidence[].quote_verification
          - metaphor_analysis.meta_metaphor.{source_phrase,transformation_phrase}_verification
        Also computes aggregate verified_quote_rate at top level.
        """
        verified = 0
        total = 0

        for ev in classification.get('evidence', []):
            v = self._verify_phrase_in_source(ev.get('span', ''), description)
            ev['quote_verification'] = v
            total += 1
            if v['verified']:
                verified += 1

        ma = classification.get('metaphor_analysis')
        if isinstance(ma, dict):
            mm = ma.get('meta_metaphor')
            if isinstance(mm, dict):
                for field in ('source_phrase', 'transformation_phrase'):
                    v = self._verify_phrase_in_source(mm.get(field, ''), description)
                    mm[f'{field}_verification'] = v
                    if mm.get(field, '').strip():
                        total += 1
                        if v['verified']:
                            verified += 1

        classification['verified_quote_rate'] = (
            round(verified / total, 3) if total else 1.0
        )
        return classification

    @staticmethod
    def _extract_json_from_response(content: str) -> str:
        """
        Robustly extract JSON from LLM response, handling markdown fences,
        preamble text, trailing garbage, etc.
        """
        # Strip markdown fences
        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
        content = re.sub(r'\s*```\s*$', '', content.strip())

        # Try to find JSON object boundaries
        brace_start = content.find('{')
        if brace_start == -1:
            return content.strip()

        # Walk forward to find balanced closing brace
        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(content)):
            ch = content[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return content[brace_start:i + 1]

        # Fallback: return from first brace to end
        return content[brace_start:]

    def _validate_classification(self, data: Dict) -> Dict:
        """
        Validate and normalise a classification dict.
        Fills in missing fields with safe defaults so downstream code never crashes.
        Returns the (possibly repaired) dict with a 'validation_warnings' list.
        """
        warnings_list: List[str] = []

        # -- primary_class --
        pc = data.get('primary_class', '')
        if pc not in self.VALID_CLASSES:
            warnings_list.append(f"Invalid primary_class '{pc}', falling back to UND")
            data['primary_class'] = 'UND'

        # -- confidence --
        conf = data.get('confidence', '')
        if conf not in self.VALID_CONFIDENCE:
            warnings_list.append(f"Invalid confidence '{conf}', defaulting to 'low'")
            data['confidence'] = 'low'

        # -- scores --
        scores = data.get('scores')
        if not isinstance(scores, dict):
            warnings_list.append("Missing or invalid 'scores', generating from pre-analysis")
            data['scores'] = {cls: 0.0 for cls in self.VALID_CLASSES}
            if data['primary_class'] != 'UND':
                data['scores'][data['primary_class']] = 0.5
        else:
            for cls in self.VALID_CLASSES:
                val = scores.get(cls)
                if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                    scores[cls] = 0.0
            data['scores'] = scores

        # -- evidence --
        evidence = data.get('evidence')
        if not isinstance(evidence, list):
            warnings_list.append("Missing or invalid 'evidence', set to empty list")
            data['evidence'] = []
        else:
            clean_evidence = []
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                # Normalise each evidence entry
                item.setdefault('class', 'UND')
                if item['class'] not in self.VALID_CLASSES:
                    item['class'] = 'UND'
                item.setdefault('metaphor_level', 'nested_metaphor')
                if item['metaphor_level'] not in self.VALID_METAPHOR_LEVELS:
                    item['metaphor_level'] = 'nested_metaphor'
                item.setdefault('metaphor_type', 'decorative')
                if item['metaphor_type'] not in self.VALID_METAPHOR_TYPES:
                    item['metaphor_type'] = 'decorative'
                item.setdefault('span', '')
                item.setdefault('weight', 0.0)
                if not isinstance(item['weight'], (int, float)):
                    item['weight'] = 0.0
                item.setdefault('reasoning', '')
                clean_evidence.append(item)
            data['evidence'] = clean_evidence

        # -- secondary_class / is_hybrid --
        sc = data.get('secondary_class', '')
        if sc and sc not in self.VALID_CLASSES:
            data['secondary_class'] = None
            data['is_hybrid'] = False
        data.setdefault('is_hybrid', False)

        # -- metaphor_analysis --
        ma = data.get('metaphor_analysis')
        if not isinstance(ma, dict):
            data['metaphor_analysis'] = {
                'meta_metaphor_detected': False,
                'meta_metaphor': {
                    'present': False, 'source_phrase': '',
                    'transformation_phrase': '', 'theory': '', 'confidence': ''
                },
                'dominant_semantic_field': 'unknown',
                'artistic_transformation': '',
                'theoretical_grounding': ''
            }
        else:
            ma.setdefault('meta_metaphor_detected', False)
            ma.setdefault('dominant_semantic_field', 'unknown')
            ma.setdefault('artistic_transformation', '')
            ma.setdefault('theoretical_grounding', '')
            # Validate nested meta_metaphor object
            mm = ma.get('meta_metaphor')
            if not isinstance(mm, dict):
                ma['meta_metaphor'] = {
                    'present': ma.get('meta_metaphor_detected', False),
                    'source_phrase': '', 'transformation_phrase': '',
                    'theory': '', 'confidence': ''
                }
            else:
                mm.setdefault('present', ma.get('meta_metaphor_detected', False))
                mm.setdefault('source_phrase', '')
                mm.setdefault('transformation_phrase', '')
                if mm.get('theory', '') not in (self.VALID_CLASSES | {''}):
                    mm['theory'] = ''
                if mm.get('confidence', '') not in ('high', 'medium', 'low', ''):
                    mm['confidence'] = ''

        data.setdefault('notes', '')
        data.setdefault('status', 'success')

        if warnings_list:
            data['validation_warnings'] = warnings_list

        return data

    def _build_fallback_from_pre_analysis(self, metaphor_hints: Dict,
                                           description: str) -> Dict:
        """
        When the LLM response is completely unparseable, build a minimal
        classification from the local pre-analysis so the pipeline never
        stops.
        """
        dominant = metaphor_hints.get('dominant_theory') or 'UND'
        if dominant not in self.VALID_CLASSES:
            dominant = 'UND'

        scores = {cls: 0.0 for cls in self.VALID_CLASSES}
        if dominant != 'UND':
            scores[dominant] = 0.4

        evidence = []
        for ep in metaphor_hints.get('evidence_preview', []):
            evidence.append({
                'class': ep.get('theory', 'UND'),
                'metaphor_level': ep.get('level', 'nested_metaphor'),
                'metaphor_type': ep.get('type', 'decorative'),
                'span': ep.get('span', ''),
                'weight': 0.3,
                'reasoning': 'Fallback: built from local pre-analysis'
            })

        return {
            'primary_class': dominant,
            'confidence': 'low',
            'scores': scores,
            'evidence': evidence,
            'is_hybrid': False,
            'metaphor_analysis': {
                'meta_metaphor_detected': metaphor_hints.get('meta_metaphor_present', False),
                'dominant_semantic_field': 'unknown',
                'artistic_transformation': '',
                'theoretical_grounding': ''
            },
            'notes': 'Fallback classification from local pre-analysis (LLM response was unparseable)',
            'status': 'fallback',
            'metaphor_pre_analysis': metaphor_hints
        }

    def classify_description(self, description: str, max_retries: int = 3) -> Dict:
        """
        Classify with enhanced metaphor analysis.
        Includes robust JSON extraction, validation and fallback.
        """
        # Pre-analyze metaphors
        metaphor_hints = self.pre_analyze_metaphors(description)

        # Create enhanced user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(description=description)

        # Add metaphor hints to help LLM
        if metaphor_hints['meta_metaphor_present']:
            user_prompt += (
                f"\n\nПОДСКАЗКА: Обнаружены метафоры метафоры. "
                f"Доминирующая теория: {metaphor_hints['dominant_theory']}"
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.0
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']

                    # Robust JSON extraction
                    cleaned = self._extract_json_from_response(content)

                    try:
                        classification = json.loads(cleaned)
                    except json.JSONDecodeError as je:
                        print(f"  JSON parse error (attempt {attempt + 1}): {je}")
                        print(f"  Raw response (first 300 chars): {content[:300]}")
                        last_error = str(je)
                        time.sleep(1)
                        continue

                    # Validate & repair
                    classification = self._validate_classification(classification)
                    classification = self._annotate_quote_verification(
                        classification, description
                    )
                    classification['metaphor_pre_analysis'] = metaphor_hints
                    classification['classifier_version'] = self.CLASSIFIER_VERSION
                    classification['model'] = self.model
                    return classification

                elif response.status_code == 429:
                    wait_time = 10 * (2 ** attempt)
                    print(f"  Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    print(f"  API Error: {last_error}")

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"  Network error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                print(f"  Unexpected error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        # All retries exhausted — use local fallback
        print(f"  All retries exhausted. Using local fallback classification.")
        fallback = self._build_fallback_from_pre_analysis(metaphor_hints, description)
        fallback['error'] = last_error or 'Max retries exceeded'
        fallback['classifier_version'] = self.CLASSIFIER_VERSION + '-fallback'
        fallback['model'] = self.model
        return fallback

    def process_excel_file(self, file_path: str) -> pd.DataFrame:
        """Process Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name='data')

            text_columns = ['descr', 'descr_en', 'descr_clean', 'descr_lemmas']
            available_cols = [col for col in text_columns if col in df.columns]

            if not available_cols:
                raise ValueError("No text columns found in Excel file")

            df['combined_description'] = df[available_cols].fillna('').agg(' '.join, axis=1)
            df['combined_description'] = df['combined_description'].str.strip()

            return df

        except Exception as e:
            print(f"Error processing Excel file: {e}")
            raise

    def classify_batch(self, descriptions: List[str], titles: Optional[List[str]] = None) -> List[Dict]:
        """Classify multiple descriptions"""
        results = []

        for idx, desc in enumerate(descriptions):
            if pd.isna(desc) or not str(desc).strip():
                results.append({
                    'index': idx,
                    'title': titles[idx] if titles else f'Item_{idx}',
                    'primary_class': 'UND',
                    'error': 'Empty description',
                    'status': 'skipped'
                })
                continue

            print(f"Processing {idx + 1}/{len(descriptions)}: {titles[idx] if titles else f'Item_{idx}'}")

            result = self.classify_description(str(desc))
            result['index'] = idx
            result['title'] = titles[idx] if titles else f'Item_{idx}'
            result['description_length'] = len(str(desc))

            results.append(result)

            time.sleep(self.request_delay)

        return results

    def create_enhanced_visualizations(self, results: List[Dict], output_dir: str):
        """Create visualizations with metaphor analysis"""
        os.makedirs(output_dir, exist_ok=True)

        df_results = pd.DataFrame(results)
        successful = df_results[df_results['status'] != 'error']

        if len(successful) == 0:
            print("No successful classifications to visualize")
            return

        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))

        # 1. Primary class distribution
        ax1 = plt.subplot(3, 3, 1)
        class_counts = successful['primary_class'].value_counts()
        class_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax1)
        ax1.set_title('Distribution of Primary Classes', fontweight='bold')
        ax1.set_xlabel('Theory')
        ax1.set_ylabel('Count')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Meta-metaphor detection rate
        ax2 = plt.subplot(3, 3, 2)
        meta_metaphor_count = 0
        for _, row in successful.iterrows():
            if 'metaphor_analysis' in row and isinstance(row['metaphor_analysis'], dict):
                if row['metaphor_analysis'].get('meta_metaphor_detected', False):
                    meta_metaphor_count += 1

        meta_data = pd.DataFrame({
            'Type': ['Meta-Metaphor', 'Standard'],
            'Count': [meta_metaphor_count, len(successful) - meta_metaphor_count]
        })
        ax2.pie(meta_data['Count'], labels=meta_data['Type'], autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff'])
        ax2.set_title('Meta-Metaphor Detection Rate', fontweight='bold')

        # 3. Hybrid theories
        ax3 = plt.subplot(3, 3, 3)
        hybrid_count = sum(1 for _, row in successful.iterrows()
                          if row.get('is_hybrid', False))
        hybrid_data = pd.DataFrame({
            'Type': ['Hybrid', 'Single Theory'],
            'Count': [hybrid_count, len(successful) - hybrid_count]
        })
        ax3.pie(hybrid_data['Count'], labels=hybrid_data['Type'], autopct='%1.1f%%',
               colors=['#ffcc99', '#99ff99'])
        ax3.set_title('Theory Hybridity', fontweight='bold')

        # 4. Average scores by theory
        ax4 = plt.subplot(3, 3, 4)
        scores_data = []
        for _, row in successful.iterrows():
            if 'scores' in row and isinstance(row['scores'], dict):
                for class_name, score in row['scores'].items():
                    scores_data.append({'class': class_name, 'score': score})

        if scores_data:
            scores_df = pd.DataFrame(scores_data)
            avg_scores = scores_df.groupby('class')['score'].mean().sort_values(ascending=False)
            avg_scores.plot(kind='barh', color='lightcoral', edgecolor='black', ax=ax4)
            ax4.set_title('Average Scores by Theory', fontweight='bold')
            ax4.set_xlabel('Average Score')

        # 5. Metaphor levels distribution
        ax5 = plt.subplot(3, 3, 5)
        metaphor_levels = []
        for _, row in successful.iterrows():
            if 'evidence' in row and isinstance(row['evidence'], list):
                for evidence in row['evidence']:
                    if 'metaphor_level' in evidence:
                        metaphor_levels.append(evidence['metaphor_level'])

        if metaphor_levels:
            level_counts = pd.Series(metaphor_levels).value_counts()
            level_counts.plot(kind='bar', color='mediumpurple', edgecolor='black', ax=ax5)
            ax5.set_title('Metaphor Levels Distribution', fontweight='bold')
            ax5.set_xlabel('Metaphor Level')
            ax5.set_ylabel('Count')
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

        # 6. Metaphor types distribution
        ax6 = plt.subplot(3, 3, 6)
        metaphor_types = []
        for _, row in successful.iterrows():
            if 'evidence' in row and isinstance(row['evidence'], list):
                for evidence in row['evidence']:
                    if 'metaphor_type' in evidence:
                        metaphor_types.append(evidence['metaphor_type'])

        if metaphor_types:
            type_counts = pd.Series(metaphor_types).value_counts()
            type_counts.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax6)
            ax6.set_title('Metaphor Types Distribution', fontweight='bold')
            ax6.set_xlabel('Metaphor Type')
            ax6.set_ylabel('Count')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        # 7. Confidence distribution
        ax7 = plt.subplot(3, 3, 7)
        if 'confidence' in successful.columns:
            confidence_counts = successful['confidence'].value_counts()
            colors_conf = {'high': 'lightgreen', 'medium': 'orange', 'low': 'lightcoral'}
            confidence_counts.plot(kind='bar',
                                  color=[colors_conf.get(x, 'gray') for x in confidence_counts.index],
                                  edgecolor='black', ax=ax7)
            ax7.set_title('Confidence Distribution', fontweight='bold')
            plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

        # 8. Description length vs classification
        ax8 = plt.subplot(3, 3, 8)
        if 'description_length' in successful.columns:
            theory_lengths = successful.groupby('primary_class')['description_length'].mean()
            theory_lengths.plot(kind='barh', color='wheat', edgecolor='black', ax=ax8)
            ax8.set_title('Avg Description Length by Theory', fontweight='bold')
            ax8.set_xlabel('Characters')

        # 9. Scores heatmap
        ax9 = plt.subplot(3, 3, 9)
        if scores_data:
            pivot_data = []
            for theory in ['COMP', 'IIT', 'PRED', 'GWT', 'ENACT', 'PAN', 'EMERG', 'UND']:
                theory_scores = [s['score'] for s in scores_data if s['class'] == theory]
                if theory_scores:
                    pivot_data.append(sum(theory_scores) / len(theory_scores))
                else:
                    pivot_data.append(0)

            sns.heatmap([pivot_data], annot=True, fmt='.2f',
                       xticklabels=['COMP', 'IIT', 'PRED', 'GWT', 'ENACT', 'PAN', 'EMERG', 'UND'],
                       yticklabels=['Avg Score'], cmap='YlOrRd', ax=ax9)
            ax9.set_title('Theory Scores Heatmap', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enhanced_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Enhanced visualizations saved to {output_dir}")

    # ------------------------------------------------------------------ #
    #  Consistency / Inter-rater Reliability Evaluation
    # ------------------------------------------------------------------ #

    def evaluate_consistency(
        self,
        descriptions: List[str],
        titles: Optional[List[str]] = None,
        num_rounds: int = 3,
        temperature_variation: bool = True,
    ) -> Dict:
        """
        Run multiple classification rounds on the same texts and compute
        agreement metrics.

        Args:
            descriptions: Texts to classify.
            titles: Optional human-readable titles.
            num_rounds: How many independent classification passes to run.
            temperature_variation: If True, use slightly different temperatures
                per round (0.0, 0.3, 0.6 …) to probe model stability.
                If False, use temperature=0.0 for all rounds.

        Returns:
            Dict with per-item and aggregate agreement statistics.
        """
        from collections import Counter

        temperatures = [0.0]
        if temperature_variation and num_rounds > 1:
            step = 0.6 / (num_rounds - 1)
            temperatures = [round(step * i, 2) for i in range(num_rounds)]

        # Pad temperatures list if needed
        while len(temperatures) < num_rounds:
            temperatures.append(temperatures[-1])

        all_rounds: List[List[Dict]] = []

        for round_idx in range(num_rounds):
            temp = temperatures[round_idx]
            print(f"\n--- Consistency round {round_idx + 1}/{num_rounds} "
                  f"(temperature={temp}) ---")

            round_results = []
            for idx, desc in enumerate(descriptions):
                if pd.isna(desc) or not str(desc).strip():
                    round_results.append({
                        'primary_class': 'UND',
                        'scores': {c: 0.0 for c in self.VALID_CLASSES},
                        'status': 'skipped'
                    })
                    continue

                title = titles[idx] if titles else f'Item_{idx}'
                print(f"  [{round_idx+1}] Processing {idx+1}/{len(descriptions)}: {title}")

                # Override temperature for this round
                orig_temp = None  # we patch the payload inside classify_description
                result = self._classify_with_temperature(str(desc), temp)
                round_results.append(result)
                time.sleep(0.5)

            all_rounds.append(round_results)

        # --- Compute agreement metrics ---
        n_items = len(descriptions)
        item_reports = []

        total_full_agreement = 0
        total_majority_agreement = 0

        for idx in range(n_items):
            classes_per_round = [
                all_rounds[r][idx].get('primary_class', 'UND')
                for r in range(num_rounds)
            ]
            counter = Counter(classes_per_round)
            majority_class, majority_count = counter.most_common(1)[0]
            agreement_ratio = majority_count / num_rounds

            is_full_agreement = (majority_count == num_rounds)
            if is_full_agreement:
                total_full_agreement += 1
            if majority_count > num_rounds / 2:
                total_majority_agreement += 1

            # Pairwise score correlation (average cosine-like similarity
            # across score vectors)
            score_vectors = []
            for r in range(num_rounds):
                scores = all_rounds[r][idx].get('scores', {})
                vec = [scores.get(c, 0.0) for c in sorted(self.VALID_CLASSES)]
                score_vectors.append(vec)

            avg_pairwise_sim = self._avg_pairwise_similarity(score_vectors)

            item_reports.append({
                'index': idx,
                'title': titles[idx] if titles else f'Item_{idx}',
                'classes_per_round': classes_per_round,
                'majority_class': majority_class,
                'agreement_ratio': round(agreement_ratio, 3),
                'full_agreement': is_full_agreement,
                'score_similarity': round(avg_pairwise_sim, 3),
            })

        # Aggregate
        n_valid = max(n_items, 1)
        aggregate = {
            'num_items': n_items,
            'num_rounds': num_rounds,
            'temperatures_used': temperatures[:num_rounds],
            'full_agreement_rate': round(total_full_agreement / n_valid, 3),
            'majority_agreement_rate': round(total_majority_agreement / n_valid, 3),
            'avg_score_similarity': round(
                sum(r['score_similarity'] for r in item_reports) / n_valid, 3
            ),
        }

        # Fleiss' kappa (simplified for multiple raters, multiple categories)
        fleiss_kappa = self._compute_fleiss_kappa(
            [[all_rounds[r][idx].get('primary_class', 'UND')
              for r in range(num_rounds)]
             for idx in range(n_items)],
            list(self.VALID_CLASSES)
        )
        aggregate['fleiss_kappa'] = round(fleiss_kappa, 4)

        return {
            'aggregate': aggregate,
            'items': item_reports,
            'raw_rounds': [
                [{'primary_class': all_rounds[r][i].get('primary_class', 'UND'),
                  'confidence': all_rounds[r][i].get('confidence', ''),
                  'scores': all_rounds[r][i].get('scores', {})}
                 for i in range(n_items)]
                for r in range(num_rounds)
            ]
        }

    def _classify_with_temperature(self, description: str, temperature: float) -> Dict:
        """Classify a single description with a specific temperature."""
        metaphor_hints = self.pre_analyze_metaphors(description)
        user_prompt = USER_PROMPT_TEMPLATE.format(description=description)
        if metaphor_hints['meta_metaphor_present']:
            user_prompt += (
                f"\n\nПОДСКАЗКА: Обнаружены метафоры метафоры. "
                f"Доминирующая теория: {metaphor_hints['dominant_theory']}"
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 3000,
            "temperature": temperature
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url, headers=self.headers,
                    json=payload, timeout=60
                )
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    cleaned = self._extract_json_from_response(content)
                    try:
                        classification = json.loads(cleaned)
                    except json.JSONDecodeError:
                        time.sleep(1)
                        continue
                    classification = self._validate_classification(classification)
                    classification['metaphor_pre_analysis'] = metaphor_hints
                    return classification
                elif response.status_code == 429:
                    time.sleep(10 * (2 ** attempt))
                    continue
            except Exception as e:
                time.sleep(10 * (2 ** attempt))

        fallback = self._build_fallback_from_pre_analysis(metaphor_hints, description)
        return fallback

    @staticmethod
    def _avg_pairwise_similarity(vectors: List[List[float]]) -> float:
        """Cosine similarity averaged over all pairs."""
        import math
        n = len(vectors)
        if n < 2:
            return 1.0

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return dot / (na * nb)

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += cosine(vectors[i], vectors[j])
                count += 1
        return total / count if count else 0.0

    @staticmethod
    def _compute_fleiss_kappa(
        ratings: List[List[str]],
        categories: List[str]
    ) -> float:
        """
        Compute Fleiss' kappa for inter-rater reliability.
        ratings: list of items, each item is a list of category labels
                 (one per rater/round).
        categories: list of all possible category labels.
        """
        n = len(ratings)       # number of items
        if n == 0:
            return 0.0
        k = len(ratings[0])   # number of raters
        if k < 2:
            return 0.0

        cat_index = {c: i for i, c in enumerate(categories)}
        num_cats = len(categories)

        # Build rating matrix: n_ij = number of raters who assigned
        # category j to item i
        matrix = []
        for item_ratings in ratings:
            row = [0] * num_cats
            for label in item_ratings:
                idx = cat_index.get(label)
                if idx is not None:
                    row[idx] += 1
            matrix.append(row)

        # P_i for each item
        P_items = []
        for row in matrix:
            s = sum(r * (r - 1) for r in row)
            P_items.append(s / (k * (k - 1)) if k > 1 else 0.0)

        P_bar = sum(P_items) / n

        # p_j: proportion of all assignments to category j
        p_j = []
        total_assignments = n * k
        for j in range(num_cats):
            col_sum = sum(matrix[i][j] for i in range(n))
            p_j.append(col_sum / total_assignments if total_assignments else 0.0)

        P_e = sum(p * p for p in p_j)

        if abs(1.0 - P_e) < 1e-9:
            return 1.0 if P_bar >= 1.0 - 1e-9 else 0.0

        kappa = (P_bar - P_e) / (1.0 - P_e)
        return kappa

    def save_consistency_report(self, report: Dict, output_dir: str):
        """Save consistency evaluation results."""
        os.makedirs(output_dir, exist_ok=True)

        # JSON
        with open(os.path.join(output_dir, 'consistency_report.json'),
                  'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Human-readable summary
        agg = report['aggregate']
        with open(os.path.join(output_dir, 'consistency_summary.txt'),
                  'w', encoding='utf-8') as f:
            f.write("INTER-RATER CONSISTENCY EVALUATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Items evaluated:        {agg['num_items']}\n")
            f.write(f"Classification rounds:  {agg['num_rounds']}\n")
            f.write(f"Temperatures used:      {agg['temperatures_used']}\n\n")

            f.write("AGGREGATE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Fleiss' kappa:          {agg['fleiss_kappa']:.4f}\n")
            f.write(f"Full agreement rate:    {agg['full_agreement_rate']:.1%}\n")
            f.write(f"Majority agreement:     {agg['majority_agreement_rate']:.1%}\n")
            f.write(f"Avg score similarity:   {agg['avg_score_similarity']:.3f}\n\n")

            # Interpretation
            kappa = agg['fleiss_kappa']
            if kappa >= 0.8:
                interp = "Almost perfect agreement"
            elif kappa >= 0.6:
                interp = "Substantial agreement"
            elif kappa >= 0.4:
                interp = "Moderate agreement"
            elif kappa >= 0.2:
                interp = "Fair agreement"
            else:
                interp = "Slight/poor agreement"
            f.write(f"Interpretation:         {interp}\n\n")

            # Per-item details
            f.write("PER-ITEM DETAILS:\n")
            f.write("-" * 60 + "\n")
            for item in report['items']:
                f.write(f"\n[{item['index']}] {item['title']}\n")
                f.write(f"  Rounds:     {' / '.join(item['classes_per_round'])}\n")
                f.write(f"  Majority:   {item['majority_class']} "
                        f"(agreement: {item['agreement_ratio']:.0%})\n")
                f.write(f"  Score sim:  {item['score_similarity']:.3f}\n")
                if not item['full_agreement']:
                    f.write(f"  ** DISAGREEMENT DETECTED **\n")

        # Visualization
        self._visualize_consistency(report, output_dir)
        print(f"Consistency report saved to {output_dir}")

    def _visualize_consistency(self, report: Dict, output_dir: str):
        """Create consistency evaluation charts."""
        items = report['items']
        if not items:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Agreement ratio distribution
        ax1 = axes[0, 0]
        ratios = [it['agreement_ratio'] for it in items]
        ax1.hist(ratios, bins=10, color='steelblue', edgecolor='black', range=(0, 1))
        ax1.set_title('Agreement Ratio Distribution', fontweight='bold')
        ax1.set_xlabel('Agreement ratio')
        ax1.set_ylabel('Count')
        ax1.axvline(x=sum(ratios)/len(ratios), color='red', linestyle='--',
                     label=f'Mean = {sum(ratios)/len(ratios):.2f}')
        ax1.legend()

        # 2. Score similarity distribution
        ax2 = axes[0, 1]
        sims = [it['score_similarity'] for it in items]
        ax2.hist(sims, bins=10, color='mediumpurple', edgecolor='black', range=(0, 1))
        ax2.set_title('Score Similarity Distribution', fontweight='bold')
        ax2.set_xlabel('Cosine similarity')
        ax2.set_ylabel('Count')
        ax2.axvline(x=sum(sims)/len(sims), color='red', linestyle='--',
                     label=f'Mean = {sum(sims)/len(sims):.2f}')
        ax2.legend()

        # 3. Full vs partial agreement pie
        ax3 = axes[1, 0]
        full = sum(1 for it in items if it['full_agreement'])
        partial = len(items) - full
        ax3.pie([full, partial],
                labels=['Full agreement', 'Disagreement'],
                autopct='%1.1f%%', colors=['#66bb6a', '#ef5350'])
        ax3.set_title('Classification Stability', fontweight='bold')

        # 4. Disagreement by majority class
        ax4 = axes[1, 1]
        from collections import Counter
        disagree_classes = Counter(
            it['majority_class'] for it in items if not it['full_agreement']
        )
        if disagree_classes:
            classes = list(disagree_classes.keys())
            counts = [disagree_classes[c] for c in classes]
            ax4.barh(classes, counts, color='salmon', edgecolor='black')
            ax4.set_title('Disagreements by Theory', fontweight='bold')
            ax4.set_xlabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No disagreements!',
                     ha='center', va='center', fontsize=14)
            ax4.set_title('Disagreements by Theory', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'consistency_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results: List[Dict], output_dir: str, original_df: pd.DataFrame):
        """Save enhanced results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed JSON
        with open(os.path.join(output_dir, 'enhanced_classification_results.json'),
                 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save enhanced summary
        with open(os.path.join(output_dir, 'enhanced_analysis_summary.txt'),
                 'w', encoding='utf-8') as f:
            f.write("ENHANCED CONSCIOUSNESS THEORY CLASSIFICATION ANALYSIS\n")
            f.write("WITH MULTI-LEVEL METAPHOR ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Items Processed: {len(results)}\n")
            f.write(f"Model Used: {self.model}\n\n")

            successful = [r for r in results if r.get('status') != 'error']
            f.write(f"Successful Classifications: {len(successful)}\n")

            # Meta-metaphor statistics
            meta_metaphor_count = sum(
                1 for r in successful
                if r.get('metaphor_analysis', {}).get('meta_metaphor_detected', False)
            )
            f.write(f"Meta-Metaphors Detected: {meta_metaphor_count} "
                   f"({meta_metaphor_count/len(successful)*100:.1f}%)\n\n")

            # Class distribution
            from collections import Counter
            classes = [r['primary_class'] for r in successful]
            class_counts = Counter(classes)

            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for class_name, count in class_counts.most_common():
                percentage = (count / len(successful)) * 100
                f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")

            f.write("\n")

            # Detailed results with metaphor analysis
            f.write("DETAILED RESULTS WITH METAPHOR ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for result in successful:
                f.write(f"\nTitle: {result.get('title', 'N/A')}\n")
                f.write(f"Primary Class: {result.get('primary_class', 'N/A')}\n")
                f.write(f"Confidence: {result.get('confidence', 'N/A')}\n")

                if result.get('is_hybrid'):
                    f.write(f"Secondary Class: {result.get('secondary_class', 'N/A')}\n")
                    f.write("⚠️  HYBRID THEORY DETECTED\n")

                # Metaphor analysis
                if 'metaphor_analysis' in result:
                    ma = result['metaphor_analysis']
                    if ma.get('meta_metaphor_detected'):
                        f.write("\n🎨 META-METAPHOR DETECTED!\n")
                        f.write(f"   Semantic Field: {ma.get('dominant_semantic_field', 'N/A')}\n")
                        f.write(f"   Transformation: {ma.get('artistic_transformation', 'N/A')}\n")

                if 'evidence' in result and isinstance(result['evidence'], list):
                    f.write("\nEvidence:\n")
                    for evidence in result['evidence'][:5]:
                        f.write(f"  - Class: {evidence.get('class', 'N/A')}\n")
                        f.write(f"    Level: {evidence.get('metaphor_level', 'N/A')}\n")
                        f.write(f"    Type: {evidence.get('metaphor_type', 'N/A')}\n")
                        f.write(f"    Weight: {evidence.get('weight', 0):.2f}\n")
                        f.write(f"    Span: {evidence.get('span', 'N/A')[:100]}...\n")
                        f.write(f"    Reasoning: {evidence.get('reasoning', 'N/A')[:150]}...\n")
                        f.write("\n")

                f.write("-" * 70 + "\n")

        # Save CSV
        if len(original_df) == len(results):
            results_df = pd.DataFrame(results)
            merged_df = pd.concat([original_df, results_df], axis=1)
            merged_df.to_csv(os.path.join(output_dir, 'enhanced_classified_data.csv'),
                           index=False, encoding='utf-8')
            # Also save Excel
            merged_df.to_excel(os.path.join(output_dir, 'enhanced_classified_data.xlsx'),
                             index=False, engine='openpyxl')

        print(f"Enhanced results saved to {output_dir}")

    def rerun_classification(
        self,
        existing_json_path: str,
        excel_path: str = 'combined_ai_preprocessed.xlsx',
        only_high_medium: bool = False
    ) -> List[Dict]:
        """
        Re-classify records from a previous run with the current classifier version.

        Default: re-classifies ALL non-error/skipped records (full corpus).
            This guarantees a homogeneous classifier_version across the dataset
            and enables aggregate reporting on meta_metaphor.* fields.

        only_high_medium=True: re-classify only high+medium confidence records,
            leaving low-confidence records on their original classifier_version.
            Use only if cost/time is a hard constraint AND your downstream analysis
            explicitly handles the mixed corpus by filtering on classifier_version.
        """
        with open(existing_json_path, encoding='utf-8') as f:
            existing: List[Dict] = json.load(f)

        df = self.process_excel_file(excel_path)
        descriptions = df['combined_description'].tolist()
        titles = df.get('title', [f'Item_{i}' for i in range(len(descriptions))]).tolist()

        if only_high_medium:
            rerun_indices = {
                r['index'] for r in existing
                if r.get('confidence') in ('high', 'medium')
                and r.get('status') not in ('error', 'skipped')
            }
            mode_desc = "high+medium only"
        else:
            rerun_indices = {
                r['index'] for r in existing
                if r.get('status') not in ('error', 'skipped')
            }
            mode_desc = "full corpus"

        print(f"Re-classifying {len(rerun_indices)} records ({mode_desc}); "
              f"{len(existing) - len(rerun_indices)} kept as-is (errors/skipped)...")

        results_by_index: Dict[int, Dict] = {r['index']: r for r in existing}

        for idx in sorted(rerun_indices):
            desc = descriptions[idx]
            title = titles[idx]
            print(f"  Re-running {idx + 1}/{len(descriptions)}: {title}")
            result = self.classify_description(str(desc))
            result['index'] = idx
            result['title'] = title
            result['description_length'] = len(str(desc))
            result['rerun'] = True
            results_by_index[idx] = result
            time.sleep(self.request_delay)

        return [results_by_index[i] for i in sorted(results_by_index)]


def main():
    """Main execution.

    Usage:
        python enhanced_classifier.py                        # standard analysis
        python enhanced_classifier.py --consistency          # inter-rater evaluation
        python enhanced_classifier.py --consistency --rounds 5
        python enhanced_classifier.py --consistency --sample 10
        python enhanced_classifier.py --rerun path/to/enhanced_classification_results.json
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced Consciousness Theory Classifier'
    )
    parser.add_argument(
        '--consistency', action='store_true',
        help='Run inter-rater consistency evaluation instead of single-pass classification'
    )
    parser.add_argument(
        '--rounds', type=int, default=3,
        help='Number of classification rounds for consistency evaluation (default: 3)'
    )
    parser.add_argument(
        '--sample', type=int, default=0,
        help='Evaluate only first N items (0 = all). Useful for quick tests.'
    )
    parser.add_argument(
        '--rerun', type=str, default=None,
        metavar='JSON_PATH',
        help='Re-classify ALL records from existing JSON results with structured '
             'meta_metaphor output. Default: full corpus reclassification for a '
             'homogeneous classifier_version.'
    )
    parser.add_argument(
        '--rerun-only-high-medium', action='store_true',
        help='With --rerun: only re-classify high+medium confidence records. '
             'WARNING: produces a mixed corpus (records with different '
             'classifier_version) — handle in downstream analysis accordingly.'
    )
    args = parser.parse_args()

    try:
        print("=" * 70)
        print("ENHANCED CONSCIOUSNESS THEORY CLASSIFIER")
        print("WITH MULTI-LEVEL METAPHOR ANALYSIS")
        print("=" * 70)
        print()

        classifier = EnhancedMetaphorClassifier()

        print("Processing Excel file...")
        df = classifier.process_excel_file('combined_ai_preprocessed.xlsx')

        descriptions = df['combined_description'].tolist()
        titles = df.get('title', [f'Item_{i}' for i in range(len(descriptions))]).tolist()

        if args.sample > 0:
            descriptions = descriptions[:args.sample]
            titles = titles[:args.sample]

        print(f"Found {len(descriptions)} items to process")
        print()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if args.rerun:
            # --- Rerun mode: re-classify with structured meta_metaphor ---
            print(f"RERUN MODE: {args.rerun}")
            mode = "high+medium only" if args.rerun_only_high_medium else "full corpus"
            print(f"Mode: {mode}\n")
            results = classifier.rerun_classification(
                args.rerun, only_high_medium=args.rerun_only_high_medium
            )
            output_dir = f'rerun_results_{timestamp}'
            classifier.save_results(results, output_dir, df)
            classifier.create_enhanced_visualizations(results, output_dir)
            rerun_count = sum(1 for r in results if r.get('rerun'))
            v_rates = [r.get('verified_quote_rate') for r in results
                       if r.get('verified_quote_rate') is not None]
            avg_v_rate = sum(v_rates) / len(v_rates) if v_rates else 0
            print("\n" + "=" * 70)
            print(f"Rerun complete! Re-classified: {rerun_count} records")
            print(f"Avg verified_quote_rate: {avg_v_rate:.1%}")
            print(f"Results saved to: {output_dir}")
            print("=" * 70)

        elif args.consistency:
            # --- Consistency evaluation mode ---
            print(f"Running consistency evaluation ({args.rounds} rounds)...")
            report = classifier.evaluate_consistency(
                descriptions, titles, num_rounds=args.rounds
            )
            output_dir = f'consistency_results_{timestamp}'
            classifier.save_consistency_report(report, output_dir)

            agg = report['aggregate']
            print("\n" + "=" * 70)
            print("CONSISTENCY EVALUATION COMPLETE")
            print(f"  Fleiss' kappa:        {agg['fleiss_kappa']:.4f}")
            print(f"  Full agreement rate:  {agg['full_agreement_rate']:.1%}")
            print(f"  Majority agreement:   {agg['majority_agreement_rate']:.1%}")
            print(f"  Avg score similarity: {agg['avg_score_similarity']:.3f}")
            print(f"\nResults saved to: {output_dir}")
            print("=" * 70)

        else:
            # --- Standard classification mode ---
            print("Starting classification with enhanced metaphor analysis...")
            results = classifier.classify_batch(descriptions, titles)

            output_dir = f'enhanced_analysis_results_{timestamp}'

            print("\nSaving results...")
            classifier.save_results(results, output_dir, df)

            print("Creating enhanced visualizations...")
            classifier.create_enhanced_visualizations(results, output_dir)

            print("\n" + "=" * 70)
            print(f"Analysis complete! Results saved to: {output_dir}")
            print("\nFiles created:")
            print(f"  {output_dir}/enhanced_analysis_summary.txt")
            print(f"  {output_dir}/enhanced_classification_results.json")
            print(f"  {output_dir}/enhanced_classified_data.csv")
            print(f"  {output_dir}/enhanced_classified_data.xlsx")
            print(f"  {output_dir}/enhanced_analysis.png")
            print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
