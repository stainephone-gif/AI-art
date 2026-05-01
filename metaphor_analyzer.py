"""
Enhanced Metaphor Analysis Module for Consciousness Theory Classification
Focuses on multi-level metaphor analysis (metaphor of metaphor)
"""

import re
import json
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum


class MetaphorLevel(Enum):
    """Levels of metaphorical expression"""
    EXPLICIT_TERM = "explicit_term"           # Direct theoretical term
    SCIENTIFIC_METAPHOR = "scientific_metaphor"  # Metaphor in scientific discourse
    META_METAPHOR = "meta_metaphor"           # Artistic interpretation of scientific metaphor
    NESTED_METAPHOR = "nested_metaphor"       # Multiple layers of metaphorical transformation


class MetaphorType(Enum):
    """Types of metaphors by their ontological function"""
    ONTOLOGICAL = "ontological"      # About nature of mind/consciousness
    STRUCTURAL = "structural"        # About structure/architecture
    ORIENTATIONAL = "orientational"  # About spatial/temporal relations
    DECORATIVE = "decorative"        # Aesthetic, not theoretical


@dataclass
class MetaphorEvidence:
    """Enhanced evidence structure for metaphor analysis"""
    span: str                           # Text fragment
    theory_class: str                   # Consciousness theory (COMP, IIT, etc.)
    level: MetaphorLevel                # Metaphor level
    type: MetaphorType                  # Metaphor type
    weight: float                       # Evidence weight
    semantic_field: str                 # Semantic domain (e.g., "computational", "organic")
    source_domain: str                  # Source domain of metaphor
    target_domain: str = "consciousness"  # Target domain (usually consciousness)
    reasoning: str = ""                 # Explanation
    related_metaphors: List[str] = field(default_factory=list)  # Connected metaphors


class MetaphorOntology:
    """
    Ontology of metaphors for consciousness theories
    Maps semantic fields to theoretical frameworks
    """

    # Computational/Functional domain
    COMPUTATIONAL = {
        'keywords': [
            'алгоритм', 'вычисление', 'процессор', 'программа', 'код',
            'input', 'output', 'обработка', 'computation', 'function',
            'символ', 'репрезентация', 'информация'
        ],
        'scientific_metaphors': [
            'мозг как компьютер', 'сознание как программное обеспечение',
            'разум как машина Тьюринга', 'мышление как вычисление'
        ],
        'artistic_transformations': [
            'данные как материал', 'алгоритм как художник',
            'код как язык искусства', 'нейросеть как сознание'
        ],
        'theory': 'COMP'
    }

    # Integrated Information Theory domain
    IIT = {
        'keywords': [
            'интеграция', 'целостность', 'несводимость', 'phi', 'квалиа',
            'integration', 'irreducibility', 'quale', 'субъективность',
            'внутренняя перспектива', 'феноменальное'
        ],
        'scientific_metaphors': [
            'сознание как интегрированная информация',
            'опыт как неделимое целое', 'phi как мера сознания'
        ],
        'artistic_transformations': [
            'единство противоположностей', 'неделимый образ',
            'целое больше суммы частей', 'интегрированное восприятие'
        ],
        'theory': 'IIT'
    }

    # Predictive Processing domain
    PRED = {
        'keywords': [
            'предсказание', 'прогноз', 'ошибка', 'байесовский', 'prior',
            'prediction', 'error', 'bayesian', 'free energy', 'минимизация',
            'неопределённость', 'вероятность', 'модель мира'
        ],
        'scientific_metaphors': [
            'мозг как байесовский предсказатель',
            'сознание как минимизация свободной энергии',
            'восприятие как контролируемая галлюцинация'
        ],
        'artistic_transformations': [
            'ожидание vs реальность', 'генеративные модели как творчество',
            'предвосхищение образа', 'синтез восприятия'
        ],
        'theory': 'PRED'
    }

    # Global Workspace Theory domain
    GWT = {
        'keywords': [
            'внимание', 'рабочее пространство', 'broadcast', 'театр',
            'spotlight', 'awareness', 'доступ', 'глобальный',
            'конкуренция', 'селекция', 'фокус'
        ],
        'scientific_metaphors': [
            'сознание как театр разума', 'внимание как прожектор',
            'осознание как трансляция информации'
        ],
        'artistic_transformations': [
            'сцена восприятия', 'фокус внимания',
            'выбор образа', 'освещение смысла'
        ],
        'theory': 'GWT'
    }

    # Enactivism domain
    ENACT = {
        'keywords': [
            'телесность', 'воплощение', 'действие', 'сенсомоторный',
            'embodied', 'enacted', 'sensorimotor', 'петля',
            'взаимодействие', 'среда', 'coupling', 'движение'
        ],
        'scientific_metaphors': [
            'познание как действие', 'разум как воплощённый процесс',
            'сознание через телесность'
        ],
        'artistic_transformations': [
            'тело как медиум', 'жест как мысль',
            'движение как познание', 'материальность разума'
        ],
        'theory': 'ENACT'
    }

    # Panpsychism domain
    PAN = {
        'keywords': [
            'протоопыт', 'панпсихизм', 'одушевление', 'всеобщность',
            'protophenomenal', 'panpsychism', 'универсальность',
            'материя чувствует', 'внутренняя жизнь', 'анимизм'
        ],
        'scientific_metaphors': [
            'материя как чувствующая', 'протоквалиа в элементах',
            'сознание как фундаментальное свойство'
        ],
        'artistic_transformations': [
            'живая материя', 'одушевлённые объекты',
            'чувствующие системы', 'агентность материала'
        ],
        'theory': 'PAN'
    }

    # Emergentism domain
    EMERG = {
        'keywords': [
            'эмерджентность', 'самоорганизация', 'сложность', 'рой',
            'emergence', 'self-organization', 'swarm', 'collective',
            'фазовый переход', 'новое качество', 'синергия'
        ],
        'scientific_metaphors': [
            'сознание как эмерджентное свойство',
            'разум как самоорганизующаяся система', 'коллективный интеллект'
        ],
        'artistic_transformations': [
            'роевой разум', 'коллективное творчество',
            'спонтанный порядок', 'множественность в единстве'
        ],
        'theory': 'EMERG'
    }

    @classmethod
    def get_all_domains(cls) -> Dict[str, Dict]:
        """Get all semantic domains"""
        return {
            'COMP': cls.COMPUTATIONAL,
            'IIT': cls.IIT,
            'PRED': cls.PRED,
            'GWT': cls.GWT,
            'ENACT': cls.ENACT,
            'PAN': cls.PAN,
            'EMERG': cls.EMERG
        }


class MetaphorAnalyzer:
    """
    Advanced metaphor analyzer with multi-level detection.
    Uses context-window matching instead of naive substring search.
    """

    # Maximum distance (in tokens) between keywords of a multi-word
    # phrase for them to count as co-occurring in the same context window.
    CONTEXT_WINDOW_SIZE = 15  # words

    def __init__(self):
        self.ontology = MetaphorOntology.get_all_domains()
        # Pre-compile a tokeniser that keeps Cyrillic, Latin and digits
        self._token_re = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)

    # ------------------------------------------------------------------ #
    #  Tokenisation & context-window helpers
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> List[str]:
        """Split text into lowercase tokens."""
        return [t.lower() for t in self._token_re.findall(text)]

    # Minimum shared prefix length for stem matching
    _MIN_STEM_LEN = 4

    def _build_token_positions(self, tokens: List[str]) -> Dict[str, List[int]]:
        """Map each unique token to a sorted list of its positions."""
        positions: Dict[str, List[int]] = {}
        for idx, tok in enumerate(tokens):
            positions.setdefault(tok, []).append(idx)
        return positions

    @staticmethod
    def _stem_match(keyword: str, token: str) -> bool:
        """
        Check if keyword and token share a common stem.
        Handles Russian morphology (inflections, conjugations) by comparing
        prefixes.  E.g. 'синтез' matches 'синтезируя', 'восприятия' matches
        'восприятие'.
        """
        min_len = MetaphorAnalyzer._MIN_STEM_LEN
        if len(keyword) < min_len or len(token) < min_len:
            return keyword == token
        # Check if one is a prefix of the other, or they share a long prefix
        shared = min(len(keyword), len(token))
        prefix_len = 0
        for a, b in zip(keyword, token):
            if a == b:
                prefix_len += 1
            else:
                break
        # Require the shared prefix to be at least min_len chars
        # AND at least 60% of the shorter word (handles short stems well)
        shorter = min(len(keyword), len(token))
        return prefix_len >= min_len and prefix_len >= shorter * 0.6

    def _find_stem_positions(self, keyword_token: str,
                             tokens: List[str],
                             token_positions: Dict[str, List[int]]) -> List[int]:
        """Find all positions where a token stem-matches the keyword."""
        positions = []
        # Fast path: exact match
        if keyword_token in token_positions:
            positions.extend(token_positions[keyword_token])

        # Stem matching for tokens not already matched
        exact_set = set(token_positions.get(keyword_token, []))
        for tok, tok_positions in token_positions.items():
            if tok == keyword_token:
                continue
            if self._stem_match(keyword_token, tok):
                for p in tok_positions:
                    if p not in exact_set:
                        positions.append(p)
        positions.sort()
        return positions

    def _phrase_in_context_window(
        self,
        phrase_tokens: List[str],
        token_positions: Dict[str, List[int]],
        total_tokens: int,
        window: Optional[int] = None,
        all_tokens: Optional[List[str]] = None,
    ) -> Tuple[bool, float, str]:
        """
        Check whether *all* tokens of a phrase appear within a sliding
        context window of ``window`` tokens.  Uses stem matching to
        handle morphological variation.

        Returns:
            (matched: bool,
             proximity_score: 0.0-1.0 — tighter cluster → higher score,
             matched_span: text slice that was matched)
        """
        if window is None:
            window = self.CONTEXT_WINDOW_SIZE
        if all_tokens is None:
            all_tokens = []

        # Every phrase token must exist in the text (via stem matching)
        positions_per_kw = []
        for kw in phrase_tokens:
            kw_positions = self._find_stem_positions(kw, all_tokens, token_positions)
            if not kw_positions:
                return (False, 0.0, "")
            positions_per_kw.append(kw_positions)

        if len(positions_per_kw) == 1:
            # Single-token phrase — always matches
            return (True, 1.0, phrase_tokens[0])

        # Sliding-pointer approach: for every position of the first keyword,
        # check whether the remaining keywords have a position within `window`.
        best_span_width = total_tokens  # worst case
        matched = False
        best_min_pos = 0
        best_max_pos = 0
        for anchor_pos in positions_per_kw[0]:
            lo = anchor_pos
            hi = anchor_pos
            all_found = True
            for other_positions in positions_per_kw[1:]:
                # Binary-ish: find closest position within [anchor-window, anchor+window]
                found_in_window = False
                for p in other_positions:
                    if abs(p - anchor_pos) <= window:
                        lo = min(lo, p)
                        hi = max(hi, p)
                        found_in_window = True
                        break
                if not found_in_window:
                    all_found = False
                    break
            if all_found:
                span_width = hi - lo
                if span_width < best_span_width:
                    best_span_width = span_width
                    best_min_pos = lo
                    best_max_pos = hi
                    matched = True

        if not matched:
            return (False, 0.0, "")

        # Proximity score: tighter cluster → higher score
        max_possible_span = window
        proximity = max(0.0, 1.0 - best_span_width / max_possible_span)
        return (True, proximity, " ".join(phrase_tokens))

    def _keyword_in_context(
        self,
        keyword: str,
        tokens: List[str],
        token_positions: Dict[str, List[int]],
    ) -> Tuple[bool, List[int]]:
        """
        Check if a keyword (single or multi-word) appears in the token stream.
        Uses stem matching to handle morphological variation (e.g.
        'восприятия' matches 'восприятие', 'синтез' matches 'синтезируя').
        Returns (found, positions_of_first_token).
        """
        kw_tokens = self._tokenize(keyword)
        if not kw_tokens:
            return (False, [])

        if len(kw_tokens) == 1:
            positions = self._find_stem_positions(kw_tokens[0], tokens, token_positions)
            return (bool(positions), positions)

        # Multi-word keyword: look for consecutive tokens (stem-matched)
        first_positions = self._find_stem_positions(kw_tokens[0], tokens, token_positions)
        match_positions = []
        for pos in first_positions:
            if pos + len(kw_tokens) - 1 >= len(tokens):
                continue
            if all(
                self._stem_match(kw_tokens[i], tokens[pos + i])
                for i in range(len(kw_tokens))
            ):
                match_positions.append(pos)

        return (bool(match_positions), match_positions)

    # ------------------------------------------------------------------ #
    #  Core detection methods (context-window based)
    # ------------------------------------------------------------------ #

    def detect_metaphor_level(self, text: str, keywords: List[str],
                             scientific_metaphors: List[str],
                             artistic_transformations: List[str]) -> Tuple[MetaphorLevel, str]:
        """
        Detect the level of metaphorical expression using context-window
        matching instead of naive substring search.
        """
        tokens = self._tokenize(text)
        token_positions = self._build_token_positions(tokens)
        total = len(tokens)

        # 1. Check for explicit terms (exact word-boundary match)
        for keyword in keywords:
            found, _ = self._keyword_in_context(keyword, tokens, token_positions)
            if found:
                return (MetaphorLevel.EXPLICIT_TERM, keyword)

        # 2. Check for scientific metaphors (context-window co-occurrence)
        best_sci = None
        best_sci_score = 0.0
        for metaphor in scientific_metaphors:
            phrase_tokens = self._tokenize(metaphor)
            if len(phrase_tokens) < 2:
                continue
            matched, proximity, span = self._phrase_in_context_window(
                phrase_tokens, token_positions, total, all_tokens=tokens
            )
            if matched and proximity > best_sci_score:
                best_sci_score = proximity
                best_sci = metaphor

        # 3. Check for artistic transformations (context-window co-occurrence)
        best_art = None
        best_art_score = 0.0
        for transformation in artistic_transformations:
            phrase_tokens = self._tokenize(transformation)
            if len(phrase_tokens) < 2:
                continue
            matched, proximity, span = self._phrase_in_context_window(
                phrase_tokens, token_positions, total, all_tokens=tokens
            )
            if matched and proximity > best_art_score:
                best_art_score = proximity
                best_art = transformation

        # Prefer the level with the highest proximity score
        if best_sci and best_sci_score >= best_art_score:
            return (MetaphorLevel.SCIENTIFIC_METAPHOR, best_sci)
        if best_art:
            return (MetaphorLevel.META_METAPHOR, best_art)

        return (MetaphorLevel.NESTED_METAPHOR, "")

    def identify_semantic_field(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Identify semantic fields present in text using context-window
        co-occurrence instead of naive substring matching.
        """
        tokens = self._tokenize(text)
        token_positions = self._build_token_positions(tokens)
        total = len(tokens)
        results = []

        for theory, domain in self.ontology.items():
            score = 0.0
            matched_keywords = []

            # Match keywords (word-boundary)
            for keyword in domain['keywords']:
                found, _ = self._keyword_in_context(keyword, tokens, token_positions)
                if found:
                    score += 0.15
                    matched_keywords.append(keyword)

            # Match scientific metaphors (context-window)
            for metaphor in domain['scientific_metaphors']:
                phrase_tokens = self._tokenize(metaphor)
                if len(phrase_tokens) < 2:
                    continue
                matched, proximity, _ = self._phrase_in_context_window(
                    phrase_tokens, token_positions, total, all_tokens=tokens
                )
                if matched:
                    score += 0.3 * proximity
                    matched_keywords.append(metaphor)

            # Match artistic transformations (context-window, higher weight)
            for transformation in domain['artistic_transformations']:
                phrase_tokens = self._tokenize(transformation)
                if len(phrase_tokens) < 2:
                    continue
                matched, proximity, _ = self._phrase_in_context_window(
                    phrase_tokens, token_positions, total, all_tokens=tokens
                )
                if matched:
                    score += 0.5 * proximity
                    matched_keywords.append(transformation)

            if score > 0:
                semantic_field = ", ".join(matched_keywords[:3])
                results.append((theory, min(score, 1.0), semantic_field))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def determine_metaphor_type(self, text: str, span: str) -> MetaphorType:
        """
        Determine if metaphor is ontological (about nature of mind) or decorative.
        Uses stem matching to handle Russian inflections (e.g. 'сознания' matches
        'сознание').
        """
        # Ontological indicators
        ontological_markers = [
            'сознание', 'разум', 'мышление', 'познание', 'опыт',
            'consciousness', 'mind', 'thinking', 'cognition', 'experience',
            'субъективность', 'квалиа', 'осознание', 'awareness',
            'природа', 'сущность', 'механизм', 'nature', 'essence'
        ]

        # Structural indicators
        structural_markers = [
            'архитектура', 'структура', 'слои', 'уровни', 'модули',
            'architecture', 'structure', 'layers', 'levels', 'modules',
            'организация', 'иерархия', 'система', 'organization'
        ]

        # Orientational indicators
        orientational_markers = [
            'внутри', 'снаружи', 'верх', 'низ', 'глубина', 'поверхность',
            'inside', 'outside', 'up', 'down', 'depth', 'surface',
            'центр', 'периферия', 'пространство', 'время'
        ]

        text_tokens = self._tokenize(text)
        span_tokens = self._tokenize(span)

        def _has_marker(tokens: List[str], markers: List[str]) -> bool:
            for marker in markers:
                marker_lower = marker.lower()
                for tok in tokens:
                    if self._stem_match(marker_lower, tok):
                        return True
            return False

        # Check ontological (against full text)
        if _has_marker(text_tokens, ontological_markers):
            return MetaphorType.ONTOLOGICAL

        # Check structural (against span)
        if _has_marker(span_tokens, structural_markers):
            return MetaphorType.STRUCTURAL

        # Check orientational (against span)
        if _has_marker(span_tokens, orientational_markers):
            return MetaphorType.ORIENTATIONAL

        return MetaphorType.DECORATIVE

    def calculate_metaphor_weight(self, level: MetaphorLevel,
                                  metaphor_type: MetaphorType,
                                  context_relevance: float = 1.0) -> float:
        """
        Calculate weight for metaphor evidence
        Enhanced weighting system that values meta-metaphors
        """
        # Base weights by level
        level_weights = {
            MetaphorLevel.EXPLICIT_TERM: 1.0,
            MetaphorLevel.SCIENTIFIC_METAPHOR: 0.7,
            MetaphorLevel.META_METAPHOR: 0.8,      # High weight for artistic interpretation
            MetaphorLevel.NESTED_METAPHOR: 0.5
        }

        # Type multipliers
        type_multipliers = {
            MetaphorType.ONTOLOGICAL: 1.2,         # Boost ontological metaphors
            MetaphorType.STRUCTURAL: 1.0,
            MetaphorType.ORIENTATIONAL: 0.8,
            MetaphorType.DECORATIVE: 0.3           # Penalize decorative
        }

        base_weight = level_weights.get(level, 0.4)
        type_mult = type_multipliers.get(metaphor_type, 0.5)

        return min(base_weight * type_mult * context_relevance, 1.0)

    def analyze_metaphor_network(self, evidence_list: List[MetaphorEvidence]) -> Dict:
        """
        Analyze network of metaphors to detect patterns.

        DEPRECATED (meta_metaphor_detected only):
            The `meta_metaphor_detected` field produced by this method is always False
            in practice because it requires co-occurrence of an exact scientific_metaphor
            phrase AND an exact artistic_transformation phrase within the same context
            window — a condition almost never satisfied in real curatorial texts.
            Use `metaphor_analysis.meta_metaphor_detected` from the LLM response instead,
            which operates at the semantic level and yields empirically valid results.
            This method is retained as a structural scaffold and for cluster/weight analysis.
        """
        # Group by theory
        theory_clusters = {}
        for evidence in evidence_list:
            theory = evidence.theory_class
            if theory not in theory_clusters:
                theory_clusters[theory] = []
            theory_clusters[theory].append(evidence)

        # Analyze each cluster
        analysis = {
            'clusters': {},
            'meta_metaphor_detected': False,
            'dominant_pattern': None,
            'hybrid_theories': []
        }

        for theory, evidences in theory_clusters.items():
            # Check for meta-metaphors
            meta_metaphors = [e for e in evidences if e.level == MetaphorLevel.META_METAPHOR]
            scientific_metaphors = [e for e in evidences if e.level == MetaphorLevel.SCIENTIFIC_METAPHOR]

            cluster_analysis = {
                'total_evidence': len(evidences),
                'meta_metaphor_count': len(meta_metaphors),
                'scientific_metaphor_count': len(scientific_metaphors),
                'ontological_count': sum(1 for e in evidences if e.type == MetaphorType.ONTOLOGICAL),
                'total_weight': sum(e.weight for e in evidences),
                'semantic_fields': list(set(e.semantic_field for e in evidences))
            }

            analysis['clusters'][theory] = cluster_analysis

            # DEPRECATED: This co-occurrence condition is almost never satisfied
            # in real texts. Kept for structural integrity; do not rely on this signal.
            if meta_metaphors and scientific_metaphors:
                analysis['meta_metaphor_detected'] = True

        # Find dominant pattern
        if theory_clusters:
            dominant = max(theory_clusters.items(),
                          key=lambda x: sum(e.weight for e in x[1]))
            analysis['dominant_pattern'] = dominant[0]

        # Detect hybrid theories (multiple strong clusters)
        strong_theories = [t for t, c in analysis['clusters'].items()
                          if c['total_weight'] >= 0.5]
        if len(strong_theories) > 1:
            analysis['hybrid_theories'] = strong_theories

        return analysis

    def extract_metaphors(self, text: str) -> List[MetaphorEvidence]:
        """
        Extract all metaphors from text with enhanced analysis
        """
        evidence_list = []

        # Identify semantic fields
        semantic_fields = self.identify_semantic_field(text)

        for theory, score, field in semantic_fields:
            if score < 0.2:  # Skip very weak matches
                continue

            domain = self.ontology[theory]

            # Detect metaphor level
            level, matched_text = self.detect_metaphor_level(
                text,
                domain['keywords'],
                domain['scientific_metaphors'],
                domain['artistic_transformations']
            )

            if not matched_text and level == MetaphorLevel.NESTED_METAPHOR:
                continue

            # Determine metaphor type
            metaphor_type = self.determine_metaphor_type(text, matched_text)

            # Calculate weight
            weight = self.calculate_metaphor_weight(level, metaphor_type, score)

            # Create evidence
            evidence = MetaphorEvidence(
                span=matched_text,
                theory_class=theory,
                level=level,
                type=metaphor_type,
                weight=weight,
                semantic_field=field,
                source_domain=field.split(',')[0] if field else "",
                reasoning=f"{level.value} в {metaphor_type.value} контексте"
            )

            evidence_list.append(evidence)

        return evidence_list

    def generate_enhanced_prompt_section(self) -> str:
        """
        Generate enhanced prompt section for LLM with metaphor analysis instructions
        """
        return """
РАСШИРЕННЫЙ АНАЛИЗ МЕТАФОР:

При анализе метафор учитывай ЧЕТЫРЕ УРОВНЯ:

1. EXPLICIT_TERM (вес 1.0): Прямой термин теории
   - Пример: "интегрированная информация", "предиктивное кодирование"

2. SCIENTIFIC_METAPHOR (вес 0.7): Метафора в научном дискурсе
   - Пример: "мозг как компьютер", "сознание как театр разума"

3. META_METAPHOR (вес 0.8): Художественная интерпретация научной метафоры
   - Пример: "алгоритм как художник", "нейросеть как сознание"
   - Это ключевой уровень для AI-арта!

4. NESTED_METAPHOR (вес 0.5): Многослойная метафорическая трансформация

ТИП МЕТАФОРЫ (влияет на вес):
- ONTOLOGICAL (×1.2): О природе сознания/разума
- STRUCTURAL (×1.0): О структуре/архитектуре
- ORIENTATIONAL (×0.8): О пространственно-временных отношениях
- DECORATIVE (×0.3): Декоративная, не теоретическая

КРИТИЧЕСКИ ВАЖНО для AI-арта:
- Если художник использует технический термин метафорически → META_METAPHOR
- Если работа превращает научную концепцию в художественный образ → META_METAPHOR
- Например: "нейросеть генерирует образы" может быть метафорой творчества (PRED/COMP)

ПАТТЕРНЫ МЕТАФОР МЕТАФОР по теориям:

COMP → META_METAPHOR:
- "алгоритм как автор/художник"
- "данные как материал искусства"
- "код как язык творчества"

PRED → META_METAPHOR:
- "генеративная модель как воображение"
- "ошибка предсказания как креативность"
- "синтез восприятия как художественный акт"

ENACT → META_METAPHOR:
- "тело как медиум познания"
- "жест/движение как мысль"
- "материальность разума"

EMERG → META_METAPHOR:
- "роевой разум"
- "коллективное творчество"
- "спонтанный порядок из хаоса"

При обнаружении META_METAPHOR обязательно указывай:
1. Исходную научную метафору
2. Художественную трансформацию
3. Связь с теорией сознания
"""


def create_enhanced_system_prompt(base_prompt: str, analyzer: MetaphorAnalyzer) -> str:
    """
    Enhance existing system prompt with metaphor analysis section
    """
    metaphor_section = analyzer.generate_enhanced_prompt_section()

    # Insert after the main class descriptions
    enhanced_prompt = base_prompt.replace(
        "ФОРМАТ ОТВЕТА (строго JSON):",
        f"{metaphor_section}\n\nФОРМАТ ОТВЕТА (строго JSON):"
    )

    return enhanced_prompt


if __name__ == "__main__":
    # Test the analyzer
    analyzer = MetaphorAnalyzer()

    # Test text with meta-metaphor
    test_text = """
    Работа исследует как нейронная сеть создаёт образы, превращая алгоритм
    в художника. Генеративная модель действует как воображение,
    синтезируя восприятие через предиктивное кодирование.
    """

    print("Testing Metaphor Analyzer")
    print("=" * 50)

    evidence = analyzer.extract_metaphors(test_text)

    for e in evidence:
        print(f"\nТеория: {e.theory_class}")
        print(f"Уровень: {e.level.value}")
        print(f"Тип: {e.type.value}")
        print(f"Вес: {e.weight:.2f}")
        print(f"Текст: {e.span}")
        print(f"Семантическое поле: {e.semantic_field}")

    # Analyze network
    network = analyzer.analyze_metaphor_network(evidence)
    print("\n" + "=" * 50)
    print("Анализ сети метафор:")
    print(json.dumps(network, indent=2, ensure_ascii=False))
