"""
Enhanced Metaphor Analysis Module for Consciousness Theory Classification
Focuses on multi-level metaphor analysis (metaphor of metaphor)
"""

import json
from typing import Dict, List, Tuple, Set
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
    Advanced metaphor analyzer with multi-level detection
    """

    def __init__(self):
        self.ontology = MetaphorOntology.get_all_domains()

    def detect_metaphor_level(self, text: str, keywords: List[str],
                             scientific_metaphors: List[str],
                             artistic_transformations: List[str]) -> Tuple[MetaphorLevel, str]:
        """
        Detect the level of metaphorical expression

        Returns:
            (MetaphorLevel, matching_text)
        """
        text_lower = text.lower()

        # Check for explicit terms
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return (MetaphorLevel.EXPLICIT_TERM, keyword)

        # Check for scientific metaphors
        for metaphor in scientific_metaphors:
            metaphor_keywords = metaphor.lower().split()
            if sum(kw in text_lower for kw in metaphor_keywords) >= len(metaphor_keywords) // 2:
                return (MetaphorLevel.SCIENTIFIC_METAPHOR, metaphor)

        # Check for artistic transformations (meta-metaphors)
        for transformation in artistic_transformations:
            trans_keywords = transformation.lower().split()
            if sum(kw in text_lower for kw in trans_keywords) >= len(trans_keywords) // 2:
                return (MetaphorLevel.META_METAPHOR, transformation)

        return (MetaphorLevel.NESTED_METAPHOR, "")

    def identify_semantic_field(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Identify semantic fields present in text

        Returns:
            List of (theory, score, dominant_semantic_field)
        """
        results = []
        text_lower = text.lower()

        for theory, domain in self.ontology.items():
            score = 0
            matched_keywords = []

            # Match keywords
            for keyword in domain['keywords']:
                if keyword.lower() in text_lower:
                    score += 0.15
                    matched_keywords.append(keyword)

            # Match scientific metaphors (higher weight)
            for metaphor in domain['scientific_metaphors']:
                metaphor_keywords = metaphor.lower().split()
                match_count = sum(kw in text_lower for kw in metaphor_keywords)
                if match_count >= 2:
                    score += 0.3 * (match_count / len(metaphor_keywords))
                    matched_keywords.append(metaphor)

            # Match artistic transformations (meta-metaphor detection)
            for transformation in domain['artistic_transformations']:
                trans_keywords = transformation.lower().split()
                match_count = sum(kw in text_lower for kw in trans_keywords)
                if match_count >= 2:
                    score += 0.5 * (match_count / len(trans_keywords))  # Higher weight for meta-metaphors
                    matched_keywords.append(transformation)

            if score > 0:
                semantic_field = ", ".join(matched_keywords[:3])
                results.append((theory, min(score, 1.0), semantic_field))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def determine_metaphor_type(self, text: str, span: str) -> MetaphorType:
        """
        Determine if metaphor is ontological (about nature of mind) or decorative
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

        text_lower = text.lower()
        span_lower = span.lower()

        # Check ontological
        if any(marker in text_lower for marker in ontological_markers):
            return MetaphorType.ONTOLOGICAL

        # Check structural
        if any(marker in span_lower for marker in structural_markers):
            return MetaphorType.STRUCTURAL

        # Check orientational
        if any(marker in span_lower for marker in orientational_markers):
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
        Analyze network of metaphors to detect patterns
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

            # Detect meta-metaphor pattern (scientific + artistic transformation)
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
