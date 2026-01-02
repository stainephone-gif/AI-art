"""
Enhanced Consciousness Theory Classifier with Advanced Metaphor Analysis
Integrates metaphor_analyzer for multi-level metaphor detection
"""

import os
import json
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
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

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
        Pre-analyze metaphors before LLM classification
        Provides structured hints to the LLM
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

    def classify_description(self, description: str, max_retries: int = 3) -> Dict:
        """
        Classify with enhanced metaphor analysis
        """
        # Pre-analyze metaphors
        metaphor_hints = self.pre_analyze_metaphors(description)

        # Create enhanced user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(description=description)

        # Add metaphor hints to help LLM
        if metaphor_hints['meta_metaphor_present']:
            user_prompt += f"\n\nПОДСКАЗКА: Обнаружены метафоры метафоры. Доминирующая теория: {metaphor_hints['dominant_theory']}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.0
        }

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

                    # Clean JSON response
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "")

                    classification = json.loads(content.strip())

                    # Add pre-analysis metadata
                    classification['metaphor_pre_analysis'] = metaphor_hints

                    return classification

                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    print(f"API Error: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        'primary_class': 'UND',
                        'error': str(e),
                        'status': 'error'
                    }
                time.sleep(2 ** attempt)

        return {
            'primary_class': 'UND',
            'error': 'Max retries exceeded',
            'status': 'error'
        }

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

            time.sleep(0.5)

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


def main():
    """Main execution"""
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

        print(f"Found {len(descriptions)} items to classify")
        print()

        print("Starting classification with enhanced metaphor analysis...")
        results = classifier.classify_batch(descriptions, titles)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'enhanced_analysis_results_{timestamp}'

        print("\nSaving results...")
        classifier.save_results(results, output_dir, df)

        print("Creating enhanced visualizations...")
        classifier.create_enhanced_visualizations(results, output_dir)

        print("\n" + "=" * 70)
        print(f"✅ Analysis complete! Results saved to: {output_dir}")
        print("\nFiles created:")
        print(f"  📊 {output_dir}/enhanced_analysis_summary.txt")
        print(f"  📋 {output_dir}/enhanced_classification_results.json")
        print(f"  📈 {output_dir}/enhanced_classified_data.csv")
        print(f"  📊 {output_dir}/enhanced_classified_data.xlsx")
        print(f"  🎨 {output_dir}/enhanced_analysis.png")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
