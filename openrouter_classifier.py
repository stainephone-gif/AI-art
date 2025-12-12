"""
OpenRouter API Classifier for Consciousness Theories in Media Art
Processes Excel file 'combined_ai_preprocessed.xlsx' and saves analysis results
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
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# System prompt definitions (moved from llm_classifier_prompt.py)
SYSTEM_PROMPT = """Вы — эксперт-социолог, специализирующийся на философии сознания и медиаискусстве.

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

УРОВНИ ИНДИКАТОРОВ (по убыванию надёжности):
- EXPLICIT: прямое упоминание термина, теории или автора (вес 1.0)
- TECHNICAL_CUE: технический сигнал без прямого термина (вес 0.6)
- METAPHOR: метафора, соответствующая онтологии теории (вес 0.4)

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
1. Для каждого класса оцените вероятность от 0.0 до 1.0
2. ДЛЯ ЛЮБОЙ ОЦЕНКИ > 0.3 обязательно приведите ТОЧНЫЕ ЦИТАТЫ из описания
3. Укажите тип каждой цитаты: explicit / technical_cue / metaphor
4. Если уверенности нет (все оценки < 0.5) → основной класс UND
5. Отвечайте ТОЛЬКО в формате JSON (без преамбулы, без markdown-блоков)

КРИТЕРИИ ОЦЕНКИ:
- COMP: алгоритмы, символьная обработка, Тьюринг, функционализм, input-output
- IIT: интеграция, phi, Tononi, несводимость к частям, квалиа
- PRED: предсказание, ошибка, приор, Bayesian, free energy, Friston
- GWT: внимание, broadcast, театр разума, Baars, рабочее пространство
- ENACT: телесность, сенсомоторные петли, Varela, embodied, действие=познание
- PAN: протоопыт, одушевление материи, Strawson, Goff, всё чувствует
- EMERG: самоорганизация, рой, сложность→новое качество, фазовый переход

ФОРМАТ ОТВЕТА (строго JSON):
{
  "primary_class": "PRED",
  "confidence": "high",  // high / medium / low
  "scores": {
    "COMP": 0.1,
    "IIT": 0.2,
    "PRED": 0.9,
    "GWT": 0.0,
    "ENACT": 0.3,
    "PAN": 0.0,
    "EMERG": 0.1,
    "UND": 0.0
  },
  "evidence": [
    {
      "class": "PRED",
      "type": "technical_cue",
      "span": "система минимизирует расхождение между прогнозом и данными",
      "weight": 0.6,
      "reasoning": "Описывает механизм предиктивного кодирования без явного термина"
    },
    {
      "class": "PRED",
      "type": "explicit",
      "span": "основана на принципе свободной энергии Фристона",
      "weight": 1.0,
      "reasoning": "Прямое упоминание Friston и free energy principle"
    },
    {
      "class": "ENACT",
      "type": "technical_cue",
      "span": "агент познаёт через телесное взаимодействие со средой",
      "weight": 0.6,
      "reasoning": "Сенсомоторная петля как основа когниции"
    }
  ],
  "secondary_class": "ENACT",  // если score ≥ 0.5
  "is_hybrid": true,  // если 2+ класса ≥ 0.5
  "notes": "Комбинация предиктивного кодирования с энактивным подходом"
}

ВАЖНО:
- Не придумывайте цитаты — копируйте ТОЧНО из описания
- Если термин есть, но используется не философски (например, "алгоритм генерации" без связи с сознанием) → не считается индикатором
- Метафоры учитываются только если они ОНТОЛОГИЧЕСКИЕ (про природу разума), не декоративные
- При сомнении между двумя классами — лучше дать обоим средние оценки, чем выбрать неправильный
"""

USER_PROMPT_TEMPLATE = """Проанализируй следующее описание художественной работы и классифицируй по теории сознания.

ОПИСАНИЕ:
\"\"\"
{description}
\"\"\"

Проведи анализ поэтапно:
1. Найди все явные термины (explicit)
2. Выяви технические сигналы (technical_cue)
3. Оцени метафоры (metaphor)
4. Рассчитай оценки для каждого класса
5. Выведи результат в JSON-формате

Ответь ТОЛЬКО JSON без дополнительного текста."""

class OpenRouterClassifier:
    def __init__(self):
        """Initialize with OpenRouter API configuration from .env"""
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv('HTTP_REFERER', 'http://localhost:3000'),
            "X-Title": os.getenv('X_TITLE', 'Consciousness Classifier')
        }
        
    def classify_description(self, description: str, max_retries: int = 3) -> dict:
        """
        Classify a single description using OpenRouter API
        
        Args:
            description: Text description to classify
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: Classification results
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(description=description)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2000,
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
                    
                    return json.loads(content.strip())
                
                elif response.status_code == 429:
                    # Rate limit - wait and retry
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
        """
        Process Excel file and extract descriptions
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            DataFrame with processed data
        """
        try:
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name='data')
            
            # Combine relevant text columns
            text_columns = ['descr', 'descr_en', 'descr_clean', 'descr_lemmas']
            available_cols = [col for col in text_columns if col in df.columns]
            
            if not available_cols:
                raise ValueError("No text columns found in Excel file")
            
            # Create combined description
            df['combined_description'] = df[available_cols].fillna('').agg(' '.join, axis=1)
            df['combined_description'] = df['combined_description'].str.strip()
            
            return df
            
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            raise
    
    def classify_batch(self, descriptions: list[str], titles: list[str] = None) -> list[dict]:
        """
        Classify multiple descriptions
        
        Args:
            descriptions: List of text descriptions
            titles: Optional list of titles for tracking
            
        Returns:
            List of classification results
        """
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
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        return results
    
    def create_visualizations(self, results: list[dict], output_dir: str):
        """
        Create visualizations and save to output directory
        
        Args:
            results: Classification results
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        df_results = pd.DataFrame(results)
        
        # Filter successful classifications
        successful = df_results[df_results['status'] != 'error']
        
        if len(successful) == 0:
            print("No successful classifications to visualize")
            return
        
        # 1. Distribution of primary classes
        plt.figure(figsize=(12, 8))
        
        # Count primary classes
        class_counts = successful['primary_class'].value_counts()
        
        # Create bar plot
        plt.subplot(2, 2, 1)
        class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Distribution of Primary Classes', fontsize=14, fontweight='bold')
        plt.xlabel('Consciousness Theory')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 2. Confidence levels
        plt.subplot(2, 2, 2)
        if 'confidence' in successful.columns:
            confidence_counts = successful['confidence'].value_counts()
            confidence_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightcoral'])
            plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
            plt.ylabel('')
        
        # 3. Average scores by class (if scores available)
        plt.subplot(2, 2, 3)
        scores_data = []
        for _, row in successful.iterrows():
            if 'scores' in row and isinstance(row['scores'], dict):
                for class_name, score in row['scores'].items():
                    scores_data.append({
                        'class': class_name,
                        'score': score
                    })
        
        if scores_data:
            scores_df = pd.DataFrame(scores_data)
            scores_df.groupby('class')['score'].mean().plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.title('Average Scores by Theory', fontsize=14, fontweight='bold')
            plt.xlabel('Theory')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
        
        # 4. Description length vs classification
        plt.subplot(2, 2, 4)
        if 'description_length' in successful.columns:
            successful.boxplot(column='description_length', by='primary_class', ax=plt.gca())
            plt.title('Description Length by Classification', fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove default title
            plt.xlabel('Theory')
            plt.ylabel('Description Length (chars)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analysis_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Detailed class distribution heatmap
        if scores_data:
            plt.figure(figsize=(10, 8))
            pivot_scores = scores_df.pivot_table(values='score', index='class', aggfunc='mean')
            sns.heatmap(pivot_scores.values.reshape(-1, 1), 
                       annot=True, 
                       fmt='.2f',
                       yticklabels=pivot_scores.index,
                       xticklabels=['Average Score'],
                       cmap='YlOrRd')
            plt.title('Average Scores Heatmap by Theory', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'scores_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, results: list[dict], output_dir: str, original_df: pd.DataFrame):
        """
        Save results to files
        
        Args:
            results: Classification results
            output_dir: Output directory
            original_df: Original DataFrame for context
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON results
        with open(os.path.join(output_dir, 'classification_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save summary text file
        with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("CONSCIOUSNESS THEORY CLASSIFICATION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Items Processed: {len(results)}\n")
            f.write(f"Model Used: {self.model}\n\n")
            
            # Summary statistics
            successful = [r for r in results if r.get('status') != 'error']
            errors = [r for r in results if r.get('status') == 'error']
            
            f.write(f"Successful Classifications: {len(successful)}\n")
            f.write(f"Errors: {len(errors)}\n\n")
            
            if successful:
                # Class distribution
                from collections import Counter
                classes = [r['primary_class'] for r in successful]
                class_counts = Counter(classes)
                
                f.write("CLASS DISTRIBUTION:\n")
                f.write("-" * 20 + "\n")
                for class_name, count in class_counts.most_common():
                    percentage = (count / len(successful)) * 100
                    f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
                
                f.write("\n")
                
                # Detailed results
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 20 + "\n")
                for result in successful:
                    f.write(f"\nTitle: {result.get('title', 'N/A')}\n")
                    f.write(f"Primary Class: {result.get('primary_class', 'N/A')}\n")
                    f.write(f"Confidence: {result.get('confidence', 'N/A')}\n")
                    
                    if 'scores' in result:
                        f.write("Scores:\n")
                        for class_name, score in result['scores'].items():
                            f.write(f"  {class_name}: {score:.2f}\n")
                    
                    if 'evidence' in result:
                        f.write("Evidence:\n")
                        for evidence in result['evidence'][:3]:  # Top 3
                            f.write(f"  - {evidence.get('type', 'N/A')}: {evidence.get('span', 'N/A')[:100]}...\n")
                    
                    f.write("-" * 40 + "\n")
        
        # Save CSV with results merged back to original data
        if len(original_df) == len(results):
            results_df = pd.DataFrame(results)
            merged_df = pd.concat([original_df, results_df], axis=1)
            merged_df.to_csv(os.path.join(output_dir, 'classified_data.csv'), index=False, encoding='utf-8')
        
        print(f"Results saved to {output_dir}")


def main():
    """Main execution function"""
    try:
        # Initialize classifier
        classifier = OpenRouterClassifier()
        
        # Process Excel file
        print("Processing Excel file...")
        df = classifier.process_excel_file('combined_ai_preprocessed.xlsx')
        
        # Get descriptions and titles
        descriptions = df['combined_description'].tolist()
        titles = df.get('title', [f'Item_{i}' for i in range(len(descriptions))]).tolist()
        
        print(f"Found {len(descriptions)} items to classify")
        
        # Classify all descriptions
        print("Starting classification...")
        results = classifier.classify_batch(descriptions, titles)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'analysis_results_{timestamp}'
        
        # Save results
        classifier.save_results(results, output_dir, df)
        
        # Create visualizations
        print("Creating visualizations...")
        classifier.create_visualizations(results, output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Files created:")
        print(f"  - {output_dir}/analysis_summary.txt")
        print(f"  - {output_dir}/classification_results.json")
        print(f"  - {output_dir}/classified_data.csv")
        print(f"  - {output_dir}/analysis_summary.png")
        print(f"  - {output_dir}/scores_heatmap.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()