# Enhanced Consciousness Theory Classifier with Multi-Level Metaphor Analysis

This project classifies consciousness theories in AI art curatorial texts using advanced metaphor analysis. It addresses the challenge of detecting "metaphor of metaphor" patterns where artists transform scientific metaphors about consciousness into artistic images.

## Key Innovation: Metaphor of Metaphor Detection

The system detects multi-level metaphorical transformations:

```
Scientific Theory → Scientific Metaphor → Artistic Meta-Metaphor → Artwork
     PRED        "brain predicts"    "generative model as imagination"   AI art
```

## Features

### Enhanced Metaphor Analysis
- **4-Level Metaphor Detection**: explicit_term, scientific_metaphor, **meta_metaphor**, nested_metaphor
- **Ontological Typing**: Prioritizes ontological metaphors (about nature of mind) over decorative
- **Semantic Ontology**: Maps 7 consciousness theories to metaphor patterns
- **Meta-Metaphor Pattern Detection**: Identifies artistic transformations of scientific concepts
- **Hybrid Theory Detection**: Recognizes when artists combine multiple theories

### Technical Features
- **OpenRouter API Integration**: Uses OpenRouter instead of direct Anthropic API
- **Excel Processing**: Automatically processes `combined_ai_preprocessed.xlsx`
- **Enhanced Visualizations**: 9 comprehensive charts including Meta-Metaphor Detection Rate
- **Multiple Output Formats**: JSON, CSV, XLSX, TXT, and PNG files
- **Pre-Analysis**: Keyword-based metaphor detection before LLM classification
- **Error Handling**: Robust retry logic and error recovery

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Copy `.env.example` to `.env` (or use the provided `.env`)
   - Add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

3. **Optional Configuration**:
   - Change the model in `.env`:
   ```
   OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
   # or
   OPENROUTER_MODEL=openai/gpt-4
   ```

## Usage

### Enhanced Classifier (Recommended)
```bash
# Run enhanced classifier with metaphor analysis
python enhanced_classifier.py
```

This produces enhanced results with:
- Multi-level metaphor detection
- Meta-metaphor pattern analysis
- 9 comprehensive visualizations
- Detailed metaphor network analysis

### Standard Classifier
```bash
# Run original classifier
python openrouter_classifier.py
```

### Test Metaphor Analysis
```bash
# Test the metaphor detection system
python test_metaphor_analysis.py
```

### Custom Configuration
Edit the `.env` file to customize:
- API key
- Model selection
- Output directory
- Retry settings

## Quick Start

1. Copy `.env.example` to `.env` and add your OpenRouter API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

2. Run the enhanced classifier:
   ```bash
   python enhanced_classifier.py
   ```

3. Check results in `enhanced_analysis_results_*/` directory

## Output Files

### Enhanced Classifier Output
The enhanced classifier creates `enhanced_analysis_results_*/` containing:

1. **enhanced_analysis_summary.txt**: Human-readable summary with metaphor analysis
2. **enhanced_classification_results.json**: Detailed JSON with metaphor levels/types
3. **enhanced_classified_data.csv**: Original data with classifications
4. **enhanced_classified_data.xlsx**: Excel format with all data
5. **enhanced_analysis.png**: 9 comprehensive charts including:
   - Primary class distribution
   - Meta-metaphor detection rate
   - Theory hybridity analysis
   - Metaphor levels distribution
   - Metaphor types distribution
   - And more...

### Standard Classifier Output
The standard classifier creates `analysis_results_*/` containing:

1. **analysis_summary.txt**: Human-readable summary
2. **classification_results.json**: Detailed JSON results
3. **classified_data.csv**: Original data with classifications
4. **analysis_summary.png**: Distribution charts
5. **scores_heatmap.png**: Theory scores visualization

## File Structure

```
.
├── enhanced_classifier.py          # Enhanced classifier with metaphor analysis (Recommended)
├── openrouter_classifier.py        # Standard classifier
├── metaphor_analyzer.py            # Core metaphor analysis engine
├── test_metaphor_analysis.py       # Test suite for metaphor detection
├── .env.example                    # Example configuration
├── .env                            # Your configuration (create from .env.example)
├── requirements.txt                # Python dependencies
├── combined_ai_preprocessed.xlsx   # Input data
├── METAPHOR_ANALYSIS_GUIDE.md      # Comprehensive documentation (600+ lines)
├── README.md                       # This file
├── enhanced_analysis_results_*/    # Enhanced output (recommended)
│   ├── enhanced_analysis_summary.txt
│   ├── enhanced_classification_results.json
│   ├── enhanced_classified_data.csv
│   ├── enhanced_classified_data.xlsx
│   └── enhanced_analysis.png (9 charts)
└── analysis_results_*/             # Standard output
    ├── analysis_summary.txt
    ├── classification_results.json
    ├── classified_data.csv
    ├── analysis_summary.png
    └── scores_heatmap.png
```

## Consciousness Theories

The classifier identifies 8 categories:
- **COMP**: Computational Functionalism
- **IIT**: Integrated Information Theory
- **PRED**: Predictive Processing/Free Energy
- **GWT**: Global Workspace Theory
- **ENACT**: Enactivism
- **PAN**: Panpsychism
- **EMERG**: Emergentism
- **UND**: Undetermined

## Meta-Metaphor Patterns by Theory

The enhanced classifier detects artistic transformations of scientific metaphors:

| Theory | Scientific Metaphor | Meta-Metaphor in AI Art |
|--------|---------------------|-------------------------|
| **COMP** | "brain as computer" | "algorithm as artist" |
| **PRED** | "perception as prediction" | "generative model as imagination" |
| **ENACT** | "cognition through body" | "gesture as thought" |
| **EMERG** | "consciousness from complexity" | "swarm mind" |
| **IIT** | "experience as whole" | "indivisible image" |
| **GWT** | "attention as spotlight" | "scene of perception" |
| **PAN** | "matter feels" | "animated objects" |

### Example Analysis

**Input text:**
```
The artist uses a generative adversarial network to create images
that emerge from the model's imagination, transforming the algorithm
into a creative agent.
```

**Detected metaphors:**
- `meta_metaphor` (weight: 0.96): "generative model as imagination" → PRED
- `meta_metaphor` (weight: 0.80): "algorithm as creative agent" → COMP
- Pattern detected: Scientific concept → Artistic practice

**Classification:**
- Primary: PRED (Predictive Processing)
- Secondary: COMP (Computational Functionalism)
- Hybrid: Yes
- Meta-metaphor detected: **True**

## Error Handling

- Automatic retry on rate limits (exponential backoff)
- Graceful handling of API errors
- Detailed error logging
- Continues processing even if individual items fail

## Rate Limiting

The script includes built-in rate limiting:
- 0.5 second delay between requests
- Exponential backoff on 429 errors
- Configurable via `.env` file

## Troubleshooting

1. **API Key Issues**: Ensure `OPENROUTER_API_KEY` is set in `.env`
2. **Rate Limits**: Increase `RATE_LIMIT_DELAY` in `.env`
3. **Excel Errors**: Check that `combined_ai_preprocessed.xlsx` has the expected structure
4. **Memory Issues**: Process smaller batches by modifying the script

## Example Output

```
Processing 1/10: Sublime landscapes/Гибридные ландшафты
Processing 2/10: All in all. Project
...
Analysis complete! Results saved to: analysis_results_20241204_123456
```

## Advanced Usage

### Programmatic Usage

**Enhanced Classifier:**
```python
from enhanced_classifier import EnhancedMetaphorClassifier

classifier = EnhancedMetaphorClassifier()

# Classify single description
result = classifier.classify_description(description)

# Access metaphor analysis
if result['metaphor_analysis']['meta_metaphor_detected']:
    print(f"Meta-metaphor detected!")
    print(f"Transformation: {result['metaphor_analysis']['artistic_transformation']}")

# Batch processing
results = classifier.classify_batch(descriptions, titles)
```

**Metaphor Analyzer (standalone):**
```python
from metaphor_analyzer import MetaphorAnalyzer

analyzer = MetaphorAnalyzer()

# Extract metaphors
evidence = analyzer.extract_metaphors(text)

for e in evidence:
    print(f"Theory: {e.theory_class}")
    print(f"Level: {e.level.value}")  # explicit_term, scientific_metaphor, meta_metaphor
    print(f"Type: {e.type.value}")    # ontological, structural, orientational, decorative
    print(f"Weight: {e.weight:.2f}")

# Analyze metaphor network
network = analyzer.analyze_metaphor_network(evidence)
print(f"Meta-metaphor detected: {network['meta_metaphor_detected']}")
print(f"Dominant theory: {network['dominant_pattern']}")
```

**Standard Classifier:**
```python
from openrouter_classifier import OpenRouterClassifier

classifier = OpenRouterClassifier()
results = classifier.classify_batch(descriptions, titles)```

## Research Applications

### Testing the Hypothesis

This system enables empirical testing of the hypothesis:

> **Artists working with ML algorithms thematize the hard problem of consciousness by artistically transforming scientific metaphors.**

### Key Research Metrics

1. **Meta-Metaphor Detection Rate** - What % of AI artworks use meta-metaphors?
2. **Dominant Theories** - Which consciousness theories are most prevalent?
3. **Theory Hybridity** - How often do artists combine theories?
4. **Ontological Depth** - Ratio of ontological vs decorative metaphors

### Interpreting Results

**Strong Theoretical Engagement:**
- ✅ `meta_metaphor_detected: true`
- ✅ `confidence: "high"`
- ✅ Multiple ontological metaphors
- ✅ Clear artistic transformation pattern

**Weak/Decorative Usage:**
- ❌ Only decorative metaphors (low weight)
- ❌ `primary_class: "UND"`
- ❌ No meta-metaphor pattern

### Sample Research Questions

1. Do artists using GANs prefer PRED (Predictive Processing) theories?
2. Are ENACT (Enactivism) metaphors more common in interactive AI art?
3. Which theories show strongest meta-metaphor patterns?
4. How do curators' descriptions differ from artists' statements?

## Documentation

- **METAPHOR_ANALYSIS_GUIDE.md** - Comprehensive 600+ line guide covering:
  - Theoretical foundation (Lakoff, Chalmers, Blumenberg)
  - Four-level metaphor model
  - Weight calculation formulas
  - Interpretation guidelines
  - 8 detailed examples
  - Methodological recommendations

- **test_metaphor_analysis.py** - Test suite with examples for all 8 theory categories

## Theoretical Foundation

This system builds on:

- **Conceptual Metaphor Theory** (Lakoff & Johnson) - Metaphors as cognitive structures
- **Hard Problem of Consciousness** (Chalmers) - Why physical processes create subjective experience
- **Metaphorology** (Blumenberg) - Metaphors as ways to conceptualize the unknowable

### The "Metaphor of Metaphor" Pattern

```
Level 1: Scientific Theory (e.g., Predictive Processing)
    ↓
Level 2: Scientific Metaphor ("brain predicts reality")
    ↓
Level 3: Artistic Meta-Metaphor ("AI imagines images")
    ↓
Level 4: Artwork (Generative AI art)
```

## Future Enhancements

Planned improvements:

1. **Multimodal Analysis** - Integration of image analysis with vision-language models
2. **Graph-based Metaphor Networks** - Tracing transformation paths from theory to artwork
3. **Fine-tuned Models** - Training on AI art corpus for better meta-metaphor detection
4. **Multilingual Support** - Expanding ontology to other languages
5. **Semantic Embeddings** - Moving beyond keyword matching to deep semantic analysis

## Citation

```
Enhanced Metaphor Analysis for Consciousness Theories in AI Art (2024)
Multi-level metaphor detection system with focus on meta-metaphor patterns
https://github.com/stainephone-gif/AI-art
```

## Contributing

For questions, suggestions, or contributions, please open an issue or pull request.

## License

See LICENSE file for details.
