# OpenRouter Consciousness Theory Classifier

This project uses OpenRouter API to classify consciousness theories in media art descriptions from the `combined_ai_preprocessed.xlsx` file.

## Features

- **OpenRouter API Integration**: Uses OpenRouter instead of direct Anthropic API
- **Excel Processing**: Automatically processes `combined_ai_preprocessed.xlsx`
- **Comprehensive Analysis**: Generates detailed classification results
- **Visualizations**: Creates charts and heatmaps
- **Multiple Output Formats**: JSON, CSV, TXT, and PNG files
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

### Basic Usage
```bash
python openrouter_classifier.py
```

### Custom Configuration
Edit the `.env` file to customize:
- API key
- Model selection
- Output directory
- Retry settings

## Output Files

The script creates a timestamped directory (e.g., `analysis_results_20241204_123456/`) containing:

1. **analysis_summary.txt**: Human-readable summary
2. **classification_results.json**: Detailed JSON results
3. **classified_data.csv**: Original data with classifications
4. **analysis_summary.png**: Distribution charts
5. **scores_heatmap.png**: Theory scores visualization

## File Structure

```
.
├── openrouter_classifier.py    # Main script
├── .env                        # Configuration file
├── requirements.txt            # Python dependencies
├── combined_ai_preprocessed.xlsx  # Input data
├── llm_classifier_prompt.py    # Original prompt definitions
└── analysis_results_*/         # Generated output directories
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

For batch processing or integration with other tools, you can import the classifier:

```python
from openrouter_classifier import OpenRouterClassifier

classifier = OpenRouterClassifier()
results = classifier.classify_batch(descriptions, titles)