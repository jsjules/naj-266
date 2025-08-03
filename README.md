# GoEmotions: Emotion Classification and Analysis

## Overview

This project builds on the [GoEmotions](https://arxiv.org/abs/2005.00547) dataset, a human-annotated corpus of 58k Reddit comments labeled for 27 fine-grained emotion categories plus Neutral. Our goal is to analyze, evaluate, and build improved emotion classification systems using both fine-grained and coarse-grained (Ekman) taxonomies.

The project involves:

- Dataset analysis and visualization
- Emotion-word association extraction
- Label remapping to higher-level categories (e.g., Ekman)
- Model training and evaluation using fine-tuned BERT

We incorporate surrounding comment context (subreddit name, author metadata) into our exploration of misclassification and emotion ambiguity.

## Dataset Summary

- **Source**: Reddit comments, curated by Google Research
- **Examples**: 58,009
- **Labels**: 27 fine-grained emotion categories + Neutral
- **Sequence length**: Max 30 tokens

Filtered by rater agreement (>=2), we use the following splits:

- **Train**: 43,410
- **Validation**: 5,426
- **Test**: 5,427

### Emotion Categories

`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise`

### Data Format

- Raw `.csv` files in `data/full_dataset/` include metadata and annotations
- Filtered and remapped `.tsv` files contain:
  1. Comment text
  2. Comma-separated emotion IDs
  3. Comment ID

## Project Components

### Data Analysis Scripts

- `analyze_data.py`: Computes label distributions, correlations, hierarchical clustering.
- `extract_words.py`: Computes top words for each emotion using log-odds ratio analysis.
- `replace_labels.py`: Maps fine-grained labels into coarse-grained categories (e.g., Ekman).
- `evaluate_predictions.py`: Evaluates classifier predictions against ground truth using accuracy, precision, recall, and F1 (macro, micro, weighted).

### Visualizations

We include support for:

- Heatmaps of label correlations
- Dendrograms of emotion label clustering
- Sentiment-colored clustermaps
- Top word bar plots by emotion

### Modeling

This project explores multiple transformer-based emotion classification models trained on the GoEmotions dataset, using both text-only and context-augmented variants.

#### **Model Scripts**

* `bert_classifier.py`: Fine-tunes **BERT-base (cased)** for **multi-label classification** on GoEmotions.

  * Supports optional **label remapping** and **hierarchical loss**.
* `roberta_classifier.py`: Fine-tunes **RoBERTa-base** for **multi-class classification** using cross-entropy loss.

  * Supports token prepending for contextual metadata (e.g., `[SUBREDDIT:]`, `[AUTHOR:]`) and dynamic embedding resizing.
  * Integrated with Hugging Face’s `Trainer` API and macro F1 as the main evaluation metric.

#### **Modeling Notebooks**

* `01_baseline_text_model.ipynb`: Trains the **baseline RoBERTa model** on comment text only (no context).
* `text_model_with_no_context_RoBERTa_cleaned_data.ipynb`: RoBERTa trained on **cleaned data** without any context.
* `text_model_with_subreddit_context_RoBERTa_cleaned_data.ipynb`: RoBERTa trained with **subreddit prepended**.
* `text_model_with_author_context_RoBERTa_cleaned_data.ipynb`: RoBERTa trained with **author identity prepended**.
* `text_model_with_subreddit_and_author_context_RoBERTa_cleaned_data.ipynb`: RoBERTa trained with **combined subreddit and author context**.
* `bert_multi_label_text_classification.ipynb`: Fine-tunes **multi-label BERT** using the raw or remapped emotion labels.
* `deberta_model_raw_data.ipynb`: DeBERTa trained on the **raw GoEmotions dataset** (unfiltered).
* `deberta_clean_data.ipynb`: DeBERTa trained on the **cleaned dataset** without context.
* `cos_deberta_context.ipynb`: DeBERTa trained on cleaned data with **author + subreddit context**.
* `text_model_with_subreddit_and_author_context_deBERTa_cleaned_data.ipynb`: Final **DeBERTa context-aware model**, also used to generate plots.


## Ekman Label Mapping

We provide a mapping file `ekman_mapping.json` that aggregates fine-grained GoEmotions into the six Ekman universal emotions + Neutral. This enables:

- Coarser-grained classification
- Emotion-level analysis at different abstraction levels

Example:

```json
{
  "anger": ["anger", "annoyance", "disapproval"],
  "joy": ["joy", "gratitude", "love"],
  "sadness": ["grief", "remorse", "sadness", "disappointment"],
  "neutral": ["neutral"]
}
```

## How to Run

1. Mount your Google Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Follow individual notebooks/scripts:

- For data analysis: see `analyze_data_colab.ipynb`
- For top words: see `extract_words_colab.ipynb`
- For label remapping: see `replace_labels_colab.ipynb`
- For evaluation: see `evaluate_predictions_colab.ipynb`

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn
- PyTorch or TensorFlow for modeling (depending on classifier)

## Baseline Results

We replicate and extend the BERT-based baseline reported in the [GoEmotions paper](https://arxiv.org/abs/2005.00547). Metrics include:

- Emotion-level F1 (macro, micro)
- Ekman-level and sentiment-level performance

## Limitations

- Reddit-based data introduces demographic and cultural bias
- Labels are context-free, often ambiguous without surrounding conversation
- Annotators were native English speakers from India, which may affect emotion perception

We highlight the importance of cautious deployment and fairness-aware modeling.

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{demszky2020goemotions,
 author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
 title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
 booktitle = {ACL},
 year = {2020}
}
```

## Contributors

- Arun Agarwal (UC Berkeley MIDS)
- [Original GoEmotions team @ Google Research](https://github.com/google-research/google-research/tree/master/goemotions)

## License

Apache 2.0. See [LICENSE](LICENSE) file for details.

---

For more on model cards, ethical use, and detailed results, see:

- [GoEmotions Model Card](goemotions_model_card.pdf)
- [GoEmotions Paper](https://arxiv.org/abs/2005.00547)

