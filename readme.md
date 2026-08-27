# e-Nurse

A retrieval-based medical triage and guidance assistant built on the [MedQuAD](https://github.com/abachaa/MedQuAD) dataset using **Sentence-BERT** and **Scikit-Learn**. No LLM required.

## Features

- Intent classification (TF-IDF + LinearSVC)
- Semantic search (SBERT + cosine similarity)
- Urgency detection with custom lexicon
- Follow-up question suggestions
- School-clinic oriented quick topics

---

## Prerequisites

- **Python 3.9+** (tested on Python 3.13)
- **pip** package manager

---

## Install & Setup

### 1. Clone or download the project

```
git clone <repo-url>
cd e-NUrse
```

Or download and extract the ZIP, then open the folder.

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
```

Activate it:

| OS | Command |
|---|---|
| Windows | `venv\Scripts\activate` |
| Mac/Linux | `source venv/bin/activate` |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.6.1 (pinned) | Intent classification |
| `sentence-transformers` | latest | Semantic search embeddings |
| `streamlit` | latest | Web UI |

### 4. First run

```bash
streamlit run app.py
```

On first launch, the SBERT model (~90 MB) downloads from HuggingFace automatically.

The app opens at **http://localhost:8501**.

---

## Project Structure

```
e-NUrse/
├── app.py                        # Streamlit web app (main)
├── chatbot.py                    # CLI chatbot (optional)
├── requirements.txt              # Python dependencies
├── intent_pipeline_model.joblib  # Pre-trained TF-IDF + LinearSVC pipeline
├── corpus_embeddings.joblib      # Pre-computed SBERT embeddings (13,124 x 384)
├── train_data.csv                # Training data (13,124 rows)
├── Health_Information.ipynb      # Training notebook (source of truth)
├── response_map.joblib           # Legacy (unused)
├── sbert_embedder.joblib         # Legacy (unused)
└── readme.md                     # This file
```

---

## How It Works

1. **Intent Classification** — User query goes through TF-IDF + LinearSVC to predict intent (symptoms, treatment, causes, etc.)
2. **Search Query Formulation** — Combines predicted intent + user query
3. **Semantic Search** — SBERT encodes the query, finds closest match in pre-computed corpus embeddings via cosine similarity
4. **Urgency Detection** — Lexicon-based scoring triggers urgent care advisory when needed
5. **Response** — Returns the best-matching answer from MedQuAD, cleaned and truncated

---

## Customization

### Adjust confidence threshold

In `app.py`, change the default:
```python
threshold = 0.60  # higher = stricter matching
```

### Edit urgency lexicon

Modify the `URGENCY_LEXICON` dictionary in `app.py`:
```python
URGENCY_LEXICON = {
    "emergency": -2,   # high urgency
    "pain": -1,        # moderate urgency
    "better": 1,       # improving
}
```

### Change quick topics

Edit the `GUIDE_TOPICS` list in the sidebar section of `app.py`.

---

## Retraining

To retrain the intent classifier from scratch, run all cells in `Health_Information.ipynb`.

**Important:** The pre-trained pipeline uses scikit-learn 1.6.1. If you retrain with a different version, the pickled model will be incompatible.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn==1.6.1` |
| `ModuleNotFoundError: torch` | Run `pip install torch` |
| SentenceTransformer error | The app loads from HuggingFace, not the pickled file. Check internet connection. |
| Corpus mismatch | Ensure `train_data.csv` (13,124 rows) and `corpus_embeddings.joblib` (13,124 x 384) are from the same training run |

---

## License

Built for **School Clinic Guidance** using publicly available MedQuAD data.
