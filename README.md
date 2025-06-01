🧠 Named Entity Recognition for Information Extraction
This project compares different Named Entity Recognition (NER) models to extract entities such as people, organizations, and locations from text using deep learning and transformer-based approaches.

🔬 Course
CSE561.1 – Advanced Natural Language Processing
Işık University
Presented by: Yiğit ÖZYAPAN

📌 Table of Contents
Overview

Dataset

Models Used

Training Setup

Evaluation Metrics

Results

Conclusion

Future Work

How to Run

Dependencies

📖 Overview
This project investigates three different model architectures for the NER task:

BiLSTM (traditional RNN-based model)

BERT (Transformer via HuggingFace)

SpaCy Transformer (RoBERTa-based pre-trained pipeline)

📊 Dataset
Name: Wikiner

Language: English

Format: CoNLL-style (BIO tagging)

Size: ~144,000 sentences

Entity Types: PER, ORG, LOC, MISC

🧠 Models Used
Model	Framework	Notes
BiLSTM	PyTorch	Lightweight, interpretable
BERT	HuggingFace	High accuracy, contextual understanding
SpaCy Transformer	SpaCy + RoBERTa	No training, fast inference

⚙️ Training Setup
BiLSTM:

Embedding Dim: 100

Hidden Units: 128

Epochs: 3

Batch Size: 32

BERT:

Model: bert-base-cased

Epochs: 3

Batch Size: 16

Trainer: HuggingFace Trainer API

SpaCy:

Used directly for inference (no training)

📈 Evaluation Metrics
F1 Score

Precision & Recall

Token-Level Accuracy

Training and Validation Loss

Model Inference Speed

Parameter Count Comparison

Tracking was done using Weights & Biases (wandb) for visualization.

🧪 Results
BERT achieved the highest F1 score and accuracy.

BiLSTM was efficient but limited in understanding context.

SpaCy Transformer was the fastest but less accurate without fine-tuning.

Sample graphs and visualizations are available in the /results folder.

✅ Conclusion
Model performance varies by trade-off:

BERT: Best accuracy, high cost

BiLSTM: Lightweight, good for small-scale

SpaCy: Ideal for fast inference with minimal setup

🔮 Future Work
Integrate a CRF layer with BiLSTM for better sequence labeling.

Try other transformer models (e.g., RoBERTa, DistilBERT).

Fine-tune SpaCy on Wikiner for improved domain accuracy.

Optimize inference speed for real-time applications.

Perform deeper error analysis.

🛠️ How to Run
bash
Kopyala
Düzenle
# Install dependencies
pip install -r requirements.txt

# Train BiLSTM
python train_bilstm.py

# Train BERT (HuggingFace)
python train_bert.py

# Evaluate with SpaCy
python evaluate_spacy.py
📦 Dependencies
Python 3.8+

PyTorch

HuggingFace Transformers

SpaCy

wandb

seqeval

scikit-learn

