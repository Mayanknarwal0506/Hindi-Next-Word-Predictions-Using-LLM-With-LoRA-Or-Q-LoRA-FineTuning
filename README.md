# Hindi Next Word Prediction Using Large Language Models (LLMs) with LoRA and QLoRA Fine-Tuning

## Overview

This project presents an efficient Hindi Next Word Prediction system using Transformer-based Large Language Models (LLMs). The system predicts the most probable next word in a Hindi sentence by leveraging pretrained language models and Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA (Low-Rank Adaptation) and QLoRA (Quantized Low-Rank Adaptation).

The project focuses on adapting powerful English-centric transformer models for Hindi language modeling while minimizing computational cost and memory requirements.

---

## Project Objectives

- Develop an efficient Hindi next-word prediction system.
- Build a large-scale Hindi text corpus.
- Train a custom Hindi tokenizer using SentencePiece.
- Compare training from scratch against pretrained transformer models.
- Apply LoRA and QLoRA for parameter-efficient fine-tuning.
- Evaluate model performance using Validation Loss and Perplexity.
- Deploy the model through a Streamlit-based web application.

---

## Problem Statement

Most state-of-the-art Large Language Models are primarily trained on English corpora, resulting in reduced performance for low-resource languages such as Hindi.

This project aims to address the following challenge:

> Develop an efficient Hindi next-word prediction model using Large Language Models and parameter-efficient fine-tuning techniques while maintaining low computational cost and high prediction accuracy.

---

## System Workflow

```text
Hindi Dataset
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
SentencePiece Tokenizer Training
      │
      ▼
Tokenized Dataset Creation
      │
      ▼
Model Training
 ┌───────────────┬──────────────────┬─────────────────────┐
 │ GPT-2 Scratch │ GPT-2 + LoRA     │ GPT-Neo + LoRA      │
 └───────────────┴──────────────────┴─────────────────────┘
      │
      ▼
Model Evaluation
(Loss & Perplexity)
      │
      ▼
Inference
      │
      ▼
Streamlit Deployment
```

---

## Dataset Information

### Dataset Name

Hindi Next Word Prediction Dataset

### Dataset Size

- Total Samples: **167,850**
- Language: Hindi
- Format: Text Corpus

### Data Sources

- Hindi Books
- Educational Material
- Online Articles
- Public Hindi Datasets
- Hindi Wikipedia Content

### Example Dataset Sample

| Input Text |
|------------|
| भारत एक महान देश है |
| मुझे हिंदी सीखना पसंद है |
| आज मौसम बहुत अच्छा है |

---

## Data Preprocessing

The following preprocessing steps were applied:

- Unicode normalization
- Removal of special symbols
- Duplicate removal
- Null value removal
- Text cleaning
- Whitespace normalization
- Sentence formatting

---

## Tokenizer Training

A custom tokenizer was trained using SentencePiece.

### Tokenizer Configuration

| Parameter | Value |
|------------|---------|
| Tokenizer | SentencePiece |
| Algorithm | BPE (Byte Pair Encoding) |
| Vocabulary Size | 8,000 |

### Example

Input Sentence:

```text
भारत एक महान देश है
```

Tokenized Output:

```text
▁भारत ▁एक ▁महान ▁देश ▁है
```

---

## Models Used

### Model 1: GPT-2 (Trained from Scratch)

- Architecture: GPT-2
- Training Strategy: Full Training
- Pretrained Weights: No

### Model 2: GPT-2 + LoRA

- Architecture: GPT-2
- Pretrained Weights: Yes
- Fine-Tuning: LoRA

### Model 3: GPT-Neo-125M + LoRA

- Architecture: GPT-Neo 125M
- Pretrained Weights: Yes
- Fine-Tuning: LoRA

---

## Parameter-Efficient Fine-Tuning (PEFT)

### LoRA (Low-Rank Adaptation)

LoRA freezes the original pretrained model weights and introduces trainable low-rank matrices into transformer layers.

Advantages:

- Reduced GPU memory usage
- Faster training
- Smaller checkpoints
- Comparable performance to full fine-tuning

### QLoRA (Quantized Low-Rank Adaptation)

QLoRA extends LoRA by quantizing model weights to lower precision while retaining model quality.

Advantages:

- Significant memory savings
- Efficient training on consumer GPUs
- Scalable fine-tuning for larger models

---

## Model Training Configuration

| Parameter | Value |
|------------|---------|
| Framework | PyTorch |
| Transformers Library | Hugging Face |
| Tokenizer | SentencePiece |
| Task | Causal Language Modeling |
| Language | Hindi |
| Context Length | 128 |
| Vocabulary Size | 8,000 |

---

## Evaluation Metrics

### Validation Loss

Loss measures prediction error between the predicted token distribution and the actual token.

Lower loss indicates better prediction quality.

---

### Perplexity

Perplexity is calculated as:

```math
Perplexity = e^{Loss}
```

Perplexity measures how confidently the model predicts the next word.

Interpretation:

| Perplexity | Meaning |
|------------|------------|
| 1 - 3 | Excellent |
| 3 - 10 | Good |
| 10 - 50 | Moderate |
| >50 | Poor |

Lower perplexity indicates better language understanding.

---

## Experimental Results

| Model | Validation Loss | Perplexity |
|---------|---------|---------|
| GPT-2 (Scratch) | 1.0879 | 12.67 |
| GPT-2 + LoRA | 0.9780 | 2.65 |
| GPT-Neo + LoRA | 0.9909 | 2.53 |

---

## Performance Analysis

### GPT-2 Scratch

- Learned Hindi patterns from scratch
- Higher perplexity
- Limited generalization

### GPT-2 + LoRA

- Strong transfer learning capability
- Lowest validation loss
- Efficient adaptation to Hindi

### GPT-Neo + LoRA

- Best perplexity score
- Strong contextual understanding
- Improved language generation quality

---

## Inference Examples

### Example 1

Input:

```text
भारत एक
```

Prediction:

```text
महान
```

---

### Example 2

Input:

```text
मुझे हिंदी
```

Prediction:

```text
सीखना
```

---

### Example 3

Input:

```text
आज मौसम
```

Prediction:

```text
अच्छा
```

---

## Project Structure

```text
Hindi-Next-Word-Prediction/
│
├── data/
│   ├── raw_dataset.csv
│   ├── processed_dataset.csv
│
├── tokenizer/
│   ├── tokenizer.model
│   ├── tokenizer.vocab
│
├── models/
│   ├── gpt2_scratch/
│   ├── gpt2_lora/
│   ├── gptneo_lora/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── tokenizer_training.ipynb
│   ├── model_training.ipynb
│
├── streamlit_app/
│   ├── app.py
│
├── results/
│   ├── loss_graph.png
│   ├── perplexity_graph.png
│
├── requirements.txt
├── README.md
└── thesis.pdf
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Hindi-Next-Word-Prediction.git
```

Move into the project directory:

```bash
cd Hindi-Next-Word-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

```bash
streamlit run app.py
```

---

## Future Work

- Fine-tune larger models such as LLaMA, Mistral, and GPT-NeoX.
- Expand to multilingual Indian languages.
- Integrate QLoRA-based training.
- Increase context length.
- Improve evaluation using BLEU and human assessment.
- Deploy using cloud-based inference services.
- Build Hindi autocomplete and chatbot applications.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- SentencePiece
- LoRA
- QLoRA
- Streamlit
- NumPy
- Pandas
- Scikit-Learn

---

## Key Contributions

- Creation of a Hindi next-word prediction dataset.
- Training a custom Hindi tokenizer.
- Comparison of GPT-2 Scratch, GPT-2 + LoRA, and GPT-Neo + LoRA.
- Demonstration of efficient Hindi language adaptation using PEFT techniques.
- Deployment of a real-time prediction system.

---

## Author

**Mayank Narwal**

M.Tech (Data Analytics)  
National Institute of Technology Tiruchirappalli (NIT Trichy)

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{narwal2026hindi,
  author = {Mayank Narwal},
  title = {Hindi Next Word Prediction Using Large Language Models with LoRA and QLoRA Fine-Tuning},
  school = {National Institute of Technology Tiruchirappalli},
  year = {2026}
}
```

---

## License

This project is intended for academic and research purposes.
