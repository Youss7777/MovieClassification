# 🎬 Movie Genre Classifier

This project implements a model capable of predicting the most probable genres of a movie based on its synopsis. The task is a **multi-label text classification problem**, as each movie can belong to multiple genres simultaneously.

---

## Overview

- **Input**: Movie synopsis (text)
- **Output**: Top *K* most probable genres (multi-label)
- **Model**: Fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) for text classification
- **Technique**: Transfer learning with class weighting
- **Serving**: REST API via FastAPI

---

## Key Features

- **Transformer-based model**: DistilBERT is fine-tuned on a custom dataset of movie synopses.
- **Multi-label support**: Movies can have multiple genres predicted.
- **Imbalanced data handling**: Uses weighted loss function to give higher importance to rare genres.
- **API endpoints**: Exposes an API with two endpoints to train the model and use it to make and return predictions.
