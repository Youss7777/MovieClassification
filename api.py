"""API endpoints."""

from io import BytesIO
from typing import Dict

import pandas as pd
import torch
from fastapi.applications import FastAPI
from fastapi.param_functions import File

import model_predict as mp
import model_train as mt
import utils as utils
from config import (BATCH_SIZE, DEVICE, DROPOUT_RATE, FINAL_PREDS_FILE,
                    GENRES_LABELS_FILE, LEARNING_RATE, MAX_TOKEN_LEN_FILE,
                    NUM_EPOCHS, SAVED_MODEL_FILE, THRESH, TRAIN_SIZE)

app = FastAPI()
device = torch.device(DEVICE)


@app.post("/genres/train")
def train(file: bytes = File(...)) -> None:
    """Train a predictive model to rank movie genres based on their synopsis."""
    # Load training data
    data_df = pd.read_csv(BytesIO(file))
    # Load pre-processed data
    train_loader, val_loader, genres_labels, class_weights, max_token_len = mt.get_processed_data(data_df, BATCH_SIZE,
                                                                                                  TRAIN_SIZE, device)
    # Instantiate model
    model = mt.DistilBertClass(num_classes=len(genres_labels), dropout_rate=DROPOUT_RATE)
    model.to(device=device)
    # Loss function & optimizer
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # Train model
    mt.train(train_loader, val_loader, NUM_EPOCHS, model, loss_function, optimizer, THRESH, device)
    # Save model & metadata
    utils.save_model(model, SAVED_MODEL_FILE)
    utils.save_to_json(genres_labels, GENRES_LABELS_FILE)
    utils.save_to_json(max_token_len, MAX_TOKEN_LEN_FILE)


@app.post("/genres/predict")
def predict(file: bytes = File(...)) -> Dict[int, Dict[int, str]]:
    """Predict the top 5 movie genres based on their synopsis."""
    # Load labels and max tokenization length
    genres_labels = utils.load_from_json(GENRES_LABELS_FILE)
    max_len = utils.load_from_json(MAX_TOKEN_LEN_FILE)
    # Load test data
    test_df = pd.read_csv(BytesIO(file))
    # Load pre-processed test data
    test_loader = mp.get_processed_data(test_df, BATCH_SIZE, max_len)
    # Instantiate & load model
    model = mt.DistilBertClass(num_classes=len(genres_labels), dropout_rate=DROPOUT_RATE)
    model.to(device=device)
    utils.load_model(model, SAVED_MODEL_FILE)
    # Predictions (top k=5)
    final_preds = mp.predict_topk(test_loader, model, genres_labels, 5, device)
    # Save output dictionary
    utils.save_to_json(final_preds, FINAL_PREDS_FILE)

    return final_preds
