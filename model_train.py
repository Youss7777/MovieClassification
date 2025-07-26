"""
Module for training a custom DistilBERT model for movie genre classification.

This module provides functions for data preprocessing, model training, and evaluation.
The DistilBERT model is fine-tuned with additional neural network layers for multi-label
classification of movie genres.
"""

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizerFast

from data_proc import MovieDataset, get_max_tokens


class DistilBertClass(nn.Module):
    """Custom model: DistilBert with added NN layers for multi-label classification."""

    def __init__(self, num_classes: int, dropout_rate: float):
        super(DistilBertClass, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bert.config.hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DistilBertClass model with additional classification layers."""
        x_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x_1.last_hidden_state
        # use vector of first token (cls) to do classification
        x = x[:, 0]
        x = self.dropout(x)
        output: torch.Tensor = self.classifier(x)
        return output


def train_epoch(
        train_loader: DataLoader[Dict[str, torch.Tensor]],
        model: nn.Module,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        threshold: float,
        device: torch.device) -> None:
    """Training loop for one epoch."""
    total_loss = 0.0
    preds_vec = []
    targets_vec = []
    model.train()
    for batch in tqdm(train_loader):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > threshold).int()
        preds_vec.extend(preds.cpu().numpy().flatten())
        targets_vec.extend(targets.cpu().numpy().flatten())
        loss.backward()     # type: ignore
        optimizer.step()
        optimizer.zero_grad()
    # Evaluation metrics
    avg_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(targets_vec, preds_vec)
    train_prec = precision_score(targets_vec, preds_vec, average='macro')
    train_recall = recall_score(targets_vec, preds_vec, average='macro')
    train_f1_macro = f1_score(targets_vec, preds_vec, average='macro')
    print(f'Training loss: {avg_loss}')
    print(f'Training accuracy: {train_acc * 100:.2f} %')
    print(f'Training precision: {train_prec * 100:.2f} %')
    print(f'Training recall: {train_recall * 100:.2f} %')
    print(f'Training f1 score: {train_f1_macro * 100:.2f} %')


def evaluate(data_loader: DataLoader[Dict[str, torch.Tensor]], model: nn.Module,
             loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             threshold: float, device: torch.device) -> None:
    """Evaluate model on given data_loader."""
    model.eval()
    total_loss = 0.0
    preds_vec = []
    targets_vec = []
    for batch in data_loader:
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > threshold).int()
        preds_vec.extend(preds.cpu().numpy().flatten())
        targets_vec.extend(targets.cpu().numpy().flatten())
    # Evaluation metrics
    avg_loss = total_loss / len(data_loader)
    eval_acc = accuracy_score(targets_vec, preds_vec)
    eval_prec = precision_score(targets_vec, preds_vec, average='macro')
    eval_recall = recall_score(targets_vec, preds_vec, average='macro')
    eval_f1_macro = f1_score(targets_vec, preds_vec, average='macro')
    print(f'Validation loss: {avg_loss}')
    print(f'Validation accuracy: {eval_acc * 100:.2f} %')
    print(f'Validation precision: {eval_prec * 100:.2f} %')
    print(f'Validation recall: {eval_recall * 100:.2f} %')
    print(f'Validation f1 score: {eval_f1_macro * 100:.2f} % \n')


def train(train_loader: DataLoader[Dict[str, torch.Tensor]],
          val_loader: DataLoader[Dict[str, torch.Tensor]], num_epochs: int, model: nn.Module,
          loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optimizer: torch.optim.Optimizer,
          threshold: float, device: torch.device) -> None:
    """Train model for specified number of epochs."""
    for epoch in range(num_epochs):
        print(f'epoch {epoch + 1} / {num_epochs} ...')
        train_epoch(train_loader, model, loss_function, optimizer, threshold, device)
        evaluate(val_loader, model, loss_function, threshold, device)


def compute_class_weights(train_labels: pd.DataFrame, num_labels: int, device: torch.device) -> torch.Tensor:
    """Compute positive class weights for multi-label classification with imbalanced dataset."""
    class_weights = np.empty(num_labels)
    for i in range(num_labels):
        class_weights[i] = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels[:, i])[1]
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def get_processed_data(data_df: pd.DataFrame, batch_size: int, train_size: float, device: torch.device) \
        -> Tuple[
            DataLoader[Dict[str, torch.Tensor]], DataLoader[Dict[str, torch.Tensor]], list[str], torch.Tensor, int]:
    """Transform data into custom datasets and data loaders and genres into binary vectors."""
    # Encode labels
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(data_df['genres'].str.split())
    # Extract labels
    genres_labels = list(mlb.classes_)
    # Create validation set
    train_data = data_df.sample(frac=train_size, random_state=42)
    val_data = data_df.drop(train_data.index)
    # Encode labels for each set
    train_encoded_genres = mlb.transform(train_data['genres'].str.split())
    val_encoded_genres = mlb.transform(val_data['genres'].str.split())
    # Compute class weights
    class_weights = compute_class_weights(encoded_genres, len(genres_labels), device)
    # Get maximum tokens length
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')  # faster tokenizer
    max_token_len = get_max_tokens(tokenizer, data_df)
    # Load processed data
    train_set = MovieDataset(train_data, tokenizer, max_token_len, genres=train_encoded_genres)
    val_set = MovieDataset(val_data, tokenizer, max_token_len, genres=val_encoded_genres)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, genres_labels, class_weights, max_token_len

# def train_plot(train_losses: list[float], val_losses: list[float], train_accs: list[float], val_accs: list[float],
#                train_f1s: list[float], val_f1s: list[float], num_epochs: int) -> None:
#     """Plot training loss, accuracy and F1 score curves."""
#     epochs = range(1, num_epochs + 1)
#     plt.figure()
#
#     plt.subplot(3, 1, 1)
#     plt.plot(epochs, train_losses, label='Training')
#     plt.plot(epochs, val_losses, label='Validation')
#     plt.xlabel('epochs')
#     plt.xticks(epochs)
#     plt.legend()
#     plt.title('Losses')
#
#     plt.subplot(3, 1, 2)
#     plt.plot(epochs, train_accs, label='Training')
#     plt.plot(epochs, val_accs, label='Validation')
#     plt.xlabel('epochs')
#     plt.xticks(epochs)
#     plt.legend()
#     plt.title('Accuracy')
#
#     plt.subplot(3, 1, 3)
#     plt.plot(epochs, train_f1s, label='Training')
#     plt.plot(epochs, val_f1s, label='Validation')
#     plt.xlabel('epochs')
#     plt.xticks(epochs)
#     plt.legend()
#     plt.title('F1 score')
#
#     plt.tight_layout()
#     plt.show()
