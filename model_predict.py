"""Provides functions to pre-process the testing data and output the top k most likely predictions from a model."""

from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

from data_proc import MovieDataset


def get_processed_data(test_df: pd.DataFrame, batch_size: int, max_len: int) -> DataLoader[Dict[str, torch.Tensor]]:
    """Transform data into custom dataset and data loader."""
    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')  # faster tokenizer
    # Load data
    test_set = MovieDataset(test_df, tokenizer, max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return test_loader


def predict_topk(data_loader: DataLoader[Dict[str, torch.Tensor]], model: nn.Module, classes_list: list[str], k: int,
                 device: torch.device) -> Dict[int, Dict[int, str]]:
    """Predict top k most likely labels."""
    final_preds: Dict[int, Dict[int, str]] = {}
    model.eval()
    for batch in tqdm(data_loader):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        outputs = model(ids, mask)
        topk_preds = torch.topk(torch.sigmoid(outputs), k)
        for i, movie_id in enumerate(batch['movie_id']):
            final_preds[movie_id.item()] = {}
            for j in range(k):
                final_preds[movie_id.item()][j] = classes_list[int(topk_preds.indices[i][j].item())]
    return final_preds
