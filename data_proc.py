"""
Module for data preprocessing for a DistilBERT model.

This module provides a custom dataset class for preprocessing text data and a function
to determine the maximum token length of the input data.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


class MovieDataset(Dataset[Dict[str, torch.Tensor]]):
    """Custom dataset for pre-processing the text data before feeding it to the network."""

    def __init__(self, data_df: pd.DataFrame, tokenizer: DistilBertTokenizerFast,
                 max_len: int, genres: Optional[NDArray[np.int_]] = None) -> None:
        self.data = data_df
        self.genres = genres
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve item (synopsis and associated data) from the dataset at the specified index."""
        movie_id = self.data.iloc[idx]['movie_id']
        synopsis = self.data.iloc[idx]['synopsis']
        inputs = self.tokenizer.encode_plus(text=synopsis, add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=True,
                                            truncation=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        item = {
            'movie_id': torch.tensor(movie_id, dtype=torch.long),
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
        if self.genres is not None:
            targets = self.genres[idx]
            item['targets'] = torch.tensor(targets, dtype=torch.long)
        return item

    def __len__(self) -> int:
        """Return length of dataset (i.e., total number of movie samples)."""
        return len(self.data)


def get_max_tokens(tokenizer: DistilBertTokenizerFast, data_df: pd.DataFrame) -> int:
    """Get the maximum number of tokens of a movie synopsis present in the data."""
    token_df = data_df['synopsis'].apply(lambda x: tokenizer.encode_plus(text=x,
                                                                         add_special_tokens=True,
                                                                         return_token_type_ids=True))
    token_length_df = token_df.apply(lambda x: len(x[0]))

    return int(token_length_df.max())
