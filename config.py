"""
Configuration file for the Movie Genre Classification project.

This file contains hyperparameters, file paths, and constants
used across different modules. Adjust these parameters to
experiment with model performance or adapt to different environments.
"""

DEVICE = 'mps'

# File paths
GENRES_LABELS_FILE = 'genres_labels.json'
MAX_TOKEN_LEN_FILE = 'max_token_len.json'
FINAL_PREDS_FILE = 'final_preds_dict.json'
SAVED_MODEL_FILE = 'saved_model_state.pt'

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
THRESH = 0.6
DROPOUT_RATE = 0.3
TRAIN_SIZE = 0.8