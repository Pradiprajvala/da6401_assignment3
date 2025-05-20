# Sequence-to-Sequence Model Implementation

This repository contains the implementation of sequence-to-sequence (Seq2Seq) models for text generation, including both vanilla Seq2Seq and attention-enhanced variants.

## Project Structure

```
├── partA
│   ├── Attention
│   │   ├── model_attention.py       # Attention-based Seq2Seq model implementation
│   │   ├── test_attention.py        # Testing script for attention model
│   │   └── train_attention.py       # Training script for attention model
│   ├── predictions_attention
│   │   └── predictions.csv          # Model predictions with attention mechanism
│   ├── predictions_vanilla
│   │   ├── model_config (1).txt     # Configuration details for vanilla model
│   │   ├── prediction_analysis (2).png  # Visualization of predictions
│   │   ├── prediction_report (1).html   # HTML report of model performance
│   │   └── test_predictions (3).csv    # Test predictions from vanilla model
│   └── Vanilla
│       ├── model_vanilla.py         # Vanilla Seq2Seq model implementation
│       ├── run_sweep.py             # Hyperparameter sweep utility
│       ├── test_vanilla.py          # Testing script for vanilla model
│       ├── train_utilities_vanilla.py  # Utility functions for training
│       ├── train_vanilla.py         # Training script for vanilla model
│       └── dataset.py               # Dataset handling and preprocessing
```

## Models

This project implements two types of sequence-to-sequence models:

1. **Vanilla Seq2Seq**: A standard encoder-decoder architecture without attention
2. **Attention-based Seq2Seq**: An enhanced model using attention mechanisms to improve performance

## Usage

### Training

#### Vanilla Seq2Seq Model

```bash
python partA/Vanilla/train_vanilla.py \
  --embed_size 64 \
  --hidden_size 128 \
  --num_layers_encoder 2 \
  --num_layers_decoder 2 \
  --cell_type LSTM \
  --dropout 0.2 \
  --lr 0.001 \
  --batch_size 64 \
  --epochs 10 \
  --beam_width 3
```

#### Attention-based Seq2Seq Model

```bash
python partA/Attention/train_attention.py \
  --embed_size 64 \
  --hidden_size 128 \
  --num_layers_encoder 2 \
  --num_layers_decoder 2 \
  --cell_type LSTM \
  --dropout 0.2 \
  --lr 0.001 \
  --batch_size 64 \
  --epochs 10 \
  --beam_width 3
```

### Testing

#### Vanilla Seq2Seq Model

```bash
python partA/Vanilla/test_vanilla.py --model_path path/to/saved/model --num_samples 10
```

#### Attention-based Seq2Seq Model

```bash
python partA/Attention/test_attention.py --model_path path/to/saved/model --num_samples 10
```

## Hyperparameter Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| embed_size | 16, 32, 64, 256 | Size of the embedding vector |
| hidden_size | 16, 32, 64, 256 | Size of the hidden layer |
| num_layers_encoder | 1, 2, 3 | Number of layers in the encoder |
| num_layers_decoder | 1, 2, 3 | Number of layers in the decoder |
| cell_type | RNN, GRU, LSTM | Type of RNN cell to use |
| dropout | 0.2, 0.3 | Dropout rate |
| lr | Any value between 0.0001 and 0.01 | Learning rate |
| batch_size | 32, 64 | Training batch size |
| epochs | Any integer (default: 10) | Number of training epochs |
| beam_width | 1, 3, 5, 10 | Beam search width during decoding |

## Hyperparameter Sweep

The project includes a hyperparameter sweep utility:

```bash
python partA/Vanilla/run_sweep.py
```

## Output Analysis

Model predictions and performance metrics are saved in the following directories:
- `predictions_vanilla/`: Predictions and analysis for the vanilla model
- `predictions_attention/`: Predictions for the attention-based model

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib (for visualizations)
- Pandas (for data handling)
