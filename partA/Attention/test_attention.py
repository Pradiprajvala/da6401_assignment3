import torch
import json
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from dataset import TransliterationDataset, build_vocab, collate_fn, load_pairs
from model_attention import Encoder, Decoder, Attention, Seq2Seq
import torch.nn as nn   

# Reconstruct vocabularies
def decode_tensor(tensor, vocab):
    idx_to_char = {v: k for k, v in vocab.items()}
    return ''.join([
        idx_to_char[idx.item()]
        for idx in tensor
        if idx.item() in idx_to_char and idx_to_char[idx.item()] not in ['<pad>', '<sos>', '<eos>']
    ])

def plot_attention(ax, attn_matrix, input_tokens, output_tokens):
    sns.heatmap(attn_matrix, xticklabels=input_tokens, yticklabels=output_tokens,
                ax=ax, cmap="viridis", cbar=False)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.tick_params(axis='x', rotation=90)

def test_attention_model():
    # Load saved best configuration
    run = wandb.init()
    print("Wandb initialization successful.")
    
    ENTITY = "da24m012-iit-madras"
    PROJECT = "dakshina-transliteration"
    SWEEP_ID = "ml2f2xwn"
    api = wandb.Api()
    
    # Get all runs in the project
    all_runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    # Filter runs that belong to the desired sweep
    sweep_runs = [run for run in all_runs if run.sweep and run.sweep.id == SWEEP_ID]
    best_run = max(sweep_runs, key=lambda run: run.summary.get("val_acc", 0.0))
    
    # Show details of the best run
    print(f"Best run ID: {best_run.id}")
    print(f"Name: {best_run.name}")
    print(f"Best Validation Accuracy: {best_run.summary['val_acc']}")
    print("Config:", dict(best_run.config))
    
    best_config = dict(best_run.config)
    best_run.file("best_model.pth").download(replace=True)
    model_path = "/kaggle/working/best_model.pth"
 
    
    train_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.train.tsv")
    val_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.dev.tsv")
    test_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.test.tsv")
    
    # === Build vocab from training ===
    input_vocab, output_vocab = build_vocab(train_pairs)
    pad_idx = output_vocab['<pad>']
    SOS_token_idx = output_vocab['<sos>']
    EOS_token_idx = output_vocab['<eos>']
            
    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    val_dataset = TransliterationDataset(val_pairs, input_vocab, output_vocab)
    test_dataset = TransliterationDataset(test_pairs, input_vocab, output_vocab)
    
    
    # Rebuild the model using best_config
    encoder = Encoder(
        input_dim=len(input_vocab),
        emb_dim=best_config['emb_dim'],
        hidden_dim=best_config['hidden_dim'],
        n_layers=best_config['n_layers'],
        cell_type=best_config['cell_type']
    )
    
    attention = Attention(
        enc_hidden_dim=best_config['hidden_dim'],
        dec_hidden_dim=best_config['hidden_dim']
    )
    
    decoder = Decoder(
        output_dim=len(output_vocab),
        emb_dim=best_config['emb_dim'],
        enc_hidden_dim=best_config['hidden_dim'],
        dec_hidden_dim=best_config['hidden_dim'],
        n_layers=best_config['n_layers'],
        cell_type=best_config['cell_type'],
        attention=attention
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare decoding dictionary
    idx_to_input = {v: k for k, v in input_vocab.items()}
    idx_to_output = {v: k for k, v in output_vocab.items()}
    pad_idx = output_vocab['<pad>']
    sos_idx = output_vocab['<sos>']
    eos_idx = output_vocab['<eos>']
    
    
    # Decode tensor to string
    def decode_output(tensor):
        result = []
        for idx in tensor:
            idx = idx.item()
            if idx == eos_idx:
                break
            if idx not in [pad_idx, sos_idx]:
                result.append(idx_to_output.get(idx, ''))
        return ''.join(result)
    
    # Prepare DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Validation
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for src, trg, _, _ in test_loader:
            src, trg = src.to(device), trg.to(device)
            output, _ = model(src, trg, teacher_forcing_ratio=0)  # no teacher forcing

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            test_loss += loss.item()
            test_correct += (output.argmax(1) == trg).sum().item()
            test_total += trg.ne(0).sum().item()

    # test_acc = test_correct / test_total
    # print("Test Acc", test_acc, "Test Loss", test_loss)
    # wandb.log({
    #     'test_acc': test_acc,
    #     'test_loss': test_loss
    # })
    
    
    print("Sample Predictions:\n")
    num_samples = 5
    count = 0
    with open('predictions.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Input (Latin)', 'Target (Devanagari)', 'Predicted (Devanagari)'])
        with torch.no_grad():
            for src, trg, _, _ in test_loader:
                src, trg = src.to(device), trg.to(device)
        
                # Inference without teacher forcing
                output, _ = model(src, trg, teacher_forcing_ratio=0)
        
                pred_ids = output.argmax(2).squeeze(0)  # (seq_len)
                true_ids = trg.squeeze(0)
        
                # Decode input
                src_text = ''.join(
                    [k for i in src.squeeze(0).tolist() for k, v in input_vocab.items() if v == i and k != '<pad>']
                )
        
                pred_text = decode_output(pred_ids)
                true_text = decode_output(true_ids)

                writer.writerow([src_text, true_text, pred_text])
        
                count += 1
                if count <= num_samples:
                    print(f"Input     : {src_text}")
                    print(f"Target    : {true_text}")
                    print(f"Predicted : {pred_text}")
                    print("-" * 30)

        
    # Prepare 3x3 plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    count = 0

    
    
    with torch.no_grad():
        for src, trg, _, _ in test_loader:
            src, trg = src.to(device), trg.to(device)
    
            # Forward pass with attention returned
            output, attn_weights = model(src, trg, teacher_forcing_ratio=0)
    
            # Decode output
            pred_ids = output.argmax(2).squeeze(0)
            attn_matrix = attn_weights.squeeze(0).cpu().numpy()  # Shape: (tgt_len, src_len)
    
            # Tokens
            input_tokens = [idx_to_input[idx.item()] for idx in src.squeeze(0) if idx.item() != input_vocab['<pad>']]
            output_tokens = [
                idx_to_output[idx.item()]
                for idx in pred_ids
                if idx.item() not in [output_vocab['<pad>'], output_vocab['<sos>'], output_vocab['<eos>']]
            ]
    
            # Crop attention matrix
            attn_matrix = attn_matrix[:len(output_tokens), :len(input_tokens)]
    
            # Plot
            plot_attention(axes[count], attn_matrix, input_tokens, output_tokens)
            axes[count].set_title(f"Input: {''.join(input_tokens)}")
    
            count += 1
            if count == 9:
                break
    
    plt.tight_layout()
    plt.savefig("attention_grid.png")
    wandb.log({"attention_grid": wandb.Image("attention_grid.png")})
    plt.show()
    
if __name__ == "__main__":
    test_attention_model()