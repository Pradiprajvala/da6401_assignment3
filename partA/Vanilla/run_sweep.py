from train import train, evaluate
from dataset import load_pairs, build_vocab, TransliterationDataset, collate_fn
import torch
from model import Encoder, Decoder, Seq2Seq
import wandb
import json
def generate_run_name(config):
    """Create a descriptive run name for wandb."""
    return (
        f"bs{config.get('batch_size')}_lr{config.get('lr'):.0e}_"
        f"emb{config.get('embed_size')}_hid{config.get('hidden_size')}_"
        f"layers{config.get('num_layers')}_{config.get('cell_type')}_"
        f"drop{int(config.get('dropout') * 100)}_beam{config.get('beam_width')}_"
        f"ep{config.get('epochs')}"
    )

def run_sweep():
    run = wandb.init()
    config = wandb.config
    run.name = generate_run_name(config)
    # === Load Data ===
    train_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.train.tsv")
    val_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.dev.tsv")
    test_pairs = load_pairs("/kaggle/input/dakshina/hi/lexicons/hi.translit.sampled.test.tsv")
    
    # === Build vocab from training ===
    input_vocab, output_vocab = build_vocab(train_pairs)
    pad_idx = output_vocab['<pad>']
    SOS_token_idx = output_vocab['<sos>']
    EOS_token_idx = output_vocab['<eos>']
    
    # === Datasets ===
    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    val_dataset = TransliterationDataset(val_pairs, input_vocab, output_vocab)
    test_dataset = TransliterationDataset(test_pairs, input_vocab, output_vocab)
    
    # === Dataloaders ===
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # === Model, Optimizer, Loss ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(len(input_vocab), config.embed_size, config.hidden_size, config.num_layers_encoder, config.cell_type, config.dropout)
    decoder = Decoder(len(output_vocab), config.embed_size, config.hidden_size, config.num_layers_decoder, config.cell_type, config.dropout)
    model = Seq2Seq(encoder, decoder, config.cell_type).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_acc = 0.0
    best_config = None
    # === Train and validate ===
    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, SOS_token_idx, EOS_token_idx, beam_width=config.beam_width)
    
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch + 1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            best_config = dict(config)
            with open('best_config.json', 'w') as f:
                json.dump(best_config, f, indent=4)
            wandb.save('best_model.pth')
            wandb.save('best_config.json')
    
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # # === Final Evaluation on Test Set ===
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device, beam_width=config.beam_width)
    # wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    # print(f"\n[Test Evaluation] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'embed_size': {'values': [16, 32, 64, 256]},
            'hidden_size': {'values': [16, 32, 64, 256]},
            'num_layers_encoder': {'values': [1, 2, 3]},
            'num_layers_decoder': {'values': [1, 2, 3]},
            'cell_type': {'values': ['RNN', 'GRU', 'LSTM']},
            'dropout': {'values': [0.2, 0.3]},
            'lr': {'min': 0.0001, 'max': 0.01},
            'batch_size': {'values': [32, 64]},
            'epochs': {'value': 10},
            'beam_width': {'values': [1, 3, 5, 10]}  # added beam_width here
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="dakshina-transliteration")
    wandb.agent(sweep_id, function=run_sweep)  # Ensure we finish the run properly
    wandb.finish()  # Finalize the wandb run