from train_utilities_vanilla import train, evaluate
from dataset import load_pairs, build_vocab, TransliterationDataset, collate_fn
import torch
from model_vanilla import Encoder, Decoder, Seq2Seq

# wandb.init(project="transliteration-seq2seq", config={
#     "embed_size": 64,
#     "hidden_size": 128,
#     "num_layers": 1,
#     "cell_type": "LSTM",
#     "batch_size": 32,
#     "epochs": 10,
#     "lr": 0.001,
#     "dropout": 0.2
# })
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

def train(config):
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
    
        # wandb.log({
        #     "train_loss": train_loss,
        #     "train_acc": train_acc,
        #     "val_loss": val_loss,
        #     "val_acc": val_acc,
        #     "epoch": epoch + 1
        # })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            best_config = dict(config)
            with open('best_config.json', 'w') as f:
                json.dump(best_config, f, indent=4)
            # wandb.save('best_model.pth')
            # wandb.save('best_config.json')
    
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
   
if __name__ == "__main__":
    # Initialize wandb
    # arg parser

    import argparse
    parser = argparse.ArgumentParser(description="Seq2Seq Hyperparameter Configuration")
    parser.add_argument('--embed_size', type=int, choices=[16, 32, 64, 256], required=True)
    parser.add_argument('--hidden_size', type=int, choices=[16, 32, 64, 256], required=True)
    parser.add_argument('--num_layers_encoder', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--num_layers_decoder', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--cell_type', type=str, choices=['RNN', 'GRU', 'LSTM'], required=True)
    parser.add_argument('--dropout', type=float, choices=[0.2, 0.3], required=True)
    parser.add_argument('--lr', type=float, required=True, help="Learning rate between 0.0001 and 0.01")
    parser.add_argument('--batch_size', type=int, choices=[32, 64], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--beam_width', type=int, choices=[1, 3, 5, 10], required=True)
    args = parser.parse_args()

    config = {
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'num_layers_encoder': args.num_layers_encoder,
        'num_layers_decoder': args.num_layers_decoder,
        'cell_type': args.cell_type,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'beam_width': args.beam_width
    }

    train(config)

    
