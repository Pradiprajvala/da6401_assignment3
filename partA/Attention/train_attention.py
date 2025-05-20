import wandb
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TransliterationDataset, build_vocab, collate_fn, load_pairs
from model_attention import Encoder, Decoder, Seq2Seq, Attention


def generate_run_name_attention(config):
    """Create a descriptive run name for wandb."""
    return (
        f"attention_bs{config.get('batch_size')}_lr{config.get('learning_rate'):.0e}_"
        f"emb{config.get('emb_size')}_hid{config.get('hidden_dim')}_"
        f"layers{config.get('n_layers')}_{config.get('cell_type')}_"
    )


def train(config=None):    
    
    config = wandb.config

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
    
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Rebuild model with sweep config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        input_dim=len(input_vocab),
        emb_dim=config.emb_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        cell_type=config.cell_type
    )
    attention = Attention(
        enc_hidden_dim=config.hidden_dim,
        dec_hidden_dim=config.hidden_dim
    )
    decoder = Decoder(
        output_dim=len(output_vocab),
        emb_dim=config.emb_dim,
        enc_hidden_dim=config.hidden_dim,
        dec_hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        cell_type=config.cell_type,
        attention=attention
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    best_val_acc = 0
    for epoch in range(10):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for src, trg, _, _ in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output, _ = model(src, trg, teacher_forcing_ratio=config.teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (output.argmax(1) == trg).sum().item()
            train_total += trg.ne(0).sum().item()

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for src, trg, _, _ in val_loader:
                src, trg = src.to(device), trg.to(device)
                output, _ = model(src, trg, teacher_forcing_ratio=0)  # no teacher forcing

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)
                val_loss += loss.item()
                val_correct += (output.argmax(1) == trg).sum().item()
                val_total += trg.ne(0).sum().item()

        val_acc = val_correct / val_total
        print("Epoch", epoch+1,"Train acc", train_acc, "Val Acc", val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            best_config = dict(config)
            with open('best_config.json', 'w') as f:
                json.dump(best_config, f, indent=4)
            wandb.save('best_model.pth')
            wandb.save('best_config.json')
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc
        })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seq2Seq Model Hyperparameter Configuration")

    parser.add_argument('--embed_size', type=int, choices=[16, 32, 64, 256], required=True,
                        help="Size of the embedding vector.")
    parser.add_argument('--hidden_size', type=int, choices=[16, 32, 64, 256], required=True,
                        help="Size of the hidden layer.")
    parser.add_argument('--num_layers_encoder', type=int, choices=[1, 2, 3], required=True,
                        help="Number of layers in the encoder.")
    parser.add_argument('--num_layers_decoder', type=int, choices=[1, 2, 3], required=True,
                        help="Number of layers in the decoder.")
    parser.add_argument('--cell_type', type=str, choices=['RNN', 'GRU', 'LSTM'], required=True,
                        help="Type of RNN cell to use.")
    parser.add_argument('--dropout', type=float, choices=[0.2, 0.3], required=True,
                        help="Dropout rate.")
    parser.add_argument('--lr', type=float, required=True,
                        help="Learning rate (must be between 0.0001 and 0.01).")
    parser.add_argument('--batch_size', type=int, choices=[32, 64], required=True,
                        help="Training batch size.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs. Default is 10.")
    parser.add_argument('--beam_width', type=int, choices=[1, 3, 5, 10], required=True,
                        help="Beam search width during decoding.")

    args = parser.parse_args()
    config = {
        'emb_dim': args.embed_size,
        'hidden_dim': args.hidden_size,
        'n_layers': args.num_layers_encoder,
        'cell_type': args.cell_type,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'beam_width': args.beam_width
    }


    train(config=config)
