import torch
from tqdm import tqdm

def accuracy(preds, targets, pad_idx=0):
    pred_tokens = preds.argmax(dim=-1)
    # print("Prediction", pred_tokens)
    correct = ((pred_tokens == targets) & (targets != pad_idx)).sum().item()
    total = (targets != pad_idx).sum().item()
    return correct / total

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0, 0

    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_lens, tgt)
        loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        acc = accuracy(output[:, 1:], tgt[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, start_token_id, end_token_id, beam_width=None, max_len=50):
    model.eval()
    total_loss, total_acc = 0, 0

    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Evaluating"):
        src, tgt = src.to(device), tgt.to(device)

        batch_size = src.size(0)

        if beam_width is not None and batch_size == 1:
            pred_seq = model.beam_search_decode(src, src_lens, start_token_id, end_token_id, beam_width=beam_width, max_len=max_len)
            pred_tensor = torch.tensor(pred_seq, device=device).unsqueeze(0)  # (1, seq_len)
            tgt_tensor = tgt[:, 1:]


            acc = (pred_tensor[:, :tgt_tensor.size(1)] == tgt_tensor[:, :pred_tensor.size(1)]).float().mean().item()
            loss = torch.tensor(0.0)  

            total_loss += loss.item()
            total_acc += acc

        else:
            output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            acc = accuracy(output[:, 1:], tgt[:, 1:])
            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)