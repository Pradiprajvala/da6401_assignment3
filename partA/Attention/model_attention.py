import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, cell_type='gru'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.cell_type = cell_type.lower()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        rnn_class = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.cell_type]
        self.rnn = rnn_class(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, src_len, hidden)
        src_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, cell_type='gru', attention=None):
        super().__init__()
        self.output_dim = output_dim
        self.cell_type = cell_type.lower()
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_class = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.cell_type]
        self.rnn = rnn_class(enc_hidden_dim + emb_dim, dec_hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + emb_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)  # (batch, 1, emb_dim)

        if isinstance(hidden, tuple):  # LSTM
            hidden_cat = hidden[0][-1]
        else:
            hidden_cat = hidden[-1]

        attn_weights = self.attention(hidden_cat, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1] if trg is not None else 30  # Max decoding steps if no trg
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        if isinstance(hidden, tuple):  # LSTM
            dec_hidden = (hidden[0], hidden[1])
        else:
            dec_hidden = hidden

        input = trg[:, 0] if trg is not None else torch.full((batch_size,), output_vocab['<sos>'], dtype=torch.long).to(self.device)

        for t in range(1, trg_len):
            output, dec_hidden, attn_weights = self.decoder(input, dec_hidden, encoder_outputs)
            outputs[:, t] = output
            attentions[:, t] = attn_weights.squeeze(1)

            if trg is not None:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input = trg[:, t] if teacher_force else output.argmax(1)
            else:
                input = output.argmax(1)

        return outputs, attentions
