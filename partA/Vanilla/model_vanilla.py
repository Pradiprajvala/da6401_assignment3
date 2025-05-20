import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, cell_type, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_class(embed_size, hidden_size, num_layers, 
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, cell_type, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_class(embed_size, hidden_size, num_layers, 
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden):
        # input_token: (batch,) -> (batch, 1)
        embedded = self.embedding(input_token.unsqueeze(1))  # (batch, 1, embed)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.squeeze(1))  # (batch, output_size)
        return output, hidden

    def beam_search_decode(self, encoder_outputs, encoder_hidden, start_token_id, end_token_id, beam_width=5, max_len=50):
        """
        Beam search decoding for the decoder.
    
        Args:
            encoder_outputs: outputs from encoder (for attention, if used)
            encoder_hidden: encoder final hidden state
            beam_width: beam size
            max_len: maximum length of decoded sequence
            start_token_id: id of start-of-sequence token
            end_token_id: id of end-of-sequence token
    
        Returns:
            best sequence (list of token IDs)
        """
    
        # Each beam is a tuple (sequence, hidden_state, score)
        beams = [([start_token_id], encoder_hidden, 0.0)]  # score is log prob sum
    
        for _ in range(max_len):
            new_beams = []
            for seq, hidden, score in beams:
                if seq[-1] == end_token_id:
                    # Already ended, keep as is
                    new_beams.append((seq, hidden, score))
                    continue
                
                # Create a batch of size 1 with the current token
                input_token = torch.tensor([seq[-1]], device=hidden[0].device if isinstance(hidden, tuple) else hidden.device)
    
                # Forward one step in decoder
                output, new_hidden = self.forward(input_token, hidden)
    
                # Get log probabilities
                log_probs = torch.log_softmax(output, dim=-1)  # (1, vocab_size)
    
                # Get top k tokens and scores
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
    
                for i in range(beam_width):
                    next_token = top_indices[0, i].item()
                    next_score = score + top_log_probs[0, i].item()
                    new_seq = seq + [next_token]
                    new_beams.append((new_seq, new_hidden, next_score))
    
            # Keep top beam_width sequences
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
    
            # If all beams ended with EOS token, stop early
            if all(seq[-1] == end_token_id for seq, _, _ in beams):
                break
    
        # Return the sequence with the highest score
        best_seq = beams[0][0]
        return best_seq

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type

    def _convert_encoder_hidden_to_decoder_hidden(self, encoder_hidden):
        """
        Convert encoder hidden state to the format expected by decoder
        Handles different number of layers between encoder and decoder
        """
        # For LSTM, encoder_hidden is a tuple (hidden_state, cell_state)
        if self.cell_type == 'LSTM':
            h, c = encoder_hidden
            
            # If encoder and decoder have different number of layers
            encoder_layers = h.size(0)
            decoder_layers = self.decoder.num_layers
            
            if encoder_layers != decoder_layers:
                # Use only the last layers of encoder if it has more layers
                if encoder_layers > decoder_layers:
                    h = h[-decoder_layers:]
                    c = c[-decoder_layers:]
                # Repeat the last layer of encoder if decoder has more layers
                else:
                    h_extra = h[-1:].repeat(decoder_layers - encoder_layers, 1, 1)
                    c_extra = c[-1:].repeat(decoder_layers - encoder_layers, 1, 1)
                    h = torch.cat([h, h_extra], dim=0)
                    c = torch.cat([c, c_extra], dim=0)
            
            return (h, c)
        
        # For RNN and GRU, encoder_hidden is just the hidden state
        else:
            h = encoder_hidden
            
            # If encoder and decoder have different number of layers
            encoder_layers = h.size(0)
            decoder_layers = self.decoder.num_layers
            
            if encoder_layers != decoder_layers:
                # Use only the last layers of encoder if it has more layers
                if encoder_layers > decoder_layers:
                    h = h[-decoder_layers:]
                # Repeat the last layer of encoder if decoder has more layers
                else:
                    h_extra = h[-1:].repeat(decoder_layers - encoder_layers, 1, 1)
                    h = torch.cat([h, h_extra], dim=0)
            
            return h

    def forward(self, src, src_lens, tgt=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len)
            src_lens: list of original lengths before padding
            tgt: (batch, tgt_len)
        Returns:
            outputs: (batch, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        device = src.device
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(device)

        # Get encoder outputs and hidden state
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        
        # Convert encoder hidden state to the format expected by decoder
        hidden = self._convert_encoder_hidden_to_decoder_hidden(encoder_hidden)

        input_token = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    def beam_search_decode(self, src, src_lens, start_token_id, end_token_id, beam_width=5, max_len=50):
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        
        # Convert encoder hidden state to the format expected by decoder
        decoder_hidden = self._convert_encoder_hidden_to_decoder_hidden(encoder_hidden)
    
        # Use decoder beam search
        return self.decoder.beam_search_decode(
            encoder_outputs, decoder_hidden,
            beam_width=beam_width, max_len=max_len,
            start_token_id=start_token_id, end_token_id=end_token_id
        )