import numpy as np
import math
from functions.params import *
import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import sys

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 48):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:x.size(1)] #[batch_size, seq_len, embedding_dim]
        return self.dropout(x)

class Trans(torch.nn.Module):

    def __init__(self):
        super(Trans, self).__init__()

        self.linear_dropout = torch.nn.Dropout(p=LINEAR_DROPOUT)
        self.gru_dropout = torch.nn.Dropout(p=GRU_DROPOUT)

        self.key_embedding = torch.nn.Embedding(KEY_SIZE, EMBED_SIZE // 2)
        self.float_embedding = torch.nn.Linear(3, EMBED_SIZE // 2)
        self.pos_embedding = PositionalEncoding(EMBED_SIZE)
        self.cond_embedding = torch.nn.Linear(MELODY_SIZE, EMBED_SIZE)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=EMBED_SIZE, nhead=ATT_HEADS, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.linear_mu = torch.nn.Linear(EMBED_SIZE, Z_SIZE)
        self.linear_std = torch.nn.Linear(EMBED_SIZE, Z_SIZE)

        self.linear_decode = torch.nn.Linear(Z_SIZE, EMBED_SIZE)

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=EMBED_SIZE, nhead=ATT_HEADS, batch_first=True)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)

        self.linear_tension_out_0 = torch.nn.Linear(EMBED_SIZE, EMBED_SIZE // 2)
        self.linear_distance_out_0 = torch.nn.Linear(EMBED_SIZE, EMBED_SIZE // 2)
        self.linear_strain_out_0 = torch.nn.Linear(EMBED_SIZE, EMBED_SIZE // 2)
        self.linear_int_out_0 = torch.nn.Linear(EMBED_SIZE, EMBED_SIZE // 2)

        self.linear_tension_out = torch.nn.Linear(EMBED_SIZE // 2, 1)
        self.linear_distance_out = torch.nn.Linear(EMBED_SIZE // 2, 1)
        self.linear_strain_out = torch.nn.Linear(EMBED_SIZE // 2, 1)
        self.linear_int_out = torch.nn.Linear(EMBED_SIZE // 2, KEY_SIZE)

    def unsqueeze_vector(self, cond1, inputs=None):
        # inputs: (B, time_len1, 4)
        # cond1: (B, time_len2, 2)
        #print(cond1.size(),cond2.size())
        if inputs != None:
            input_float_part = inputs[..., :3]
            input_int_part = inputs[..., 3].long()
            #print("input_int_part:", input_int_part)

            input_float_part = self.float_embedding(input_float_part)
            input_onehot_part = self.key_embedding(input_int_part)

            new_inputs = torch.cat([input_float_part, input_onehot_part], dim=-1)
            # new_inputs: (B, time_len1, 27)
        else:
            new_inputs = None

        cond1_hot_part = cond1[..., 0].long()
        cond1_onehot_part = F.one_hot(cond1_hot_part, num_classes = PITCH_SIZE).float()

        cond1_int_part = cond1[..., 1].float()
        if inputs != None:
            cond1_int_part = cond1_int_part.unsqueeze(-1)
            SOS = torch.ones(BATCH_SIZE, cond1_int_part.size(1), 1).to(cond1_int_part.device).float()
        else:
            cond1_int_part = cond1_int_part.unsqueeze(-1)
            SOS = torch.ones(cond1_int_part.size(0), 1).to(cond1_int_part.device).float()

        #print(SOS.size(), cond1_onehot_part.size(), cond1_int_part.size())
        new_cond1 = torch.cat([SOS, cond1_onehot_part, SOS, cond1_int_part], dim=-1) #1+128+1+1=131
        new_cond1 = self.cond_embedding(new_cond1)
        # new_cond1: (B, time_len2, 132)
        return new_inputs, new_cond1

    def encoder(self, c1, lens, generate):

        if not generate:
            x_padding_mask = torch.arange(c1.size(1), device=c1.device).expand(BATCH_SIZE, c1.size(1)) >= torch.tensor(lens).to(c1.device).unsqueeze(1)
            x_padding_mask.to(c1.device)

            x_emb = self.pos_embedding(c1)
            memory = self.transformer_encoder(x_emb, src_key_padding_mask=x_padding_mask)
        else:
            x_emb = self.pos_embedding(c1)
            memory = self.transformer_encoder(x_emb)
        return memory

    def decoder(self, src, memory, x, lens, generate):

        if not generate:
            maxlen = x.size(1)
            x = torch.cat([src.mean(dim=1, keepdim=True).to(x.device), x[:, :-1, :]], dim=1)
            x_emb = self.pos_embedding(x + src)
            emb_padding_mask = torch.arange(x_emb.size(1), device=x_emb.device).expand(BATCH_SIZE, x_emb.size(1)) >= torch.tensor(lens).to(x_emb.device).unsqueeze(1)
            emb_padding_mask.to(x_emb.device)
            memory_key_padding_mask = torch.arange(memory.size(1), device=memory.device).expand(BATCH_SIZE, memory.size(1)) >= torch.tensor(lens).to(memory.device).unsqueeze(1)
            memory_key_padding_mask.to(memory.device)

            tgt_mask = torch.triu(torch.ones(maxlen, maxlen) * float('-inf'), diagonal=1).to(x.device)
            #print(x_emb.size(), memory.size())
            output = self.transformer_decoder(
                tgt=x_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=emb_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

            output = self.linear_dropout(output)

            output_t = F.leaky_relu(self.linear_tension_out_0(output))
            output_t = self.linear_dropout(output_t)
            output_d = F.leaky_relu(self.linear_distance_out_0(output))
            output_d = self.linear_dropout(output_d)
            output_s = F.leaky_relu(self.linear_strain_out_0(output))
            output_s = self.linear_dropout(output_s)
            output_i = F.leaky_relu(self.linear_int_out_0(output))
            output_i = self.linear_dropout(output_i)

            output_tension = F.leaky_relu(self.linear_tension_out(output_t))
            output_distance = F.leaky_relu(self.linear_distance_out(output_d))
            output_strain = F.leaky_relu(self.linear_strain_out(output_s))
            output_int = self.linear_int_out(output_i)
            output = torch.cat([output_tension, output_distance, output_strain, output_int], dim=-1)

        else:
            
            output_seq = []
            next_input = src.mean(dim=1, keepdim=True).to(memory.device)
            #print(memory.size())
            #onset_float = torch.tensor([[[1.85472, 0.0, 0.39273]]]).to(memory.device)
            #onset_key = torch.tensor([[5]]).to(memory.device).long()
            #onset = torch.cat([self.float_embedding(onset_float), self.key_embedding(onset_key)], dim=-1)
            #next_input = onset
            #print(onset_float.size(), onset_key.size(), next_input)
            #print(torch.triu(torch.ones(maxlen, maxlen) * float('-inf'), diagonal=1).to(c1.device))
            #tgt_mask = torch.triu(torch.ones(2, 2) * float('-inf'), diagonal=1).to(z.device)
            #print(tgt_mask)
            for i in range(lens):
                x_emb = self.pos_embedding(next_input + src[:, :(i+1), :])
                #print(next_input.size(), c1[:, :2, :].size())
                #tgt_mask = torch.triu(torch.ones(next_input.size(1), next_input.size(1)) * float('-inf'), diagonal=1).to(memory.device)
                #print(tgt_mask)
                decoded = self.transformer_decoder(
                    tgt=x_emb,
                    memory=memory,
                    #tgt_mask=tgt_mask
                )
                decoded = decoded[:, -1, :].unsqueeze(1)
                #print(decoded.size())
                output_t = F.leaky_relu(self.linear_tension_out_0(decoded))
                output_d = F.leaky_relu(self.linear_distance_out_0(decoded))
                output_s = F.leaky_relu(self.linear_strain_out_0(decoded))
                output_i = F.leaky_relu(self.linear_int_out_0(decoded))

                output_tension = F.leaky_relu(self.linear_tension_out(output_t))
                output_distance = F.leaky_relu(self.linear_distance_out(output_d))
                output_strain = F.leaky_relu(self.linear_strain_out(output_s))
                output_i = self.linear_int_out(output_i)
                output_int = torch.argmax(torch.softmax(output_i, dim=-1), dim=-1).unsqueeze(1)

                #print(output_tension.size(), output_int.size())
                current = torch.cat([output_tension, output_distance, output_strain, output_int], dim=-1)
                current_out = torch.cat([self.float_embedding(current[..., :3]), self.key_embedding(current[..., 3].long())], dim=-1)
                #print(next_input.size(), current_out.size())
                next_input = torch.cat([next_input, current_out], dim=1)
                output_seq.append(torch.cat([output_tension, output_distance, output_strain, output_i], dim=-1))

            output = torch.cat(output_seq, dim=1)  # [1, maxlen, EMBED_SIZE]

        return output

    def forward(self, batch):

        x, c1 = self.unsqueeze_vector(batch['melody'], batch['input'])
        #print(x[0, ...])
        #print(c1[0, ...])
        #sys.exit()
        #inputs =  self.pos_embedding(torch.cat([x, c1], dim=-1))
        #packed_inputs = pack_padded_sequence(inputs, batch['len'], batch_first=True, enforce_sorted=False)
        #packed_c1 = pack_padded_sequence(c1, batch['len'], batch_first=True, enforce_sorted=False)
        memory = self.encoder(c1, batch['len'], False)
        #z = noise.rsample()
        #print('z shape: ', z.size())
        output = self.decoder(c1, memory, x, batch['len'], False)

        return output

    def generate(self, one):

        _, c1 = self.unsqueeze_vector(one['melody'])
        c1 = c1.unsqueeze(0)
        #print(c1.size(), c2.size())
        memory = self.encoder(c1, None, True)

        output = self.decoder(c1, memory, None, c1.size(1), True)

        return output