import numpy as np
from functions.params import *
import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import sys

class CVAE_GRU(torch.nn.Module):

    def __init__(self):
        super(CVAE_GRU, self).__init__()

        self.linear_dropout = torch.nn.Dropout(p=LINEAR_DROPOUT)
        self.gru_dropout = torch.nn.Dropout(p=GRU_DROPOUT)

        self.gru_encode = torch.nn.GRU(CURVE_SIZE + MELODY_SIZE, CURVE_EMB_SIZE + MELODY_EMB_SIZE, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=True)
        self.gru_encode_melody = torch.nn.GRU(MELODY_SIZE, MELODY_EMB_SIZE, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=True)

        self.linear_mu = torch.nn.Linear(2 * GRU_NUM_LAYERS * (CURVE_EMB_SIZE + MELODY_EMB_SIZE), Z_SIZE)
        self.linear_std = torch.nn.Linear(2 * GRU_NUM_LAYERS * (CURVE_EMB_SIZE + MELODY_EMB_SIZE), Z_SIZE)

        self.gru_decode = torch.nn.GRU(Z_SIZE + 2 * GRU_NUM_LAYERS * MELODY_EMB_SIZE, OUTPUT_EMB_SIZE * 2, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=True)

        self.linear_tension_out_0 = torch.nn.Linear(OUTPUT_EMB_SIZE * 2 * 2, OUTPUT_EMB_SIZE)
        self.linear_distance_out_0 = torch.nn.Linear(OUTPUT_EMB_SIZE * 2 * 2, OUTPUT_EMB_SIZE)
        self.linear_strain_out_0 = torch.nn.Linear(OUTPUT_EMB_SIZE * 2 * 2, OUTPUT_EMB_SIZE)
        self.linear_int_out_0 = torch.nn.Linear(OUTPUT_EMB_SIZE * 2 * 2, OUTPUT_EMB_SIZE)

        self.linear_tension_out = torch.nn.Linear(OUTPUT_EMB_SIZE, 1)
        self.linear_distance_out = torch.nn.Linear(OUTPUT_EMB_SIZE, 1)
        self.linear_strain_out = torch.nn.Linear(OUTPUT_EMB_SIZE, 1)
        self.linear_int_out = torch.nn.Linear(OUTPUT_EMB_SIZE, KEY_SIZE)

    def unsqueeze_vector(self, cond1, inputs=None):
        # inputs: (B, time_len1, 4)
        # cond1: (B, time_len2, 2)
        #print(cond1.size(),cond2.size())
        if inputs != None:
            input_float_part = inputs[..., :3]
            input_int_part = inputs[..., 3].long()
            #print("input_int_part:", input_int_part)
            input_onehot_part = F.one_hot(input_int_part, num_classes = KEY_SIZE).float()
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
        # new_cond1: (B, time_len2, 132)
        return new_inputs, new_cond1

    def encoder(self, x):

        _, x_gru_emb = self.gru_encode(x)
        x_gru_emb = x_gru_emb.transpose(0, 1).contiguous() # (B, 2*GRU_NUM_LAYERS, CURVE_EMB_SIZE + MELODY_EMB_SIZE)

        gru_emb = x_gru_emb.view(BATCH_SIZE, -1)
        gru_emb = self.linear_dropout(gru_emb)
        mu = self.linear_mu(gru_emb)
        std = torch.exp(self.linear_std(gru_emb)) #(B, Z_SIZE)
        noise = Normal(mu, std)
        return mu, std, noise

    def decoder(self, z, c1, maxlen, generate):

        c1_gru, c1_gru_emb = self.gru_encode_melody(c1)
        c1_gru_emb = c1_gru_emb.transpose(0, 1).contiguous() # (B, 2*GRU_NUM_LAYERS, MELODY_EMB_SIZE)

        if not generate:
            gru_emb = torch.cat([z, c1_gru_emb.view(BATCH_SIZE, -1)], dim=-1)
        else:
            gru_emb = torch.cat([z, c1_gru_emb.view(1, -1)], dim=-1)

        gru_emb = gru_emb.unsqueeze(1).repeat(1, maxlen, 1) # (B, output_len, Z_SIZE + 2*MELODY_EMB_SIZE)
        gru_emb = self.gru_dropout(gru_emb)
        
        output, _ = self.gru_decode(gru_emb)
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

        return output

    def forward(self, batch):

        x, c1 = self.unsqueeze_vector(batch['melody'], batch['input'])
        #print(x[0, ...])
        #print(c1[0, ...])
        #sys.exit()
        inputs = torch.cat([x, c1], dim=-1)
        packed_inputs = pack_padded_sequence(inputs, batch['len'], batch_first=True, enforce_sorted=False)
        packed_c1 = pack_padded_sequence(c1, batch['len'], batch_first=True, enforce_sorted=False)
        mu, std, noise = self.encoder(packed_inputs)
        z = noise.rsample()
        #print('z shape: ', z.size())
        output = self.decoder(z, packed_c1, c1.size(1), False)

        return mu, std, noise, output

    def generate(self, one):

        _, c1 = self.unsqueeze_vector(one['melody'])
        c1 = c1.unsqueeze(0)
        #print(c1.size(), c2.size())
        z = torch.randn(1, Z_SIZE).to(c1.device)
        #z = torch.zeros(1, Z_SIZE).to(c1.device).float()
        output = self.decoder(z, c1, c1.size(1), True)

        return output