import numpy as np
from functions.params import *
import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import sys

class CPFG(torch.nn.Module):

    def __init__(self):
        super(CPFG, self).__init__()

        self.linear_dropout = torch.nn.Dropout(p=LINEAR_DROPOUT)
        self.gru_dropout = torch.nn.Dropout(p=GRU_DROPOUT)

        self.encode_linear_to_attention = torch.nn.Linear(CURVE_SIZE + MELODY_SIZE, 256)
        self.encode_attention = torch.nn.MultiheadAttention(256, ATT_HEADS, dropout=ATT_DROPOUT, batch_first=True)
        self.decode_linear_to_attention = torch.nn.Linear(Z_SIZE + 2 * GRU_NUM_LAYERS * MELODY_EMB_SIZE + MELODY_SIZE, 512)
        self.decode_attention = torch.nn.MultiheadAttention(512, ATT_HEADS, dropout=ATT_DROPOUT, batch_first=True)

        self.gru_encode = torch.nn.GRU(256, CURVE_EMB_SIZE + MELODY_EMB_SIZE, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=True)
        self.gru_encode_melody = torch.nn.GRU(MELODY_SIZE, MELODY_EMB_SIZE, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=True)

        self.linear_mu = torch.nn.Linear(2 * GRU_NUM_LAYERS * (CURVE_EMB_SIZE + MELODY_EMB_SIZE), Z_SIZE)
        self.linear_std = torch.nn.Linear(2 * GRU_NUM_LAYERS * (CURVE_EMB_SIZE + MELODY_EMB_SIZE), Z_SIZE)

        self.gru_decode = torch.nn.GRU(512, OUTPUT_EMB_SIZE * 2, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=True)

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
            SOS = torch.ones(cond1_int_part.size(0), cond1_int_part.size(1), 1).to(cond1_int_part.device).float()
        else:
            cond1_int_part = cond1_int_part.unsqueeze(-1)
            SOS = torch.ones(cond1_int_part.size(0), 1).to(cond1_int_part.device).float()

        #print(SOS.size(), cond1_onehot_part.size(), cond1_int_part.size())
        new_cond1 = torch.cat([SOS, cond1_onehot_part, SOS, cond1_int_part], dim=-1) #1+128+1+1=131
        # new_cond1: (B, time_len2, 132)
        return new_inputs, new_cond1

    def encoder(self, x, lens, generate=False):

        x = F.leaky_relu(self.encode_linear_to_attention(x))
        if not generate:
            x_padding_mask = torch.arange(x.size(1), device=x.device).expand(BATCH_SIZE, x.size(1)) >= torch.tensor(lens).to(x.device).unsqueeze(1)
            x_padding_mask.to(x.device)
            x = self.linear_dropout(x)
            x_attn, _ = self.encode_attention(x, x, x, key_padding_mask=x_padding_mask)
            #x_attn, _ = self.encode_attention_2(x_attn, x_attn, x_attn, key_padding_mask=x_padding_mask)
        else:
            x = self.linear_dropout(x)
            x_attn, _ = self.encode_attention(x, x, x)
            #x_attn, _ = self.encode_attention_2(x_attn, x_attn, x_attn)
        x_attn = self.gru_dropout(x_attn)
        if not generate:
            packed_x = pack_padded_sequence(x_attn, lens, batch_first=True, enforce_sorted=False)
        else:
            packed_x = x_attn
        _, x_gru_emb = self.gru_encode(packed_x)
        x_gru_emb = x_gru_emb.transpose(0, 1).contiguous() # (B, 2*GRU_NUM_LAYERS, CURVE_EMB_SIZE + MELODY_EMB_SIZE)
        if not generate:
            gru_emb = x_gru_emb.view(BATCH_SIZE, -1)
        else:
            gru_emb = x_gru_emb.view(1, -1)
        gru_emb = self.linear_dropout(gru_emb)
        mu = self.linear_mu(gru_emb)
        std = torch.exp(self.linear_std(gru_emb)) #(B, Z_SIZE)
        noise = Normal(mu, std)
        return mu, std, noise

    def decoder(self, z, c1, lens, generate):

        maxlen = c1.size(1)
        if not generate:
            packed_c1 = pack_padded_sequence(c1, lens, batch_first=True, enforce_sorted=False)
        else:
            packed_c1 = c1

        _, c1_gru_emb = self.gru_encode_melody(packed_c1)
        c1_gru_emb = c1_gru_emb.transpose(0, 1).contiguous() # (B, 2*GRU_NUM_LAYERS, MELODY_EMB_SIZE)

        if not generate:
            gru_emb = torch.cat([z, c1_gru_emb.view(BATCH_SIZE, -1)], dim=-1)
        else:
            gru_emb = torch.cat([z, c1_gru_emb.view(1, -1)], dim=-1)

        gru_emb = gru_emb.unsqueeze(1).repeat(1, maxlen, 1) # (B, output_len, Z_SIZE + 2*GRU_NUM_LAYERS*MELODY_EMB_SIZE)
        gru_emb = torch.cat([gru_emb, c1], dim=-1) # (B, output_len, Z_SIZE + 2*GRU_NUM_LAYERS*MELODY_EMB_SIZE + MELODY_SIZE)

        gru_emb = F.leaky_relu(self.decode_linear_to_attention(gru_emb))
        if not generate:
            emb_padding_mask = torch.arange(gru_emb.size(1), device=gru_emb.device).expand(BATCH_SIZE, gru_emb.size(1)) >= torch.tensor(lens).to(gru_emb.device).unsqueeze(1)
            emb_padding_mask.to(gru_emb.device)
            gru_emb = self.linear_dropout(gru_emb)
            emb_attn, _ = self.decode_attention(gru_emb, gru_emb, gru_emb, key_padding_mask=emb_padding_mask)
        else:
            gru_emb = self.linear_dropout(gru_emb)
            emb_attn, _ = self.decode_attention(gru_emb, gru_emb, gru_emb)

        emb_attn = self.gru_dropout(emb_attn)
        if not generate:
            packed_emb = pack_padded_sequence(emb_attn, lens, batch_first=True, enforce_sorted=False)
            output, _ = self.gru_decode(packed_emb)
            output_padded, lengths = pad_packed_sequence(output, batch_first=True)
        else:
            packed_emb = emb_attn
            output, _ = self.gru_decode(packed_emb)
            output_padded = output

        output = self.linear_dropout(output_padded)

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
        #packed_inputs = pack_padded_sequence(inputs, batch['len'], batch_first=True, enforce_sorted=False)
        #packed_c1 = pack_padded_sequence(c1, batch['len'], batch_first=True, enforce_sorted=False)
        mu, std, noise = self.encoder(inputs, batch['len'])
        z = noise.rsample()
        #print('z shape: ', z.size())
        output = self.decoder(z, c1, batch['len'], False)

        return mu, std, noise, output
#--------------------------------------------------------------------------------------------------------------
#Generate functions
    def generate(self, one, indices, nums):

        _, c1 = self.unsqueeze_vector(one['melody'])
        c1 = c1.unsqueeze(0)
        #print(c1.size(), c2.size())
        z = torch.randn(1, Z_SIZE).to(c1.device)
        for i in range(len(indices)):
            z[0][indices[i]] += nums[i]
        #z = torch.zeros(1, Z_SIZE).to(c1.device).float()
        output = self.decoder(z, c1, None, True)

        return output

    def generate_2(self, tension, distance, strain, key, cond):
        tension = [10*(x - 1.46969) / (4.62169 - 1.46969) for x in tension]
        distance = [10*(x / 3.56156) for x in distance]
        strain = [10*(x - 0.0586) / (3.08 - 0.0586) for x in strain]
        inputs = [[tension[i], distance[i], strain[i], key[i]] for i in range(len(tension))]
        one = {
            'input': torch.tensor(inputs).to(cond['melody'].device).unsqueeze(0),
            'melody': cond['melody'].unsqueeze(0),
        }
        #print(one['input'].size(), one['melody'].size())
        x, c1 = self.unsqueeze_vector(one['melody'], one['input'])
        inputs = torch.cat([x, c1], dim=-1)
        _, _, noise = self.encoder(inputs, None, True)
        z = noise.rsample()
        #print(z.size())
        output = self.decoder(z, c1, None, True)

        return output

    def generate_3(self, tension, distance, strain, key, cond):
        tension = [10*(x - 1.46969) / (4.62169 - 1.46969) for x in tension]
        distance = [10*(x / 3.56156) for x in distance]
        strain = [10*(x - 0.0586) / (3.08 - 0.0586) for x in strain]
        inputs = [[tension[i], distance[i], strain[i], key[i]] for i in range(len(tension))]
        one = {
            'input': torch.tensor(inputs).to(cond['melody'].device).unsqueeze(0),
            'melody': cond['melody'].unsqueeze(0),
        }
        #print(one['input'].size(), one['melody'].size())
        x, c1 = self.unsqueeze_vector(one['melody'], one['input'])
        inputs = torch.cat([x, c1], dim=-1)
        mu, std, noise = self.encoder(inputs, None, True)
        z = noise.rsample()

        return mu.view(-1), std.view(-1), z.view(-1)


    def smooth_generate(self, tension, distance, strain, key, cond, indices, nums):
        tension = [10*(x - 1.46969) / (4.62169 - 1.46969) for x in tension]
        distance = [10*(x / 3.56156) for x in distance]
        strain = [10*(x - 0.0586) / (3.08 - 0.0586) for x in strain]
        inputs = [[tension[i], distance[i], strain[i], key[i]] for i in range(len(tension))]
        one = {
            'input': torch.tensor(inputs).to(cond['melody'].device).unsqueeze(0),
            'melody': cond['melody'].unsqueeze(0),
        }
        #print(one['input'].size(), one['melody'].size())
        x, c1 = self.unsqueeze_vector(one['melody'], one['input'])
        inputs = torch.cat([x, c1], dim=-1)
        mu, std, _ = self.encoder(inputs, None, True)
        #print(mu)
        #for i in range(len(indices)):
        #    mu[0][indices[i]] = mu[0][indices[i]] + nums[i]
        noise = Normal(mu, std)
        z = noise.rsample()
        for i in range(len(indices)):
            z[0][indices[i]] = z[0][indices[i]] + nums[i]
        #print(z.size())
        output = self.decoder(z, c1, None, True)

        return output

    def generate_4(self, batch):

        x, c1 = self.unsqueeze_vector(batch['melody'], batch['input'])
        inputs = torch.cat([x, c1], dim=-1)

        mu, std, noise = self.encoder(inputs, batch['len'])

        return mu, std, noise