from functions.curve_dataset import EmotionCurveDataset, pad_timeline_function
import functions.params as params
import functions.analyse as analyse
import torch
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence, Normal
from torch.utils.tensorboard import SummaryWriter
from model import CPFG
import math
import time
import sys
import os

ctime = time.localtime()
system_time = f'{ctime.tm_year}-{ctime.tm_mon}-{ctime.tm_mday}_{ctime.tm_hour}-{ctime.tm_min}'
train_label = 'getModel'
file_name = f'{system_time}_{train_label}' #Name label for tensorboard SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'model state dict path for training save'
train_dataset_path = 'path of train part of augmented dataset'
eval_dataset_path = 'path of evaluation part of augmented dataset'
dataset_train = EmotionCurveDataset(train_dataset_path)
dataset_eval = EmotionCurveDataset(eval_dataset_path)
train_loader = DataLoader(dataset_train, params.BATCH_SIZE, shuffle=True, collate_fn=pad_timeline_function, drop_last=True)
eval_loader = DataLoader(dataset_eval, params.BATCH_SIZE, shuffle=True, collate_fn=pad_timeline_function, drop_last=True)

model = CPFG()

optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.GAMMA)
params_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
print('Model total params num: ', params_num)
key_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
tension_criterion = torch.nn.MSELoss(reduction='mean')
distance_criterion = torch.nn.MSELoss(reduction='mean')
strain_criterion = torch.nn.MSELoss(reduction='mean')

if params.PARALLEL:
    model = torch.nn.DataParallel(model, device_ids=params.CUDA_DEVICES)
    model = model.to(device)
else:
    model = model.to(device)

# training losses store path for tensorboard
result_path = os.path.abspath(os.path.dirname(__file__))
result_path = os.path.dirname(result_path)
result_path = os.path.join(result_path, f'tf-logs/{file_name}')

is_record = False
if is_record:
    print('loss result path: ', result_path)
    train_loss_writer = SummaryWriter(result_path)
    variables = {key: value for key, value in params.__dict__.items() if not key.startswith("__")}
    information = ''
    for var, val in variables.items():
        information += f"{var} = {val}\n"
    train_loss_writer.add_text('Hparams', information, 0)

# Seed for randomness reproduci
seed=77
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# process mask for variance sequence
def fix_batch_for_loss_calculate(batch, output):
    # turn the padded '0' into '-1'
    key_ints = batch['input'][:, :, -1]
    key_lens = torch.tensor(batch['len']).to(device)
    mask = torch.arange(key_ints.size(1), device=device)[None, :] < key_lens[:, None] #(B, time_len1)
    mask.to(device)
    key_ints = key_ints.masked_fill(~mask, -1)
    key_ints = key_ints.view(-1).long() # (B*time_len1)
    #print('key_ints: ', key_ints)

    float_input = batch['input'][:, :, :3]
    float_output = output[:, :, :3]
    filter_float_input = float_input.masked_select(mask.unsqueeze(-1)).view(-1, 3)
    filter_float_output = float_output.masked_select(mask.unsqueeze(-1)).view(-1, 3)

    return key_ints, filter_float_input, filter_float_output

def loss_calculate(mu, std, noise, output, batch, epoch, show = False):
    standard_normal = Normal(torch.zeros(params.BATCH_SIZE, params.Z_SIZE).to(device), torch.ones(params.BATCH_SIZE, params.Z_SIZE).to(device))
    key_ints, filter_float_input, filter_float_output = fix_batch_for_loss_calculate(batch, output)
    int_out = output[:, :, 3:]
    
    kl_loss = kl_divergence(noise, standard_normal).mean()
    cross_loss = key_criterion(int_out.view(-1, params.KEY_SIZE), key_ints)
    tension_loss = tension_criterion(filter_float_input[..., 0], filter_float_output[..., 0])
    distance_loss = distance_criterion(filter_float_input[..., 1], filter_float_output[..., 1])
    strain_loss = strain_criterion(filter_float_input[..., 2], filter_float_output[..., 2])

    # record objective matrics
    if (epoch in [2, 3, 5, 20, 50, 100, 150]) and show:
        print(epoch, '\n')
        show_tension_input = batch['input'][1, :batch['len'][1], 0]
        show_tension_output = output[1, :batch['len'][1], 0]
        show_distance_input = batch['input'][1, :batch['len'][1], 1]
        show_distance_output = output[1, :batch['len'][1], 1]
        show_strain_input = batch['input'][1, :batch['len'][1], 2]
        show_strain_output = output[1, :batch['len'][1], 2]
        show_key_input = batch['input'][1, :batch['len'][1], -1].int()
        show_keyhot_output = output[1, :batch['len'][1], 3:]
        show_key_output = torch.argmax(torch.softmax(show_keyhot_output, dim=-1), dim=-1)
        print(show_tension_input, '\n#######\n', show_tension_output, '\nSpearman 相关系数：', analyse.calculate_spear(show_tension_input, show_tension_output))
        print(show_distance_input, '\n#######\n', show_distance_output, '\nSpearman 相关系数：', analyse.calculate_spear(show_distance_input, show_distance_output))
        print(show_strain_input, '\n#######\n', show_strain_output, '\nSpearman 相关系数：', analyse.calculate_spear(show_strain_input, show_strain_output))
        print(show_key_input, '\n#######\n', show_key_output, '\n\n')
        #analyse.calculate_pca(mu, noise, epoch)
        tension_mean_spear = distance_mean_spear = strain_mean_spear = 0.
        for i in range(params.BATCH_SIZE):
            t_input = batch['input'][i, :batch['len'][i], 0]
            t_output = output[i, :batch['len'][i], 0]
            d_input = batch['input'][i, :batch['len'][i], 1]
            d_output = output[i, :batch['len'][i], 1]
            s_input = batch['input'][i, :batch['len'][i], 2]
            s_output = output[i, :batch['len'][i], 2]

            current_t = analyse.calculate_spear(t_input, t_output)
            if current_t and current_t != None and not math.isnan(current_t):
                tension_mean_spear += current_t
            current_d = analyse.calculate_spear(d_input, d_output)
            if current_d and current_d != None and not math.isnan(current_d):
                distance_mean_spear += current_d
            current_s = analyse.calculate_spear(s_input, s_output)
            if current_s and current_s != None and not math.isnan(current_s):
                strain_mean_spear += current_s
        print('Mean tension spearman similarity: ', tension_mean_spear / params.BATCH_SIZE)
        print('Mean distance spearman similarity: ', distance_mean_spear / params.BATCH_SIZE)
        print('Mean strain spearman similarity: ', strain_mean_spear / params.BATCH_SIZE)
        print('\nMean std: ', std.mean().item())

    # Total loss
    loss = params.TENSION_LOSS_WEIGHT * tension_loss + params.DISTANCE_LOSS_WEIGHT * distance_loss + params.STRAIN_LOSS_WEIGHT * strain_loss +\
        params.CROSS_LOSS_WEIGHT * cross_loss +\
        min(params.KL_LOSS_WEIGHT, max(0, (epoch - params.MIN_KL_IMPROVE_T))/(params.MAX_KL_IMPROVE_T - params.MIN_KL_IMPROVE_T)) * kl_loss

    return loss, tension_loss, distance_loss, strain_loss, cross_loss, kl_loss

def train_part(epoch):
    model.train()
    batch_num = len(train_loader)
    batch_index = 0
    average_loss = average_cross_loss = average_tension_loss = average_distance_loss = average_strain_loss = average_kl_loss = 0.
    first_batch = True
    if (epoch in [2, 3, 5, 20, 50, 100, 150]) and is_record:
        mus = None
        name_tags = []
        key_tags = []
        tension_std_tags = []
        distance_std_tags = []
        strain_std_tags = []
        index_tags = []
        keys_tags = []

    for batch in train_loader:
        batch['input'] = batch['input'].to(device)
        batch['melody'] = batch['melody'].to(device)
        optimizer.zero_grad()
        mu, std, noise, output = model(batch)
        loss, tension_loss, distance_loss, strain_loss, cross_loss, kl_loss = loss_calculate(mu, std, noise, output, batch, epoch, first_batch)
        loss.backward()
        optimizer.step()

        if (epoch in [2, 3, 5, 20, 50, 100, 150]) and is_record:

            first_batch = False
            if mus == None:
                mus = mu
            else:
                mus = torch.cat([mus, mu], dim=0)

            name_tags += [dataset_train.get_sample_tags(ind)[0] for ind in batch['index']]
            key_tags += [dataset_train.get_sample_tags(ind)[1] for ind in batch['index']]
            index_tags += [ind for ind in batch['index']]
            keys_tags += [dataset_train.get_sample_key(ind) for ind in batch['index']]

            tension_std_tags += [dataset_train.get_sample_tags(ind)[2] for ind in batch['index']]
            distance_std_tags += [dataset_train.get_sample_tags(ind)[8] for ind in batch['index']]
            strain_std_tags += [dataset_train.get_sample_tags(ind)[14] for ind in batch['index']]

            #tension_range_tags = analyse.bin_data_to_tags([dataset_train.get_sample_tags(ind)[3] for ind in batch['index']], 3)
            #distance_range_tags = analyse.bin_data_to_tags([dataset_train.get_sample_tags(ind)[9] for ind in batch['index']], 3)
            #strain_range_tags = analyse.bin_data_to_tags([dataset_train.get_sample_tags(ind)[15] for ind in batch['index']], 3)

            #tension_gradient_tags = analyse.bin_data_to_tags([dataset_train.get_sample_tags(ind)[6] for ind in batch['index']], 3)
            #distance_gradient_tags = analyse.bin_data_to_tags([dataset_train.get_sample_tags(ind)[12] for ind in batch['index']], 3)
            #strain_gradient_tags = analyse.bin_data_to_tags([dataset_train.get_sample_tags(ind)[18] for ind in batch['index']], 3)

            #train_loss_writer.add_embedding(mu, metadata=name_tags, tag=f'{epoch}_latent_mu_name')
            #train_loss_writer.add_embedding(mu, metadata=key_tags, tag=f'{epoch}_latent_mu_key')

            #train_loss_writer.add_embedding(mu, metadata=tension_std_tags, tag=f'{epoch}_latent_mu_tension_std')
            #train_loss_writer.add_embedding(mu, metadata=distance_std_tags, tag=f'{epoch}_latent_mu_distance_std')
            #train_loss_writer.add_embedding(mu, metadata=strain_std_tags, tag=f'{epoch}_latent_mu_strain_std')

            #train_loss_writer.add_embedding(mu, metadata=tension_range_tags, tag=f'{epoch}_latent_mu_tension_range')
            #train_loss_writer.add_embedding(mu, metadata=distance_range_tags, tag=f'{epoch}_latent_mu_distance_range')
            #train_loss_writer.add_embedding(mu, metadata=strain_range_tags, tag=f'{epoch}_latent_mu_strain_range')

            #train_loss_writer.add_embedding(mu, metadata=tension_gradient_tags, tag=f'{epoch}_latent_mu_tension_gradient')
            #train_loss_writer.add_embedding(mu, metadata=distance_gradient_tags, tag=f'{epoch}_latent_mu_distance_gradient')
            #train_loss_writer.add_embedding(mu, metadata=strain_gradient_tags, tag=f'{epoch}_latent_mu_strain_gradient')

        average_loss += loss.item()
        average_cross_loss += cross_loss.item()
        average_tension_loss += tension_loss.item()
        average_distance_loss += distance_loss.item()
        average_strain_loss += strain_loss.item()
        average_kl_loss += kl_loss.item()

        loss_index = epoch * batch_num + batch_index
        if is_record:
            train_loss_writer.add_scalars(f'{file_name}/train_batch_loss', {
                'loss': loss.item(),
                'cross': cross_loss.item(),
                'tension': tension_loss.item(),
                'distance': distance_loss.item(),
                'strain': strain_loss.item(),
                'kl': kl_loss.item()}, loss_index)
        batch_index += 1

    if (epoch in [2, 3, 5, 20, 50, 100, 150]) and is_record:

        tension_std_tags = analyse.bin_data_to_tags(tension_std_tags, 5)
        distance_std_tags = analyse.bin_data_to_tags(distance_std_tags, 5)
        strain_std_tags = analyse.bin_data_to_tags(strain_std_tags, 5)

        train_loss_writer.add_embedding(mus, metadata=[[k, t, d, s, n, i, kn] for k, t, d, s, n, i, kn in zip(key_tags, tension_std_tags, distance_std_tags, strain_std_tags, name_tags, index_tags, keys_tags)
            ], tag=f'{epoch}_latent_mu', metadata_header=['Key', 'Tension', 'Distance', 'Strain', 'Name', 'Index', 'Key_num'])
        #train_loss_writer.add_embedding(mus, metadata=tension_std_tags, tag=f'{epoch}_latent_mu_tension_std')
        #train_loss_writer.add_embedding(mus, metadata=distance_std_tags, tag=f'{epoch}_latent_mu_distance_std')
        #train_loss_writer.add_embedding(mus, metadata=strain_std_tags, tag=f'{epoch}_latent_mu_strain_std')

    average_loss /= batch_num
    average_cross_loss /= batch_num
    average_tension_loss /= batch_num
    average_distance_loss /= batch_num
    average_strain_loss /= batch_num
    average_kl_loss /= batch_num

    return {
        'loss': average_loss,
        'cross': average_cross_loss,
        'tension': average_tension_loss,
        'distance': average_distance_loss,
        'strain': average_strain_loss,
        'kl': average_kl_loss,
    }

def evaluate_part(epoch):
    model.eval()
    batch_num = len(eval_loader)
    batch_index = 0
    average_loss = average_cross_loss = average_tension_loss = average_distance_loss = average_strain_loss = average_kl_loss = 0.
    with torch.no_grad():
        for batch in eval_loader:
            batch['input'] = batch['input'].to(device)
            batch['melody'] = batch['melody'].to(device)
            mu, std, noise, output = model(batch)
            loss, tension_loss, distance_loss, strain_loss, cross_loss, kl_loss = loss_calculate(mu, std, noise, output, batch, epoch)

            average_loss += loss.item()
            average_cross_loss += cross_loss.item()
            average_tension_loss += tension_loss.item()
            average_distance_loss += distance_loss.item()
            average_strain_loss += strain_loss.item()
            average_kl_loss += kl_loss.item()

            loss_index = epoch * batch_num + batch_index
            if is_record:
                train_loss_writer.add_scalars(f'{file_name}/eval_batch_loss', {
                    'loss': loss.item(),
                    'cross': cross_loss.item(),
                    'tension': tension_loss.item(),
                    'distance': distance_loss.item(),
                    'strain': strain_loss.item(),
                    'kl': kl_loss.item()}, loss_index)
            batch_index += 1

        average_loss /= batch_num
        average_cross_loss /= batch_num
        average_tension_loss /= batch_num
        average_distance_loss /= batch_num
        average_strain_loss /= batch_num
        average_kl_loss /= batch_num

    return {
        'loss': average_loss,
        'cross': average_cross_loss,
        'tension': average_tension_loss,
        'distance': average_distance_loss,
        'strain': average_strain_loss,
        'kl': average_kl_loss,
    }

def training_model():
    print('Training model at time: ', system_time)
    for epoch in range(params.EPOCH_NUM):
        print('--- Current epoch: ', epoch+1)
        start_time = time.time()
        train_epoch_losses = train_part(epoch)

        if is_record:
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.numel() > 0:
                    train_loss_writer.add_histogram(f'{file_name}/grad_histogram', param.grad, epoch)
                    train_loss_writer.add_scalar(f'{file_name}/grad_norm', param.grad.data.norm(2), epoch)

        eval_epoch_losses = evaluate_part(epoch)
        scheduler.step()
        print('---------- lr: ', scheduler.get_last_lr())

        if is_record:
            train_loss_writer.add_scalars(f'{file_name}/train_epoch_loss', train_epoch_losses, epoch+1)
            train_loss_writer.add_scalars(f'{file_name}/eval_epoch_loss', eval_epoch_losses, epoch+1)

        if eval_epoch_losses['loss'] < params.BEST_EVAL_LOSS:
            params.BEST_EVAL_LOSS = eval_epoch_losses['loss']
            torch.save(model.state_dict(), model_path)

        print('---------- epoch cost: ', time.time() - start_time, 's')

    #torch.save(model.state_dict(), model_path)

training_model()
if is_record:
    train_loss_writer.close()