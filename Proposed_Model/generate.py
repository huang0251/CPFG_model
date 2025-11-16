import torch
from torch.utils.data import DataLoader
from model import CPFG
import functions.process as process
import os
import numpy as np
import functions.analyse as analyse
from functions.curve_dataset import EmotionCurveDataset, pad_timeline_function
import functions.params as params
import math
import random

seed=77
random.seed(seed)
np.random.seed(seed)

data = np.load('Dataset_augment/dataset_evaluate.npy', allow_pickle=True)

model_path = 'your state_dict file path of model'
dataset_path = 'Dataset_augment/dataset_evaluate.npy'

dataset_train = EmotionCurveDataset(dataset_path)
train_loader = DataLoader(dataset_train, params.BATCH_SIZE, shuffle=True, collate_fn=pad_timeline_function, drop_last=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# An example sample in Case study
example_melody = [53, 60, 57, 53, 60, 62, 62, 62, 62, 60, 60, 62, 64, 64, 65, 65, 64, 64]
example_melody_weight = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
example_melody_len = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
example_tension = [1.85472, 1.85472, 1.85472, 1.85472, 1.85472, 1.85472, 2.15407, 1.85472, 3.1241, 1.85472, 1.85472, 1.85472, 3.1241, 3.1241, 1.85472, 3.1241, 1.6, 1.85472]
example_distance = [0.0, 1.12744, 1.12744, 0.71802, 0.71802, 1.12744, 0.42622, 0.42622, 0.6069, 0.71764, 0.0, 1.12744, 1.04563, 0.38873, 1.06249, 1.08321, 1.33333, 0.4899]
example_strain = [0.39273, 1.04091, 0.39273, 0.7871, 0.39273, 0.89356, 0.49475, 0.89356, 0.53574, 0.39273, 0.39273, 0.89356, 0.69486, 0.8105, 0.39273, 0.99681, 1.1525, 1.04091]
example_key = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
example_sample = [example_melody, example_melody_len, example_melody_weight, example_tension, example_distance, example_strain, example_key]

# Batch encode test samples and analyze latent representation using PCA, corresponding to Appendex III
def encode_batch(new_model_path):
    model = CPFG()
    model.load_state_dict(torch.load(new_model_path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch['input'] = batch['input'].to(device)
            batch['melody'] = batch['melody'].to(device)
            mu, std, noise = model.generate_4(batch)
            break
    return analyse.major_component(mu)

# Reconstruction test set and compute objective metrics, corresponding to Table 2 and Table 5 in paper.
def restruction_batch(new_model_path):
    model = CPFG()
    model.load_state_dict(torch.load(new_model_path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():

        tension_spears = []
        distance_spears = []
        strain_spears = []
        tension_mses = []
        distance_mses = []
        strain_mses = []
        key_cels = []

        for batch in train_loader:
            batch['input'] = batch['input'].to(device)
            batch['melody'] = batch['melody'].to(device)
            _, _, _, output = model(batch)


            for i in range(batch['input'].shape[0]):
                t_input = batch['input'][i, :batch['len'][i], 0]
                t_output = output[i, :batch['len'][i], 0]
                d_input = batch['input'][i, :batch['len'][i], 1]
                d_output = output[i, :batch['len'][i], 1]
                s_input = batch['input'][i, :batch['len'][i], 2]
                s_output = output[i, :batch['len'][i], 2]
                k_input = batch['input'][i, :batch['len'][i], 3]
                k_output = output[i, :batch['len'][i], 3:]

                current_t = analyse.calculate_spear(t_input, t_output)
                if current_t and current_t != None and not math.isnan(current_t):
                    tension_spears.append(current_t)
                current_d = analyse.calculate_spear(d_input, d_output)
                if current_d and current_d != None and not math.isnan(current_d):
                    distance_spears.append(current_d)
                current_s = analyse.calculate_spear(s_input, s_output)
                if current_s and current_s != None and not math.isnan(current_s):
                    strain_spears.append(current_s)
                tension_mses.append(analyse.calculate_mse(t_input, t_output))
                distance_mses.append(analyse.calculate_mse(d_input, d_output))
                strain_mses.append(analyse.calculate_mse(s_input, s_output))
                key_cels.append(analyse.calculate_cel(k_input, k_output))

        tension_spear_std = np.std(np.array(tension_spears))
        distance_spear_std = np.std(np.array(distance_spears))
        strain_spear_std = np.std(np.array(strain_spears))
        tension_mse_std = np.std(np.array(tension_mses))
        distance_mse_std = np.std(np.array(distance_mses))
        strain_mse_std = np.std(np.array(strain_mses))
        key_cel_std = np.std(np.array(key_cels))
        length = (len(data)//params.BATCH_SIZE)*params.BATCH_SIZE
        print('Tension spearman similarity: ', np.mean(np.array(tension_spears)))
        print('Distance spearman similarity: ', np.mean(np.array(distance_spears)))
        print('Strain spearman similarity: ', np.mean(np.array(strain_spears)))
        print('Tension mse: ', np.mean(np.array(tension_mses)))
        print('Distance mse: ', np.mean(np.array(distance_mses)))
        print('Strain mse: ', np.mean(np.array(strain_mses)))
        print('Key cel: ', np.mean(np.array(key_cels)))

# Predict results from given melodies in test set, and calculate tonality accuracy, corresponding to Table 5 in paper.
def predict_batch(new_model_path):
    model = CPFG()
    model.load_state_dict(torch.load(new_model_path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():

        key_acc = 0
        key_all = 0

        for batch in train_loader:
            batch['input'] = batch['input'].to(device)
            batch['melody'] = batch['melody'].to(device)
            output = model.generate_batch(batch)

            for i in range(batch['melody'].shape[0]):

                key_int = batch['input'][i, :batch['len'][i], 3].cpu()
                key_out = torch.argmax(torch.softmax(output[i, :batch['len'][i], 3:], dim=-1), dim=-1).squeeze(0).cpu().numpy()
                
                int_key_values, int_key_counts = np.unique(key_int, return_counts = True)
                out_key_values, out_key_counts = np.unique(key_out, return_counts = True)

                int_key_num = int_key_values[np.argmax(int_key_counts)]
                out_key_num = out_key_values[np.argmax(out_key_counts)]

                key_all += 1
                if int_key_num == out_key_num:
                    key_acc += 1


        print(key_acc, key_all)

# A basic function to predict MTT curves from given melody.
def model_result(melody, melody_weight, indices, nums):
    condition_1 = np.stack((melody, melody_weight), axis=1)
    cond_1 = torch.tensor(condition_1, dtype = torch.float32).to(device)
    conditions = {
        'melody': cond_1,
    }
    model = CPFG()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(conditions, indices, nums)
        tension_out = output[..., 0].squeeze(0).cpu().numpy()
        distance_out = output[..., 1].squeeze(0).cpu().numpy()
        tensile_out = output[..., 2].squeeze(0).cpu().numpy()
        key_out = torch.argmax(output[..., 3:], dim=-1).squeeze(0).cpu().numpy()

    tension_out = [x if x >= 0 else 0.0 for x in tension_out]
    distance_out = [x if x >= 0 else 0.0 for x in distance_out]
    tensile_out = [x if x >= 0 else 0.0 for x in tensile_out]

    tension_out = [round(x / 10 * (4.62169 - 1.46969) + 1.46969, 5) for x in tension_out]
    distance_out = [round(x / 10 * 3.56156, 5) for x in distance_out]
    tensile_out = [round(x / 10 * (3.08 - 0.0586) + 0.0586, 5) for x in tensile_out]
    
    return key_out, tension_out, distance_out, tensile_out

# Predict a chord progression from given melody, and calculate RD for Equation (18) in paper.
def generate_chords_for_melody(indices, nums, file_path='', new_path = 'generated_midi.mid'):

    if file_path == '':
        index = random.randint(0, len(data)-1)
        melody, melody_weight, offset_len = data[index][0], data[index][2], data[index][1]
    else:
        melody, melody_weight, melody_offset = process.get_net_input(file_path)
        offset_len = []
        for i in range(len(melody_offset)):
            if (i+1) == len(melody_offset):
                offset_len.append(1.0)
            else:
                offset_len.append(float(melody_offset[i+1]) - float(melody_offset[i]))

    key_out, tension_out, distance_out, tensile_out = model_result(melody, melody_weight, indices, nums)

    print(tension_out, distance_out, tensile_out, key_out)
    key_values, key_counts = np.unique(key_out, return_counts = True)
    key_num = key_values[np.argmax(key_counts)]
    key_nums = [key_num] * len(key_out)

    cost = process.make_midi_only_chords(tension_out, distance_out, tensile_out, offset_len, key_nums, new_path)

    return cost/len(melody)/100

# Generate a Noise sample, the melody is optional, corresponding to 'Noise' in paper.
def random_generate_chords_for_melody(file_path='', new_path = 'generated_midi.mid'):
    
    if file_path == '':
        index = random.randint(0, len(data)-1)
        melody, melody_weight, offset_len = data[index][0], data[index][2], data[index][1]
        example_key = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    else:
        melody, melody_weight, melody_offset = process.get_net_input(file_path)
        offset_len = []
        for i in range(len(melody_offset)):
            if (i+1) == len(melody_offset):
                offset_len.append(1.0)
            else:
                offset_len.append(float(melody_offset[i+1]) - float(melody_offset[i]))

    tension_out = [np.random.random() * (4.62169 - 1.46969) + 1.46969 for i in range(len(melody))]
    distance_out = [np.random.random() * 3.56156 for i in range(len(melody))]
    tensile_out = [np.random.random() * (3.08 - 0.0586) + 0.0586 for i in range(len(melody))]

    key_values, key_counts = np.unique(example_key, return_counts = True)
    key_num = key_values[np.argmax(key_counts)]
    key_nums = [key_num] * len(melody)

    cost = process.make_midi_only_chords(tension_out, distance_out, tensile_out, offset_len, key_nums, new_path)

    return cost/len(melody)/100

# A basic function : Encode one sample and return latent representation
def generate_z_of_sample(array, path=model_path):

    condition_1 = np.stack((array[0], array[2]), axis=1)
    cond_1 = torch.tensor(condition_1, dtype = torch.float32).to(device)
    conditions = {
        'melody': cond_1,
    }
    model = CPFG()
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        mu, std, z = model.generate_3(array[3], array[4], array[5], array[6], conditions)

    return mu.clone().detach().cpu().numpy(), std.clone().detach().cpu().numpy(), z.clone().detach().cpu().numpy()

# A basic function : Encode one sample and transform latent representation then decode for output
def shift_z_from_sample(array, indices, nums):

    condition_1 = np.stack((array[0], array[2]), axis=1)
    cond_1 = torch.tensor(condition_1, dtype = torch.float32).to(device)
    conditions = {
        'melody': cond_1,
    }
    model = CPFG()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.smooth_generate(array[3], array[4], array[5], array[6], conditions, indices, nums)

        tension_out = output[..., 0].squeeze(0).cpu().numpy()
        distance_out = output[..., 1].squeeze(0).cpu().numpy()
        tensile_out = output[..., 2].squeeze(0).cpu().numpy()
        key_out = torch.argmax(output[..., 3:], dim=-1).squeeze(0).cpu().numpy()


    tension_out = [x if x >= 0 else 0.0 for x in tension_out]
    distance_out = [x if x >= 0 else 0.0 for x in distance_out]
    tensile_out = [x if x >= 0 else 0.0 for x in tensile_out]

    tension_out = [round(x / 10 * (4.62169 - 1.46969) + 1.46969, 5) for x in tension_out]
    distance_out = [round(x / 10 * 3.56156, 5) for x in distance_out]
    tensile_out = [round(x / 10 * (3.08 - 0.0586) + 0.0586, 5) for x in tensile_out]

    return key_out, tension_out, distance_out, tensile_out

# Control and Modify chords for one sample in reconstruction process, corresponding to Case Study in paper.
def change_chords_for_melody(array, indices, nums):

    key_out, tension_out, distance_out, tensile_out = shift_z_from_sample(array, indices, nums)

    key_values, key_counts = np.unique(key_out, return_counts = True)
    key_num = key_values[np.argmax(key_counts)]
    key_nums = [key_num] * len(key_out)

    new_path = 'generated_midi.mid'
    offset_len = [n * 2 for n in array[1]]
    process.make_midi(array[0], array[2], tension_out, distance_out, tensile_out, offset_len, key_nums, new_path)

#Illustrate tension curves that decoded from modified latent representation, corresponding to Case Study and Figure 4 in paper.
def loop_show():
    # amplification  degree
    num_try_list = [-10, -3, -1, 0, 1, 3, 10]
    # amplification  dimensions (may change depending on your model)
    indices_try_list = [0, 0, 27, 0, 3, 27, 20]
    indices_try_list_2 = [60, 3, 48, 0, 3, 27, 20]
    indices_try_list_3 = [25, 27, 31, 0, 3, 27, 20]

    all_tensions = []
    all_distances = []
    all_strains = []
    for i in range(len(indices_try_list)):
        tension_outs = []
        distance_outs = []
        strain_outs = []
        for num in num_try_list:
            _, tension_out, distance_out, tensile_out = shift_z_from_sample(example_sample, [indices_try_list[i], indices_try_list_2[i], indices_try_list_3[i]], [num, num, num])
            #key_out, tension_out, distance_out, tensile_out = model_result([60]*12, [1,0,0,1,0,0,1,0,0,1,0,0], [indices_try_list[i]], [num])
            tension_outs.append(tension_out)
            distance_outs.append(distance_out)
            strain_outs.append(tensile_out)

        all_tensions.append(tension_outs)
        all_distances.append(distance_outs)
        all_strains.append(strain_outs)

    analyse.show_curves(indices_try_list, num_try_list, all_tensions, all_distances, all_strains)

#Loop generation process and see the tonality prediction effect, corresponding to Table 6 in paper.
def loop_key_types():
    summary = [0]*24
    index = random.randint(0, len(data)-1)
    melody, melody_weight, offset_len = data[index][0], data[index][2], data[index][1]
    print(data[index][6])
    for i in range(100):
        key_out, tension_out, distance_out, tensile_out = model_result(melody, melody_weight, [0], [0])
        key_values, key_counts = np.unique(key_out, return_counts = True)
        key_num = key_values[np.argmax(key_counts)]
        summary[key_num] += 1
    print(summary)

# Calculate MRDA one time, corresponding to Table 3 in paper.
def calculate_mrda():
    values = []
    for i in range(1000):
        print('current process: ', i)
        values.append(generate_chords_for_melody([0],[0]))
    mean = np.mean(np.array(values))

    print(mean)

#loop_key_types()
#generate_chords_for_melody([27, 0, 3], [0, 0, 0])
#loop_show()
#restruction_batch(model_path)

#predict_batch(model_path)

'''
v_ts = []
v_ds = []
v_ss = []
v_tm = []
v_dm = []
v_sm = []
v_kc = []
for i in range(10):
    t_s, d_s, s_s, t_m, d_m, s_m, k_c = restruction_batch(model_path)
    v_ts.append(t_s)
    v_ds.append(d_s)
    v_ss.append(s_s)
    v_tm.append(t_m)
    v_dm.append(d_m)
    v_sm.append(s_m)
    v_kc.append(k_c)
for v in [v_ts, v_ds, v_ss, v_tm, v_dm, v_sm, v_kc]:
    mean = np.mean(np.array(v))
    std = np.std(np.array(v))
    ci = 1.96*(std/np.sqrt(len(v)))
    print(mean, std, ci)
'''