import numpy as np
import json
from generate import generate_z_of_sample, encode_batch, restruction_batch, generate_chords_for_melody, random_generate_chords_for_melody
import functions.analyse as analyse
import os

data = np.load('Dataset_augment/dataset_train.npy', allow_pickle=True)
print(len(data))
# Several variant models for decision of parameter settings, corresponding to Appendix III.
paths = ['modelSave/cvae_model_001.mdl', 'modelSave/cvae_model_01.mdl', 'modelSave/cvae_model_03.mdl', 'modelSave/cvae_model_05.mdl', 'modelSave/cvae_model_1.mdl', 'modelSave/cvae_model_2.mdl', 'modelSave/cvae_model_3.mdl']
labels = ['β=0.01', 'β=0.1', 'β=0.3', 'β=0.5', 'β=1', 'β=2', 'β=3']
paths_2 = ['modelSave/cvae_model_001_.mdl', 'modelSave/cvae_model_01_.mdl', 'modelSave/cvae_model_06_.mdl', 'modelSave/cvae_model_1_.mdl', 'modelSave/cvae_model_2_.mdl']
labels_2 = ['β=0.01', 'β=0.1', 'β=0.6', 'β=1', 'β=2']

#Get sample indices in numpy dataset, the samples are selected according to their factor value, corresponding to Case Study in paper.
def get_rank_indices(data, key_word='tension-std'):
    tension_rank_sigma = []
    for i in range(len(data)):
        for tag in data[i][-1]:
            if key_word in tag and data[i][6][0] < 12 and len(data[i][0])==8:
                tension_sigma = float(tag.split('_')[-1])
                tension_rank_sigma.append({'index': i, 'value': tension_sigma, 'len': len(data[i][0])})

    tension_rank_sigma.sort(key=lambda x: x['value'])
    print(tension_rank_sigma[0:10], tension_rank_sigma[-10:])

    min_list = tension_rank_sigma[0:10]
    max_list = tension_rank_sigma[-10:]

    return min_list, max_list

#Encode the new sample that modified with Controlled Variable Methodology, and return latent variable, corresponding to Case Study in paper.
def get_z_from_sample(index, diff_row, model_path=''):

    offset = data[index][6][0]
    length = len(data[index][0])
    #new_melody_seg_number = [x-offset for x in data[index][0]]
    melody_list = [60] * length
    melody_weight = [0] * length
    tension_list = [1.85472] * length
    distance_list = [0.0] * length
    strain_list = [0.3] * length
    key_list = [0] * length
    key_list_2 = [12] * length

    new_data = [
        melody_list if diff_row!=0 else data[index][0],
        [],
        melody_weight if diff_row!=2 else data[index][2],
        tension_list if diff_row!=3 else data[index][3],
        distance_list if diff_row!=4 else data[index][4],
        strain_list if diff_row!=5 else data[index][5],
        key_list if diff_row!=6 else key_list_2,
    ]
    mu, std, z = generate_z_of_sample(new_data, model_path)

    return mu

#Analyze the main dimensions contribute to certain feature label, i.e., z_diff in paper.
def analyze_main_vectors(model_path='', feature_index=3, feature_tag='tension-std'):
    #min_list, max_list = [{'index': 0, 'value': 0}] * 10, [{'index': 0, 'value': 0}] * 10
    min_list, max_list = get_rank_indices(data, feature_tag)

    min_lists = []
    for i in min_list:
        z = get_z_from_sample(i['index'], feature_index, model_path)
        #print(z)
        min_lists.append(z)
    mean_min = np.mean(np.stack(min_lists), axis=0)

    max_lists = []
    for i in max_list:
        z = get_z_from_sample(i['index'], feature_index, model_path)
        #print(z)
        max_lists.append(z)
    mean_max = np.mean(np.stack(max_lists), axis=0)

    differences = []
    for i in range(len(mean_min)):
        differences.append({'index': i, 'gap': abs(mean_min[i]-mean_max[i])})
    differences.sort(key=lambda x: x['gap'])

    #print('------\n', mean_min, '\n', mean_max)
    for t in differences[-10:]:
        print(f'{t}\n') 
    return differences

#Analyze models with different hyperparameters, show their dimension contributions to certain feature label
def loop_models_show_feature_influence():
    results = []
    for path in paths_2:
        diff = analyze_main_vectors(path, 3, 'tension-std')
        curve = [f['gap'] for f in diff]
        result = [x / max(curve) for x in curve]
        results.append(result)
    analyse.show_curve(results, labels_2, 'Dimension (Sorted)', 'Influence (Normalized)')

#Batch encode samples in dataset, analyze PCA from models of different hyperparameters
def loop_models_show_pca_variance():
    results = []
    for path in paths_2:
        result = encode_batch(path)
        result = [x / max(result) for x in result]
        results.append(result)
    analyse.show_curve(results, labels_2)

#Analyze objective metrics of models of different hyperparameters
def loop_models_show_accuracy():
    results = []
    for path in paths:
        print(path)
        restruction_batch(path)

# Generate Noise samples for the chords quality evaluation, corresponding to Table 4 in paper.
def loop_random_harmonies_midi():

    melodies_path = 'your melody files, maybe in Subjective_Eval provided by us'
    destination_path = 'destination path for generated chord progressions'
    file_names = os.listdir(melodies_path)
    for i in range(len(file_names)):
        print(f'iterate: {i+1}, melody: {file_names[i]}')
        melody_path = f'{melodies_path}/{file_names[i]}'
        new_path = f'{destination_path}/{file_names[i]}'
        random_generate_chords_for_melody(file_path=melody_path, new_path=new_path)

# Generate chord progressions samples for the chords quality evaluation, corresponding to Table 4 and Figure 3 in paper.
def loop_generate_for_melodies():

    melodies_path = 'your melody files, maybe in Subjective_Eval provided by us'
    destination_path = 'destination path for generated chord progressions'
    file_names = os.listdir(melodies_path)
    for i in range(len(file_names)):
        print(f'iterate: {i+1}, melody: {file_names[i]}')
        melody_path = f'{melodies_path}/{file_names[i]}'
        new_path = f'{destination_path}/{file_names[i]}'
        generate_chords_for_melody([0], [0], file_path=melody_path, new_path=new_path)

#loop_models_show_accuracy()
#analyze_main_vectors('modelSave/cvae_model_1.mdl', 3, 'tension-std')
#analyze_main_vectors('modelSave/cvae_model_1.mdl', 4, 'distance-mean')
#analyze_main_vectors('modelSave/cvae_model_1.mdl', 5, 'strain-gradient')
#rank_vectors(100)