import os
import numpy as np
import random
import csv
import sys
from scipy.fft import rfft, rfftfreq

max_sigma = 0
min_sigma = float('inf')
max_range = 0
min_range = float('inf')
max_mc_density = 0
min_mc_density = float('inf')
max_mdc_density = 0
min_mdc_density = float('inf')
max_gradient = 0
min_gradient = float('inf')
max_fft = 0
min_fft = float('inf')

seed=77
random.seed(seed)
np.random.seed(seed)

#Main function that sugment metadata and analyze samples.
def get_curve_lists(file_path):

    offset_list = []
    melody_lists = []
    weight_list = []
    tension_list = []
    distance_list = []
    strain_list = []
    key_list = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(zip(*reader))
    for row in rows:
        row = list(row)
        if 'melody' in row[0] and 'offset' not in row[0] and 'weight' not in row[0]:
            melody_lists.append(row[1:])
        if 'offset' in row[0] and len(offset_list) == 0:
            offset_list = row[1:]
        if 'weight' in row[0] and len(weight_list) == 0:
            weight_list = row[1:]
        if 'tension' in row[0]:
            tension_list = row[1:]
        if 'distance' in row[0]:
            distance_list = row[1:]
        if 'strain' in row[0]:
            strain_list = row[1:]
        if 'key' in row[0]:
            key_list = row[1:]

    lists = []

    segments = []
    for i in range(len(weight_list)):
        if int(weight_list[i]) == 1:
            segments.append(i)
    segments.append(len(weight_list))
    #print(segments)
    #----------------------------------------------------------------------------
    def get_segment_curves(start, end_not_include):
        #print(start, end_not_include)
        arrays = []
        table = {
            'C': 0,
            'D': 2,
            'E': 4,
            'F': 5,
            'G': 7,
            'A': 9,
            'B': 11,
        }
        seg_offset_list = []
        for i in range(start, end_not_include):
            if i+1 == len(weight_list):
                seg_offset_list.append(1.0)
            else:
                seg_offset_list.append(float(offset_list[i+1]) - float(offset_list[i]))
        seg_weight_list = [int(x) for x in weight_list[start: end_not_include]]
        seg_tension_list = [float(x) for x in tension_list[start: end_not_include]]
        seg_distance_list = [float(x) for x in distance_list[start: end_not_include]]
        seg_distance_list[0] = 0.0
        seg_strain_list = [float(x) for x in strain_list[start: end_not_include]]
        seg_key_list = key_list[start: end_not_include]

        key_seg_number = []
        for k in seg_key_list:
            n = table[k[0]]
            if 'b' in k:
                n -= 1
            if 'minor' in k:
                n += 12
            key_seg_number.append(n)

        for m in melody_lists:
            seg_melody_list = m[start: end_not_include]
            melody_seg_number = []
            for string in seg_melody_list:
                n = int(string[-1]) * 12 + table[string[0]]
                if '-' in string:
                    n -= 1
                elif '#' in string:
                    n += 1
                melody_seg_number.append(n)
            arrays.append([
                melody_seg_number,
                seg_offset_list,
                seg_weight_list,
                seg_tension_list,
                seg_distance_list,
                seg_strain_list,
                key_seg_number,
            ])
            
            #for t in range(1, 12):
            #    new_melody_seg_number = [x+t if x+t<=120 else x-(12-t) for x in melody_seg_number]
            #    new_key_seg_number = [(k+t)%12 if k<12 else (k-12+t)%12+12 for k in key_seg_number]
            #    arrays.append([
            #        new_melody_seg_number,
            #        seg_offset_list,
            #        seg_weight_list,
            #        seg_tension_list,
            #        seg_distance_list,
            #        seg_strain_list,
            #        new_key_seg_number,
            #    ])
        def noise_melody(melody_list, noise_num):
            noise_index = random.sample(range(0, len(melody_list)), noise_num)
            new_melody_list = []
            for i in range(0, len(melody_list)):
                if i in noise_index:
                    r = random.random()
                    if r <= 0.4:
                        new_melody_list.append(melody_list[i]+12 if melody_list[i]+12<=120 else melody_list[i]-12)
                    elif 0.4 < r <= 0.7:
                        new_melody_list.append(melody_list[i]+7 if melody_list[i]+7<=120 else melody_list[i]-(12-7))
                    elif 0.7 < r <= 0.9:
                        new_melody_list.append(melody_list[i]+5 if melody_list[i]+5<=120 else melody_list[i]-(12-5))
                    else:
                        new_melody_list.append(melody_list[i]+4 if melody_list[i]+4<=120 else melody_list[i]-(12-4))
                else:
                    new_melody_list.append(melody_list[i])

            return new_melody_list

        noise_arrays = []
        for arr in arrays:
            melody_noise_num = random.randint(len(arr[0])//6, len(arr[0])//3)
            changed_melody_list = noise_melody(arr[0], melody_noise_num)
            noise_arrays.append([
                changed_melody_list,
                arr[1],
                arr[2],
                arr[3],
                arr[4],
                arr[5],
                arr[6],
            ])
        arrays += noise_arrays

        return arrays
        
    #----------------------------------------------------------------------------
    if len(segments) <= 5:
        seg_lists = get_segment_curves(segments[0], segments[-1])
        lists += seg_lists
    else:
        for i in range(len(segments)-4):
            seg_lists = get_segment_curves(segments[i], segments[i+4])
            lists += seg_lists

    return lists

#Make tag labels for samples, such as SD/ZCR/FFT... described in paper.
def make_tags(lists, file_name):

    def standard_deviation(curve):
        global max_sigma, min_sigma
        curve = np.array(curve)
        sigma = np.std(curve)
        if sigma > max_sigma:
            max_sigma = sigma
        if sigma < min_sigma:
            min_sigma = sigma
        return sigma
    def value_range(curve):
        global max_range, min_range
        r = max(curve)-min(curve)
        if r > max_range:
            max_range = r
        if r < min_range:
            min_range = r
        return r
    def mean_cross_density(curve):
        global max_mc_density, min_mc_density
        mean_val = np.mean(curve)
        signs = np.sign(np.array(curve) - mean_val)
        crossings = np.count_nonzero(np.diff(signs))
        density = crossings / (len(curve) - 1)
        if density > max_mc_density:
            max_mc_density = density
        if density < min_mc_density:
            min_mc_density = density
        return density
    def median_cross_density(curve):
        global max_mdc_density, min_mdc_density
        median_val = (max(curve)+min(curve)) / 2
        signs = np.sign(np.array(curve) - median_val)
        crossings = np.count_nonzero(np.diff(signs))
        density = crossings / (len(curve) - 1)
        if density > max_mdc_density:
            max_mdc_density = density
        if density < min_mdc_density:
            min_mdc_density = density
        return density
    def gradient_change(curve):
        global max_gradient, min_gradient
        diffs = np.diff(curve)
        signs = np.sign(diffs)
        turns = np.count_nonzero(np.diff(signs))
        turn_density = turns / (len(curve) - 2)
        if turn_density > max_gradient:
            max_gradient = turn_density
        if turn_density < min_gradient:
            min_gradient = turn_density
        return turn_density
    def main_fft(curve):
        global max_fft, min_fft
        fft_vals = np.abs(rfft(curve - np.mean(curve)))
        freqs = rfftfreq(len(curve), d=1)
        dominant_freq_index = np.argmax(fft_vals[1:]) + 1
        dominant_freq = freqs[dominant_freq_index]
        if dominant_freq > max_fft:
            max_fft = dominant_freq
        if dominant_freq < min_fft:
            min_fft = dominant_freq
        return dominant_freq
    def value_mean(curve):
        return np.mean(curve)

    name_tag = file_name.split('_datas')[0]
    for array in lists:
        tags = []
        tags.append(name_tag)
        tags.append('major' if array[6][0] < 12 else 'minor')
        tension = array[3]
        distance = array[4]
        strain = array[5]

        tags.append(f'tension-std_{standard_deviation(tension)}')
        tags.append(f'tension-range_{value_range(tension)}')
        tags.append(f'tension-mc-density_{mean_cross_density(tension)}')
        tags.append(f'tension-mdc-density_{median_cross_density(tension)}')
        tags.append(f'tension-gradient_{gradient_change(tension)}')
        tags.append(f'tension-fft_{main_fft(tension)}')
        tags.append(f'tension-mean_{value_mean(tension)}')

        tags.append(f'distance-std_{standard_deviation(distance)}')
        tags.append(f'distance-range_{value_range(distance)}')
        tags.append(f'distance-mc-density_{mean_cross_density(distance)}')
        tags.append(f'distance-mdc-density_{median_cross_density(distance)}')
        tags.append(f'distance-gradient_{gradient_change(distance)}')
        tags.append(f'distance-fft_{main_fft(distance)}')
        tags.append(f'distance-mean_{value_mean(distance)}')

        tags.append(f'strain-std_{standard_deviation(strain)}')
        tags.append(f'strain-range_{value_range(strain)}')
        tags.append(f'strain-mc-density_{mean_cross_density(strain)}')
        tags.append(f'strain-mdc-density_{median_cross_density(strain)}')
        tags.append(f'strain-gradient_{gradient_change(strain)}')
        tags.append(f'strain-fft_{main_fft(strain)}')
        tags.append(f'strain-mean_{value_mean(strain)}')

        array.append(tags)

    return lists

#Balace tonality distribution, and augment samples in dataset
def change_keys(lists):

    summarize = [0]*24
    new_lists = lists.copy()
    random.shuffle(lists)
    for l in lists:
        summarize[l[6][0]] += 1
    print(summarize)
    for l in lists:
        for t in range(1, 12):
            num = (l[6][0]+t)%12 if l[6][0]<12 else (l[6][0]-12+t)%12+12
            if summarize[num] < 10000:
                new_melody_seg_number = [x+t if x+t<=120 else x-(12-t) for x in l[0]]
                new_key_seg_number = [(k+t)%12 if k<12 else (k-12+t)%12+12 for k in l[6]]
                new_lists.append([
                    new_melody_seg_number,
                    l[1],
                    l[2],
                    l[3],
                    l[4],
                    l[5],
                    new_key_seg_number,
                    l[7],
                ])
                summarize[num] += 1
    print(summarize)
    return new_lists

def get_translated_datas():
    train_arr = []
    evaluate_arr = []
    arrs = []
    invalide_num = 0

    path = 'Your Bach_Tension_Metadata folder path'
    file_names = os.listdir(path)
    for file in file_names:
        print('current data_file: ' + file)
        lists = get_curve_lists(path + '/' + file)
        if len(lists) != 0:
            #print(len(lists))
            arrs += make_tags(lists, file)
        else:
            invalide_num += 1
    arrs2 = change_keys(arrs)
    for l in arrs2:
        if random.random() <= 0.8:
            train_arr.append(l)
        else:
            evaluate_arr.append(l)

    np.save('Dataset_augment/dataset_train.npy', np.array(train_arr, dtype=object))
    np.save('Dataset_augment/dataset_evaluate.npy', np.array(evaluate_arr, dtype=object))

    print('train sample number: ', len(train_arr))
    print('evaluate sample number: ', len(evaluate_arr))
    print('invalide number: ', invalide_num)

    print('max_sigma: ', max_sigma)
    print('min_sigma: ', min_sigma)
    print('max_range: ', max_range)
    print('min_range: ', min_range)
    print('max_mc_density: ', max_mc_density)
    print('min_mc_density: ', min_mc_density)
    print('max_mdc_density: ', max_mdc_density)
    print('min_mdc_density: ', min_mdc_density)
    print('max_gradient: ', max_gradient)
    print('min_gradient: ', min_gradient)
    print('max_fft: ', max_fft)
    print('min_fft: ', min_fft)

get_translated_datas()