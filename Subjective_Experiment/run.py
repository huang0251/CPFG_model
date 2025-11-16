import os
import numpy as np
import music21
import math
import spiralArray as sp
from statistical_test import plot_bar_with_significance

# Uncompose all sampled for fair comparision
def uncompose_chords(rawpath):

    def uncompose(path, file):

        midi = music21.converter.parse(path + '/' + file)
        chords = midi.flat.getElementsByClass('Chord')
        offsets = []
        chord_list = []
        lens = []
        for chord in chords:
            notes = []
            octaves = []
            for note in chord.pitches:
                n = note.name
                if len(n) == 2 and n[1] == '-':
                    n = n[0] + 'b'
                notes.append(n)
                octaves.append(note.octave)
            if len(notes) > 1:
                chord_list.append(notes)
                offsets.append(chord.offset)
                lens.append(chord.quarterLength)

        score = music21.stream.Score()
        harmony_part = music21.stream.Part()
        harmony_part.partName = 'CHORDS'
        for i in range(len(chord_list)):
            chord = music21.chord.Chord(chord_list[i])
            #chord.inversion(0)
            chord.quarterLength = lens[i]
            harmony_part.insert(offsets[i], chord)
        score.insert(0, harmony_part)
        score.write('midi', path + '_uncomposed/' + file)

    file_names = os.listdir(rawpath)
    for file in file_names:
        uncompose(rawpath, file)

def get_chords_chordify(chordifypath):

    def chordifies(path):
        midi = music21.converter.parse(path)
        midi.chordify().write('midi', fp=path)

    file_names = os.listdir(chordifypath)
    for file in file_names:
        chordifies(chordifypath + '/' + file)

# Calculate MCTD metric for one sample in Table 4 in paper.
def calculate_mctd(melody_file, chord_file):

    chord_midi = music21.converter.parse(chord_file)
    chords = chord_midi.flat.getElementsByClass('Chord')

    melody_midi = music21.converter.parse(melody_file)
    notes = melody_midi.flat.getElementsByClass('Note')

    total_distance = []

    for i in range(len(chords)):
        if i != 0 and chords[i].pitchedCommonName == chords[i-1].pitchedCommonName:
            continue
        for j in range(len(notes)):
            if chords[i].offset >= notes[j].offset:

                pitches = []
                for note in chords[i].pitches:
                    n = note.name
                    if len(n) == 2 and n[1] == '-':
                        n = n[0] + 'b'
                    pitches.append(n)
                m_name = notes[j].name
                if len(m_name) == 2 and m_name[1] == '-':
                        m_name = m_name[0] + 'b'

                distance = sp.single_distance(pitches, [m_name])
                total_distance.append(distance)
                break

    return np.mean(np.array(total_distance))

def calculate_chord_num_type(path):

    def calculate_che(num, cc_dict):

        total = 0
        for key, value in cc_dict.items():
            p = value / num
            v = p * math.log(p)
            total += v
        return total

    midi = music21.converter.parse(path)
    chords = midi.flat.getElementsByClass('Chord')
    nums = 0
    cach = {}
    for i in range(len(chords)):
        if i != 0 and chords[i].pitchedCommonName == chords[i-1].pitchedCommonName:
            continue
        nums += 1
        if chords[i].commonName not in cach:
            cach[f'{chords[i].commonName}'] = 1
        else:
            cach[f'{chords[i].commonName}'] += 1
    #print(nums, len(cach))
    return nums, len(cach), calculate_che(nums, cach)

# Calculate Mean CC and CHE metrics for one sample in Table 4 in paper.
def mean_chord_coverage():

    model_name = ['ground_truth', 'coconet', 'deepbach', 'CPFG', 'noise']
    pathes = ['D:/DeepLearning/DL_project/Music-research-project/CPFG_melody_harmonization/Subjective_Eval/' + p + '_uncomposed' for p in model_name]
    cc_values = [[] for v in model_name]
    che_values = [[] for v in model_name]

    for p in range(len(pathes)):
        file_names = os.listdir(pathes[p])
        for file in file_names:
            c_num, c_type, che = calculate_chord_num_type(pathes[p] + '/' + file)
            cc_values[p].append(c_type/c_num)
            che_values[p].append(che)

    print('CC: ')
    for v in range(len(cc_values)):
        std = np.std(np.array(cc_values[v]))
        mean = np.mean(np.array(cc_values[v]))
        ci = 1.96*(std/len(cc_values[v]))
        print(mean, std, ci)
    print('CHE: ')
    for v in range(len(che_values)):
        std = np.std(np.array(che_values[v]))
        mean = np.mean(np.array(che_values[v]))
        ci = 1.96*(std/len(che_values[v]))
        print(mean, std, ci)

# Calculate MCTD metric for one sample in Table 4 in paper.
def mean_mctd():

    sp.initSpiralArray()
    model_name = ['ground_truth', 'coconet', 'deepbach', 'CPFG', 'noise']
    pathes = ['D:/DeepLearning/DL_project/Music-research-project/CPFG_melody_harmonization/Subjective_Eval/' + p + '_uncomposed' for p in model_name]
    melodies_path = 'D:/DeepLearning/DL_project/Music-research-project/CPFG_melody_harmonization/Subjective_Eval/2_melodies'
    values = [[] for v in model_name]

    for p in range(len(pathes)):
        file_names = os.listdir(pathes[p])
        for file in file_names:
            mctd = calculate_mctd(melodies_path + '/' + file, pathes[p] + '/' + file)
            values[p].append(mctd)
    print('MCTD: ')
    for v in range(len(values)):
        std = np.std(np.array(values[v]))
        mean = np.mean(np.array(values[v]))
        ci = 1.96*(std/len(values[v]))
        print(mean, std, ci)

#uncompose_chords('D:/DeepLearning/DL_project/Music-research-project/CPFG_melody_harmonization/Subjective_Eval/noise')
#get_chords_chordify('D:/DeepLearning/DL_project/Music-research-project/CPFG_melody_harmonization/Subjective_Eval/deepbach')

mean_chord_coverage()
mean_mctd()
plot_bar_with_significance(df, metric_name='CN')
plot_bar_with_significance(df, metric_name='HC')