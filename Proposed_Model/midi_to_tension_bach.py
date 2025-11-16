import os
import numpy as np
import csv
import music21
from functions.spiralArray import initSpiralArray, analyze_chords_spiral, chord_tension_spiral, chord_distance_spiral, chord_tensile_strain_spiral
import random

seed=77
random.seed(seed)
np.random.seed(seed)

#Adopt consistent pitch representation label
translate_Flat = {
    'C#':'Db',
    'D#':'Eb',
    'F#':'Gb',
    'G#':'Ab',
    'A#':'Bb',
}

#Standard pitch offsets
def fix_offsets(offsets): #threshold: 0.0 - 1.0
    new_offsets = []
    for offset in offsets:
        int_part = int(offset)
        decimal_part = offset - int_part
        targets = [0.0, 0.25, 0.5, 0.75, 1.0]
        rounded_decimal = min(targets, key=lambda t: abs(decimal_part - t))
        new_offsets.append(int_part + rounded_decimal)
    return new_offsets

#Detect key labels using Music21 module
def music21_analyze_key(score):
    
    key = score.analyze('key')
    _key = str(key).split('minor')[0].split('major')[0][:-1]
    _key = _key.upper()
    if _key[-1] == '-':
        _key = _key[:-1] + 'b'
    if _key in translate_Flat:
        _key = translate_Flat[_key]
    if 'minor' in str(key):
        _key = _key + ' minor'
    elif 'major' in str(key):
        _key = _key  + ' major'
    spiral_key_name = _key
    
    return spiral_key_name

#Get block chords
def get_chords_chordify(score):

    chords = score.chordify().flat.getElementsByClass('Chord')
    offsets = []
    chord_list = []
    octave_list = []
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
            octave_list.append(octaves)
            offsets.append(chord.offset)
    #print(list(zip(chord_list, offsets)))
    return chord_list, octave_list, fix_offsets(offsets)

#Get beats
def get_beat_input(track):

    timeSig = None
    beat_weights = []
    beat_offsets = []
    measures = track.getElementsByClass('Measure')
    for m in measures:
        if(m.hasElementOfClass('TimeSignature')):
            timeSig = m.getElementsByClass('TimeSignature')[0]#.ratioString
            #print(timeSig.ratioString)
        num_beats = int(m.quarterLength / (4 / int(timeSig.ratioString.split("/")[1])))
        #print(m.quarterLength, timeSig.ratioString.split("/")[1])
        beats = [i * (4 / int(timeSig.ratioString.split("/")[1])) + m.offset for i in range(num_beats)]
        weights = [1]
        weights += [0 for i in range(num_beats-1)]
        #print(weights)
        beat_weights += weights
        beat_offsets += beats

    return beat_weights, fix_offsets(beat_offsets)

#Get melodys
def get_melodys_input(score):

    melodys = []
    melodys_offsets = []
    track_names = []
    for part in score.parts:
        melody_pitch_with_octave = []
        melody_offset = []
        for el in part.flat.notes:
            if el.isNote:
                melody_pitch_with_octave.append(el.pitch.nameWithOctave)
            else:
                melody_pitch_with_octave.append(el.pitches[-1].nameWithOctave)
            melody_offset.append(float(el.offset))
        melodys.append(melody_pitch_with_octave)
        melodys_offsets.append(fix_offsets(melody_offset))
        track_names.append(part.partName)

    return melodys, melodys_offsets, track_names

#Get tonalities
def get_key_input(score):

    keys = {'None': 0}
    for part in score.parts:
        if part.flat.hasElementOfClass('Key'):
            #print('key', str(key))
            key = part.flat.getElementsByClass('Key')[0]
            _key = str(key).split('minor')[0].split('major')[0][:-1]
            _key = _key.upper()
            if _key[-1] == '-':
                _key = _key[:-1] + 'b'
            if _key in translate_Flat:
                _key = translate_Flat[_key]
            if 'minor' in str(key):
                _key = _key + ' minor'
            elif 'major' in str(key):
                _key = _key  + ' major'
            if _key in keys:
                keys[_key] += 1
            else:
                keys[_key] = 1
    result_key = None
    max_value = 0
    for k, v in keys.items():
        if v >= max_value:
            result_key = k
    #print(result_key)
    return result_key

#Main function to calculate tension from one bach file
def get_curves(file_path):

    bach_score = music21.corpus.parse(file_path)
    #bach_score.show('text')
    for part in bach_score.parts:
        #print(part.partName)
        if part.partName and 'Soprano' not in part.partName and 'Alto' not in part.partName and 'Tenor' not in part.partName and 'Bass' not in part.partName:
            bach_score.remove(part)
    melodys, melodys_offsets, track_names = get_melodys_input(bach_score)

    key_name = get_key_input(bach_score)
    if key_name == None or key_name == 'None':
        print('use music21 key this sample')
        try:
            key_name = music21_analyze_key(bach_score)
        except:
            print('detect key false')
            return None, None

    beat_weights = []
    beat_offsets = []
    for part in bach_score.parts:
        if part.partName and 'Bass' in part.partName:
            beat_weights, beat_offsets = get_beat_input(part)
            break

    chords_pitch_list, chords_octave_list, chords_offset_list = get_chords_chordify(bach_score)

    for i in range(len(beat_offsets)):
        if beat_weights[i] == 1 and beat_offsets[i] not in chords_offset_list:
            for j in range(len(chords_offset_list)):
                if (j != len(chords_offset_list)-1 and chords_offset_list[j]<beat_offsets[i] and chords_offset_list[j+1]>beat_offsets[i])\
                    or (j == len(chords_offset_list)-1 and chords_offset_list[j]<beat_offsets[i]):
                    chords_offset_list.insert(j+1, beat_offsets[i])
                    chords_pitch_list.insert(j+1, chords_pitch_list[j])
                    chords_octave_list.insert(j+1, chords_octave_list[j])
                    break
    max_offset_start = 0
    for melody, offset in zip(melodys, melodys_offsets):
        if offset[0] >= max_offset_start:
            max_offset_start = offset[0]
        for i in range(len(chords_offset_list)):
            if chords_offset_list[i] not in offset:
                for j in range(len(offset)):
                    if (j != len(offset)-1 and offset[j]<chords_offset_list[i] and offset[j+1]>chords_offset_list[i])\
                        or (j == len(offset)-1 and offset[j]<chords_offset_list[i]):
                        offset.insert(j+1, chords_offset_list[i])
                        melody.insert(j+1, melody[j])
                        
    if len(chords_offset_list) < 2:
        return None, None

    for melody, offset in zip(melodys, melodys_offsets):
        index = 0
        for j in range(len(offset)):
            if offset[j] >= max_offset_start:
                index = j
                break
        del offset[0:index]
        del melody[0:index]
    del chords_offset_list[0:index]
    del chords_pitch_list[0:index]
    del chords_octave_list[0:index]

    if not (melodys_offsets[0][-1] == melodys_offsets[1][-1] == melodys_offsets[2][-1] == chords_offset_list[-1]):
        raise 'not same long'

    melody_weights = [[0] * len(row) for row in melodys]
    for weight, offset in zip(melody_weights, melodys_offsets):
        for i in range(len(offset)):
            for j in range(len(beat_offsets)):
                if offset[i] == beat_offsets[j]:
                    weight[i] = beat_weights[j]
    

    analyze_chords_spiral(spiralChordList = chords_pitch_list, key_detect_function = '')
    global_keys_string = []
    for i in range(len(chords_offset_list)):
        global_keys_string.append(key_name)

    spiral_tension = chord_tension_spiral()
    spiral_distance = chord_distance_spiral()
    spiral_tensile = chord_tensile_strain_spiral(global_keys_string)
    
    #print(list(zip(beat_offsets, beat_weights)))
    dic = {}
    for t in range(len(track_names)):
        dic[f'{track_names[t]}_melody'] = melodys[t]
        dic[f'{track_names[t]}_melody_offset'] = melodys_offsets[t]
        dic[f'{track_names[t]}_melody_weight'] = melody_weights[t]
    dic['chord_pitches'] = chords_pitch_list
    dic['chord_octaves'] = chords_octave_list
    dic['chord_offset'] = chords_offset_list
    dic['key'] = global_keys_string
    dic['tension_value'] = [round(v, 5) for v in spiral_tension]
    dic['distance_value'] = [round(v, 5) for v in spiral_distance]
    dic['strain_value'] = [round(v, 5) for v in spiral_tensile]
    return dic, bach_score

#Loop function to get all metadatas
def calculate_curves():

    initSpiralArray()
    bach_works = music21.corpus.getComposer('bach')
    for work in bach_works:
        file_name = os.path.splitext(os.path.basename(work))[0]
        if 'riemenschneider' in file_name:
            continue
        print('current process: ' + file_name)
        datas, new_score = get_curves(work)
        if datas == None:
            continue
        keys = datas.keys()
        rows = list(zip(*datas.values()))
        #results.insert(0, [])
        with open('Bach_Tension_Metadata folder path that you want save metadatas/' + file_name + '_datas.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(keys)
            writer.writerows(rows)

#The test function
def calculate_curves_once():

    initSpiralArray()
    bach_works = music21.corpus.getComposer('bach')
    work = random.choice(bach_works)
    work = 'D:/Download/anaconda3/Lib/site-packages/music21/corpus/bach/bwv289.mxl'
    print(work)
    datas, new_score = get_curves(work)
    #print(results)
    if datas == None:
        return 0
    keys = datas.keys()
    rows = list(zip(*datas.values()))
    #with open('datas.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(keys)
    #    writer.writerows(rows)
    new_score.write("midi", 'new_bach.mid')

calculate_curves()
#calculate_curves_once()