import math
import numpy as np

# parameters
ToneType = ["G","D","A","E","B","Gb","Db","Ab","Eb","Bb","F","C","G","D","A","E","B","Gb","Db","Ab","Eb","Bb","F"]
weight = [0.536, 0.274, 0.19]  
h = 0.4 #0.365
radius = 1
alpha = 0.75
beta = 0.75
translate_Flat = {
    'C#':'Db',
    'D#':'Eb',
    'F#':'Gb',
    'G#':'Ab',
    'A#':'Bb',
}

# Basic class for object coordinates
class Vector3:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def distance(self, other):
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2) + math.pow(self.z - other.z, 2))
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __mul__(self, num):
        return Vector3(self.x * num, self.y * num, self.z * num)

# Pitch coordinate class, corresbouding to equation (1) in paper
class Tone:
    def __init__(self, num, name):
        self.position = Vector3(radius*math.sin(num*math.pi/2), num * h, -radius*math.cos(num*math.pi/2))
        self.name = name

# Chord coordinate class, corresbouding to euqation (2) in paper
class Chord:
    def __init__(self, toneList, _type, rootname):
        self.chordType = _type
        self.root = rootname
        self.toneList = toneList
        pos = Vector3(0,0,0)
        if self.chordType == 'major' or self.chordType == 'minor':
            for i in range(3):
                pos += self.toneList[i].position * weight[i]
        else:
            for i in range(len(self.toneList)):
                pos += self.toneList[i].position * (1 / len(self.toneList))
        self.position = pos

# Key coordinate class, corresbouding to equation (3) in paper
class Key:
    def __init__(self, chordList, _type, rootname):
        self.keyType = _type
        self.root = rootname
        self.chordList = chordList
        pos = Vector3(0,0,0)
        if self.keyType == 'major':
            for i in range(3):
                pos += self.chordList[i].position * weight[i]
        elif self.keyType == 'minor':
            pos += self.chordList[0].position * weight[0]
            pos += self.chordList[1].position * weight[1] * alpha
            pos += self.chordList[2].position * weight[1] * (1 - alpha)
            pos += self.chordList[3].position * weight[2] * beta
            pos += self.chordList[4].position * weight[2] * (1 - beta)
        else:
            for i in range(3):
                pos += self.chordList[i].position * weight[i]
        self.position = pos

# Make fundamental elements within Spiral Array
Tones = []
majorChords = []
minorChords = []
Chords = []
majorKeys = []
minorKeys = []
natural_minorKeys = []
harmonic_minorKeys = []
melodic_minorKeys = []
Keys = []
# Final point coordinates list computed from input chord list
midiObjectList = []

def initSpiralArray(): #must initiate Spiral Array using this function
    global Tones, majorChords, minorChords, Chords, majorKeys, minorKeys, natural_minorKeys, harmonic_minorKeys, melodic_minorKeys, Keys
    Tones = [Tone((j - 11), ToneType[j]) for j in range(len(ToneType))]
    for i in range(len(ToneType) - 4):
        majorChords.append(Chord([Tones[i], Tones[i + 1], Tones[i + 4]], 'major', ToneType[i]))
    for i in range(1, len(ToneType) - 4):
        minorChords.append(Chord([Tones[i + 3], Tones[i + 4], Tones[i]], 'minor', ToneType[i + 3]))
    for i in range(len(majorChords) - 2):
        majorKeys.append(Key([majorChords[i + 1], majorChords[i + 2], majorChords[i]], 'major', majorChords[i + 1].root))
    for i in range(len(minorChords) - 5):
        minorKeys.append(Key([minorChords[i + 1], majorChords[i + 6], minorChords[i + 2], minorChords[i], majorChords[i + 4]], 'minor', minorChords[i + 1].root))
    for i in range(len(minorChords) - 5):
        natural_minorKeys.append(Key([minorChords[i + 1], minorChords[i + 2], minorChords[i]], 'natural_minor', minorChords[i + 1].root))
    for i in range(len(minorChords) - 5):
        harmonic_minorKeys.append(Key([minorChords[i + 1], majorChords[i + 6], minorChords[i]], 'harmonic_minor', minorChords[i + 1].root))
    for i in range(len(minorChords) - 5):
        melodic_minorKeys.append(Key([minorChords[i + 1], majorChords[i + 6], majorChords[i + 4]], 'melodic_minor', minorChords[i + 1].root))
    Chords = majorChords + minorChords
    Keys = majorKeys + minorKeys + natural_minorKeys + harmonic_minorKeys + melodic_minorKeys

# Find best coordinate sets in one time step, corresbouding to 'f' function in Algorithm (1) in paper
def findBestChords(vectors):
    def generateCombinations(arrays):
        results = []
        _max = len(arrays) - 1
        def helper(arr, i):
            for j in range(len(arrays[i])):
                a = arr.copy()
                a.append(arrays[i][j])
                if i == _max:
                    results.append(a)
                else:
                    helper(a, i + 1)
        helper([], 0)
        return results
    allCombinations = generateCombinations(vectors)
    minDist = float('inf')
    bestCombinations = []
    for combination in allCombinations:
        totalDist = 0
        for i in range(len(combination)):
            for j in range(i + 1, len(combination)):
                totalDist += combination[i].position.distance(combination[j].position)
        totalDist = round(totalDist * 100000, 0) / 100000
        if totalDist < minDist:
            minDist = totalDist
            bestCombinations = [combination]
        elif totalDist == minDist:
            bestCombinations.append(combination)
    
    return bestCombinations

#Use beam search to get optimal coordinates, corrsbouding to last 'argmin' function in Algorithm (1) in paper
def best_chords_list_by_beam_search(arrays, beam_width):
    beams = [{'path': [array], 'cost': 0} for array in arrays[0]]

    for i in range(1, len(arrays)):
        new_beams = []
        for beam in beams:
            for obj in arrays[i]:
                new_path = beam['path'] + [obj]
                new_cost = beam['cost'] + beam['path'][-1].position.distance(obj.position)
                new_beams.append({'path': new_path, 'cost': new_cost})
        
        new_beams.sort(key=lambda x: x['cost'])
        beams = new_beams[:beam_width]
    
    beams.sort(key=lambda x: x['cost'])
    return beams[0]['path']

# Get key result for equation (8) in paper
def calculate_key_one(chords, tones_num):
    total = Vector3(0,0,0)
    count = 0
    for i in range(len(chords)):
        if isinstance(chords[i], Tone):
            for j in range(tones_num[i][0]):
                total += chords[i].position
                count += 1
        elif isinstance(chords[i], Chord):
            for j in range(len(chords[i].toneList)):
                for k in range(tones_num[i][j]):
                    total += chords[i].toneList[j].position
                    count += 1
    key_pos = total * (1 / count)
    minDistance = float('inf')
    closer_key_name = ''
    for k in Keys:
        dist = k.position.distance(key_pos)
        if dist < minDistance:
            minDistance = dist
            closer_key_name = f'{k.root} {k.keyType}'
    return closer_key_name, key_pos

#The main function that return pitch coordinate set with determined 'k' indices, corrsbouding to Algorithm (1) in paper
'''
Input chord list like [['A', 'Bb', 'F'], [...], [...]]
So far, the last two parameter are ignored.
'''
def analyze_chords_spiral(spiralChordList, key_detect_function='', threshold=0):
    global midiObjectList
    scale = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
    chords_list = []
    # Tranlate pitch labels for consistent representation
    for chord in spiralChordList:
        tone_list = []
        for tone in chord:
            if tone in translate_Flat:
                tone_list.append(translate_Flat[tone])
            else:
                tone_list.append(tone)
        chords_list.append(tone_list)
    midiObjectList_prepare = []
    tones_num = []
    #Main traverse for each time step
    for i in range(len(chords_list)):
        ifAllDouble = True
        tones = []
        num = []
        toneObjList = []
        for char in chords_list[i]:
            if char in scale:
                if char not in tones:
                    tones.append(char)
                    num.append(1)
                else:
                    num[tones.index(char)] += 1
        for j in range(len(tones)):
            toneObjs = list(filter(lambda x: x.name == tones[j], Tones))
            toneObjList.append(toneObjs)
            if ifAllDouble:
                ifAllDouble = False if len(toneObjs) == 1 else True
        if len(tones) > 1:
            current_chords = []
            currentChords = findBestChords(toneObjList)
            for c in range(len(currentChords)):
                stringName = ''
                for iname in range(len(currentChords[c])):
                    stringName += currentChords[c][iname].name + '|'
                current_chords.append(Chord(currentChords[c], stringName, ''))
            midiObjectList_prepare.append(current_chords)
            tones_num.append(num)
        elif len(tones) != 0:
            midiObjectList_prepare.append(toneObjList[0])
            tones_num.append(num)
    if len(midiObjectList_prepare) < 2:
        raise 'Error: less than two chords in Spiral Array.'
    midiObjectList = best_chords_list_by_beam_search(midiObjectList_prepare, 10)

# Calculate tension feature for equation (4) in paper
def chord_tension_spiral():
    diameters = []
    for chord_or_tone in midiObjectList:
        if isinstance(chord_or_tone, Tone):
            diameters.append(1.46969)
        elif isinstance(chord_or_tone, Chord):
            _max = 0
            for i in range(len(chord_or_tone.toneList)):
                for j in range(i + 1, len(chord_or_tone.toneList)):
                    distance = chord_or_tone.toneList[i].position.distance(chord_or_tone.toneList[j].position)
                    _max = distance if distance > _max else _max
            diameters.append(_max)
    return diameters

# Calculate distance feature for equation (5) in paper
def chord_distance_spiral():
    distances = []
    distances.append(0)
    for i in range(1, len(midiObjectList)):
        distances.append(midiObjectList[i].position.distance(midiObjectList[i-1].position))
    return distances

# Calculate strain feature for equation (7) in paper
def chord_tensile_strain_spiral(global_keys_string):
    tensile = []
    for i in range(len(global_keys_string)):
        key_objs = []
        # Find nearest key point for reason of multiple point with the same label exist
        for k in Keys:
            if f'{k.root} {k.keyType}' == global_keys_string[i]:
                key_objs.append(k)
        minDist = float('inf')
        for k in key_objs:
            distance = k.position.distance(midiObjectList[i].position)
            if distance < minDist:
                minDist = distance
                final_key = k
        tensile.append(minDist)
    return tensile