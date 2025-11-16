import music21
import functions.chordFinder as cf
import functions.chords_database as cd

#Generate piano roll from tension


cf.initChordFinder(cd.Complex_Seventh, departure_threshold=100)
def fix_offsets(offsets): #threshold: 0.0 - 1.0
    new_offsets = []
    for offset in offsets:
        int_part = int(offset)
        decimal_part = offset - int_part
        targets = [0.0, 0.25, 0.5, 0.75, 1.0]
        rounded_decimal = min(targets, key=lambda t: abs(decimal_part - t))
        new_offsets.append(int_part + rounded_decimal)
    return new_offsets

def get_melody_input(midifile):
    
    midi = music21.midi.MidiFile()
    midi.open(midifile)
    midi.read()
    midi.close()
    midiScore = music21.midi.translate.midiFileToStream(midi, quantizePost=False)
    #midiScore_quantized = music21.midi.translate.midiFileToStream(midi)

    melody_pitch_with_octave = []
    melody_offset = []
    melody_offset_quantized = []
    if len(midiScore.parts) == 1:
        track = midiScore.parts[0]
    else:
        track = list(filter(lambda x: x.partName == 'MELODY' or x.partName == 'Melody' or x.partName == 'melody', midiScore.parts))[0]
    #print(track.partName)
    for el in track.notes:
        if el.isNote:
            melody_pitch_with_octave.append(el.pitch.nameWithOctave)
        else:
            melody_pitch_with_octave.append(el.pitches[-1].nameWithOctave)
        melody_offset.append(format(float(el.offset), '.2f'))
    #_track = list(filter(lambda x: x.partName == track_name, midiScore_quantized.parts))[0]
    for el in track.notes:
        melody_offset_quantized.append(float(el.offset))

    melody_pitches = []
    table = {
        'C': 0,
        'D': 2,
        'E': 4,
        'F': 5,
        'G': 7,
        'A': 9,
        'B': 11,
    }
    for string in melody_pitch_with_octave:
        n = int(string[-1]) * 12 + table[string[0]]
        if '-' in string:
            n -= 1
        elif '#' in string:
            n += 1
        melody_pitches.append(n)

    return melody_pitches, melody_offset, fix_offsets(melody_offset_quantized)

def get_beat_input(midifile):

    score = music21.converter.parse(midifile)
    new_score = score.makeNotation()

    if len(new_score.parts) == 1:
        track = new_score.parts[0]
    else:
        track = list(filter(lambda x: x.partName == 'MELODY' or x.partName == 'Melody' or x.partName == 'melody', new_score.parts))[0]

    timeSig = ''
    measure_dur = 0
    beat_weights = []
    beat_offsets = []
    first_beat_measure = False
    measures = track.getElementsByClass('Measure')
    for m in measures:
        if(m.measureNumber == 1 and m.hasElementOfClass('TimeSignature')):
            timeSig = m.getElementsByClass('TimeSignature')[0].ratioString
            print('timeSig: ', timeSig)
            measure_dur = m.quarterLength
        if len(m.flat.notes) != 0:
            offset = fix_offsets([m.flat.notes[0].offset])[0]
            if offset == 0.0:
                first_beat_measure = True
            break
    for m in measures:
        if len(m.flat.notes) == 0 and len(beat_offsets) == 0:
            continue
        if len(beat_offsets) == 0:
            if first_beat_measure:
                for i in range(int(timeSig[0])):
                    beat_offsets.append(m.offset + i * measure_dur / int(timeSig[0]))
                continue
            else:
                beat_offsets.append(0.0)
                continue

        for i in range(int(timeSig[0])):
            beat_offsets.append(m.offset + i * measure_dur / int(timeSig[0]))

    if not first_beat_measure:
        del beat_offsets[0]

    if int(timeSig[0])%3 == 0:
        for i in range(len(beat_offsets)):
            if i%6 == 0:
                beat_weights.append(4)
            elif i%6 in {1, 2, 4, 5}:
                beat_weights.append(1)
            elif i%6 == 3:
                beat_weights.append(2)
    elif int(timeSig[0])%2 == 0:
        for i in range(len(beat_offsets)):
            if i%4 == 0:
                beat_weights.append(4)
            elif i%4 in {1, 3}:
                beat_weights.append(1)
            elif i%4 == 2:
                beat_weights.append(2)

    return beat_weights, fix_offsets(beat_offsets)

def get_net_input(midifile):

    melody_pitches, _, melody_offsets = get_melody_input(midifile)
    beat_weights, beat_offsets = get_beat_input(midifile)

    melody_offset_list = []
    melody_pitch_list = []
    for i in range(len(melody_offsets)):
        if melody_offsets[i] >= beat_offsets[0]:
            melody_offset_list.append(melody_offsets[i])
            melody_pitch_list.append(melody_pitches[i])
    melody_weight_list = [0] * len(melody_pitch_list)
    for i in range(len(melody_offset_list)):
            for j in range(len(beat_offsets)):
                if melody_offset_list[i] == beat_offsets[j]:
                    melody_weight_list[i] = beat_weights[j]

    return melody_pitch_list, melody_weight_list

def make_midi(melody, melody_weight, tension, distance, strain, offset_len, key, destination_file_path):

    cost = 0
    table = {
        0: 'C',
        1: 'Db',
        2: 'D',
        3: 'Eb',
        4: 'E',
        5: 'F',
        6: 'Gb',
        7: 'G',
        8: 'Ab',
        9: 'A',
        10: 'Bb',
        11: 'B',
    }
    key_sig = []
    for k in key:
        if k >= 12:
            key_sig.append(f'{table[k-12]} minor')
        else:
            key_sig.append(f'{table[k]} major')
    melody_sig = []
    for m in melody:
        melody_sig.append(f'{table[m%12]}{m//12}')
    chords = []
    onset_chord, departure = cf.get_next_chord(tension[0], key_sig[0], strain[0], distance[0])
    cost += departure
    chords.append(onset_chord)
    for i in range(1, len(tension)):
        one_chord, departure = cf.get_next_chord(tension[i], key_sig[i], strain[i], distance[i], chords[-1], key_sig[i-1])
        chords.append(one_chord)
        cost += departure
    score = music21.stream.Score()
    melody_part = music21.stream.Part()
    melody_part.partName = 'MELODY'
    harmony_part = music21.stream.Part()
    harmony_part.partName = 'CHORDS'
    for i in range(len(melody_sig)):
        note = music21.note.Note(melody_sig[i])
        note.quarterLength = offset_len[i]
        if i == 0:
            melody_part.insert(0, note)
        else:
            melody_part.insert(sum(offset_len[:i]), note)
    for i in range(len(chords)):
        chord = music21.chord.Chord(chords[i])
        #chord.inversion(0)
        chord.quarterLength = offset_len[i]
        if i == 0:
            harmony_part.insert(0, chord)
        else:
            harmony_part.insert(sum(offset_len[:i]), chord)
    score.insert(0, melody_part)
    score.insert(0, harmony_part)
    score.write('midi', destination_file_path)

    return cost