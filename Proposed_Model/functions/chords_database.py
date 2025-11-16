from enum import Enum

#Chords Library for chords recovery

SCALE = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']

class Chord_Types(Enum):
    note = 1
    interval_3_5 = 2 # C - E,Eb,G,A,Ab,F
    interval_2_4_6 = 3 # C - D,Bb
    interval_7_tritone = 4 # C - Db,Fb,B
    default_3 = 5
    dim_aug_3 = 6
    default_7 = 7 # dominant_7, Major_7, minor_7
    sus = 8
    dim_7 = 9
    other_triad = 10
    default_9 = 11 # Major_9, minor_9, dominant_9, 
    triad_add = 12
    seventh_add = 13

Simple_Triads = [1, 2, 5]
Common_Triads = [1, 2, 3, 4, 5, 6, 8]
Simple_Seventh = [1, 2, 3, 4, 5, 6, 7]
Common_Seventh = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Complex_Triads = [1, 2, 3, 4, 5, 6, 8, 10]
Complex_Seventh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
Complex_All = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
Custom_types = []

# Initiate chord library (almost 1600 types)
def generate_chords_with_types(types):

    chords = []

    if Chord_Types.note.value in types:
        for c in SCALE:
            chords.append([c])

    if Chord_Types.interval_3_5.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 5) % 12]]))

    if Chord_Types.interval_2_4_6.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 2) % 12]]))

    if Chord_Types.interval_7_tritone.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 1) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 6) % 12]]))

    if Chord_Types.default_3.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12], SCALE[(i + 7) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12], SCALE[(i + 7) % 12]]))

    if Chord_Types.dim_aug_3.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12], SCALE[(i + 6) % 12]]))
        for i in range(len(SCALE) - 4):
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12], SCALE[(i + 8) % 12]]))

    if Chord_Types.default_7.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12], SCALE[(i + 7) % 12], SCALE[(i + 11) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12], SCALE[(i + 7) % 12], SCALE[(i + 10) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12], SCALE[(i + 7) % 12], SCALE[(i + 10) % 12]]))

    if Chord_Types.sus.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 2) % 12], SCALE[(i + 7) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 5) % 12], SCALE[(i + 7) % 12]]))
        if Chord_Types.default_7.value in types:
            for i in range(len(SCALE)):
                chords.append(sorted([SCALE[i], SCALE[(i + 2) % 12], SCALE[(i + 7) % 12], SCALE[(i + 10) % 12]]))
                chords.append(sorted([SCALE[i], SCALE[(i + 2) % 12], SCALE[(i + 7) % 12], SCALE[(i + 11) % 12]]))
                chords.append(sorted([SCALE[i], SCALE[(i + 5) % 12], SCALE[(i + 7) % 12], SCALE[(i + 10) % 12]]))
                chords.append(sorted([SCALE[i], SCALE[(i + 5) % 12], SCALE[(i + 7) % 12], SCALE[(i + 11) % 12]]))

    if Chord_Types.dim_7.value in types:
        for i in range(len(SCALE) - 9):
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12], SCALE[(i + 6) % 12], SCALE[(i + 9) % 12]]))
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12], SCALE[(i + 6) % 12], SCALE[(i + 10) % 12]]))

    if Chord_Types.default_9.value in types:
        for i in range(len(SCALE)):
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12], SCALE[(i + 7) % 12], SCALE[(i + 11) % 12], SCALE[(i + 14) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 4) % 12], SCALE[(i + 7) % 12], SCALE[(i + 10) % 12], SCALE[(i + 14) % 12]]))
            chords.append(sorted([SCALE[i], SCALE[(i + 3) % 12], SCALE[(i + 7) % 12], SCALE[(i + 10) % 12], SCALE[(i + 14) % 12]]))

    if Chord_Types.other_triad.value in types:
        for c1 in range(len(SCALE)):
            for c2 in range(c1+1, len(SCALE)+c1-1):
                for c3 in range(c2+1, len(SCALE)+c1):
                    chord = sorted([SCALE[c1], SCALE[c2 % 12], SCALE[c3 % 12]])
                    if Chord_Types.interval_7_tritone.value not in types:
                        indices = [c1, c2%12, c3%12]
                        if any(abs(indices[i] - indices[j]) in {1, 6, 11} for i in range(3) for j in range(i+1, 3)):
                            continue
                    if chord not in chords:
                        chords.append(chord)

    if Chord_Types.triad_add.value in types:
        triad_chords = [chords[i] for i in range(len(chords)) if len(chords[i]) == 3]
        for triad_chord in triad_chords:
            for new_c in SCALE:
                if new_c not in triad_chord:
                    chord = triad_chord.copy()
                    chord.append(new_c)
                    chord = sorted(chord)
                    if Chord_Types.interval_7_tritone.value not in types:
                        indices = [SCALE.index(c) for c in chord]
                        if any(abs(indices[i] - indices[j]) in {1, 6, 11} for i in range(4) for j in range(i+1, 4)):
                            continue
                    if chord not in chords:
                        chords.append(chord)

    if Chord_Types.seventh_add.value in types:
        seven_chords = [chords[i] for i in range(len(chords)) if len(chords[i]) == 4]
        for seven_chord in seven_chords:
            for new_c in SCALE:
                if new_c not in seven_chord:
                    chord = seven_chord.copy()
                    chord.append(new_c)
                    chord = sorted(chord)
                    if Chord_Types.interval_7_tritone.value not in types:
                        indices = [SCALE.index(c) for c in chord]
                        if any(abs(indices[i] - indices[j]) in {1, 6, 11} for i in range(5) for j in range(i+1, 5)):
                            continue
                    if chord not in chords:
                        chords.append(chord)

    #print(chords, len(chords))
    return chords

#cs = generate_chords_with_types(Complex_All)
#print(len(cs))