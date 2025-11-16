import functions.chords_database as cd
from functions.spiralArray import initSpiralArray, analyze_chords_spiral, chord_tension_spiral, chord_distance_spiral, chord_tensile_strain_spiral
import music21

chords = []
# Max deviation can be received, larger value will cost more memory. Default 20 is better.
departure = 50

#Initiate script before use
def initChordFinder(chords_type=cd.Common_Seventh, departure_threshold=50):

    global chords, departure, acceptable
    chords = cd.generate_chords_with_types(chords_type)
    departure = departure_threshold
    initSpiralArray()

def get_pair_data(chord_pair, current_key, next_key):

    analyze_chords_spiral(spiralChordList = chord_pair, key_detect_function = '')
    spiral_tension = chord_tension_spiral()
    spiral_distance = chord_distance_spiral()
    spiral_tensile = chord_tensile_strain_spiral([current_key, next_key])

    return spiral_tension, spiral_distance, spiral_tensile

#Autogressive find next step in progress of equation (10) in paper
def get_next_chord(next_tension, next_key, next_tensile, distance, current_chord=None, current_key=None):

    candidate_chords = []
    for i in range(len(chords)):
        pair = [current_chord if current_chord != None else chords[i], chords[i]]
        i_tension, i_distance, i_tensile = get_pair_data(pair, current_key if current_key != None else next_key, next_key)
        tension_deviation = abs(next_tension - i_tension[1])
        distance_deviation = abs(distance - i_distance[1])
        tensile_deviation = abs(next_tensile - i_tensile[1])

        current_departure = (100*tension_deviation + 100*distance_deviation + 100*tensile_deviation) / 3
        if current_departure <= departure:
            candidate_chords.append({'chord': chords[i], 'departure': current_departure})

    if len(candidate_chords) == 0:
        raise 'find chord error'
    # Auto sort chord candidate by their deviation
    candidate_chords.sort(key=lambda x: x['departure'])
    print(candidate_chords[0], '\n')
    #result = candidate_chords[0]['chord']
    #print(result)

    #Also return the statistic deviation for equation (18) in paper
    return candidate_chords[0]['chord'], candidate_chords[0]['departure']

# Tests
#initChordFinder()
#get_next_chord(next_tension=3.1241, next_key='C major', next_tensile=0.88398, distance=0.96695, current_chord=['F', 'C', 'A', 'E'], current_key='C major')
#get_next_chord(next_tension=1.4, next_key='C major', next_tensile=0.5, distance=0.0)