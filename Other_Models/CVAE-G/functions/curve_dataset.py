import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class EmotionCurveDataset(Dataset):

    def __init__(self, file_path):

        super(EmotionCurveDataset, self).__init__()
        self.file_path = file_path
        self.data = np.load(self.file_path, allow_pickle=True)

        #print(self.data[0], self.data[1])

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, index):
        
        #print(len(self.data[index]))
        #print(len(self.data[index][4]), len(self.data[index][0]), len(self.data[index][2]))
        input_list  = np.stack((
            [10*(x - 1.46969) / (4.62169 - 1.46969) for x in self.data[index][3]], #tension
            [10*(x / 3.56156) for x in self.data[index][4]],                         #distance
            [10*(x - 0.0586) / (3.08 - 0.0586) for x in self.data[index][5]],      #strain
            #self.data[index][3], #tension
            #self.data[index][4], #distance
            #self.data[index][5], #strain
            self.data[index][6]), axis=1)                                       #key
        condition_1_list = np.stack((
            self.data[index][0],                #melody
            self.data[index][2]), axis=1)       #melody weight

        return torch.tensor(input_list, dtype = torch.float32), torch.tensor(condition_1_list, dtype = torch.float32), index

    
    def get_sample_tags(self, index):
        
        tags = self.data[index][-1]
        
        return tags
    def get_sample_key(self, index):

        return self.data[index][6][0]

def pad_timeline_function(batch):

    inputs, cond1, indices = zip(*batch)

    input_lengths = [x.shape[0] for x in inputs]
    
    inputs_padded = pad_sequence(inputs, batch_first=True)
    cond1_padded = pad_sequence(cond1, batch_first=True)
    
    return {
    	'input': inputs_padded,
    	'melody': cond1_padded,
        'len': input_lengths,
        'index': indices,
    }
