import random
import numpy as np

class Dataset:
    def __init__(self, token_ids, context_length, stride, eos_token):
        """Assumes masked attention"""
        self.context_length = context_length
        self.stride = stride
        self.inputs = []
        self.targets = []
        for i in range(0,len(token_ids),stride):
            if i+context_length+1 > len(token_ids):
                input_end = min(i+context_length, len(token_ids))
                input_token_ids = token_ids[i:input_end]
                input_token_ids.extend([eos_token] * (context_length - len(input_token_ids)))

                target_end = min(i+context_length+1, len(token_ids))
                target_token_ids = token_ids[i+1:target_end]
                target_token_ids.extend([eos_token] * (context_length - len(target_token_ids)))

                self.inputs.append(input_token_ids)
                self.targets.append(target_token_ids)
            else:
                input_token_ids = token_ids[i:i+context_length]
                target_token_ids = token_ids[i+1:i+context_length+1]
                self.inputs.append(input_token_ids)
                self.targets.append(target_token_ids)
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,i):
        return self.inputs[i],self.targets[i]

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.indices) // self.batch_size
        
    def __iter__(self):
        if self.shuffle:
            self.indices = random.sample(self.indices, len(self.indices))
            
        for i in range(0,len(self.indices),self.batch_size):
            if (i+self.batch_size) > len(self.indices):
                return # this is drop_last=True behavior
            batch_indices = self.indices[i:i+self.batch_size]
            inputs_tensor = np.array([self.dataset[j][0] for j in batch_indices])
            targets_tensor = np.array([self.dataset[j][1] for j in batch_indices])
            yield (inputs_tensor,targets_tensor)
            
