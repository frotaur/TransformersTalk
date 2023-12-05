from torch.utils.data import Dataset
import torch,os, h5py

class TokenTexth5(Dataset):
    """
        Dataset used to store tokenized text. Produces tuples of text, and the text shifted by one
        token, to be used as input and target for language modelling. Uses memory mapping, with hdf5.

        If we notice that creation of the data is SLOW, we may use batched calls like I did the the cellular automata, to be seen.
        Args:
        text_location : location of the tokenized text tensor
        attn_length : size of the attention window
        stride : by how many tokens to stride to get the next example. Default is half the attention length.
    """

    def __init__(self,h5_file :str, attn_length:int, stride:int=None, backwards=False):
        self.h5_file = h5_file
        self.attn_length = attn_length

        self.backwards = backwards

        
        if(stride is None):
            self.stride=self.attn_length//2
        else :
            self.stride = stride

        if(not os.path.isfile(self.h5_file)):
            raise ValueError(f'File/Folder {self.h5_file} not found')
        
        self.h5_file = h5py.File(self.h5_file, 'r')
        self.text_tensor = self.h5_file['tokens']


        self.num_tokens = len(self.text_tensor)
        self.length = (self.num_tokens-self.attn_length-1)//(self.stride) # -1 because we need to have a target for each input
    
        print(f'Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length} examples.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            Returns a tuple of (input, target) tensors, each of shape (attn_length)

            For now, when backwards, we still give the examples in the 'forward' way, but
            we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """
        if(self.backwards):
            true_idx = self.stride*(idx)+self.attn_length+1 # We still start from the front

            return torch.tensor(self.text_tensor[true_idx-self.attn_length:true_idx],dtype=torch.long).flip(dims=(0,)), \
                torch.tensor(self.text_tensor[true_idx-self.attn_length-1:true_idx-1],dtype=torch.long).flip(dims=(0,))
        else :
            true_idx = idx*self.stride
            return torch.tensor(self.text_tensor[true_idx:true_idx+self.attn_length],dtype=torch.long), \
            torch.tensor(self.text_tensor[true_idx+1:true_idx+self.attn_length+1],dtype=torch.long)