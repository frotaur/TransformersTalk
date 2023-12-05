import h5py , os, torch
from pathlib import Path
from tqdm import tqdm
from modules import tokenizer



toki = tokenizer.get_tokenizer(m_path='en_tokenizer',m_name='en')

def make_h5(pt_data_folder, destination_folder = None):
    if(destination_folder is None):
        destination_folder= Path(__file__).parent.as_posix()
    
    tarname = os.path.join(destination_folder,f'{os.path.basename(pt_data_folder)}.h5')
    os.makedirs(os.path.dirname(tarname),exist_ok=True)


    if(os.path.isdir(pt_data_folder)):
        with h5py.File(tarname, 'w') as f:
            dset = f.create_dataset("tokens", (0,), maxshape=(None,), dtype='int32')  # note the maxshape parameter
            
            current_index = 0
            for file in tqdm(os.listdir(pt_data_folder)):
                if os.path.splitext(file)[1]=='.pt':
                    pt_file = os.path.join(pt_data_folder,file)
                    tensor = torch.load(pt_file,map_location=torch.device('cpu'))
                    length = tensor.shape[1]
                    print('snippet', toki.detokenize(tensor[:,:40]))
                    # Resize the dataset to accommodate the new data
                    dset.resize((current_index + length,))
                    
                    # Add the new data to the dataset
                    dset[current_index:current_index+length] = tensor.numpy().squeeze()
                    
                    # Update the current ind
                    current_index += length
    else :
        raise ValueError(f'{pt_data_folder} not found')

make_h5('test_text/', destination_folder='.')