from typing import Literal
import torch

from TorchJaekwon.Util import UtilData, UtilAudio

class MedleySoloSet(torch.utils.data.Dataset):
    def __init__(
        self, 
        meta_data_dir_path:str = 'Data/Dataset/MedleySolosDBSplit', 
        subset_type:Literal['train', 'valid'] = 'train'
    ) -> None:
        super().__init__()
        self.meta_data = UtilData.pickle_load(f'{meta_data_dir_path}/{subset_type}.pkl')
        self.sample_rate:int = 44100    

    def __getitem__(self, index):
        audio, _ = UtilAudio.read(self.meta_data[index]['file_path'], self.sample_rate, mono = False)
        return audio

    def __len__(self):
        return len(self.meta_data)