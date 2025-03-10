from typing import Optional, Tuple, Union
from torch import Tensor

import torch

from TorchJaekwon.Util import UtilData, UtilAudio
from TorchJaekwon.Model.FlowMatching.FlowMatching import FlowMatching

from Model.StableAudioOpen.AutoencoderPretransform import AutoencoderPretransform
from Model.StableAudioOpen.CLAP.CLAPConditioner import CLAPConditioner

class InstFlow(FlowMatching):
    def __init__(
        self, 
        autoencoder_ckpt_path:str = 'CKPT/autoencoder.pth', 
        clap_ckpt_path:str = 'CKPT/music_speech_audioset_epoch_15_esc_89.98.pt',
        sample_rate:int = 44100,
        **kwargs 
    ) -> None:
        super().__init__(**kwargs)
        self.audio_length:int = 131072
        self.autoencoder_dim:int = 64
        self.autoencoder_downsampling_ratio:int = 2048
        self.sample_rate:int = sample_rate
        # Autoencoder
        self.autoencoder: AutoencoderPretransform
        self.__dict__["autoencoder"] = AutoencoderPretransform()
        autoencoder_ckpt = torch.load(autoencoder_ckpt_path, map_location='cpu')
        self.autoencoder.load_state_dict(autoencoder_ckpt)
        # CLAP
        self.clap = CLAPConditioner(output_dim=self.model.transformer.dim, clap_ckpt_path = clap_ckpt_path)
    
    def to(self, device:torch.device) -> None:
        super().to(device)
        self.autoencoder.to(device)
        self.clap.model.to(device)
        
    def preprocess(
        self,
        x_start:Tensor, 
        cond:dict = None, # {'audio': [batch, 2, self.audio_length], 'text': List[str]}
    ) -> Tuple[Tensor,Tensor]: 
        if x_start is not None:
            x_start = UtilData.fix_length(x_start, self.audio_length)
            with torch.no_grad():
                z:Tensor = self.autoencoder.encode(x_start)
        else:
            z = None
        
        if 'audio' in cond:
            clap_sample_rate:int = self.clap.model.model_cfg['audio_cfg']['sample_rate']
            clap_audio = UtilAudio.resample_audio(cond['audio'], self.sample_rate, clap_sample_rate)
            cond = {'global_embed': self.clap(clap_audio, type='audio')}
        elif 'text' in cond:
            cond = {'global_embed': self.clap(cond['text'], type='text')}
        additional_data_dict = None
        return z, cond, additional_data_dict
        
    def postprocess(
        self, 
        x: Tensor, #[batch_size, self.autoencoder_dim, self.audio_length // self.autoencoder_downsampling_ratio]
        additional_data_dict
    ) -> Tensor:
        with torch.no_grad():
            pred_audio:Tensor = self.autoencoder.decode(x)
        return pred_audio
    
    def get_x_shape(self, cond) -> tuple:
        batch_size:int = len(cond[list(cond.keys())[0]])
        return (batch_size, self.autoencoder_dim, self.audio_length // self.autoencoder_downsampling_ratio)
    
    def get_unconditional_condition(self,
                                    cond:Optional[Union[dict,Tensor]] = None, 
                                    cond_shape:Optional[tuple] = None,
                                    condition_device:Optional[torch.device] = None
                                    ) -> dict:
        batch_size:int = len(cond[list(cond.keys())[0]])
        null_embed:Tensor = self.clap([''], type='text')
        return {'global_embed': null_embed.repeat(batch_size, 1)}

if __name__ == '__main__':
    from TorchJaekwon.Util import Util
    root_path:str = Util.get_ancestor_dir_path(__file__, 1)
    device = torch.device('cuda:5')

    inst_flow = InstFlow(
        model_class_name = 'DiffusionTransformer',
        timestep_sampler = 'logit_normal',
        cfg_scale = 3.5
    )
    inst_flow.to(device)

    audio = torch.randn(8, 2, 131072)
    audio = audio.to(device)
    flow_loss = inst_flow( x_start = audio , cond = {'audio': audio})
    audio = inst_flow.infer(
        cond = {
            'text': [
                'clarinet',
                'electric guitar',
                'singer',
                'flute',
                'piano',
                'saxophone',
                'trumpet',
                'violin'
            ]
        }
    )
    print('finish', audio.shape)
    
    