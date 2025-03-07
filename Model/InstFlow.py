from typing import Any, Iterator, Optional, Tuple, Union
from torch import Tensor
from torch.nn.parameter import Parameter

import torch

from TorchJaekwon.Util import UtilData
from TorchJaekwon.Model.FlowMatching.FlowMatching import FlowMatching

from Model.StableAudioOpen.AutoencoderPretransform import AutoencoderPretransform

class InstFlow(FlowMatching):
    def __init__(
        self, 
        autoencoder_ckpt_path:str = 'CKPT/autoencoder.pth', 
        **kwargs 
    ) -> None:
        super().__init__(**kwargs)
        self.autoencoder = AutoencoderPretransform()
        autoencoder_ckpt = torch.load(autoencoder_ckpt_path, map_location='cpu')
        self.autoencoder.load_state_dict(autoencoder_ckpt)

        self.audio_length:int = 131072
        self.autoencoder_dim:int = 64
        self.autoencoder_downsampling_ratio:int = 2048
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            if 'autoencoder' not in name:
                yield param
    
    def load_state_dict( self, state_dict: Any ) -> None:
        super().load_state_dict(state_dict, strict=False)
    
    def state_dict(self) -> dict:
        state_dict_all = super().state_dict()
        return {k: v for k, v in state_dict_all.items() if 'autoencoder' not in k}
        
    def preprocess(
        self,
        x_start:Tensor, 
        cond:Optional[Union[dict, Tensor]] = None,
    ) -> Tuple[Tensor,Tensor]: 
        if x_start is not None:
            x_start = UtilData.fix_length(x_start, self.audio_length)
            with torch.no_grad():
                z:Tensor = self.autoencoder.encode(x_start)
        else:
            z = None
        cond = None
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
    
    def get_x_shape(self, _) -> tuple:
        batch_size:int = 1
        return (batch_size, self.autoencoder_dim, self.audio_length // self.autoencoder_downsampling_ratio)
    
    def get_unconditional_condition(self,
                                    cond:Optional[Union[dict,Tensor]] = None, 
                                    cond_shape:Optional[tuple] = None,
                                    condition_device:Optional[torch.device] = None
                                    ) -> dict:
        return {'class_label': torch.tensor([[11] for _ in range(cond["class_label"].shape[0])]).to(condition_device)}

if __name__ == '__main__':
    from TorchJaekwon.Util import Util
    root_path:str = Util.get_ancestor_dir_path(__file__, 1)

    inst_flow = InstFlow(
        model_class_name = 'DiffusionTransformer',
        timestep_sampler = 'logit_normal',
    )

    audio = torch.randn(8, 2, 131072)
    flow_loss = inst_flow( x_start = audio , is_cond_unpack = True)
    audio = inst_flow.infer()
    print('finish', audio.shape)
    
    