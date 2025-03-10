import typing as tp

import logging, warnings
import gc
import torch
from torch import nn

from Model.StableAudioOpen.CLAP.Conditioner import Conditioner

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)
        self.finetune = False

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from Model.StableAudioOpen.CLAP.CLAPTextConditioner import clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]]) -> tp.Any:

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        # Cast audio embedding to same type as proj_out, unless proj_out is Identity

        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        return self.proj_out(audio_embedding)
