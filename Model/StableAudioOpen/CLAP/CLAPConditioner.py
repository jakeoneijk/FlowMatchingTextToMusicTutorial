import typing as tp

import logging, warnings
import gc
import torch
from torch import nn

from Model.StableAudioOpen.CLAP.Conditioner import Conditioner

class CLAPConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(512, output_dim, project_out=project_out)
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, input: tp.Union[torch.Tensor, tp.List[torch.Tensor],tp.List[str]], type:tp.Literal['text', 'audio'] = 'text') -> tp.Any:
        # Fix for CLAP bug when only one text is passed
        if type == 'text':
            if len(input) == 1:
                embedding = self.model.get_text_embedding([input[0], ""], use_tensor=True)[:1, ...]
            else:
                embedding = self.model.get_text_embedding(input, use_tensor=True)
        else:
            if isinstance(input, list) or isinstance(input, tuple):
                input = torch.cat(input, dim=0)
            # Convert to mono
            mono_audios = input.mean(dim=1)
            with torch.amp.autocast(enabled=False, device_type='cuda'):
                embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        # Cast embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embedding = embedding.to(proj_out_dtype)

        return self.proj_out(embedding)

def clap_load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict
