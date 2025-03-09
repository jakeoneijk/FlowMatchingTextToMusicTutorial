import typing as tp

import logging, warnings
import gc
import torch
from torch import nn

from Model.StableAudioOpen.CLAP.Conditioner import Conditioner

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
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
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to( non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str]) -> tp.Any:
        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)                        

            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return self.proj_out(text_embedding)

def clap_load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # removing position_ids to maintain compatibility with latest transformers update        
        #if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
        #    del state_dict["text_branch.embeddings.position_ids"]
    # for k in state_dict:
    #     if k.startswith('transformer'):
    #         v = state_dict.pop(k)
    #         state_dict['text_branch.' + k[12:]] = v
    return state_dict
