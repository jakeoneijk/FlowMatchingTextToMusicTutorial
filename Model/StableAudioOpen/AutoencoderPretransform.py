from TorchJaekwon.Util import Util
Util.set_sys_path_to_parent_dir(__file__,2)

import torch
import torch.nn as nn

from Model.StableAudioOpen.AudioAutoencoder import AudioAutoencoder

class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
    
    def tokenize(self, x):
        raise NotImplementedError
    
    def decode_tokens(self, tokens):
        raise NotImplementedError
    
class AutoencoderPretransform(Pretransform):
    def __init__(
        self,
        model = AudioAutoencoder(), 
        scale=1.0, 
        model_half=False, 
        iterate_batch=True, 
        chunked=False
    ) -> None:
        super().__init__(enable_grad=False, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()
    
    def encode(self, x, **kwargs):
        
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]
    
    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

if __name__ == '__main__':
    from TorchJaekwon.Util import UtilAudio

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    auto_encoder = AutoencoderPretransform()
    root_path:str = Util.get_ancestor_dir_path(__file__, 2)
    ckpt_path:str = f'{root_path}/CKPT/autoencoder.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    auto_encoder.load_state_dict(ckpt)
    auto_encoder.to(device)
    
    audio_path:str = f'{root_path}/Data/Dataset/MedleySolosDB/Medley-solos-DB_test-0_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav'
    audio, sr = UtilAudio.read(audio_path, mono=False)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)
    z = auto_encoder.encode(audio)
    recon_audio = auto_encoder.decode(z)
    UtilAudio.write(f'{root_path}/autoencoder_test_gt.wav', audio, 44100)
    UtilAudio.write(f'{root_path}/autoencoder_test_recon.wav', recon_audio, 44100)
    print('finish')
